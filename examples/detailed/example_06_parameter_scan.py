"""
example_06_parameter_scan.py
============================
How to run a parameter scan with CosmoTransitions.

This example scans the cubic coupling ``E`` in the single-field finite-T
potential, showing how the transition strength (T_n/T_c) depends on the
coupling.  It also demonstrates:

  * looping over model parameters and collecting transition data
  * graceful failure handling when ``findAllTransitions`` returns nothing
  * auto-selecting ``TunnelingConfig`` based on the output of a first-pass scan
  * a parallel-ready helper pattern using ``concurrent.futures``
  * plotting scan results

Model:

    V(φ, T) = D (T² - T₀²) φ² − E T φ³ + (λ/4) φ⁴

Scan:  E ∈ [0.01, 0.14]  with D=0.10, T0=80, λ=0.10 fixed.

Physical intuition
------------------
The cubic term ``−E T φ³`` creates a potential barrier at finite T.
Larger E → larger barrier → stronger transition → more supercooling (lower Tn/Tc).
At E → 0 the transition becomes second-order (Tn/Tc → 1).
At large E, ``supercooling_preset()`` may be needed when Tn/Tc ≲ 0.5.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cosmoTransitions import generic_potential
from cosmoTransitions.config import TunnelingConfig


# ============================================================================
#   Model
# ============================================================================

class FiniteT_SingleField(generic_potential.generic_potential):
    """Single-field finite-temperature potential (high-T expansion)."""

    def init(self, D=0.10, E=0.02, T0=80.0, lam=0.10):
        self.Ndim = 1
        self.D = D
        self.E = E
        self.T0 = T0
        self.lam = lam
        self.Tmax = 3.0 * T0     # extend Tmax to catch large-E transitions
        self.x_eps = 0.001
        # Zero-T VEV seed for phase tracing (essential for correct seeding)
        self.phi_v = T0 * np.sqrt(2.0 * D / lam)

    def approxZeroTMin(self):
        """Seed the broken phase at the approximate zero-T VEV."""
        return [np.array([self.phi_v])]

    def Vtot(self, X, T, include_radiation=False):
        phi = np.asanyarray(X, dtype=float)[..., 0]
        T   = np.asanyarray(T, dtype=float)
        return (self.D * (T**2 - self.T0**2) * phi**2
                - self.E * T * phi**3
                + self.lam / 4.0 * phi**4)

    def V1T_from_X(self, X, T, include_radiation=True):
        phi = np.asanyarray(X, dtype=float)[..., 0]
        T   = np.asanyarray(T, dtype=float)
        return self.D * T**2 * phi**2 - self.E * T * phi**3


# ============================================================================
#   Scan result container
# ============================================================================

@dataclass
class ScanPoint:
    E: float
    Tc: Optional[float]
    Tn: Optional[float]
    S3_over_Tn: Optional[float]
    elapsed: float
    note: str = ""

    @property
    def ratio(self) -> Optional[float]:
        if self.Tc and self.Tn:
            return self.Tn / self.Tc
        return None


# ============================================================================
#   Single scan point evaluation
# ============================================================================

def scan_one(E: float, cfg: TunnelingConfig, verbose: bool = True) -> ScanPoint:
    """Run the full pipeline for a single value of E."""
    t0 = time.perf_counter()
    note = ""
    try:
        m = FiniteT_SingleField(E=E)
        m.getPhases()
        Tc_list = m.calcTcTrans()
        Tc = Tc_list[0]["Tcrit"] if Tc_list else None

        Tn_list = m.findAllTransitions(tunneling_config=cfg)
        if Tn_list:
            t = Tn_list[0]
            Tn          = t.get("Tnuc")
            S3_over_Tn  = t.get("action") / Tn if (Tn and t.get("action")) else None
        else:
            Tn = S3_over_Tn = None
            note = "no transition found"
    except Exception as exc:
        Tc = Tn = S3_over_Tn = None
        note = f"error: {exc}"

    elapsed = time.perf_counter() - t0
    if verbose:
        Tc_s   = f"{Tc:.4f} GeV" if Tc  else "N/A"
        Tn_s   = f"{Tn:.4f} GeV" if Tn  else "N/A"
        ratio_s = f"{Tn/Tc:.4f}" if (Tc and Tn) else "N/A"
        print(f"    E={E:.3f}  Tc={Tc_s:>12}  Tn={Tn_s:>12}  "
              f"Tn/Tc={ratio_s}  ({elapsed:.1f} s)"
              + (f"  [{note}]" if note else ""))
    return ScanPoint(E=E, Tc=Tc, Tn=Tn, S3_over_Tn=S3_over_Tn,
                     elapsed=elapsed, note=note)


# ============================================================================
#   Parallel-ready wrapper  (drop-in replacement for scan_one above)
# ============================================================================

def run_scan(
    E_values: list[float],
    cfg: TunnelingConfig,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[ScanPoint]:
    """
    Run ``scan_one`` for each E in *E_values*.

    Parameters
    ----------
    E_values    : list of cubic coupling values to scan
    cfg         : TunnelingConfig applied to every scan point
    parallel    : if True, use ProcessPoolExecutor (faster for large scans)
    max_workers : worker processes for parallel mode

    Returns
    -------
    list[ScanPoint] in the same order as E_values
    """
    if parallel:
        # ProcessPoolExecutor requires the scan function and model to be
        # importable from a module (cannot use local classes defined at
        # __main__ scope).  Switch to sequential for self-contained examples.
        print("  [INFO] parallel=True requested but model is defined locally.")
        print("  [INFO] Falling back to sequential scan.  For module-level models,")
        print("  [INFO] replace scan_one with a top-level importable function.")
        print()
        parallel = False

    if not parallel:
        results = []
        for E in E_values:
            results.append(scan_one(E, cfg, verbose=True))
        return results

    # Parallel path (for importable functions / larger scans)
    import concurrent.futures
    import functools
    fn = functools.partial(scan_one, cfg=cfg, verbose=False)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, E): E for E in E_values}
        results_dict = {}
        for fut in concurrent.futures.as_completed(futures):
            E = futures[fut]
            try:
                results_dict[E] = fut.result()
            except Exception as exc:
                results_dict[E] = ScanPoint(E=E, Tc=None, Tn=None,
                                            S3_over_Tn=None, elapsed=0,
                                            note=f"worker error: {exc}")
    return [results_dict[E] for E in E_values]


# ============================================================================
#   Adaptive config selection
# ============================================================================

def adaptive_rescan(
    results: list[ScanPoint],
    threshold_ratio: float = 0.5,
) -> list[ScanPoint]:
    """
    Re-run scan points where Tn/Tc < threshold_ratio with supercooling_preset().

    This pattern is useful when a coarse first-pass scan reveals which
    parameter points require tighter numerics.
    """
    cfg_preset = TunnelingConfig.supercooling_preset()
    updated = list(results)
    rerun_count = 0

    for i, pt in enumerate(results):
        if pt.Tn is None:
            print(f"    Re-running E={pt.E:.3f} with supercooling_preset()  ...")
            new_pt = scan_one(pt.E, cfg_preset, verbose=True)
            updated[i] = new_pt
            rerun_count += 1
    if rerun_count == 0:
        print("    (no points needed re-running with preset)")
    return updated


# ============================================================================
#   Plotting
# ============================================================================

def make_plots(results: list[ScanPoint], out_path: str) -> None:
    valid = [r for r in results if r.Tc and r.Tn]
    if not valid:
        print("  No valid results to plot.")
        return

    E_vals  = np.array([r.E     for r in valid])
    Tc_vals = np.array([r.Tc    for r in valid])
    Tn_vals = np.array([r.Tn    for r in valid])
    ratio   = Tn_vals / Tc_vals
    S3T     = np.array([r.S3_over_Tn if r.S3_over_Tn else np.nan for r in valid])

    # Flag points where Tn/Tc < 0.5 (may benefit from supercooling_preset)
    supercooled = ratio < 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("example_06 — Parameter scan: E vs nucleation temperature", y=1.01)

    # --- Panel 1: Tc and Tn vs E -----------------------------------------
    ax = axes[0]
    ax.plot(E_vals, Tc_vals, "C0o-",  lw=2, ms=5, label="$T_c$")
    ax.plot(E_vals, Tn_vals, "C1s--", lw=2, ms=5, label="$T_n$")
    ax.fill_between(E_vals, Tn_vals, Tc_vals, alpha=0.12, color="C2",
                    label="supercooled window $T_n < T < T_c$")
    ax.set_xlabel("Cubic coupling $E$")
    ax.set_ylabel("Temperature (GeV)")
    ax.set_title("$T_c$ and $T_n$ vs $E$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Tn/Tc vs E  -------------------------------------------
    ax2 = axes[1]
    ax2.plot(E_vals[~supercooled], ratio[~supercooled], "C0o-",
             lw=2, ms=6, label="$T_n/T_c$ (default config)")
    if np.any(supercooled):
        ax2.plot(E_vals[supercooled], ratio[supercooled], "C1^",
                 ms=8, zorder=5,
                 label=r"$T_n/T_c < 0.5$ (use supercooling\_preset)")
    ax2.axhline(0.5, color="C3", ls=":", lw=1.5, label="preset threshold $T_n/T_c=0.5$")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("Cubic coupling $E$")
    ax2.set_ylabel("$T_n / T_c$")
    ax2.set_title("Supercooling depth")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: S3/Tn vs E  -------------------------------------------
    ax3 = axes[2]
    mask = ~np.isnan(S3T)
    ax3.plot(E_vals[mask], S3T[mask], "C4D-", lw=2, ms=5)
    ax3.axhline(140, color="k", ls="--", lw=1.5, label="fixed_140 threshold")
    ax3.set_xlabel("Cubic coupling $E$")
    ax3.set_ylabel("$S_3(T_n) / T_n$")
    ax3.set_title("Action at nucleation")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path}")


# ============================================================================
#   Main
# ============================================================================

def main():
    sep = "=" * 70
    print()
    print(sep)
    print("  example_06 — Parameter scan: E (cubic coupling) vs Tn")
    print(sep)
    print()
    print("  Model:  V(φ,T) = D(T²−T₀²)φ² − E·T·φ³ + (λ/4)φ⁴")
    print("  Fixed:  D=0.10, T0=80 GeV, λ=0.10")
    print()

    # ── Scan setup -----------------------------------------------------------
    E_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
    # TunnelingConfig with T_scan_extension enabled for robustness
    cfg = TunnelingConfig(T_scan_extension=True, T_scan_max_extend=3)

    E_str = ", ".join(str(e) for e in E_values)
    print(f"  Scan:   E \u2208 [{E_str}]")
    print()
    print(f"  Scanning {len(E_values)} values of E  (sequential, ~10 s each)...")
    print(f"  Config:  T_scan_extension=True, T_scan_max_extend=3, nuclCriterion='fixed_140'")
    print()

    t_total = time.perf_counter()
    results = run_scan(E_values, cfg)
    elapsed_total = time.perf_counter() - t_total

    print()
    print(f"  Scan complete in {elapsed_total:.1f} s")
    print()

    # ── Analytical Tc check + adaptive re-scan ----------------------------------
    import math
    E_crit = math.sqrt(0.10 * 0.10)  # sqrt(D * lam) = sqrt(0.10 * 0.10) = 0.10
    failed_rescuable = [r for r in results
                        if r.Tn is None and r.E < E_crit]
    failed_no_tc     = [r for r in results
                        if r.Tc is None and r.E >= E_crit]

    if failed_no_tc:
        print(f"  NOTE: For E ≥ E_crit = {E_crit:.3f} (= √(D·λ)),")
        print(f"  Tc = T₀ / √(1 - E²/(Dλ)) diverges → no finite critical temperature.")
        print(f"  findAllTransitions cannot bound its T scan without Tc.")
        print(f"  These {len(failed_no_tc)} point(s) require a different scan strategy (e.g.,")
        print("  fixing T_scan_Tmax or using a spinodal temperature estimate).")
        print()

    if failed_rescuable:
        print(f"  {len(failed_rescuable)} scan point(s) returned no transition with default config.")
        print("  Attempting adaptive re-scan with supercooling_preset() ...")
        print()
        results = adaptive_rescan(results, threshold_ratio=0.5)
        print()

    # ── Summary table --------------------------------------------------------
    print(sep)
    print("  SCAN RESULTS")
    print(sep)
    print()
    print(f"  {'E':>6}  {'Tc (GeV)':>10}  {'Tn (GeV)':>10}  "
          f"{'Tn/Tc':>7}  {'S3/Tn':>7}  {'time (s)':>8}  note")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*8}  ──────")
    for r in results:
        Tc_s  = f"{r.Tc:.4f}"  if r.Tc  else "    N/A"
        Tn_s  = f"{r.Tn:.4f}"  if r.Tn  else "    N/A"
        rat_s = f"{r.ratio:.4f}" if r.ratio else "    N/A"
        S3_s  = f"{r.S3_over_Tn:.2f}" if r.S3_over_Tn else "   N/A"
        flag  = " ◄ supercooled" if (r.ratio and r.ratio < 0.5) else ""
        print(f"  {r.E:6.3f}  {Tc_s:>10}  {Tn_s:>10}  "
              f"{rat_s:>7}  {S3_s:>7}  {r.elapsed:8.2f}  {r.note}{flag}")
    print()

    # ── Physical interpretation -----------------------------------------------
    print("  Physical interpretation:")
    print("  ─────────────────────────────────────────────────────────────────────")
    print("  Analytical Tc for this model:  Tc = T₀ / √(1 - E²/(Dλ))")
    print("  At D=0.10, λ=0.10:  Tc = 80 / √(1 - 100E²)")
    print("  Critical point at E = √(Dλ) = 0.10 → Tc → ∞.")
    print("  For E ≥ 0.10: no finite Tc → the two minima never degenerate.")
    print()
    print("  Transition strength:  larger E → larger cubic barrier")
    print("  → more supercooling (lower Tn/Tc) → stronger GW signal.")
    print()
    # Compare analytical Tc with numerical:
    import math
    print("  Analytical vs numerical Tc comparison:")
    print(f"  {'E':>6}  {'Tc_analytic (GeV)':>20}  {'Tc_numeric (GeV)':>18}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*18}")
    for r in results:
        denom = 1.0 - r.E**2 / (0.10 * 0.10)
        Tc_a = 80.0 / math.sqrt(denom) if denom > 0 else float('inf')
        Tc_a_str = f"{Tc_a:.4f}" if Tc_a < 1e6 else "   ∞ (E ≥ E_crit)"
        Tc_n_str = f"{r.Tc:.4f}" if r.Tc else "      N/A"
        print(f"  {r.E:6.3f}  {Tc_a_str:>20}  {Tc_n_str:>18}")
    print()

    ratios = [(r.E, r.ratio) for r in results if r.ratio is not None]
    if ratios:
        E_min, ratio_min = min(ratios, key=lambda x: x[1])
        E_max, ratio_max = max(ratios, key=lambda x: x[1])
        print(f"  Strongest supercooling: E={E_min:.2f}, Tn/Tc={ratio_min:.4f}")
        print(f"  Weakest  supercooling: E={E_max:.2f}, Tn/Tc={ratio_max:.4f}")
    print()

    print("  Config selection guide based on Tn/Tc:")
    print("    Tn/Tc > 0.8  → TunnelingConfig() defaults (fast)")
    print("    0.5 < Tn/Tc ≤ 0.8  → TunnelingConfig(T_scan_extension=True)")
    print("    Tn/Tc ≤ 0.5  → TunnelingConfig.supercooling_preset()  (required)")
    print("    See example_04 for Tn/Tc ~ 0.03 (extreme supercooling)")
    print()

    # ── Parallel scan code snippet -------------------------------------------
    print("  Parallel scan pattern (for module-level scan functions):")
    print()
    print("    import concurrent.futures, functools")
    print("    fn = functools.partial(scan_one, cfg=TunnelingConfig())")
    print("    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:")
    print("        results = list(pool.map(fn, E_values))")
    print()
    print("  NOTE: The scan function and model class must be importable from a")
    print("  module (not defined in __main__) for ProcessPoolExecutor to work.")
    print()

    # ── Plot -----------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "example_06_output.png")
    make_plots(results, out_path)

    print()
    print(sep)
    print("  example_06 complete.")
    print(sep)


if __name__ == "__main__":
    main()
