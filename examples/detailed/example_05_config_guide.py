"""
example_05_config_guide.py
==========================
TunnelingConfig — a guided tour of all numerical parameters.

This example walks through every parameter group in ``TunnelingConfig`` with
default values and practical tuning advice, then runs several live
demonstrations on a simple single-field model.

Sections
--------
A. Reference table: all parameters, defaults, descriptions
B. Adaptive ε-tier selection (thinCutoff / rmin)
C. write_default() and from_file() — TOML round-trip
D. Nucleation criterion: 'fixed_140' vs 'cosmological' vs custom callable
E. Logging with enable_logging()
F. supercooling_preset() vs default config (recap from example_04)
"""

from __future__ import annotations

import os
import sys
import logging
import tempfile
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cosmoTransitions import generic_potential
from cosmoTransitions.config import (
    TunnelingConfig,
    enable_logging,
    fixed_140_nucl_criterion,
    cosmological_nucl_criterion,
    _epsilon_to_params,
    _PROFILE_PARAM_TIERS,
)


# ============================================================================
#   Helper model (same as example_01)
# ============================================================================

class FiniteT_SingleField(generic_potential.generic_potential):
    """Simple finite-temperature single-field potential."""

    def init(self, D=0.10, E=0.02, T0=80.0, lam=0.10):
        self.Ndim = 1
        self.D = D
        self.E = E
        self.T0 = T0
        self.lam = lam
        self.Tmax = 2.5 * T0
        self.x_eps = 0.001

    def Vtot(self, X, T, include_radiation=False):
        X = np.asanyarray(X, dtype=float)
        phi = X[..., 0]
        D, E, T0, lam = self.D, self.E, self.T0, self.lam
        T = np.asanyarray(T, dtype=float)
        return (D * (T ** 2 - T0 ** 2) * phi ** 2
                - E * T * phi ** 3
                + lam * phi ** 4)


def _run_model(cfg: TunnelingConfig, verbose: bool = False):
    """Run getPhases + findAllTransitions on the example model."""
    m = FiniteT_SingleField()
    m.getPhases()
    Tc_list = m.calcTcTrans()
    Tc = Tc_list[0]["Tcrit"] if Tc_list else None
    Tn_list = m.findAllTransitions(tunneling_config=cfg)
    Tn = Tn_list[0]["Tnuc"] if Tn_list else None
    return Tc, Tn


# ============================================================================
#  SECTION A — Parameter reference table
# ============================================================================

_SECTION_A = """
╔══════════════════════════════════════════════════════════════════════════╗
║  SECTION A — TunnelingConfig parameter reference                         ║
╚══════════════════════════════════════════════════════════════════════════╝

TunnelingConfig is a dataclass.  Instantiate with keyword overrides:

    from cosmoTransitions.config import TunnelingConfig
    cfg = TunnelingConfig()                   # all defaults
    cfg = TunnelingConfig(Ttol=1.0, nuclCriterion='cosmological')
    model.findAllTransitions(tunneling_config=cfg)

Parameter groups and their defaults
────────────────────────────────────

 ① findProfile (1D instanton ODE)
 ─────────────────────────────────
   thinCutoff = "auto"   adaptive via ε-tiers (see Section B); historical: 0.01
   rmin       = "auto"   same adaptive tier selection; historical: 1e-4
   xtol       = 1e-6     bracketing tolerance for the shooting solution
   phitol     = 1e-6     ODE integrator field-value tolerance
   rmax       = 1e4      maximum integration radius (× rscale)
   npoints    = 500      number of radial output points

 ② SplinePath / fullTunneling
 ────────────────────────────
   V_spline_samples    = 100  pre-sample V(φ) at N uniform points; None = direct Vtot()
   maxiter_fullTunneling = 20  max outer iterations (path-deform + 1D tunneling)

 ③ Path deformation
 ───────────────────
   deform_fRatioConv = 0.02  stop when |F_⊥| / |∇V| < this value
   deform_maxiter    = 500   max deformation iterations

 ④ Nucleation search
 ────────────────────
   nuclCriterion  = 'fixed_140'  'fixed_140', 'cosmological', or callable(S,T)→float
   Ttol           = 1e-3         temperature tolerance for T_n (GeV)
   maxiter_tunnel = 100          brentq / fmin iterations
   phitol_tunnel  = 1e-8         L-BFGS-B gtol for refining the true-vacuum location
   overlapAngle   = 45.0°        min angular separation to attempt both directions

 ⑤ Temperature scan extension
 ─────────────────────────────
   T_scan_extension  = True   extend scan below T_min if no nucleation found
   T_scan_max_extend = 3      each extension: T_min → 0.1 × T_min (3 covers T_n/Tc ≲ 1e-3)

 ⑥ Phase tracing
 ────────────────
   dtstart = 1e-3  initial T step (relative to T-span); reduce to avoid skipping phases
   tjump   = 1e-3  T jump when searching for next phase start

 ⑦ Adaptive retries
 ───────────────────
   enable_profile_retry = True   retry findProfile with tighter tiers on step-size crash
   max_profile_retries  = 3      maximum retry attempts
   use_adaptive_grad    = True   use adaptive finite-difference step in gradV / d2V

 ⑧ Logging
 ──────────
   log_level = None   set to logging.INFO (20), logging.DEBUG (10), etc.
   log_file  = None   write to file instead of stderr; call cfg.apply_log_level()
"""


# ============================================================================
#  SECTION B — ε-tier adaptive selection
# ============================================================================

def section_b_epsilon_tiers():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SECTION B — Adaptive ε-tier selection for thinCutoff / rmin            ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("The thin-wall ratio  ε = ΔV / ΔV_barrier  measures how close the system")
    print("is to the thin-wall limit (ε → 0) or thick-wall limit (ε → 1).")
    print()
    print("When thinCutoff='auto' (default) the code calls _estimate_epsilon() once")
    print("during findProfile and looks up the recommended tier:")
    print()
    print("  ε > 0.1      →  thick-wall tier   thinCutoff=1e-2,  rmin=1e-4")
    print("  ε > 1e-3     →  medium tier        thinCutoff=1e-3,  rmin=1e-6")
    print("  ε ≤ 1e-3     →  thin-wall tier     thinCutoff=1e-4,  rmin=1e-7")
    print()
    print("  ε         thinCutoff    rmin")
    print("  ──────────────────────────────────────────────────────────")
    for eps in [1.0, 0.5, 0.11, 0.099, 0.01, 0.0011, 0.001, 1e-4, 1e-6]:
        tc, rm = _epsilon_to_params(eps)
        print(f"  {eps:.3e}   {tc:.0e}        {rm:.0e}")
    print()
    print("  ► Thick-wall tiers match the historical CosmoTransitions defaults,")
    print("    so backward compatibility is preserved for typical EW models.")
    print("  ► Thin-wall tiers handle extreme supercooling with narrow bubbles.")
    print("  ► You can always override: TunnelingConfig(thinCutoff=1e-3, rmin=1e-6)")
    print()


# ============================================================================
#  SECTION C — TOML round-trip (write_default + from_file)
# ============================================================================

def section_c_toml_round_trip():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SECTION C — write_default() and from_file() round-trip                 ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        toml_path = os.path.join(tmp, "my_config.toml")

        # --- write default template -----------------------------------------
        TunnelingConfig.write_default(toml_path)
        print(f"  Written to: {toml_path}")

        # Read and show first 30 lines of the generated file
        with open(toml_path) as f:
            lines = f.readlines()
        print(f"  File size: {len(lines)} lines. First 25 lines:")
        print()
        for line in lines[:25]:
            print("  | " + line.rstrip())
        print("  | ...")
        print()

        # --- from_file round-trip -------------------------------------------
        cfg = TunnelingConfig.from_file(toml_path)
        print("  Loaded back with from_file():")
        print(f"    thinCutoff        = {cfg.thinCutoff!r}")
        print(f"    rmin              = {cfg.rmin!r}")
        print(f"    V_spline_samples  = {cfg.V_spline_samples!r}")
        print(f"    nuclCriterion     = {cfg.nuclCriterion!r}")
        print(f"    Ttol              = {cfg.Ttol!r}")
        print(f"    T_scan_extension  = {cfg.T_scan_extension!r}")
        print(f"    log_level         = {cfg.log_level!r}")
        print()
        print("  ► Modify the TOML file to save parameter sets for reproducible scans.")
        print("  ► V_spline_samples = \"none\"  →  V_spline_samples = None in Python.")
        print("  ► log_level = \"INFO\"  →  log_level = logging.INFO = 20.")
        print()


# ============================================================================
#  SECTION D — Nucleation criteria comparison
# ============================================================================

def section_d_nucleation_criteria():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SECTION D — Nucleation criteria: fixed_140, cosmological, custom       ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    # --- threshold table ---
    print("  The nucleation criterion determines the temperature T_n at which")
    print("  Γ/H⁴ ~ 1 (one bubble per Hubble volume per Hubble time).")
    print()
    print("  criterion     formula                                  threshold at 100 GeV")
    print("  ─────────────────────────────────────────────────────────────────────────────")
    print("  fixed_140     S₃/T = 140                              140.00")
    print("  cosmological  S₃/T = 4 ln(M_Pl/T)                    {:.2f}".format(
        4 * np.log(1.22e19 / 100)))
    print()
    print("  Threshold S₃/T as function of temperature:")
    print()
    print("  T (GeV)      fixed_140    cosmological    Δ = cosmo − fixed")
    print("  ──────────────────────────────────────────────────────────────")
    for T in [1, 10, 50, 100, 500, 1000, 1e4, 1e5, 1e6, 1e7]:
        cosmo_thresh = 4 * np.log(1.22e19 / T)
        delta = cosmo_thresh - 140.0
        print(f"  {T:.0e}        140.00       {cosmo_thresh:7.2f}         {delta:+.2f}")
    print()
    print("  ► Below T ~ 7700 GeV: cosmo threshold > 140 (LOOSER) → higher T_n, less supercooling.")
    print("  ► Above T ~ 7700 GeV: cosmo threshold < 140 (STRICTER) → lower T_n, more supercooling.")
    print("    Crossover: 4·ln(M_Pl/T) = 140 at T ≈ 7700 GeV.")
    print()

    # --- custom callable demo ---
    print("  Custom callable demo  — slightly relaxed threshold (S₃/T > 120):")
    print()
    cfg_custom = TunnelingConfig(nuclCriterion=lambda S, T: S / (T + 1e-100) - 120.0)
    fn = cfg_custom.get_nucl_criterion()
    S_test = 14000.0  # GeV
    T_test = 100.0    # GeV
    val = fn(S_test, T_test)
    threshold = S_test / T_test
    print(f"    At S₃ = {S_test:.0f} GeV, T = {T_test:.0f} GeV:")
    print(f"      S₃/T = {threshold:.1f}")
    print(f"      custom(S,T) = {val:.1f}  → {'nucleated' if val < 0 else 'not nucleated'}")
    print()

    # --- live run comparison: fixed_140 vs cosmological on single-field model ---
    print("  Running single-field model with 3 criteria  (takes ~10 s)...")
    results = {}
    for label, crit in [
        ("fixed_140",    "fixed_140"),
        ("cosmological", "cosmological"),
        ("custom_120",   lambda S, T: S / (T + 1e-100) - 120.0),
    ]:
        cfg = TunnelingConfig(nuclCriterion=crit)
        Tc, Tn = _run_model(cfg)
        results[label] = (Tc, Tn)

    Tc_ref = results["fixed_140"][0]
    print()
    print(f"  {'criterion':<18s}  {'T_n (GeV)':>10s}  {'T_c (GeV)':>10s}  {'T_n/T_c':>8s}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*8}")
    for label, (Tc, Tn) in results.items():
        ratio = f"{Tn / Tc:.4f}" if (Tc and Tn) else "N/A"
        print(f"  {label:<18s}  {Tn:10.4f}  {Tc:10.4f}  {ratio:>8}")
    print()
    print("  NOTE: For T_n ~ 80 GeV cosmological threshold ≈ 157 > 140 (LOOSER).")
    print("  nucleation occurs at a slightly HIGHER T_n (less supercooling needed).")
    print("  The difference is tiny here because the barrier forms abruptly near T_c.")
    print("  For extreme supercooling at T ~ 10^5 GeV, threshold ≈ 130 < 140 (STRICTER)")
    print("  → significantly lower T_n; see example_04 for a concrete demonstration.")
    print()


# ============================================================================
#  SECTION E — Logging demonstration
# ============================================================================

def section_e_logging():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SECTION E — Logging with enable_logging()                               ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("  cosmoTransitions emits log messages via Python's standard logging module.")
    print("  By default no handler is attached so nothing is shown.")
    print()
    print("  ① Direct function call:")
    print("      from cosmoTransitions.config import enable_logging")
    print("      enable_logging(logging.INFO)   # INFO and above → stderr")
    print("      enable_logging(logging.DEBUG)  # all messages   → stderr")
    print()
    print("  ② Via TunnelingConfig:")
    print("      cfg = TunnelingConfig(log_level=logging.INFO)")
    print("      cfg.apply_log_level()    # same effect as enable_logging(INFO)")
    print()
    print("  ③ Log to file:")
    print("      cfg = TunnelingConfig(log_level=logging.DEBUG, log_file='ct.log')")
    print("      cfg.apply_log_level()    # all DEBUG → ct.log")
    print()
    print("  ④ In TOML:")
    print("      log_level = \"INFO\"      # or 20")
    print("      log_file  = \"run.log\"   # omit for stderr")
    print()

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
        log_path = f.name

    try:
        cfg_log = TunnelingConfig(log_level=logging.INFO, log_file=log_path)
        cfg_log.apply_log_level()

        # Run a quick transition search so the logger emits something
        m = FiniteT_SingleField()
        m.getPhases()
        m.findAllTransitions(tunneling_config=cfg_log)

        # Show what was written
        with open(log_path) as flog:
            log_lines = flog.readlines()

        if log_lines:
            print(f"  Log file contains {len(log_lines)} lines.  First 15:")
            print()
            for line in log_lines[:15]:
                print("  | " + line.rstrip())
            if len(log_lines) > 15:
                print(f"  | ... ({len(log_lines) - 15} more lines)")
        else:
            print("  (No log lines were written — transitions may not have emitted INFO messages.)")
    finally:
        os.unlink(log_path)
        # Disable added file handler to keep subsequent output clean
        pkg_logger = logging.getLogger("cosmoTransitions")
        pkg_logger.handlers = [
            h for h in pkg_logger.handlers
            if not isinstance(h, logging.FileHandler)
        ]
    print()

    print("  Tip: Use logging.DEBUG during model development to trace exactly which")
    print("  temperatures are scanned and how the instanton action evolves.")
    print()


# ============================================================================
#  SECTION F — supercooling_preset() recap
# ============================================================================

def section_f_supercooling_preset():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SECTION F — supercooling_preset() vs TunnelingConfig()                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    cfg_default = TunnelingConfig()
    cfg_preset  = TunnelingConfig.supercooling_preset()

    fields_to_show = [
        "V_spline_samples", "thinCutoff", "rmin",
        "T_scan_extension", "T_scan_max_extend", "Ttol", "nuclCriterion",
    ]
    print(f"  {'field':<22s}  {'default':>15s}  {'supercooling_preset':>20s}")
    print(f"  {'─'*22}  {'─'*15}  {'─'*20}")
    for field in fields_to_show:
        dval = repr(getattr(cfg_default, field))
        pval = repr(getattr(cfg_preset,  field))
        marker = "◄" if dval != pval else ""
        print(f"  {field:<22s}  {dval:>15s}  {pval:>20s}  {marker}")
    print()
    print("  ► Highlighted (◄) fields are the ones changed by supercooling_preset().")
    print()
    print("  Key insight (see example_04 for full demo):")
    print("  ─────────────────────────────────────────────────────────────────────")
    print("  V_spline_samples=100 builds a coarse spline on [0, w] and MISSES the")
    print("  narrow thermal barrier near φ ~ T_n when T_n ≪ w.")
    print("  V_spline_samples=None calls Vtot() directly → barrier resolved correctly.")
    print()
    print("  T_scan_max_extend=5 allows scanning 5 orders of magnitude below T_c,")
    print("  enough to find T_n/T_c ~ 10⁻⁵.")
    print()
    print("  Ttol=1.0 GeV is adequate when T_n ~ 10⁴–10⁵ GeV (< 0.01% relative error)")
    print("  and avoids unnecessary brentq iterations.")
    print()

    # Quick printout: how to use the preset
    print("  Usage example:")
    print()
    print("      from cosmoTransitions.config import TunnelingConfig")
    print("      cfg = TunnelingConfig.supercooling_preset()")
    print("      # optionally switch to cosmological criterion:")
    print("      cfg.nuclCriterion = 'cosmological'")
    print("      results = model.findAllTransitions(tunneling_config=cfg)")
    print()


# ============================================================================
#  SECTION G — Visual summary plot
# ============================================================================

def section_g_plot():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SECTION G — Nucleation threshold visualization                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("  Generating threshold comparison plot ...")

    T_vals = np.logspace(-1, 19, 500)  # 0.1 GeV to 10^19 GeV
    thresh_fixed = np.full_like(T_vals, 140.0)
    thresh_cosmo = 4.0 * np.log(1.22e19 / T_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("example_05 — Nucleation criterion: fixed_140 vs cosmological", y=1.01)

    # --- left panel: threshold vs T ---
    ax = axes[0]
    ax.semilogx(T_vals, thresh_fixed, "k--", lw=2, label="fixed: $S_3/T = 140$")
    ax.semilogx(T_vals, thresh_cosmo, "C0-",  lw=2, label=r"cosmo: $4\ln(M_{Pl}/T)$")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(100, color="C1", lw=1, ls=":", label="T = 100 GeV (EW scale)")
    ax.axvline(1e6, color="C2", lw=1, ls=":", label="T = 10⁶ GeV (conformal)")
    ax.fill_between(T_vals, thresh_fixed, thresh_cosmo,
                    where=thresh_cosmo < thresh_fixed,
                    alpha=0.15, color="C0", label="cosmo stricter (lower T)")
    ax.fill_between(T_vals, thresh_fixed, thresh_cosmo,
                    where=thresh_cosmo > thresh_fixed,
                    alpha=0.15, color="C3", label="cosmo looser (higher T)")
    ax.set_xlabel("Temperature $T$ (GeV)")
    ax.set_ylabel("Nucleation threshold $S_3 / T$")
    ax.set_title("Threshold comparison")
    ax.set_ylim(50, 200)
    ax.set_xlim(1e-1, 1e19)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # --- right panel: ε-tier boundaries ---
    ax2 = axes[1]
    eps_range = np.logspace(-8, 0.5, 300)
    tc_vals = np.array([_epsilon_to_params(e)[0] for e in eps_range])
    rm_vals = np.array([_epsilon_to_params(e)[1] for e in eps_range])

    ax2.step(eps_range, tc_vals, where="post", lw=2, color="C0", label="thinCutoff")
    ax2.step(eps_range, rm_vals, where="post", lw=2, color="C1", label="rmin")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axvline(0.1,  color="gray", lw=1, ls="--")
    ax2.axvline(1e-3, color="gray", lw=1, ls="--")
    ax2.text(0.13,  1.5e-2, "thick-wall\n(ε > 0.1)",  fontsize=8, color="gray")
    ax2.text(2e-3,  1.5e-3, "medium\n(ε > 1e-3)",     fontsize=8, color="gray")
    ax2.text(1e-7,  1.5e-4, "thin-wall\n(ε ≤ 1e-3)",  fontsize=8, color="gray")
    ax2.set_xlabel(r"Thin-wall ratio $\epsilon = \Delta V / \Delta V_{\rm barrier}$")
    ax2.set_ylabel("Parameter value")
    ax2.set_title("Adaptive ε-tier selection")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "example_05_output.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path}")
    print()


# ============================================================================
#  Main
# ============================================================================

def main():
    sep = "=" * 70

    print()
    print(sep)
    print("  example_05 — TunnelingConfig: a guided tour")
    print(sep)
    print()

    # Section A is a static text block
    print(_SECTION_A)

    section_b_epsilon_tiers()
    section_c_toml_round_trip()
    section_d_nucleation_criteria()
    section_e_logging()
    section_f_supercooling_preset()
    section_g_plot()

    print(sep)
    print("  example_05 complete.")
    print(sep)


if __name__ == "__main__":
    main()
