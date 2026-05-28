r"""
example_04_extreme_supercooled_ewpt.py — Extreme supercooling with a conformal model
======================================================================================

Demonstrates the CosmoTransitions pipeline for a **conformal first-order phase
transition** driven entirely by finite-temperature corrections.  The tree-level
potential is zero; the electroweak symmetry is broken by a combination of the
Coleman-Weinberg (CW) one-loop potential and thermal corrections.

        $T_n / T_c \approx 0.15$ (85% supercooling)

Model — Dark U(1)\ :sub:`X` conformal extension (KPX model)
------------------------------------------------------------
A dark scalar $h$ acquires a VEV via the CW mechanism.  A dark photon of mass
:math:`m_X = g_X \langle h \rangle` is the only new particle.

.. math::
    V_0 &= 0  \qquad \text{(conformal: no tree-level mass)} \\
    V_\text{CW}(h) &= \frac{3 g_X^4}{32\pi^2} h^4
                       \!\left(\ln\frac{h}{w} - \frac{1}{4}\right),
                       \quad w \equiv m_X / g_X \\
    V_T(h,T) &= \frac{3 T^4}{2\pi^2} J_b\!\left(\frac{g_X^2 h^2}{T^2}\right)
               - \frac{g_X^3 T}{12\pi}
                 \Bigl[(h^2+T^2)^{3/2} - h^3\Bigr]

Parameters (default): ``gX = 0.65``, ``mX = 1e7`` GeV → ``w ≈ 1.54e7`` GeV.

Key topics covered
------------------
1. Why the **default** ``TunnelingConfig()`` under-estimates :math:`T_n`
   (``V_spline_samples=100`` misses the narrow thermal barrier).
2. How ``TunnelingConfig.supercooling_preset()`` corrects this.
3. **Nucleation criterion comparison**: ``nuclCriterion="fixed_140"`` vs
   ``"cosmological"`` (:math:`S_3/T < 4\ln(M_\text{Pl}/T)`).
4. Visualization of the :math:`S_3(T)/T` curve and the two thresholds.

TunnelingConfig guide for this example
---------------------------------------
``V_spline_samples`` is the single most important parameter for conformal models:

* **Default** ``V_spline_samples=100``: the potential :math:`V(h)` is pre-sampled
  at 100 equally-spaced field values and a PCHIP spline is built.  For conformal
  models the thermal barrier is narrow (width :math:`\sim T_n \ll w`) and lies
  near :math:`h \sim T_n`.  A 100-point grid covering :math:`[0, w]` cannot
  resolve this barrier → ``tunnelFromPhase`` receives an inaccurate potential
  and finds a spuriously low :math:`T_n`.

* ``V_spline_samples=None`` (set by ``supercooling_preset()``): every call to
  :math:`V(h)` goes to ``Vtot`` directly.  The narrow thermal barrier is
  reproduced exactly → correct :math:`T_n`.

Usage::

    python examples/example_04_extreme_supercooled_ewpt.py

Output: terminal summary + figure saved as ``example_04_output.png`` in the
script directory.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cosmoTransitions import generic_potential
from cosmoTransitions.config import TunnelingConfig, cosmological_nucl_criterion
from cosmoTransitions.finiteT import Jb_spline as Jb


# ─────────────────────────────────────────────────────────────────────────────
# Conformal U(1)X model (KPX model)
# ─────────────────────────────────────────────────────────────────────────────

class ConformaU1X(generic_potential.generic_potential):
    r"""
    Conformal dark U(1)\ :sub:`X` extension — single dark scalar.

    The zero-temperature potential is **identically zero** (conformal symmetry).
    The broken-phase VEV is generated purely by the Coleman-Weinberg mechanism
    supplemented by the finite-temperature correction :math:`V_T` and the
    leading Arnold-McLerran ring resummation (Daisy term).

    Parameters
    ----------
    gX : float
        Dark gauge coupling.
    mX : float
        Dark photon mass in GeV.  Sets the VEV ``w = mX / gX``.
    """

    def init(self, gX: float = 0.65, mX: float = 1e7) -> None:
        self.Ndim = 1
        self.gX = gX
        self.mX = mX
        self.w = mX / gX  # zero-T VEV (broken-phase)
        self.renormScaleSq = self.w ** 2
        self.x_eps = 1e-7 * self.w
        self.T_eps = 1e-7 * self.w
        self.Tmax = self.w

    def forbidPhaseCrit(self, X) -> bool:
        """Restrict to h > 0 (the cubic term breaks Z2; negative branch is unphysical)."""
        return (np.asarray(X)[..., 0] < -5.0).any()

    # ── Potential pieces ──────────────────────────────────────────────────────

    def V1(self, X: np.ndarray) -> np.ndarray:
        r"""Coleman-Weinberg one-loop potential: $V_{\rm CW}=\frac{3g_X^4}{32\pi^2}h^4(\ln(h/w)-1/4)$."""
        h = np.asanyarray(X)[..., 0]
        gX, w = self.gX, self.w
        # Replace |h|=0 with tiny positive to avoid log(0)
        h_safe = np.where(np.abs(h) < 1e-100 * w, 1e-100 * w, np.abs(h))
        return 3.0 * gX ** 4 / (32.0 * np.pi ** 2) * h ** 4 * (np.log(h_safe / w) - 0.25)

    def V1T_from_X(
        self,
        X: np.ndarray,
        T: float,
        include_radiation: bool = True,
    ) -> np.ndarray:
        r"""Thermal correction + Arnold-McLerran ring resummation for the dark photon."""
        h = np.asanyarray(X)[..., 0]
        gX = self.gX
        T2 = float(T) ** 2 + 1e-100  # protect against T=0

        # Thermal correction from 3 dark-photon d.o.f.
        VT = 3.0 * float(T) ** 4 / (2.0 * np.pi ** 2) * Jb(gX ** 2 * h ** 2 / T2)

        # Arnold-McLerran ring resummation
        Vdaisy = -gX ** 3 * float(T) / (12.0 * np.pi) * (
            (h ** 2 + float(T) ** 2) ** 1.5 - h ** 3
        )

        return VT + Vdaisy

    def Vtot(
        self,
        X: np.ndarray,
        T: float,
        include_radiation: bool = False,
    ) -> np.ndarray:
        r"""Full finite-temperature effective potential :math:`V(h, T) = V_\text{CW} + V_T`."""
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        return self.V1(X) + self.V1T_from_X(X, T, include_radiation)

    def approxZeroTMin(self) -> list:
        r"""Return the zero-temperature CW minimum at :math:`h = w` as seed."""
        return [np.array([self.w])]


# ─────────────────────────────────────────────────────────────────────────────
# Main computation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    gX: float = 0.65,
    mX: float = 1e7,
    verbose: bool = True,
) -> tuple:
    """Run the extreme-supercooling pipeline and compare nucleation criteria."""
    w = mX / gX

    if verbose:
        sep = "=" * 66
        print(sep)
        print("  example_04 — Extreme supercooling: conformal U(1)X model")
        print(sep)
        print(f"  Parameters: gX = {gX},  mX = {mX:.2e} GeV")
        print(f"  VEV:        w  = mX/gX = {w:.4e} GeV")
        print(f"  V0 ≡ 0 (conformal):  symmetry broken purely by CW + finite-T")
        print()

    # ── Step 1: phase tracing ─────────────────────────────────────────────────
    if verbose:
        print("Step 1: getPhases() — tracing phases...")

    m = ConformaU1X(gX, mX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.getPhases()

    if verbose:
        print(f"  Found {len(m.phases)} phases")
        for key, ph in m.phases.items():
            x0 = ph.valAt(ph.T[0])
            x1 = ph.valAt(ph.T[-1])
            sym = "symmetric (h≈0)" if abs(x1[0]) < 0.01 * w else "broken (h≈w)"
            print(f"    Phase {key} ({sym}): "
                  f"T in [{ph.T[0]:.3e}, {ph.T[-1]:.3e}] GeV")
        print()

    # ── Step 2: critical temperature ──────────────────────────────────────────
    if verbose:
        print("Step 2: calcTcTrans() — computing T_c...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TcTrans = m.calcTcTrans()

    Tc = TcTrans[0]["Tcrit"] if TcTrans else None
    if verbose:
        if Tc is not None:
            print(f"  T_c = {Tc:.4e} GeV")
            print(f"  T_c / w = {Tc / w:.4f}  (conformal models typically T_c ~ 0.2 w)")
        else:
            print("  WARNING: T_c not found.")
        print()

    # ── Step 3A: default TunnelingConfig ──────────────────────────────────────
    if verbose:
        print("Step 3A: findAllTransitions() with TunnelingConfig() defaults")
        print("         (V_spline_samples=100 — may miss narrow thermal barrier)")

    cfg_default = TunnelingConfig()
    m_def = ConformaU1X(gX, mX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_def.getPhases()
        TnTrans_def = m_def.findAllTransitions(tunneling_config=cfg_default)

    Tn_default = TnTrans_def[0]["Tnuc"] if TnTrans_def else None

    if verbose:
        if Tn_default is not None:
            label = "(⚠ likely inaccurate — see note below)" if Tc and Tn_default < 0.05 * Tc else ""
            print(f"  T_n (default) = {Tn_default:.4e} GeV  {label}")
            if Tc:
                print(f"  T_n / T_c = {Tn_default / Tc:.4f}")
        else:
            print("  T_n (default) = not found")
        print()
        print("  NOTE: V_spline_samples=100 pre-samples V(h) on a coarse grid.")
        print("  For a conformal model the thermal barrier is narrow")
        print(f"  (width ~ T_n << w = {w:.1e} GeV) and located near h ~ T_n.")
        print("  A 100-point grid over [0, w] cannot resolve this barrier.")
        print("  The instanton solver sees a smooth, barrier-free spline and")
        print("  underestimates S₃/T → finds nucleation at a spuriously HIGH")
        print("  temperature (too early); the real barrier only manifests at lower T.")
        print()

    # ── Step 3B: supercooling_preset ──────────────────────────────────────────
    if verbose:
        print("Step 3B: findAllTransitions() with TunnelingConfig.supercooling_preset()")
        print("         (V_spline_samples=None — direct Vtot calls, resolves barrier)")

    cfg_preset = TunnelingConfig.supercooling_preset()
    if verbose:
        print(f"  Key preset params: V_spline_samples={cfg_preset.V_spline_samples!r}, "
              f"thinCutoff={cfg_preset.thinCutoff}, rmin={cfg_preset.rmin}, "
              f"T_scan_max_extend={cfg_preset.T_scan_max_extend}")
        print()

    m_preset = ConformaU1X(gX, mX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_preset.getPhases()
        TnTrans_preset = m_preset.findAllTransitions(tunneling_config=cfg_preset)

    Tn_preset = TnTrans_preset[0]["Tnuc"] if TnTrans_preset else None
    S3_preset = TnTrans_preset[0]["action"] if TnTrans_preset else None

    if verbose:
        if Tn_preset is not None:
            S3T_preset = S3_preset / Tn_preset if S3_preset else float("nan")
            print(f"  T_n (preset, fixed_140)  = {Tn_preset:.4e} GeV")
            print(f"  S_3(T_n) = {S3_preset:.4e} GeV")
            print(f"  S_3/T_n  = {S3T_preset:.4f}  (should be ~ 140)")
            if Tc:
                print(f"  T_n / T_c = {Tn_preset / Tc:.4f}  "
                      f"(supercooling = {(1 - Tn_preset / Tc) * 100:.1f}%)")
            if Tn_default is not None:
                ratio = Tn_default / Tn_preset  # > 1 means default found higher T (too early)
                direction = "higher" if ratio > 1 else "lower"
                print(f"  Compare: T_n(default) / T_n(preset) = {ratio:.2f}x")
                print(f"  → default reported T_n {ratio:.2f}x {direction} than preset")
                print(f"    (coarse V-spline flattens the barrier → S₃ underestimated → nucleation found too early)")
        else:
            print("  T_n (preset) = not found")
        print()

    # ── Step 3C: cosmological criterion ───────────────────────────────────────
    if verbose:
        print("Step 3C: nuclCriterion='cosmological' vs 'fixed_140'")
        print()
        print("  The cosmological nucleation criterion for a radiation-dominated")
        print("  Universe sets:")
        print()
        print("      Γ/H⁴ ~ 1  →  S₃/T = 4 ln(M_Pl / T)")
        print()
        if Tn_preset:
            thresh_at_Tn = 4.0 * np.log(1.22e19 / Tn_preset)
            print(f"  At T_n = {Tn_preset:.1e} GeV: threshold = {thresh_at_Tn:.2f}  (vs fixed 140)")
        print()

    # Build cosmological preset (same as preset but nuclCriterion="cosmological")
    cfg_cosmo = TunnelingConfig(
        V_spline_samples=None,
        thinCutoff=1e-4,
        rmin=1e-7,
        T_scan_extension=True,
        T_scan_max_extend=5,
        Ttol=1.0,
        nuclCriterion="cosmological",
    )

    m_cosmo = ConformaU1X(gX, mX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_cosmo.getPhases()
        TnTrans_cosmo = m_cosmo.findAllTransitions(tunneling_config=cfg_cosmo)

    Tn_cosmo = TnTrans_cosmo[0]["Tnuc"] if TnTrans_cosmo else None
    S3_cosmo = TnTrans_cosmo[0]["action"] if TnTrans_cosmo else None

    if verbose:
        if Tn_cosmo is not None:
            S3T_cosmo = S3_cosmo / Tn_cosmo if S3_cosmo else float("nan")
            thresh_cosmo = 4.0 * np.log(1.22e19 / Tn_cosmo)
            print(f"  T_n (cosmological) = {Tn_cosmo:.4e} GeV")
            print(f"  S_3/T_n            = {S3T_cosmo:.4f}  (should be ~ {thresh_cosmo:.2f})")
            if Tn_preset is not None:
                delta_Tn = (Tn_cosmo - Tn_preset) / Tn_preset * 100.0
                print(f"  Delta T_n = {delta_Tn:+.2f}%  relative to fixed_140")
            print()
            print("  Interpretation:")
            if abs(Tn_cosmo - Tn_preset) / max(Tn_preset, 1.0) < 0.01:
                print("  At this T_n the two criteria agree closely because")
                thresh_p = 4.0 * np.log(1.22e19 / Tn_preset)
                print(f"  4 ln(M_Pl/T_n) = {thresh_p:.2f} ≈ 140 at T_n = {Tn_preset:.2e} GeV.")
                print("  The difference is larger at lower T_n (EW scale ~ 100 GeV)")
                print("  where 4 ln(M_Pl/T) ≈ 157, ~12% above 140.")
            else:
                direction = "earlier (higher T)" if Tn_cosmo > Tn_preset else "later (lower T)"
                print(f"  The cosmological criterion nucleates {direction}.")
        else:
            print("  T_n (cosmological) = not found")
        print()

    # Summary
    if verbose:
        print("=" * 66)
        print("  SUMMARY")
        print("=" * 66)
        rows = [
            ("gX", f"{gX}"),
            ("mX", f"{mX:.2e} GeV"),
            ("w = mX/gX", f"{w:.4e} GeV"),
            ("T_c", f"{Tc:.4e} GeV" if Tc else "---"),
            ("T_n (default config, inaccurate)", f"{Tn_default:.4e} GeV" if Tn_default else "---"),
            ("T_n (supercooling_preset, fixed_140)", f"{Tn_preset:.4e} GeV" if Tn_preset else "---"),
            ("T_n (supercooling_preset, cosmo)",   f"{Tn_cosmo:.4e} GeV" if Tn_cosmo else "---"),
        ]
        if Tc and Tn_preset:
            rows.append(("T_n / T_c", f"{Tn_preset / Tc:.4f}  (supercooling = {(1-Tn_preset/Tc)*100:.1f}%)"))
        for label, val in rows:
            print(f"  {label:<40s}: {val}")
        print()

    return dict(
        model=m,
        TcTrans=TcTrans,
        Tc=Tc,
        Tn_default=Tn_default,
        Tn_preset=Tn_preset,
        Tn_cosmo=Tn_cosmo,
        S3_preset=S3_preset,
        S3_cosmo=S3_cosmo,
        TnTrans_preset=TnTrans_preset,
        TnTrans_cosmo=TnTrans_cosmo,
    )


# ─────────────────────────────────────────────────────────────────────────────
# S₃(T)/T curve via direct instanton calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_S3T_curve(
    gX: float = 0.65,
    mX: float = 1e7,
    Tc: float = None,
    n_T: int = 60,
    T_low_frac: float = 0.03,
) -> dict:
    r"""Scan $S_3(T)/T$ on a log-spaced grid using ``SingleFieldInstanton``."""
    from scipy.optimize import minimize_scalar
    from cosmoTransitions.tunneling1D import SingleFieldInstanton

    m = ConformaU1X(gX, mX)
    w = mX / gX

    if Tc is None:
        m.getPhases()
        Tc_list = m.calcTcTrans()
        Tc = Tc_list[0]["Tcrit"] if Tc_list else 0.2 * w

    T_low = max(T_low_frac * Tc, 1.0)
    T_arr = np.exp(np.linspace(np.log(T_low), np.log(0.998 * Tc), n_T))
    S3T_arr = np.full(n_T, np.inf)

    def _V(h, T_val):
        return float(np.asarray(m.Vtot([[h]], T_val)).flat[0])

    step = max(w * 1e-5, 1.0)

    def _dV(h, T_val):
        return (_V(h + step, T_val) - _V(h - step, T_val)) / (2.0 * step)

    for i, T_val in enumerate(T_arr):
        try:
            # Find broken minimum at T_val
            res = minimize_scalar(
                lambda h: _V(h, T_val),
                bounds=(w * 0.01, w * 2.0),
                method="bounded",
            )
            phi_brk = res.x
            if _V(phi_brk, T_val) >= _V(0.0, T_val):
                continue  # not yet a true vacuum

            V_ = lambda h: _V(h, T_val)
            dV_ = lambda h: _dV(h, T_val)

            inst = SingleFieldInstanton(phi_brk, 0.0, V_, dV_, phi_eps=1e-7)
            profile = inst.findProfile()
            action = inst.findAction(profile)

            if np.isfinite(action) and action > 0:
                S3T_arr[i] = action / T_val
        except Exception:
            pass

    return {"T_arr": T_arr, "S3T_arr": S3T_arr, "Tc": Tc}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(results: dict, save_path: str = None) -> None:
    """Three-panel figure: S3(T)/T curve, phase trajectories, and V(h,T) at T_n."""
    if save_path is None:
        save_path = os.path.join(_here, "example_04_output.png")

    Tc = results["Tc"]
    Tn_preset = results["Tn_preset"]
    Tn_cosmo = results["Tn_cosmo"]
    model = results["model"]
    gX = model.gX
    mX = model.mX
    w = mX / gX

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        rf"Conformal U(1)$_X$ — Extreme Supercooling   "
        rf"$(g_X={gX},\; m_X={mX:.0e}\ \mathrm{{GeV}})$",
        fontsize=13,
    )

    # ── Panel 1: S3(T)/T curve ────────────────────────────────────────────────
    ax = axes[0]

    print("  Computing S₃(T)/T curve for plot (may take ~30 s)...")
    curve = compute_S3T_curve(gX, mX, Tc=Tc, n_T=50)
    T_arr = curve["T_arr"]
    S3T_arr = curve["S3T_arr"]
    finite = np.isfinite(S3T_arr)

    if finite.any():
        ax.semilogy(T_arr[finite] / Tc, S3T_arr[finite], "b-", lw=2, label=r"$S_3(T)/T$")

    # Fixed-140 threshold (constant horizontal line)
    T_plot = np.linspace(T_arr[finite].min(), T_arr[finite].max(), 200) if finite.any() else np.linspace(0.01, 0.99, 200) * Tc
    ax.axhline(140.0, color="darkorange", ls="--", lw=1.5, label=r"Fixed $S_3/T = 140$")

    # Cosmological threshold (temperature-dependent)
    T_ratio_plot = T_plot / Tc
    cosmo_thresh = 4.0 * np.log(1.22e19 / T_plot)
    ax.semilogy(T_ratio_plot, cosmo_thresh, color="purple", ls="--", lw=1.5,
                label=r"Cosmo: $4\ln(M_{\rm Pl}/T)$")

    if Tn_preset and Tc:
        ax.axvline(Tn_preset / Tc, color="darkorange", ls=":", lw=1.2,
                   label=fr"$T_n^{{140}}$ = {Tn_preset:.2e} GeV")
    if Tn_cosmo and Tc and abs(Tn_cosmo - Tn_preset) / Tn_preset > 0.01:
        ax.axvline(Tn_cosmo / Tc, color="purple", ls=":", lw=1.2,
                   label=fr"$T_n^{{\rm cosmo}}$ = {Tn_cosmo:.2e} GeV")

    ax.set_xlabel(r"$T / T_c$", fontsize=12)
    ax.set_ylabel(r"$S_3(T)/T$", fontsize=12)
    ax.set_title(r"Euclidean action $S_3(T)/T$", fontsize=12)
    ax.set_ylim(50, 5000)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 2: phase trajectories ───────────────────────────────────────────
    ax = axes[1]

    colors = plt.cm.tab10.colors
    for i, (key, ph) in enumerate(model.phases.items()):
        h_vals = np.array([x[0] for x in ph.X])
        sym = abs(h_vals[-1]) < 0.01 * w
        label = r"Symmetric ($h \approx 0$)" if sym else r"Broken ($h \approx w$)"
        ls = "--" if sym else "-"
        ax.semilogy(ph.T, np.where(np.abs(h_vals) < 1.0, 1.0, np.abs(h_vals)),
                    ls, color=colors[i % 10], lw=2, label=label)

    if Tc:
        ax.axvline(Tc, color="green", ls=":", lw=1.5, label=fr"$T_c$ = {Tc:.2e} GeV")
    if Tn_preset:
        ax.axvline(Tn_preset, color="darkorange", ls=":", lw=1.5,
                   label=fr"$T_n$ = {Tn_preset:.2e} GeV")

    ax.set_xlabel("Temperature $T$ / GeV", fontsize=12)
    ax.set_ylabel(r"$h(T)$ / GeV", fontsize=12)
    ax.set_title("Phase trajectories", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 3: V(h) at T_n ─────────────────────────────────────────────────
    ax = axes[2]

    if Tn_preset:
        m_plot = ConformaU1X(gX, mX)
        T_vals_plot = [Tn_preset * f for f in [2.0, 1.0, 0.5]]
        labels_V = [r"$T = 2\,T_n$", r"$T = T_n$", r"$T = 0.5\,T_n$"]
        colors_V = ["royalblue", "darkorange", "crimson"]

        h_arr = np.linspace(0.0, min(w * 0.3, Tn_preset * 20), 400)

        for T_val, lbl, col in zip(T_vals_plot, labels_V, colors_V):
            V_arr = np.array([float(np.asarray(m_plot.Vtot([[h]], T_val)).flat[0])
                              for h in h_arr])
            # Normalize by V(0, T) for visibility
            V0_val = float(np.asarray(m_plot.Vtot([[0.0]], T_val)).flat[0])
            ax.plot(h_arr / Tn_preset, (V_arr - V0_val) / Tn_preset ** 4,
                    color=col, lw=2, label=lbl)

        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"$h / T_n$", fontsize=12)
        ax.set_ylabel(r"$[V(h,T) - V(0,T)] / T_n^4$", fontsize=12)
        ax.set_title(r"Potential $V(h,T)$ near $T_n$", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # Zoom into the region where the barrier lives
        ymin = min(0, ax.get_ylim()[0])
        ax.set_ylim(ymin * 1.2, None)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_pipeline(gX=0.65, mX=1e7, verbose=True)
    make_plots(results)
