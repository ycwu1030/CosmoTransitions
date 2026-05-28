r"""
example_03_supercooled_ewpt.py — Supercooled single-field first-order EWPT
===========================================================================

Demonstrates the CosmoTransitions pipeline for a **moderately supercooled**
electroweak phase transition (EWPT), where the nucleation temperature
$T_n$ lies noticeably below the critical temperature $T_c$:

        $T_n / T_c \approx 0.74$ (26% supercooling)

Model (high-temperature expansion with strong cubic term):

        $V(\phi, T) = D\,(T^2 - T_0^2)\,\phi^2 - E\,T\,\phi^3 + \tfrac{\lambda}{4}\,\phi^4$

Parameters (default):
    - D = 0.10  : thermal mass coefficient (from BSM gauge/scalar loops)
    - E = 0.09  : cubic term coefficient (enhanced by extra BSM gauge bosons)
    - T0 = 50 GeV : characteristic temperature (spinodal ≈ T0)
    - λ = 0.10  : quartic self-coupling

Compared to example_01, the larger cubic coefficient E (0.09 vs 0.02)
makes the transition significantly stronger ($\phi_c / T_c \approx 1.8$ vs 0.4)
and pushes $T_n$ well below $T_c$.

Key topics covered:
    1. Full ``getPhases → calcTcTrans → findAllTransitions`` pipeline
    2. $S_3(T)/T$ profile computed via ``SingleFieldInstanton``
    3. When to use ``TunnelingConfig.supercooling_preset()``

TunnelingConfig guide:
    The default ``TunnelingConfig()`` is sufficient for this model
    ($T_n/T_c \approx 0.74$, thick-wall regime).
    Use ``TunnelingConfig.supercooling_preset()`` for extreme supercooling
    ($T_n/T_c \lesssim 0.1$); see example_04 for a concrete comparison.

Usage::

        python examples/example_03_supercooled_ewpt.py

Output: terminal summary + figure saved as ``example_03_output.png`` in the
script directory.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from cosmoTransitions import generic_potential
from cosmoTransitions.config import TunnelingConfig, enable_logging
from cosmoTransitions.tunneling1D import SingleFieldInstanton


# ─────────────────────────────────────────────────────────────────────────────
# Physics model
# ─────────────────────────────────────────────────────────────────────────────

class SupercooledSingleField(generic_potential.generic_potential):
    r"""
    Single-field effective potential with an enhanced thermal cubic term.

    $V(\phi, T) = D\,(T^2 - T_0^2)\,\phi^2 - E\,T\,\phi^3 + \tfrac{\lambda}{4}\,\phi^4$

    Physical motivation:
        The cubic term $-E\,T\,\phi^3$ arises from the thermal resummation of
        gauge-boson and scalar-loop contributions.  When $E$ is large (here
        $E = 0.09$, close to $E_\text{max} = \sqrt{D\lambda} = 0.10$), the
        broken minimum at $T_c$ is deep ($\phi_c / T_c \approx 2E/\lambda = 1.8$)
        and the transition is strongly first-order.
    """

    def init(
        self,
        D: float = 0.10,
        E: float = 0.09,
        T0: float = 50.0,
        lam: float = 0.10,
    ) -> None:
        """
        Parameters
        ----------
        D   : thermal mass coefficient (default 0.10)
        E   : cubic term coefficient (default 0.09; E_max = sqrt(D*lam) = 0.10)
        T0  : characteristic temperature / GeV (default 50.0)
              equals the spinodal temperature T_sp where V''(0,T_sp) = 0
        lam : quartic coupling (default 0.10)
        """
        self.Ndim = 1
        self.D = D
        self.E = E
        self.T0 = T0
        self.lam = lam

        # Analytic Tc estimate: Tc = T0 / sqrt(1 - E^2/(D*lam))
        _ratio = E**2 / (D * lam)
        if _ratio >= 1.0:
            raise ValueError(
                f"E={E} >= E_max=sqrt(D*lam)={D*lam**0.5:.4f}: "
                "no first-order transition with these parameters."
            )
        self.Tc_analytic = T0 / np.sqrt(1.0 - _ratio)

        # Zero-T broken minimum: phi_v = T0 * sqrt(2D/lam)
        self.phi_v = T0 * np.sqrt(2.0 * D / lam)
        self.x_eps = 0.001

        # Tmax covers the symmetric phase up to well above T_c
        self.Tmax = 3.5 * self.Tc_analytic

    # ── Potential / energy terms ──────────────────────────────────────────────

    def Vtot(
        self,
        X: np.ndarray,
        T: ArrayLike,
        include_radiation: bool = True,
    ) -> np.ndarray:
        phi = np.asanyarray(X)[..., 0]
        T = np.asanyarray(T, dtype=float)
        return (
            self.D * (T**2 - self.T0**2) * phi**2
            - self.E * T * phi**3
            + self.lam / 4.0 * phi**4
        )

    # ── Helper / seed methods ─────────────────────────────────────────────────

    def approxZeroTMin(self) -> list:
        """Seed the broken phase at the zero-T VEV."""
        return [np.array([self.phi_v])]

    def forbidPhaseCrit(self, X) -> bool:
        """Exclude unphysical negative-field branches created by the cubic term."""
        return (np.array([X])[..., 0] < -5.0).any()


# ─────────────────────────────────────────────────────────────────────────────
# S3(T)/T curve via direct instanton calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_S3T_curve(
    model: SupercooledSingleField,
    Tc: float,
    n_T: int = 150,
    T_low_frac: float = 0.45,
) -> dict:
    r"""Scan $S_3(T)/T$ on a log-spaced grid using ``SingleFieldInstanton``."""
    # Temperature grid: log-spaced from T_low to just below Tc.
    # T_low must stay above the spinodal temperature T_sp = model.T0 (where
    # V''(0, T_sp) = 0 and the false vacuum disappears).
    T_low = max(model.T0 * 1.02, T_low_frac * Tc)
    T_arr = np.exp(np.linspace(np.log(T_low), np.log(0.998 * Tc), n_T))

    def make_V_dV(T_val: float):
        """Build scalar/array (phi) → V and dV/dphi callables at fixed T."""
        h = max(model.phi_v * 1e-5, 1e-3)

        def V_1d(phi):
            phi = np.asarray(phi, dtype=float)
            scalar = phi.ndim == 0
            phi1 = np.atleast_1d(phi)
            X = phi1[:, np.newaxis]  # shape (N, 1)
            result = np.ravel(model.Vtot(X, T_val))
            return float(result[0]) if scalar else result

        def dV_1d(phi):
            return (V_1d(phi + h) - V_1d(phi - h)) / (2.0 * h)

        return V_1d, dV_1d

    S3T_arr = np.full(n_T, np.inf)

    for i in range(n_T - 1, -1, -1):
        T_val = T_arr[i]
        V_, dV_ = make_V_dV(T_val)

        # Locate the two vacua at this T
        phi_sym = 0.0
        try:
            res = minimize_scalar(
                lambda p: V_(p),
                bounds=(model.phi_v * 0.01, model.phi_v * 4.0),
                method="bounded",
            )
            phi_brk = res.x
        except Exception:
            continue

        if V_(phi_brk) >= V_(phi_sym):
            continue  # not yet a true vacuum

        try:
            inst = SingleFieldInstanton(phi_brk, phi_sym, V_, dV_, phi_eps=1e-7)
            profile = inst.findProfile()
            action = inst.findAction(profile)
        except Exception:
            continue

        if np.isfinite(action) and action > 0:
            S3T_arr[i] = action / T_val

    finite_mask = np.isfinite(S3T_arr)
    return {
        "T_arr": T_arr,
        "S3T_arr": S3T_arr,
        "Tc": Tc,
        "min_S3T": float(np.nanmin(S3T_arr[finite_mask])) if finite_mask.any() else np.inf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main computation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(verbose: bool = True) -> tuple:
    """Run the supercooled EWPT pipeline and return (model, TcTrans, TnTrans, S3T_data)."""
    # ── Model ─────────────────────────────────────────────────────────────────
    m = SupercooledSingleField(D=0.10, E=0.09, T0=50.0, lam=0.10)

    # Analytic strength estimate: phi_c/T_c ~ 2E/lambda
    strength_analytic = 2.0 * m.E / m.lam  # ≈ 1.8 (strongly first-order)

    if verbose:
        sep = "=" * 64
        print(sep)
        print("  example_03 — Supercooled single-field first-order EWPT")
        print(sep)
        print(f"  Model: D={m.D}, E={m.E}, T0={m.T0} GeV, lambda={m.lam}")
        print(f"  E_max = sqrt(D*lam) = {np.sqrt(m.D*m.lam):.4f}  "
              f"(E/E_max = {m.E/np.sqrt(m.D*m.lam):.2f})")
        print(f"  phi_v (zero-T VEV) = {m.phi_v:.2f} GeV")
        print(f"  Analytic T_c estimate = {m.Tc_analytic:.2f} GeV")
        print(f"  Analytic phi_c/T_c = 2E/lambda = {strength_analytic:.2f}  "
              f"(strong FOPT for > 1)")
        print(f"  Spinodal T_sp = T0 = {m.T0:.0f} GeV  "
              f"(V''(0,T_sp)=0; false vacuum becomes unstable below T_sp)")
        print()

    # ── Step 1: phase tracing ─────────────────────────────────────────────────
    if verbose:
        print("Step 1: getPhases() — tracing phases...")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phases = m.getPhases()

    if verbose:
        print(f"  Found {len(phases)} phases")
        for key, ph in phases.items():
            phi_lo = ph.X[0][0]
            phi_hi = ph.X[-1][0]
            tag = "symmetric" if abs(phi_hi) < 5.0 else "broken"
            print(f"    Phase {key} ({tag}):  "
                  f"T in [{ph.T[0]:.1f}, {ph.T[-1]:.1f}] GeV,  "
                  f"phi: {phi_lo:.1f} → {phi_hi:.1f} GeV")
        print()

    # ── Step 2: critical temperature ──────────────────────────────────────────
    if verbose:
        print("Step 2: calcTcTrans() — computing T_c...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TcTrans = m.calcTcTrans()

    if verbose:
        if not TcTrans:
            print("  WARNING: no T_c found.")
        else:
            tc = TcTrans[0]
            Tc = tc["Tcrit"]
            phi_c_num = tc["low_vev"][0]
            print(f"  T_c = {Tc:.4f} GeV  (analytic: {m.Tc_analytic:.4f} GeV)")
            print(f"  phi_c(T_c) = {phi_c_num:.2f} GeV  "
                  f"(analytic 2ET_c/lambda = {2*m.E*Tc/m.lam:.2f} GeV)")
            print(f"  phi_c / T_c = {phi_c_num/Tc:.3f}  "
                  f"(>> 1 → strongly first-order)")
        print()

    # ── Step 3: nucleation temperature ────────────────────────────────────────
    if verbose:
        print("Step 3: findAllTransitions() — searching for T_n (S3/T = 140)...")
        print()
        print("  --- Run A: TunnelingConfig() defaults ---")

    cfg_default = TunnelingConfig()
    # nuclCriterion="fixed_140" by default; epsilon auto-derivation active
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TnTrans_A = m.findAllTransitions(tunneling_config=cfg_default)

    if TnTrans_A:
        Tn_A = TnTrans_A[0]["Tnuc"]
        S3_A = TnTrans_A[0]["action"]
        phi_n_A = TnTrans_A[0]["low_vev"][0]
        if verbose:
            Tc_num = TcTrans[0]["Tcrit"] if TcTrans else float("nan")
            print(f"  T_n = {Tn_A:.4f} GeV,  S3/T_n = {S3_A/Tn_A:.4f}")
            print(f"  phi_n = {phi_n_A:.2f} GeV,  phi_n/T_n = {phi_n_A/Tn_A:.3f}")
            print(f"  T_n / T_c = {Tn_A/Tc_num:.4f}  (supercooling = "
                  f"{(1-Tn_A/Tc_num)*100:.1f}%)")
    else:
        Tn_A = None
        if verbose:
            print("  WARNING: no nucleation found with default config.")

    if verbose and TnTrans_A:
        print("  NOTE: ε = ΔV/ΔV_barrier >> 1 at T_n (thick-wall regime).")
        print("  Default TunnelingConfig is sufficient; supercooling_preset()")
        print("  gives the same T_n but is needed for ε < 0.001 (see example_04).")

    TnTrans = TnTrans_A

    # ── Step 4: S3/T curve ────────────────────────────────────────────────────
    if verbose:
        print()
        print("Step 4: computing S3(T)/T profile via SingleFieldInstanton...")

    if TcTrans:
        S3T_data = compute_S3T_curve(m, TcTrans[0]["Tcrit"], n_T=150)
        if verbose:
            print(f"  Scanned {np.isfinite(S3T_data['S3T_arr']).sum()} points "
                  f"in T ∈ [{S3T_data['T_arr'][0]:.1f}, "
                  f"{S3T_data['T_arr'][-1]:.1f}] GeV")
            print(f"  min S3/T = {S3T_data['min_S3T']:.3f}  "
                  f"(at low T, where the barrier shrinks toward T_sp)")
    else:
        S3T_data = {}

    if verbose:
        print()

    return m, TcTrans, TnTrans, S3T_data


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(
    model: SupercooledSingleField,
    TcTrans: list,
    TnTrans: list,
    S3T_data: dict,
    save_path: str | None = None,
) -> None:
    """Three-panel figure: phase trajectories, S3(T)/T profile, and bounce profile."""
    if save_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(here, "example_03_output.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        r"Supercooled EWPT — "
        r"$V(\phi,T)=D(T^2-T_0^2)\phi^2-ET\phi^3+\frac{\lambda}{4}\phi^4$"
        r"  ($E=0.09$, $T_0=50$ GeV)",
        fontsize=12,
    )

    colors = plt.cm.tab10.colors
    Tc_num = TcTrans[0]["Tcrit"] if TcTrans else None
    Tn_num = TnTrans[0]["Tnuc"] if TnTrans else None
    phi_n = TnTrans[0]["low_vev"][0] if TnTrans else None

    # ── Panel 1: phase trajectories ───────────────────────────────────────────
    ax = axes[0]
    for i, (key, ph) in enumerate(model.phases.items()):
        phi_vals = np.array([X[0] for X in ph.X])
        is_sym = abs(phi_vals[-1]) < 5.0
        label = (r"symmetric ($\phi \approx 0$)" if is_sym
                 else r"broken ($\phi \approx \phi_v$)")
        ls = "--" if is_sym else "-"
        ax.plot(ph.T, phi_vals, ls, color=colors[i % 10], lw=2, label=label)

    if Tc_num is not None:
        ax.axvline(Tc_num, color="darkorange", ls=":", lw=1.5,
                   label=f"$T_c$ = {Tc_num:.1f} GeV")
        phi_c = TcTrans[0]["low_vev"][0]
        ax.plot(Tc_num, phi_c, "o", color="darkorange", ms=8, zorder=5)

    if Tn_num is not None:
        ax.axvline(Tn_num, color="crimson", ls=":", lw=1.5,
                   label=f"$T_n$ = {Tn_num:.1f} GeV")
        ax.plot(Tn_num, phi_n, "s", color="crimson", ms=8, zorder=5)

    # Spinodal T0
    ax.axvline(model.T0, color="gray", ls="-.", lw=1.0, alpha=0.7,
               label=f"$T_{{sp}}$ = $T_0$ = {model.T0:.0f} GeV")

    ax.set_xlabel("Temperature $T$ / GeV", fontsize=11)
    ax.set_ylabel(r"Field value $\phi$ / GeV", fontsize=11)
    ax.set_title("Phase trajectories", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ── Panel 2: S3(T)/T profile ──────────────────────────────────────────────
    ax = axes[1]

    if S3T_data and "T_arr" in S3T_data:
        T_arr = S3T_data["T_arr"]
        S3T_arr = S3T_data["S3T_arr"]
        finite = np.isfinite(S3T_arr)

        if finite.any():
            ax.semilogy(T_arr[finite], S3T_arr[finite], "b-", lw=2,
                        label=r"$S_3(T)/T$")

        # Nucleation criterion line
        ax.axhline(140, color="red", ls="--", lw=1.5,
                   label=r"nucleation criterion $S_3/T = 140$")

        if Tc_num is not None:
            ax.axvline(Tc_num, color="darkorange", ls=":", lw=1.5,
                       label=f"$T_c$ = {Tc_num:.1f} GeV")

        if Tn_num is not None:
            ax.axvline(Tn_num, color="crimson", ls=":", lw=1.5,
                       label=f"$T_n$ = {Tn_num:.1f} GeV")
            # Scatter point at Tn only when finite S3/T values bracket Tn
            if finite.any():
                T_fin = T_arr[finite]
                S3T_fin = S3T_arr[finite]
                if T_fin[0] <= Tn_num <= T_fin[-1]:
                    crit_at_Tn = np.interp(Tn_num, T_fin, S3T_fin)
                    ax.scatter([Tn_num], [crit_at_Tn], color="crimson",
                               s=80, zorder=5)

        ax.axvline(model.T0, color="gray", ls="-.", lw=1.0, alpha=0.7,
                   label=f"$T_{{sp}} = {model.T0:.0f}$ GeV")

    else:
        ax.text(0.5, 0.5, "$S_3/T$ not computed",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Temperature $T$ / GeV", fontsize=11)
    ax.set_ylabel(r"$S_3(T)/T$", fontsize=11)
    ax.set_title(r"$S_3(T)/T$ profile", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    if Tc_num is not None:
        ax.set_xlim(left=max(0, model.T0 * 0.5), right=Tc_num * 1.05)
    ax.set_ylim(bottom=50, top=None)

    # ── Panel 3: bounce profile ───────────────────────────────────────────────
    ax = axes[2]

    if TnTrans and TnTrans[0].get("instanton") is not None:
        inst = TnTrans[0]["instanton"]
        prof = inst.profile1D
        R = prof.R
        Phi = prof.Phi
        phi_true_val = TnTrans[0]["low_vev"][0]
        phi_false_val = TnTrans[0]["high_vev"][0]

        ax.plot(R, Phi, "b-", lw=2,
                label=fr"bounce profile ($T_n$={Tn_num:.1f} GeV)")
        ax.axhline(phi_false_val, color="gray", ls="--", lw=1, alpha=0.6,
                   label=fr"$\phi_\text{{false}}$ = {phi_false_val:.1f} GeV")
        ax.axhline(phi_true_val, color="blue", ls="--", lw=1, alpha=0.6,
                   label=fr"$\phi_\text{{true}}$ = {phi_true_val:.1f} GeV")

        # 10%–90% wall region
        phi_lo_w = phi_false_val + 0.1 * (phi_true_val - phi_false_val)
        phi_hi_w = phi_false_val + 0.9 * (phi_true_val - phi_false_val)
        mask_wall = (Phi >= phi_lo_w) & (Phi <= phi_hi_w)
        if mask_wall.any():
            R_wall = R[mask_wall]
            ax.axvspan(R_wall[0], R_wall[-1], alpha=0.08, color="blue",
                       label="wall (10%–90%)")

        ax.set_xscale("log")
        ax.set_xlabel(r"$r$ / GeV$^{-1}$", fontsize=11)
        ax.set_ylabel(r"$\phi(r)$ / GeV", fontsize=11)
        ax.set_title(f"Bounce profile ($T_n$ = {Tn_num:.1f} GeV)", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

    else:
        ax.text(
            0.5, 0.5,
            "Bounce profile not available.\n\n"
            "• Check that findAllTransitions succeeded.\n"
            "• Try enable_logging('INFO') for diagnostics.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, multialignment="left",
        )
        ax.set_title("Bounce profile", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging

    # ── Logging ───────────────────────────────────────────────────────────────
    # Uncomment ONE line below to enable CosmoTransitions diagnostic output:
    # enable_logging(level=_logging.INFO)   # phase-tracing and S3/T progress
    # enable_logging(level=_logging.DEBUG)  # full instanton solver output

    print("CosmoTransitions — example_03_supercooled_ewpt.py\n")

    model, TcTrans, TnTrans, S3T_data = run_pipeline(verbose=True)
    make_plots(model, TcTrans, TnTrans, S3T_data)
