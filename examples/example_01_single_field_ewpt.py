r"""
example_01_single_field_ewpt.py — Single-field weak first-order EWPT example
=====================================================================

Demonstrates the full CosmoTransitions pipeline for a single-field weak
electroweak first-order phase transition (EWPT):

        getPhases()  →  calcTcTrans()  →  findAllTransitions()

Model (high-temperature expansion):

        $V(\phi, T) = D\,(T^2 - T_0^2)\,\phi^2 - E\,T\,\phi^3 + (\lambda/4)\,\phi^4$

Parameters:
    - D : thermal mass coefficient
    - E : cubic term coefficient
    - T0: characteristic temperature
    - \lambda : quartic self-coupling

Outputs:
    - $T_c$: critical temperature (degenerate minima)
    - $T_n$: nucleation temperature (criterion $S_3/T \approx 140$ by default)
    - $S_3/T_n$: Euclidean action at nucleation
    - $v_n/T_n$: order parameter at nucleation

Usage::

        python examples/example_01_single_field_ewpt.py

Output: terminal summary + figure saved as ``example_01_output.png`` in the
script directory.

Notes on TunnelingConfig:
    This example uses ``TunnelingConfig()`` default (``nuclCriterion="fixed_140"``).
    For deep supercooling ($T_n/T_c \ll 1$) use::

        from cosmoTransitions.config import TunnelingConfig
        cfg = TunnelingConfig.supercooling_preset()
        model.findAllTransitions(tunneling_config=cfg)

    Or use the cosmological nucleation criterion::

        cfg = TunnelingConfig(nuclCriterion="cosmological")
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from cosmoTransitions import generic_potential
from cosmoTransitions.config import TunnelingConfig, enable_logging


# ─────────────────────────────────────────────────────────────────────────────
# Physics model
# ─────────────────────────────────────────────────────────────────────────────

class FiniteT_SingleField(generic_potential.generic_potential):
    r"""
    Single-field finite-temperature effective potential (high-T expansion).

    $V(\phi, T) = D\,(T^2 - T_0^2)\,\phi^2 - E\,T\,\phi^3 + (\lambda/4)\,\phi^4$

    Typical phase structure for default parameters (D=0.1, E=0.02,
    T0=80, \lambda=0.1):
        - $T > T_c$: symmetric phase ($\phi \approx 0$)
        - $T_n < T < T_c$: two coexisting minima (requires tunneling)
        - $T < T_n$: broken phase (true vacuum)

    Implementation note:
        This example overrides ``Vtot`` and ``V1T_from_X`` and omits Coleman-
        Weinberg loop corrections for pedagogical clarity. For a full QFT
        treatment include field-dependent masses via ``boson_massSq``/
        ``fermion_massSq`` and let the base class compute loop and thermal
        corrections.
    """

    def init(
        self,
        D: float = 0.10,
        E: float = 0.02,
        T0: float = 80.0,
        lam: float = 0.10,
    ) -> None:
        """
        Parameters
        ----------
        D   : thermal mass coefficient (default 0.10)
        E   : cubic term coefficient (default 0.02)
        T0  : characteristic temperature / GeV (default 80.0)
        lam : quartic coupling (default 0.10)
        """
        self.Ndim = 1  # single-field model

        self.D = D
        self.E = E
        self.T0 = T0
        self.lam = lam

        # Zero-T VEV: from V0'(phi)=0 -> phi_v = T0 * sqrt(2D/lam)
        self.phi_v = T0 * np.sqrt(2.0 * D / lam)

        # renormScaleSq is a placeholder for the base-class interface
        # (this model does not compute Coleman-Weinberg corrections)
        self.renormScaleSq = self.phi_v**2

        # Tmax: high enough to cover the symmetric phase (T = 2.5 T0)
        self.Tmax = 2.5 * T0

        # Finite-difference step (field scale ~ phi_v). Default 0.001.
        self.x_eps = 0.001

    # ── Potential / energy terms ──────────────────────────────────────────────

    def V0(self, X: np.ndarray) -> np.ndarray:
        """
        Zero-temperature tree-level potential (equivalent to Vtot(X, T=0)).
        Used internally by d2V / massSqMatrix.
        """
        phi = np.asanyarray(X)[..., 0]
        # At T=0: $V = -D\,T_0^2\,\phi^2 + (\lambda/4)\,\phi^4$
        return -self.D * self.T0**2 * phi**2 + self.lam / 4.0 * phi**4

    def Vtot(
        self,
        X: np.ndarray,
        T: float,
        include_radiation: bool = True,
    ) -> np.ndarray:
        """
        Full finite-temperature effective potential V(φ, T).

        Override base class to use the high-T analytic form and omit
        field-independent radiation constants (these do not affect
        relative phase stability or transition searches).
        """
        phi = np.asanyarray(X)[..., 0]
        T = np.asanyarray(T, dtype=float)
        return (
            self.D * (T**2 - self.T0**2) * phi**2
            - self.E * T * phi**3
            + self.lam / 4.0 * phi**4
        )

    def V1T_from_X(
        self,
        X: np.ndarray,
        T: float,
        include_radiation: bool = True,
    ) -> np.ndarray:
        """
        Temperature-dependent part of the potential: V_T(φ, T) = D T^2 φ^2 - E T φ^3

        Called by ``dgradV_dT`` to provide the temperature gradient for
        the phase-tracing ODE.
        """
        phi = np.asanyarray(X)[..., 0]
        T = np.asanyarray(T, dtype=float)
        return self.D * T**2 * phi**2 - self.E * T * phi**3

    # ── Helpers / utilities ──────────────────────────────────────────────────

    def approxZeroTMin(self) -> list:
        """
        Provide approximate zero-temperature minima as seeds for getPhases().

        Returns
        -------
        list of array_like
            Each element is an array of field values for a candidate minimum.
        """
        # Approximate zero-temperature minimum for the broken (positive VEV) branch.
        # The spurious negative branch (phi ≈ -phi_v) is excluded by forbidPhaseCrit.
        return [np.array([self.phi_v])]

    def forbidPhaseCrit(self, X) -> bool:
        """
        Exclude physically irrelevant negative-field minima (the cubic term
        breaks Z2 symmetry).

        Return True when phi < -5 GeV to tell the tracer to drop that
        candidate. The threshold -5 is much smaller than the positive VEV
        (~113 GeV) but excludes the spurious -phi_v branch.
        """
        return (np.array([X])[..., 0] < -5.0).any()


# ─────────────────────────────────────────────────────────────────────────────
# Computation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ewpt_pipeline(
    verbose: bool = True,
) -> tuple:
    """
    Run the full EWPT computation pipeline.

    The pipeline performs three steps:
      1. ``getPhases()``          — trace phases (phi vs T)
      2. ``calcTcTrans()``        — compute critical temperatures
      3. ``findAllTransitions()`` — search for nucleation temperatures (S3/T~140)

    Parameters
    ----------
    verbose : bool
        If True, print progress and results to the terminal.

    Returns
    -------
    model : FiniteT_SingleField
        Completed model (TcTrans, TnTrans fields populated).
    TcTrans : list of dict
    TnTrans : list of dict
    """
    # ── Model initialization ───────────────────────────────────────────────────
    m = FiniteT_SingleField(D=0.10, E=0.02, T0=80.0, lam=0.10)

    # Analytic estimates (for comparison with numerical results)
    # From degenerate minima conditions V(phi_c,T_c)=0 and V'(phi_c,T_c)=0:
    #   phi_c = 2E/λ * T_c, leading to T_c^2 = D*T0^2/(D - 0.4E + 0.04λ)
    denom = m.D - 0.4 * m.E + 0.04 * m.lam
    Tc_analytic = m.T0 * np.sqrt(m.D / max(denom, 1e-30))
    phi_c_analytic = 2.0 * m.E / m.lam * Tc_analytic
    strength_analytic = 2.0 * m.E / m.lam  # approximate v_c / T_c

    if verbose:
        sep = "=" * 62
        print(sep)
        print("  example_01 — Single-field weak first-order EWPT")
        print(sep)
        print(f"  Model parameters: D={m.D}, E={m.E}, T0={m.T0} GeV, lambda={m.lam}")
        print(f"  Zero-T VEV phi_v = {m.phi_v:.3f} GeV")
        print(f"  Analytic estimate  T_c ≈ {Tc_analytic:.2f} GeV, "
              f"phi_c/T_c ≈ {strength_analytic:.3f}")
        print(f"  Tmax = {m.Tmax:.1f} GeV\n")

    # ── Step 1: phase tracing ───────────────────────────────────────────────────
    if verbose:
        print("Step 1: getPhases() — tracing phases...")

    phases = m.getPhases()

    if verbose:
        print(f"  Found {len(phases)} phases")
        for key, ph in phases.items():
            phi_lo = ph.X[0][0]
            phi_hi = ph.X[-1][0]
            tag = "symmetric" if abs(phi_hi) < 10 else "broken"
            print(f"    Phase {key} ({tag}): "
                  f"T in [{ph.T[0]:.2f}, {ph.T[-1]:.2f}] GeV, "
                  f"phi: {phi_lo:.2f} → {phi_hi:.2f} GeV")
        print()

    # ── Step 2: critical temperature search ───────────────────────────────────
    if verbose:
        print("Step 2: calcTcTrans() — computing critical temperatures...")

    TcTrans = m.calcTcTrans()

    if verbose:
        if not TcTrans:
            print("  ⚠ No critical temperature found. Check phases or lower Tmax.")
        else:
            for tc in TcTrans:
                phi_lo = tc['low_vev'][0]
                phi_hi = tc['high_vev'][0]
                print(f"  T_c = {tc['Tcrit']:.4f} GeV  "
                      f"(analytic estimate {Tc_analytic:.4f} GeV)")
                print(f"    high-T phase VEV = {phi_hi:.4f} GeV (symmetric)")
                print(f"    low-T phase VEV = {phi_lo:.4f} GeV (broken)")
                print(f"    phi_low / T_c = {phi_lo / tc['Tcrit']:.4f}  "
                      f"(analytic {phi_c_analytic / Tc_analytic:.4f})")
                print(f"    Delta_rho = {tc['Delta_rho']:.4e} GeV^4")
        print()

    # ── Step 3: nucleation temperature search ─────────────────────────────────
    if verbose:
        print("Step 3: findAllTransitions() — searching for nucleation temperatures (S3/T=140)...")

    # TunnelingConfig() defaults to nuclCriterion="fixed_140".
    # For deep supercooling use TunnelingConfig.supercooling_preset().
    cfg = TunnelingConfig()

    TnTrans = m.findAllTransitions(tunneling_config=cfg)

    if verbose:
        if not TnTrans:
            print("  ⚠ No nucleation temperature found. Possible reasons:")
            print("    1. S3/T > 140 for all T < T_c (barrier too high)")
            print("    2. Temperature search range too narrow (increase Tmax or lower x_eps)")
            print("    Tip: enable_logging('DEBUG') for detailed output.")
        else:
            for tn in TnTrans:
                phi_n = tn['low_vev'][0]
                Tn = tn['Tnuc']
                S3 = tn['action']
                Tc_num = (tn['crit_trans']['Tcrit']
                          if tn.get('crit_trans') is not None else float('nan'))
                print(f"  T_n = {Tn:.4f} GeV")
                print(f"    S3(T_n) = {S3:.4f} GeV, S3/T_n = {S3/Tn:.4f}")
                print(f"    low-T (broken) VEV phi_n = {phi_n:.4f} GeV")
                print(f"    phi_n / T_n = {phi_n / Tn:.4f}  (order parameter)")
                print(f"    T_n / T_c = {Tn / Tc_num:.4f}" if np.isfinite(Tc_num)
                      else "    T_n / T_c = N/A (T_c not found)")
                print(f"    alpha_GW (GW strength) = {tn['alpha_GW']:.4e}")
        print()

    return m, TcTrans, TnTrans


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(
    model: FiniteT_SingleField,
    TcTrans: list,
    TnTrans: list,
    save_path: str | None = None,
) -> None:
    r"""
    Prepare and save a two-panel figure.

    Left: phase trajectories $\phi(T)$ with $T_c$ and $T_n$ marked.
    Right: bounce profile $\phi(r)$ at nucleation (log radial scale).

    Parameters
    ----------
    model : FiniteT_SingleField
        Completed model instance (with TcTrans/TnTrans filled).
    TcTrans : list
        Output from ``calcTcTrans()``.
    TnTrans : list
        Output from ``findAllTransitions()``.
    save_path : str or None
        Path to save the figure. If None, saves to the script directory.
    """
    if save_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(here, "example_01_output.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"Single-field weak first-order EWPT — $V(\phi,T)=D(T^2-T_0^2)\phi^2-ET\phi^3+\frac{\lambda}{4}\phi^4$",
        fontsize=13,
    )

    # ── Left panel: phase trajectories ─────────────────────────────────────────
    ax = axes[0]
    colors = plt.cm.tab10.colors

    for i, (key, ph) in enumerate(model.phases.items()):
        phi_vals = np.array([X[0] for X in ph.X])
        # classify symmetric vs broken by high-T endpoint
        is_sym = abs(phi_vals[-1]) < 10.0
        label = (r"symmetric phase ($\phi\approx0$)" if is_sym
             else r"broken phase ($\phi\approx\phi_v$)")
        ls = "--" if is_sym else "-"
        ax.plot(ph.T, phi_vals, ls, color=colors[i % 10], lw=2, label=label)

    # Mark T_c (first-order transitions only)
    if TcTrans:
        Tc = TcTrans[0]['Tcrit']
        phi_c = TcTrans[0]['low_vev'][0]
        ax.axvline(Tc, color="darkorange", ls=":", lw=1.5,
                   label=f"$T_c$ = {Tc:.1f} GeV")
        ax.plot(Tc, phi_c, "o", color="darkorange", ms=8, zorder=5)

    # Mark T_n
    if TnTrans:
        Tn = TnTrans[0]['Tnuc']
        phi_n = TnTrans[0]['low_vev'][0]
        ax.axvline(Tn, color="crimson", ls=":", lw=1.5,
                   label=f"$T_n$ = {Tn:.1f} GeV")
        ax.plot(Tn, phi_n, "s", color="crimson", ms=8, zorder=5)

    ax.set_xlabel("Temperature $T$ / GeV", fontsize=12)
    ax.set_ylabel(r"Field value $\phi$ / GeV", fontsize=12)
    ax.set_title("Phase trajectories", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ── Right panel: bounce profile ──────────────────────────────────────────
    ax = axes[1]

    if TnTrans and TnTrans[0].get("instanton") is not None:
        inst = TnTrans[0]["instanton"]
        Tn = TnTrans[0]["Tnuc"]
        # low_vev = broken-phase VEV (true vacuum, inside the bubble)
        # high_vev = symmetric-phase VEV (metastable, outside the bubble)
        phi_true = TnTrans[0]["low_vev"][0]   # ≈ φᵥ(T_n)
        phi_meta = TnTrans[0]["high_vev"][0]  # ≈ 0

        # profile1D.Phi: runs from true vacuum (r~0, bubble center)
        # to the metastable vacuum (r->infty, outside the bubble)
        prof = inst.profile1D
        R = prof.R        # radial coordinate in GeV^-1
        # CosmoTransitions convention:
        #   Phi[0]  ≈ phi_meta (metastable / false vacuum, small r)
        #   Phi[-1] ≈ phi_abs  (true vacuum, large r)
        Phi = prof.Phi

        ax.plot(R, Phi, "b-", lw=2, label=fr"bounce profile ($T_n$={Tn:.1f} GeV)")
        ax.axhline(phi_meta, color="gray", ls="--", lw=1, alpha=0.6,
               label=fr"metastable $\phi_{{meta}}$ = {phi_meta:.1f} GeV")
        ax.axhline(phi_true, color="blue", ls="--", lw=1, alpha=0.6,
               label=fr"true vacuum $\phi_{{true}}$ = {phi_true:.1f} GeV")

        # Wall thickness region (radial range where φ is 10%–90% between vacua)
        phi_wall_lo = phi_meta + 0.1 * (phi_true - phi_meta)
        phi_wall_hi = phi_meta + 0.9 * (phi_true - phi_meta)
        mask_wall = (Phi >= phi_wall_lo) & (Phi <= phi_wall_hi)
        if mask_wall.any():
            R_wall = R[mask_wall]
            ax.axvspan(R_wall[0], R_wall[-1], alpha=0.08, color="blue",
                       label="wall region (10%–90%)")

        ax.set_xscale("log")
        ax.set_xlabel("$r$ / GeV$^{-1}$", fontsize=12)
        ax.set_ylabel(r"$\phi(r)$ / GeV", fontsize=12)
        ax.set_title(f"Bounce profile ($T_n$ = {Tn:.2f} GeV)", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

    else:
        ax.text(
            0.5, 0.5,
            "Nucleation temperature not found.\n\n"
            "Suggestions:\n"
            "• enable_logging('INFO') to inspect the calculation\n"
            "• Increase E to make the transition stronger (e.g. E=0.03)\n"
            "• Or use TunnelingConfig.supercooling_preset()",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, multialignment="left",
        )
        ax.set_title("Bounce profile (not computed)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging

    # ── Logging configuration ──────────────────────────────────────────────
    # By default logs go to stderr.  Set LOG_FILE to a path to save them to a
    # file instead (the file is created / appended in the same directory as
    # this script).  Set LOG_FILE = None to keep logging on screen.
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "example_01.log")

    # Uncomment ONE of the lines below to enable logging:
    # enable_logging(level=_logging.INFO)                        # INFO  → stderr
    # enable_logging(level=_logging.DEBUG)                       # DEBUG → stderr
    # enable_logging(level=_logging.INFO,  log_file=LOG_FILE)    # INFO  → file
    # enable_logging(level=_logging.DEBUG, log_file=LOG_FILE)    # DEBUG → file

    print("CosmoTransitions — example_01_single_field_ewpt.py\n")

    model, TcTrans, TnTrans = run_ewpt_pipeline(verbose=True)
    make_plots(model, TcTrans, TnTrans)
