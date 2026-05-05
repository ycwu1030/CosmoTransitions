#!/usr/bin/env python3
r"""
example_02_multifield_ewpt.py — Two-field weak FOPT + path deformation demo
=====================================================================
Phase E · Tier 1 — multi-field path deformation

This example uses the demo two-field model ``testModel1.model1`` to show:

  1. Phase structure of a two-field potential (three phases: low, mid, high)
  2. Transition finding via ``calcTcTrans`` and ``findAllTransitions``
  3. Path-deformation convergence and the role of ``TunnelingConfig.deform_fRatioConv``
  4. Visualization of the bounce path on the 2D potential surface

Usage::

    python examples/example_02_multifield_ewpt.py

    # or from Python:
    from examples.example_02_multifield_ewpt import run_pipeline, make_plots
    m, TcTrans, TnTrans = run_pipeline()
    make_plots(m, TcTrans, TnTrans)

Model
-----
The demo model ``testModel1.model1`` contains two scalar fields $\phi_1,\phi_2$
and extra bosons that provide finite-temperature corrections. Default parameters:

  - m1 = 120 GeV, m2 = 50 GeV  (tree-level masses at T=0)
  - mu = 25 GeV                (mixing mass coefficient)
  - Y1 = 0.1, Y2 = 0.15        (extra boson couplings)
  - n  = 30                   (degrees of freedom for extra bosons)

Approximate zero-T minima: [246, 246] GeV and [246, -246] GeV.

Path-deformation settings
------------------------
``TunnelingConfig.deform_fRatioConv``
    Path convergence criterion: stop when max normal force / max grad V < value.
    Smaller -> more accurate. Typical 0.02 (default); sensitive models may use 0.005.

``TunnelingConfig.deform_maxiter``
    Maximum iterations for path deformation (default 500).

``TunnelingConfig.maxiter_fullTunneling``
    Outer iterations for full tunneling (path deformation + 1D tunneling alternation). Default 20.
"""

import os
import sys

# Ensure running from examples/ or project root works
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np

from cosmoTransitions.config import TunnelingConfig

# Import testModel1 (support two path prefixes)
try:
    from examples.testModel1 import model1
except ImportError:
    from testModel1 import model1


# ============================================================
#  Helper functions
# ============================================================

def _phase_label(phases, key, Tref=None):
    """Construct a short descriptive label for a phase."""
    p = phases[key]
    if Tref is None:
        Tref = p.T[0]
    x = p.valAt(Tref)
    return f"Phase {key}: [{x[0]:.0f}, {x[1]:.0f}] GeV"


def _find_first_fopt(TnTrans):
    """Find the first FOPT with an instanton in TnTrans."""
    for t in TnTrans:
        if t.get('trantype') == 1 and 'Tnuc' in t and t.get('instanton') is not None:
            return t
    return None


# ============================================================
#  1. Full pipeline
# ============================================================

def run_pipeline(verbose: bool = True):
    """
    Build the model and run the pipeline: ``getPhases`` → ``calcTcTrans`` → ``findAllTransitions``.

    Returns
    -------
    m : model1
        Model instance with phases and transitions computed.
    TcTrans : list[dict]
        Output from ``calcTcTrans()``.
    TnTrans : list[dict]
        Output from ``findAllTransitions()``.
    """
    m = model1()

    if verbose:
        print()
        print("=" * 65)
        print("  example_02 — Two-field weak FOPT + path deformation demo")
        print("=" * 65)
        print("  Model parameters: m1=120, m2=50, mu=25, Y1=0.1, Y2=0.15, n=30")
        print()

    # ── Step 1: phase tracing ──
    if verbose:
        print("Step 1: getPhases() — tracing phases...")
    phases = m.getPhases()
    if verbose:
        print(f"  Found {len(phases)} phases")
        for key in sorted(phases.keys()):
            p = phases[key]
            x0 = p.valAt(p.T[0])
            xf = p.valAt(p.T[-1])
            print(f"    Phase {key}: T in [{p.T[0]:.1f}, {p.T[-1]:.1f}] GeV  |  "
                  f"phi(T_min)=[{x0[0]:.1f},{x0[1]:.1f}]  phi(T_max)=[{xf[0]:.1f},{xf[1]:.1f}]")
        print()

    # ── Step 2: critical temperature search ──
    if verbose:
        print("Step 2: calcTcTrans() — computing critical temperatures...")
    TcTrans = m.calcTcTrans()
    if verbose:
        if TcTrans:
            for i, tc in enumerate(TcTrans):
                ttype = 'FOPT' if tc.get('trantype') == 1 else 'second-order/continuous'
                print(f"  Transition {i}: T_c = {tc['Tcrit']:.4f} GeV  ({ttype})")
                print(f"    high-phase Phase {tc['high_phase']} -> low-phase Phase {tc['low_phase']}")
                print(f"    high_vev = [{tc['high_vev'][0]:.2f}, {tc['high_vev'][1]:.2f}] GeV")
                print(f"    low_vev  = [{tc['low_vev'][0]:.2f}, {tc['low_vev'][1]:.2f}] GeV")
        else:
            print("  No critical temperature found")
        print()

    # ── Step 3: nucleation temperature search (default config) ──
    if verbose:
        print("Step 3: findAllTransitions() — searching for nucleation temperatures (default TunnelingConfig)...")
    cfg = TunnelingConfig()
    TnTrans = m.findAllTransitions(tunneling_config=cfg)
    if verbose:
        fopt_n = sum(1 for t in TnTrans if t.get('trantype') == 1)
        print(f"  Found {len(TnTrans)} transitions, of which {fopt_n} are FOPT")
        for i, t in enumerate(TnTrans):
            ttype = t.get('trantype')
            if ttype == 1 and 'Tnuc' in t:
                Tn = t['Tnuc']
                inst = t.get('instanton')
                if inst is not None:
                    S3T = inst.action / Tn
                    print(f"  FOPT {i}: T_n = {Tn:.4f} GeV,  S3/T_n = {S3T:.4f},  fRatio = {inst.fRatio:.4e}")
                    print(f"    high_vev = [{t['high_vev'][0]:.2f}, {t['high_vev'][1]:.2f}] GeV")
                    print(f"    low_vev  = [{t['low_vev'][0]:.2f}, {t['low_vev'][1]:.2f}] GeV")
                else:
                    print(f"  FOPT {i}: T_n = {Tn:.4f} GeV  (instanton not computed)")
            elif ttype == 2:
                Tn_val = t.get('Tnuc')
                Tn_str = f"{Tn_val:.2f}" if Tn_val is not None else "?"
                print(f"  Second-order {i}: T_n = {Tn_str} GeV  (no instanton)")
        print()

    return m, TcTrans, TnTrans


# ============================================================
#  2. deform_fRatioConv comparison
# ============================================================

def compare_deform_conv(verbose: bool = True):
    """
    Compare deform_fRatioConv=0.02 (default) vs 0.005 (tight) and their
    effect on T_n and S3/T.

    Returns
    -------
    dict
        Keys are labels and values are dicts with keys ``{'Tn','S3T','fRatio'}``.
    """
    if verbose:
        print("=" * 65)
        print("  deform_fRatioConv comparison: 0.02 (default) vs 0.005 (tight)")
        print("  Note: smaller fRatioConv -> better convergence, more computing time")
        print("=" * 65)

    results = {}
    for label, frc in [("default fRatioConv=0.02", 0.02), ("tight fRatioConv=0.005", 0.005)]:
        m = model1()
        cfg = TunnelingConfig(deform_fRatioConv=frc)
        TnTrans = m.findAllTransitions(tunneling_config=cfg)

        fopt = _find_first_fopt(TnTrans)
        if fopt is not None:
            Tn = fopt['Tnuc']
            inst = fopt['instanton']
            S3T = inst.action / Tn
            fr = inst.fRatio
            results[label] = {'Tn': Tn, 'S3T': S3T, 'fRatio': fr}
            if verbose:
                print(f"  [{label}]")
                print(f"    T_n = {Tn:.6f} GeV")
                print(f"    S3/T_n = {S3T:.4f}")
                print(f"    final fRatio = {fr:.4e}  (smaller -> more converged)")
        else:
            if verbose:
                print(f"  [{label}] — No FOPT found")

    if verbose:
        if len(results) == 2:
            vals = list(results.values())
            dS3T = abs(vals[0]['S3T'] - vals[1]['S3T'])
            print(f"\n  Delta S3/T (difference) = {dS3T:.4f}  (relative {dS3T/vals[0]['S3T']*100:.2f}%)")
        print()

    return results


# ============================================================
#  3. Visualization
# ============================================================

def make_plots(m, TcTrans, TnTrans, save_path=None):
    """
    Create three subplots and save the figure:

    Left  (ax_phase)  : phase trajectories (phi1, phi2 vs T)
    Middle(ax_2d)     : 2D potential at T_n + deformed bounce path
    Right (ax_profile): bounce profile (phi1(r), phi2(r) vs r, log scale)

    Parameters
    ----------
    m : model1
    TcTrans, TnTrans : list[dict]
    save_path : str, optional
        Output PNG path. Defaults to examples/example_02_output.png.
    """
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "example_02_output.png",
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ax_phase, ax_2d, ax_profile = axes

    fig.suptitle(
        "example_02 — Two-field FOPT + path deformation (testModel1)",
        fontsize=13, y=1.01,
    )

    phases = m.phases
    fopt = _find_first_fopt(TnTrans)

    # ── (1) Phase trajectories ──────────────────────────────────────────
    phase_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    ls_styles = ['-', '--', '-.']
    for i, key in enumerate(sorted(phases.keys())):
        p = phases[key]
        T_arr = p.T
        c = phase_colors[i % len(phase_colors)]
        phi1_arr = np.array([p.valAt(T)[0] for T in T_arr])
        phi2_arr = np.array([p.valAt(T)[1] for T in T_arr])
        ax_phase.plot(T_arr, phi1_arr, '-',  color=c, lw=2,
                  label=fr"Phase {key}  $\phi_1$")
        ax_phase.plot(T_arr, phi2_arr, '--', color=c, lw=2,
                  label=fr"Phase {key}  $\phi_2$", alpha=0.7)

    # mark T_c (FOPT only)
    for tc in TcTrans:
        if tc.get('trantype') == 1:
            ax_phase.axvline(tc['Tcrit'], color='gray', ls=':', lw=1.2)
            ax_phase.text(tc['Tcrit'] + 2, -200,
                          f"T_c={tc['Tcrit']:.0f}", fontsize=7, color='gray', rotation=90, va='bottom')

    # mark T_n
    if fopt is not None:
        ax_phase.axvline(fopt['Tnuc'], color='red', ls=':', lw=1.5,
                         label=f"$T_n$ = {fopt['Tnuc']:.1f} GeV")

    # Limit x-axis to the interesting transition range (avoid empty high-T region)
    T_max_plot = max(
        (p.T[-1] for p in phases.values()),
        default=300.0,
    )
    ax_phase.set_xlim(0, min(T_max_plot, 300.0))
    ax_phase.set_xlabel("T / GeV", fontsize=12)
    ax_phase.set_ylabel(r"Field values $\phi$ / GeV", fontsize=12)
    ax_phase.set_title(r"Phase trajectories ($\phi_1, \phi_2$ vs $T$)", fontsize=12)
    ax_phase.legend(fontsize=7, ncol=1, loc='best')
    ax_phase.grid(True, alpha=0.3)

    # ── (2) 2D potential at T_n + bounce path ──────────────────────────
    if fopt is not None:
        Tn = fopt['Tnuc']
        inst = fopt['instanton']
        low_vev = fopt['low_vev']
        high_vev = fopt['high_vev']

        # Plotting range: include both vacua + margin
        margin = 100.0
        phi1_lo = min(low_vev[0], high_vev[0]) - margin
        phi1_hi = max(low_vev[0], high_vev[0]) + margin
        phi2_lo = min(low_vev[1], high_vev[1]) - margin
        phi2_hi = max(low_vev[1], high_vev[1]) + margin

        n_grid = 90
        phi1_grid = np.linspace(phi1_lo, phi1_hi, n_grid)
        phi2_grid = np.linspace(phi2_lo, phi2_hi, n_grid)
        P1, P2 = np.meshgrid(phi1_grid, phi2_grid)
        X_grid = np.stack([P1, P2], axis=-1)   # shape (n_grid, n_grid, 2)
        V_grid = m.Vtot(X_grid, Tn)

        # Clip extremes for clearer contours
        V_lo = np.percentile(V_grid, 1)
        V_hi = np.percentile(V_grid, 98)
        V_plot = np.clip(V_grid, V_lo, V_hi)

        cnt = ax_2d.contourf(P1, P2, V_plot, levels=40, cmap='RdYlBu_r')
        ax_2d.contour(P1, P2, V_plot, levels=15,
                      colors='k', linewidths=0.3, alpha=0.4)
        fig.colorbar(cnt, ax=ax_2d, shrink=0.85, label=r"$V(\phi_1,\phi_2,T_n)$ / GeV$^4$")

        # Bounce path (inst.Phi: deformed path points, shape (N, 2))
        if inst is not None:
            path_pts = np.array(inst.Phi)   # (N, 2)
            ax_2d.plot(path_pts[:, 0], path_pts[:, 1], 'w-', lw=2.0,
                       label="bounce path (deformed)", zorder=4)
            ax_2d.plot(path_pts[0, 0], path_pts[0, 1], 'ws', ms=9, zorder=5,
                       label="start (low_vev side)")
            ax_2d.plot(path_pts[-1, 0], path_pts[-1, 1], 'w^', ms=9, zorder=5,
                       label="end (high_vev side)")

        # vacuum markers
        ax_2d.plot(*low_vev, 'g*', ms=15, zorder=6,
               label=fr"low_vev (true)\n[{low_vev[0]:.0f}, {low_vev[1]:.0f}]")
        ax_2d.plot(*high_vev, 'r*', ms=15, zorder=6,
               label=fr"high_vev (meta)\n[{high_vev[0]:.0f}, {high_vev[1]:.0f}]")

        ax_2d.set_xlim(phi1_lo, phi1_hi)
        ax_2d.set_ylim(phi2_lo, phi2_hi)
        ax_2d.set_xlabel(r"$\phi_1$ / GeV", fontsize=12)
        ax_2d.set_ylabel(r"$\phi_2$ / GeV", fontsize=12)
        ax_2d.set_title(f"V($\\phi_1,\\phi_2$, $T_n$={Tn:.1f} GeV)\n+ bounce path (deformed)", fontsize=11)
        ax_2d.legend(fontsize=7, loc='upper right')
    else:
        ax_2d.text(0.5, 0.5, "No FOPT found", ha='center', va='center',
                   transform=ax_2d.transAxes, fontsize=12)
        ax_2d.set_title("2D potential surface", fontsize=12)

    # ── (3) Bounce profile ────────────────────────────────────────
    # inst.Phi: shape (N,2), each row is the field values at one radial point
    # CosmoTransitions convention:
    #   inst.Phi[0]  ≈ path_pts[0]   (low_vev side)
    #   inst.Phi[-1] ≈ path_pts[-1]  (high_vev side)
    if fopt is not None and fopt.get('instanton') is not None:
        Tn = fopt['Tnuc']
        inst = fopt['instanton']
        R = inst.profile1D.R
        Phi2D = np.array(inst.Phi)       # shape (N, 2)
        phi1_r = Phi2D[:, 0]
        phi2_r = Phi2D[:, 1]

        ax_profile.plot(R, phi1_r, 'b-',  lw=2, label=r"$\phi_1(r)$")
        ax_profile.plot(R, phi2_r, 'r--', lw=2, label=r"$\phi_2(r)$")

        # vacuum asymptotes
        low_vev = fopt['low_vev']
        high_vev = fopt['high_vev']
        ax_profile.axhline(low_vev[0],  color='b', ls=':', lw=0.9, alpha=0.6,
                   label=fr"low_vev $\phi_1$={low_vev[0]:.0f}")
        ax_profile.axhline(low_vev[1],  color='r', ls=':', lw=0.9, alpha=0.6,
                   label=fr"low_vev $\phi_2$={low_vev[1]:.0f}")
        ax_profile.axhline(high_vev[0], color='b', ls='-.', lw=0.9, alpha=0.6,
                   label=fr"high_vev $\phi_1$={high_vev[0]:.0f}")
        ax_profile.axhline(high_vev[1], color='r', ls='-.', lw=0.9, alpha=0.6,
                   label=fr"high_vev $\phi_2$={high_vev[1]:.0f}")

        # Wall thickness: radial range where |Phi2D - high_vev| drops to 90%
        dist = np.sqrt(np.sum((Phi2D - high_vev)**2, axis=-1))
        dist_max = dist.max()
        if dist_max > 0:
            mask_wall = dist > 0.1 * dist_max
            if mask_wall.any():
                R_wall = R[mask_wall]
                ax_profile.axvspan(R_wall[0], R_wall[-1], alpha=0.08, color='blue',
                                   label="transition region")

        ax_profile.set_xscale("log")
        # Truncate unphysically small r (show only region with field variation)
        if len(R) > 1:
            # find index where Phi2D deviates 1% from start
            start_val = Phi2D[0]
            dist_from_start = np.sqrt(np.sum((Phi2D - start_val)**2, axis=-1))
            total_dist = dist_from_start.max()
            if total_dist > 0:
                idx_start = np.argmax(dist_from_start > 0.01 * total_dist)
                idx_start = max(0, idx_start - 2)  # leave a small margin
                ax_profile.set_xlim(R[idx_start], R[-1] * 5)
        ax_profile.set_xlabel(r"$r$ / GeV$^{-1}$", fontsize=12)
        ax_profile.set_ylabel(r"$\phi(r)$ / GeV", fontsize=12)
        S3T = inst.action / Tn
        ax_profile.set_title(
            f"Bounce profile (T_n={Tn:.1f} GeV)\n$S_3/T_n$ = {S3T:.2f},  "
            f"fRatio = {inst.fRatio:.3e}",
            fontsize=10,
        )
        ax_profile.legend(fontsize=7, ncol=2, loc='best')
        ax_profile.grid(True, alpha=0.3, which='both')
    else:
        ax_profile.text(0.5, 0.5, "No instanton found", ha='center', va='center',
                        transform=ax_profile.transAxes, fontsize=12)
        ax_profile.set_title("Bounce profile", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")
    plt.show()


# ============================================================
#  4. Main
# ============================================================

if __name__ == "__main__":
    # Full pipeline
    m, TcTrans, TnTrans = run_pipeline(verbose=True)

    # Parameter comparison
    compare_deform_conv(verbose=True)

    # Visualization
    make_plots(m, TcTrans, TnTrans)
