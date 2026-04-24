"""
tests/conftest.py
-----------------
Shared pytest fixtures for CosmoTransitions regression tests (Phase 0 baseline).

All session-scoped fixtures are computed once per test run and cached, to avoid
re-running expensive phase-tracing computations multiple times.
"""
import sys
import os
import numpy as np
import pytest

# Ensure project root is on sys.path regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Model fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model1_instance():
    """examples/testModel1.py with default parameters."""
    from examples.testModel1 import model1
    return model1()


@pytest.fixture(scope="session")
def model1_tc_transitions(model1_instance):
    """
    model1.calcTcTrans() result, session-cached (~30 s on first run).

    Baseline values (locked 2026-04-24):
      Tc[0] = 222.94942912744762
      Tc[1] = 109.40840756818058
    """
    return model1_instance.calcTcTrans()


@pytest.fixture(scope="session")
def model1_phases(model1_instance):
    """model1.getPhases() result, session-cached."""
    return model1_instance.getPhases()


# ── Simple 1D potentials ──────────────────────────────────────────────────────

@pytest.fixture
def quartic_well_delta03():
    """
    Asymmetric double-well:  V(φ) = (φ²−1)² − 0.3·φ

    Baseline values (locked 2026-04-24, conda base / scipy 1.13.1):
      phi_absMin  ≈  1.03558030   (true minimum, lower energy)
      phi_metaMin ≈ -0.96014959   (metastable minimum)
      S3 (O3, α=2) ≈ 302.69415144
      S4 (O4, α=3) ≈ 7532.45798088
    """
    delta = 0.3
    from scipy.optimize import minimize_scalar

    def V(phi, d=delta):
        return (phi**2 - 1.0)**2 - d * phi

    def dV(phi, d=delta):
        return 4.0 * phi * (phi**2 - 1.0) - d

    res_abs  = minimize_scalar(V, bounds=( 0.5, 1.5), method="bounded")
    res_meta = minimize_scalar(V, bounds=(-1.5,-0.5), method="bounded")
    return dict(
        V=V, dV=dV,
        phi_absMin=res_abs.x,
        phi_metaMin=res_meta.x,
        V_absMin=res_abs.fun,
        V_metaMin=res_meta.fun,
        S3_ref=302.69415144,
        S4_ref=7532.45798088,
    )


@pytest.fixture
def quartic_well_delta05():
    """
    Asymmetric double-well:  V(φ) = (φ²−1)² − 0.5·φ

    Baseline values (locked 2026-04-24):
      S3 (O3) ≈ 102.92078560
      S4 (O4) ≈ 1525.09307116
    """
    delta = 0.5
    from scipy.optimize import minimize_scalar

    def V(phi, d=delta):
        return (phi**2 - 1.0)**2 - d * phi

    def dV(phi, d=delta):
        return 4.0 * phi * (phi**2 - 1.0) - d

    res_abs  = minimize_scalar(V, bounds=( 0.5, 1.5), method="bounded")
    res_meta = minimize_scalar(V, bounds=(-1.5,-0.5), method="bounded")
    return dict(
        V=V, dV=dV,
        phi_absMin=res_abs.x,
        phi_metaMin=res_meta.x,
        S3_ref=102.92078560,
        S4_ref=1525.09307116,
    )


# ── 2D potential (pathDeformation) ───────────────────────────────────────────

@pytest.fixture
def full_tunneling_potential():
    """
    examples/fullTunneling.py Potential with default params (c=5, fx=10, fy=10).

    Minima (locked 2026-04-24):
      true min  ≈ (1.0, 1.0),   V ≈ -1.66666667
      false min ≈ (0.0, 0.0),   V ≈  0.0
    """
    from examples.fullTunneling import Potential
    return Potential(c=5., fx=10., fy=10.)
