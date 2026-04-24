"""
tests/test_pathDeformation.py
------------------------------
Regression tests for cosmoTransitions.pathDeformation.fullTunneling.

Golden reference values locked on 2026-04-24 using conda base
(Python 3.12, numpy 1.26.4, scipy 1.13.1).

Potential used:  examples.fullTunneling.Potential(c=5, fx=10, fy=10)
  True  minimum: (1.0, 1.0),  V ≈ -1.66666667
  False minimum: (0.0, 0.0),  V =  0.0

API notes (discovered during probe):
  fullTunneling(path_pts, V, dV, ...)
  - path_pts[0]  = LOWER (true) minimum  → e.g. [1., 1.]
  - path_pts[-1] = METASTABLE minimum    → e.g. [0., 0.]
  Returns: fullTunneling_rval(profile1D, Phi, action, fRatio, saved_steps)

The action reference value for the default Potential is obtained by running
the probe script; it will be filled in after a successful run of
probe_all_baselines.py with the corrected path ordering.
"""
import numpy as np
import pytest

from cosmoTransitions import pathDeformation as pd
from cosmoTransitions import tunneling1D as t1


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pot_default():
    """Default Potential(c=5, fx=10, fy=10)."""
    from examples.fullTunneling import Potential
    return Potential(c=5., fx=10., fy=10.)


@pytest.fixture(scope="module")
def pot_thin_wall():
    """Thin-walled potential Potential(c=5, fx=0, fy=2) from the examples."""
    from examples.fullTunneling import Potential
    return Potential(c=5., fx=0., fy=2.)


def _linear_path(start, end, n=20):
    """Straight-line path from start (true min) to end (false min)."""
    return np.array([start + t * (np.array(end) - np.array(start))
                     for t in np.linspace(0, 1, n)])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module API
# ─────────────────────────────────────────────────────────────────────────────

def test_module_imports():
    """pathDeformation must import and expose fullTunneling."""
    import importlib
    mod = importlib.import_module("cosmoTransitions.pathDeformation")
    assert hasattr(mod, "fullTunneling")
    assert hasattr(mod, "Deformation_Spline")
    assert hasattr(mod, "DeformationError")
    assert hasattr(mod, "SplinePath")


# ─────────────────────────────────────────────────────────────────────────────
# 2. fullTunneling return-value structure
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestFullTunnelingReturnValue:
    """Verify the named fields of the fullTunneling return value."""

    @pytest.fixture(scope="class")
    def result_default(self, pot_default):
        # Correct ordering: path_pts[0] = true min, path_pts[-1] = metastable
        path = _linear_path([1., 1.], [0., 0.], n=20)
        return pd.fullTunneling(path, pot_default.V, pot_default.dV)

    def test_result_has_profile1D(self, result_default):
        assert hasattr(result_default, "profile1D")

    def test_result_has_Phi(self, result_default):
        assert hasattr(result_default, "Phi")

    def test_result_has_action(self, result_default):
        assert hasattr(result_default, "action")

    def test_result_has_fRatio(self, result_default):
        assert hasattr(result_default, "fRatio")

    def test_action_is_positive(self, result_default):
        """Euclidean action must be positive."""
        assert result_default.action > 0

    def test_action_is_finite(self, result_default):
        assert np.isfinite(result_default.action)

    def test_profile1D_R_increasing(self, result_default):
        """Radial coordinate must be monotonically increasing."""
        R = result_default.profile1D.R
        assert np.all(np.diff(R) > 0)

    def test_profile1D_fields(self, result_default):
        """profile1D must have R, Phi, dPhi, Rerr."""
        for f in ("R", "Phi", "dPhi", "Rerr"):
            assert hasattr(result_default.profile1D, f)

    def test_Phi_shape(self, result_default):
        """Phi (2D path) must have shape (n_points, 2)."""
        Phi = result_default.Phi
        assert Phi.ndim == 2
        assert Phi.shape[1] == 2

    def test_fRatio_small_at_convergence(self, result_default):
        """
        fRatio = max(transverse force) / max(|∇V|) should be small at convergence.
        The code's own convergence threshold is typically < 0.02.
        """
        assert result_default.fRatio < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Path endpoints
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_path_endpoint_near_true_minimum(pot_default):
    """The first point of the deformed path should be near (1,1)."""
    path = _linear_path([1., 1.], [0., 0.], n=20)
    result = pd.fullTunneling(path, pot_default.V, pot_default.dV)
    start = result.Phi[0]
    assert np.linalg.norm(start - np.array([1., 1.])) < 0.2


@pytest.mark.slow
def test_path_endpoint_near_false_minimum(pot_default):
    """The last point of the deformed path should be near (0,0)."""
    path = _linear_path([1., 1.], [0., 0.], n=20)
    result = pd.fullTunneling(path, pot_default.V, pot_default.dV)
    end = result.Phi[-1]
    assert np.linalg.norm(end - np.array([0., 0.])) < 0.2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Deformation_Spline initialisation (unit test, no tunneling)
# ─────────────────────────────────────────────────────────────────────────────

def test_spline_path_init_does_not_crash(pot_default):
    """
    SplinePath can be initialised with a linear path without error.
    SplinePath(pts, V, dV) — a higher-level wrapper used inside fullTunneling.
    """
    path = _linear_path([1., 1.], [0., 0.], n=15)
    try:
        sp = pd.SplinePath(path, pot_default.V, pot_default.dV)
        assert sp is not None
        assert sp.L > 0
    except Exception as e:
        pytest.fail(f"SplinePath init raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Thin-wall example from the package docs
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_thin_wall_example_converges(pot_thin_wall):
    """
    The thin-wall example from examples/fullTunneling.py should converge.
    This is a direct replication of the canonical usage example.
    """
    try:
        result = pd.fullTunneling([[1, 1.], [0, 0]], pot_thin_wall.V, pot_thin_wall.dV)
        assert result.action > 0
        assert np.isfinite(result.action)
    except Exception as e:
        pytest.fail(f"Thin-wall example failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Two-point path input (minimal path)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_two_point_path_accepted(pot_default):
    """fullTunneling should accept a 2-point path (just start and end)."""
    try:
        result = pd.fullTunneling([[1., 1.], [0., 0.]], pot_default.V, pot_default.dV)
        assert result.action > 0
    except Exception as e:
        pytest.fail(f"Two-point path failed: {e}")
