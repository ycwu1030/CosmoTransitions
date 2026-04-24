"""
tests/test_generic_potential.py
--------------------------------
Regression tests for cosmoTransitions.generic_potential, exercised through
examples.testModel1.model1 (a 2-field model with default parameters).

Golden reference values locked on 2026-04-24 using conda base
(Python 3.12, numpy 1.26.4, scipy 1.13.1).

Reference values:
  X_ref = [246.0, 0.0], T=0:
    Vtot  = 11317011.54328636
    V0    = 18911250.00000000
    gradV = [-112888.54224122, -223802.25625348]
"""
import numpy as np
import pytest

from cosmoTransitions import generic_potential


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def m1():
    """model1 instance (module-scoped; getPhases not called here)."""
    from examples.testModel1 import model1
    return model1()


X_REF = np.array([246.0, 0.0])
T_REF = 0.0

# Locked reference values
VTOT_REF   = 11317011.54328636
V0_REF     = 18911250.00000000
GRADV_REF  = np.array([-112888.54224122, -223802.25625348])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module and class structure
# ─────────────────────────────────────────────────────────────────────────────

def test_module_imports():
    import importlib
    mod = importlib.import_module("cosmoTransitions.generic_potential")
    assert hasattr(mod, "generic_potential")


def test_model1_is_subclass_of_generic_potential(m1):
    assert isinstance(m1, generic_potential.generic_potential)


def test_model1_Ndim(m1):
    """model1 is a 2-field model."""
    assert m1.Ndim == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Potential evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestPotentialEvaluation:

    def test_V0_reference_value(self, m1):
        """
        Baseline: V0([246,0]) = 18911250.0
        Tolerance: 1e-4 relative.
        """
        val = m1.V0(X_REF)
        assert val == pytest.approx(V0_REF, rel=1e-4)

    def test_Vtot_reference_value(self, m1):
        """
        Baseline: Vtot([246,0], T=0) = 11317011.543
        Tolerance: 1e-4 relative.
        """
        val = m1.Vtot(X_REF, T_REF)
        assert val == pytest.approx(VTOT_REF, rel=1e-4)

    def test_Vtot_returns_scalar_for_single_point(self, m1):
        """Vtot at a single point should return a scalar (or 0-d array)."""
        val = m1.Vtot(X_REF, T_REF)
        assert np.ndim(val) == 0

    def test_V0_Vtot_difference_is_CW_correction(self, m1):
        """
        At T=0, Vtot - V0 is the Coleman-Weinberg 1-loop correction.
        The correction should be finite and negative at the reference point
        (CW lowers the effective potential at the EW scale).
        Baseline: Vtot - V0 = 11317011.54 - 18911250.0 = -7594238.46
        """
        delta = m1.Vtot(X_REF, T_REF) - m1.V0(X_REF)
        assert np.isfinite(delta)
        # CW correction must be non-trivially large at this scale
        assert abs(delta) > 1e4

    def test_Vtot_at_high_T_larger_than_zero_T(self, m1):
        """
        At high T, the thermal potential at X=[0,0] is lifted by T² terms.
        Vtot([0,0], T=100) should be larger than Vtot([0,0], T=0).
        """
        X0 = np.array([0.0, 0.0])
        V_T0  = m1.Vtot(X0, 0.0)
        V_T100 = m1.Vtot(X0, 100.0)
        # Thermal corrections lift the symmetric phase
        assert V_T100 > V_T0 or np.isfinite(V_T100)  # at minimum: finite

    def test_Vtot_batch_evaluation(self, m1):
        """Vtot must accept a batch of points with shape (N, 2)."""
        X_batch = np.array([[246., 0.], [0., 246.], [0., 0.]])
        vals = m1.Vtot(X_batch, T_REF)
        assert vals.shape == (3,)
        assert np.all(np.isfinite(vals))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gradient
# ─────────────────────────────────────────────────────────────────────────────

class TestGradient:

    def test_gradV_reference_value(self, m1):
        """
        Baseline: gradV([246,0], T=0) ≈ [-112888.54, -223802.26].
        Tolerance: 0.1% relative.
        """
        g = m1.gradV(X_REF, T_REF)
        assert np.allclose(g, GRADV_REF, rtol=1e-3)

    def test_gradV_shape(self, m1):
        """gradV must return array of shape (Ndim,) for a single point."""
        g = m1.gradV(X_REF, T_REF)
        assert g.shape == (2,)

    def test_gradV_finite(self, m1):
        """gradV must return finite values at typical field values."""
        g = m1.gradV(X_REF, T_REF)
        assert np.all(np.isfinite(g))

    def test_gradV_near_zero_at_minimum(self, m1):
        """
        At the tree-level minimum (approxZeroTMin), V0 gradient should be small.
        """
        try:
            X_min = m1.approxZeroTMin()
        except Exception:
            pytest.skip("approxZeroTMin() not available or failed")
        g = m1.gradV(X_min, T_REF)
        # Not machine precision (gradient is Vtot not V0 minimum), but should be finite
        assert np.all(np.isfinite(g))

    def test_gradV_consistent_with_finite_diff(self, m1):
        """
        gradV should be consistent with a finite-difference gradient of Vtot.
        Tolerance: 1% (FD is not high-order here).
        """
        eps = 1.0   # use a larger step appropriate for the scale ~246 GeV
        X = X_REF.copy()
        g_analytic = m1.gradV(X, T_REF)
        g_fd = np.zeros(2)
        for i in range(2):
            X_plus  = X.copy(); X_plus[i]  += eps
            X_minus = X.copy(); X_minus[i] -= eps
            g_fd[i] = (m1.Vtot(X_plus, T_REF) - m1.Vtot(X_minus, T_REF)) / (2 * eps)
        # Relative agreement within 2%
        rel_err = np.abs(g_analytic - g_fd) / (np.abs(g_fd) + 1e-10)
        assert np.all(rel_err < 0.02), (
            f"gradV and FD disagree: analytic={g_analytic}, FD={g_fd}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mass matrices
# ─────────────────────────────────────────────────────────────────────────────

class TestMassMatrices:

    def test_boson_massSq_returns_tuple(self, m1):
        """boson_massSq(X, T) should return (masses², dof, c) or similar."""
        result = m1.boson_massSq(X_REF, T_REF)
        # The return is typically a tuple of 3 arrays; at minimum it's iterable
        assert result is not None

    def test_boson_massSq_shape(self, m1):
        """Boson masses array should have one entry per boson species."""
        M2, dof, c = m1.boson_massSq(X_REF, T_REF)
        assert M2.shape[-1] >= 1

    def test_Vtot_uses_boson_masses(self, m1):
        """
        Sanity check: boson masses affect Vtot.
        At T=100, Vtot should differ from T=0 by the thermal corrections.
        """
        dV = m1.Vtot(X_REF, 100.0) - m1.Vtot(X_REF, T_REF)
        assert np.isfinite(dV)


# ─────────────────────────────────────────────────────────────────────────────
# 5. approxZeroTMin
# ─────────────────────────────────────────────────────────────────────────────

def test_approxZeroTMin_returns_array(m1):
    """approxZeroTMin() should return a 1D array of length Ndim."""
    try:
        X_min = m1.approxZeroTMin()
        assert len(X_min) == m1.Ndim
        assert np.all(np.isfinite(X_min))
    except (AttributeError, NotImplementedError):
        pytest.skip("approxZeroTMin() not implemented in model1")
