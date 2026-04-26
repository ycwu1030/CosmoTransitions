"""
tests/test_phase5_pathdeformation.py
--------------------------------------
Unit and regression tests for Phase 5 improvements to
pathDeformation.py and transitionFinder.Phase.

5.7  – PCHIP deployment: Phase.valAt and SplinePath.V/dV/d2V
5.1  – Force normalisation (delta_V denominator)
5.3  – Scale-invariant convergence criterion
5.2  – Adaptive step size (Armijo back-tracking)
5.4  – Oscillation / stagnation detection
5.5  – Endpoint minimisation (L-BFGS-B)
5.6  – Adaptive path resampling
"""
import numpy as np
import pytest

from cosmoTransitions import transitionFinder, pathDeformation as pd


# ─────────────────────────────────────────────────────────────────────────────
# 5.7  PCHIP deployment
# ─────────────────────────────────────────────────────────────────────────────

class TestPHCIPPhaseValAt:
    """Phase.valAt uses PchipInterpolator – no spurious extrema between nodes."""

    @staticmethod
    def _make_phase(T_data, X_data):
        dXdT = np.gradient(X_data, T_data, axis=0)
        return transitionFinder.Phase(0, X_data, T_data, dXdT)

    def test_no_oscillation_1d(self):
        """1-D field: interpolated values must stay within the data range."""
        T = np.array([100., 120., 200., 300.])
        X = np.array([[1.0], [1.5], [0.5], [0.1]])   # non-monotone
        phase = self._make_phase(T, X)

        T_fine = np.linspace(T[0], T[-1], 2000)
        vals = phase.valAt(T_fine)          # shape (2000, 1)

        assert vals.min() >= X.min() - 1e-10, "PCHIP introduced undershoot"
        assert vals.max() <= X.max() + 1e-10, "PCHIP introduced overshoot"

    def test_scalar_input_shape(self):
        """valAt(scalar) → shape (Ndim,)."""
        T = np.linspace(100., 300., 10)
        X = np.column_stack([np.sin(T / 100.), np.cos(T / 100.)])
        phase = self._make_phase(T, X)

        result = phase.valAt(200.)
        assert result.shape == (2,), f"Expected (2,), got {result.shape}"

    def test_array_input_shape(self):
        """valAt(array of m) → shape (m, Ndim)."""
        T = np.linspace(100., 300., 10)
        X = np.column_stack([T / 300., (T / 300.) ** 2])
        phase = self._make_phase(T, X)

        T_eval = np.array([120., 150., 200., 250.])
        result = phase.valAt(T_eval)
        assert result.shape == (4, 2), f"Expected (4, 2), got {result.shape}"

    def test_derivative_order_1(self):
        """valAt with deriv=1 returns dX/dT, consistent with finite differences."""
        T = np.linspace(50., 250., 20)
        X = np.column_stack([(T / 200.) ** 2])   # X = (T/200)^2 → dX/dT = T/20000
        phase = self._make_phase(T, X)

        T_mid = 150.0
        dxdt_pchip = phase.valAt(T_mid, deriv=1).ravel()[0]
        dxdt_exact = 2 * T_mid / 200.**2   # = 0.0075
        assert abs(dxdt_pchip - dxdt_exact) < 1e-3

    def test_no_oscillation_2d(self):
        """2-D field: both components must stay within data range."""
        T = np.linspace(0., 100., 6)
        X = np.column_stack([np.array([0., 5., 3., 8., 2., 10.]),
                              np.array([10., 7., 9., 4., 6., 1.])])
        phase = self._make_phase(T, X)

        T_fine = np.linspace(T[0], T[-1], 3000)
        vals = phase.valAt(T_fine)
        for dim in range(2):
            lo, hi = X[:, dim].min(), X[:, dim].max()
            assert vals[:, dim].min() >= lo - 1e-10
            assert vals[:, dim].max() <= hi + 1e-10

    def test_tck_attribute_none(self):
        """tck attribute exists but is None (deprecated sentinel)."""
        T = np.linspace(100., 200., 5)
        X = np.column_stack([T / 200.])
        phase = self._make_phase(T, X)
        assert hasattr(phase, "tck")
        assert phase.tck is None


class TestPHCIPSplinePath:
    """SplinePath.V/dV/d2V use PchipInterpolator – no spurious extrema."""

    @staticmethod
    def _double_well_V(phi):
        """V(x,y) = (x^2-1)^2 + y^2/2  — simple double-well in 2D."""
        x, y = phi[..., 0], phi[..., 1]
        return (x**2 - 1.)**2 + 0.5*y**2

    @staticmethod
    def _double_well_dV(phi):
        x, y = phi[..., 0], phi[..., 1]
        dVdx = 4.*x*(x**2 - 1.)
        dVdy = y
        return np.stack([dVdx, dVdy], axis=-1)

    @pytest.fixture(scope="class")
    def spline_path(self):
        """SplinePath built on a straight line between (-1,0) and (1,0)."""
        pts = np.column_stack([np.linspace(-1., 1., 30),
                               np.zeros(30)])
        return pd.SplinePath(pts, self._double_well_V, self._double_well_dV,
                             V_spline_samples=100)

    def test_V_pchip_attribute_set(self, spline_path):
        """_V_pchip must be set (not None) when V_spline_samples is given."""
        assert spline_path._V_pchip is not None

    def test_V_no_oscillation(self, spline_path):
        """V(x) must not introduce extrema between sample points."""
        x_fine = np.linspace(0., spline_path.L, 5000)
        V_fine = spline_path.V(x_fine)
        # The double-well potential rises from -1 to 0 and back; no dip below
        # min sample and no peak above max sample expected.
        x_samples = np.linspace(0., spline_path.L, 100)
        V_samples = spline_path.V(x_samples)
        assert V_fine.min() >= V_samples.min() - 1e-6
        assert V_fine.max() <= V_samples.max() + 1e-6

    def test_dV_finite_diff_consistent(self, spline_path):
        """dV(x) consistent with finite-difference estimate of V(x)."""
        x = np.linspace(spline_path.L * 0.2, spline_path.L * 0.8, 30)
        eps = 1e-5
        dV_pchip = spline_path.dV(x)
        dV_fd = (spline_path.V(x + eps) - spline_path.V(x - eps)) / (2.*eps)
        np.testing.assert_allclose(dV_pchip, dV_fd, rtol=1e-3, atol=1e-6)

    def test_d2V_finite_diff_consistent(self, spline_path):
        """d2V(x) consistent with finite-difference estimate of dV(x)."""
        x = np.linspace(spline_path.L * 0.2, spline_path.L * 0.8, 30)
        eps = 1e-4
        d2V_pchip = spline_path.d2V(x)
        d2V_fd = (spline_path.dV(x + eps) - spline_path.dV(x - eps)) / (2.*eps)
        # PCHIP d2V is piecewise-linear (C0), so allow looser tolerance
        np.testing.assert_allclose(d2V_pchip, d2V_fd, rtol=0.05, atol=1e-4)

    def test_no_V_spline_samples_fallback(self):
        """When V_spline_samples=None, _V_pchip is None and V uses direct eval."""
        pts = np.column_stack([np.linspace(-1., 1., 10), np.zeros(10)])
        sp = pd.SplinePath(pts, self._double_well_V, self._double_well_dV,
                           V_spline_samples=None)
        assert sp._V_pchip is None
        # V() must still work via direct evaluation
        x = np.linspace(0., sp.L, 5)
        V_vals = sp.V(x)
        assert np.all(np.isfinite(V_vals))
