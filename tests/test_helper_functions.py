"""
tests/test_helper_functions.py
-------------------------------
Regression tests for cosmoTransitions.helper_functions.

Covers: rkqs, deriv14, deriv23, deriv14_const_dx, Nbspl, cubicInterpFunction.

All golden reference values locked on 2026-04-24 using conda base
(Python 3.12, numpy 1.26.4, scipy 1.13.1).
"""
import numpy as np
import pytest

from cosmoTransitions import helper_functions as hf


# ─────────────────────────────────────────────────────────────────────────────
# Baseline defect markers
# ─────────────────────────────────────────────────────────────────────────────

SYNTAX_WARNINGS = pytest.mark.xfail(
    strict=False,
    reason="B-01/B-02: SyntaxWarning from invalid escape sequences in docstrings "
           "(helper_functions lines 313, 354, 391; tunneling1D line 50). "
           "Fix in Phase 1 modernisation.",
)

DEPRECATED_GETARGSPEC = pytest.mark.xfail(
    strict=False,
    reason="B-06: setDefaultArgs uses inspect.getargspec which was removed in "
           "Python 3.12. Fix in Phase 1.",
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  rkqs — Cash-Karp RK5 single-step integrator
# ─────────────────────────────────────────────────────────────────────────────

def _decay(y, t, *args):
    """dy/dt = -y  →  y(t) = exp(-t)"""
    return np.array([-y[0]])


def _harmonic(y, t, *args):
    """
    d²x/dt² = -x  written as  d/dt [x, v] = [v, -x]
    Solution: x(t) = cos(t), v(t) = -sin(t)
    """
    return np.array([y[1], -y[0]])


class TestRkqs:
    """Cash-Karp RK5 single step: rkqs(y, dydt, t, f, dt_try, epsfrac, epsabs)
    Returns _rkqs_rval(Delta_y, Delta_t, dtnxt).
    """

    def test_fields_exist(self):
        """Return value exposes the expected named fields."""
        y0 = np.array([1.0])
        result = hf.rkqs(y0, _decay(y0, 0.0), 0.0, _decay, 0.1, 1e-8, np.array([1e-8]))
        assert hasattr(result, "Delta_y")
        assert hasattr(result, "Delta_t")
        assert hasattr(result, "dtnxt")

    def test_decay_accuracy(self):
        """
        Baseline (locked): y(0.1) = exp(-0.1) = 0.904837418036...
        rkqs gives y_new = 0.904837417917 → abs error < 1.5e-10.
        """
        y0 = np.array([1.0])
        result = hf.rkqs(y0, _decay(y0, 0.0), 0.0, _decay, 0.1, 1e-8, np.array([1e-8]))
        y_new = y0 + result.Delta_y
        expected = np.exp(-result.Delta_t)
        assert abs(y_new[0] - expected) < 1.5e-10

    def test_decay_step_taken(self):
        """Delta_t should equal the requested dt_try when error is small enough."""
        y0 = np.array([1.0])
        dt_try = 0.1
        result = hf.rkqs(y0, _decay(y0, 0.0), 0.0, _decay, dt_try, 1e-8, np.array([1e-8]))
        assert result.Delta_t == pytest.approx(dt_try, rel=1e-12)

    def test_decay_next_step_larger(self):
        """After a successful step the suggested next step should be >= current."""
        y0 = np.array([1.0])
        result = hf.rkqs(y0, _decay(y0, 0.0), 0.0, _decay, 0.1, 1e-8, np.array([1e-8]))
        assert result.dtnxt >= result.Delta_t

    def test_harmonic_accuracy(self):
        """2D harmonic oscillator: error should be < 1e-9 per component."""
        y0 = np.array([1.0, 0.0])   # x=1, v=0
        result = hf.rkqs(y0, _harmonic(y0, 0.0), 0.0, _harmonic, 0.1, 1e-10, np.array([1e-10, 1e-10]))
        y_new = y0 + result.Delta_y
        dt = result.Delta_t
        expected = np.array([np.cos(dt), -np.sin(dt)])
        assert np.max(np.abs(y_new - expected)) < 1e-9

    def test_delta_y_shape_matches_input(self):
        """Output Delta_y must have the same shape as the input y."""
        y0 = np.array([1.0, 2.0, 3.0])
        dydt = -y0
        result = hf.rkqs(y0, dydt, 0.0, lambda y, t: -y, 0.05, 1e-8, np.ones(3) * 1e-8)
        assert result.Delta_y.shape == y0.shape


# ─────────────────────────────────────────────────────────────────────────────
# 2.  deriv14 — 4th-order accurate 1st derivative
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriv14:
    """deriv14(y, x) → first derivative using 4th-order finite differences."""

    def test_sin_interior_error(self):
        """
        Baseline: deriv14(sin, x) interior max error vs cos ≈ 8.98e-06.
        Tolerance: < 1e-5 (strictly better than order-of-magnitude).
        """
        n = 50
        x = np.linspace(0, 2 * np.pi, n)
        dy = hf.deriv14(np.sin(x), x)
        err = np.max(np.abs(dy[5:-5] - np.cos(x[5:-5])))
        assert err < 1e-5

    def test_linear_exact(self):
        """deriv14 should be exact (to machine precision) for linear functions."""
        x = np.linspace(0, 5, 40)
        y = 3.0 * x + 7.0
        dy = hf.deriv14(y, x)
        # Avoid boundary effects
        assert np.max(np.abs(dy[4:-4] - 3.0)) < 1e-10

    def test_quadratic_accuracy(self):
        """For y=x², derivative should be 2x with error < 1e-8."""
        x = np.linspace(1, 4, 60)
        dy = hf.deriv14(x**2, x)
        assert np.max(np.abs(dy[5:-5] - 2 * x[5:-5])) < 1e-8

    def test_output_shape(self):
        """Output shape must match input."""
        x = np.linspace(0, 1, 30)
        dy = hf.deriv14(np.sin(x), x)
        assert dy.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# 3.  deriv14_const_dx — 4th-order derivative on uniform grid
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriv14ConstDx:

    def test_matches_deriv14_uniform(self):
        """On a uniform grid, deriv14_const_dx should agree with deriv14."""
        n = 60
        x = np.linspace(0, 2 * np.pi, n)
        dx = x[1] - x[0]
        y = np.sin(x)
        d1 = hf.deriv14(y, x)
        d2 = hf.deriv14_const_dx(y, dx=dx)
        assert np.max(np.abs(d1[5:-5] - d2[5:-5])) < 1e-12

    def test_sin_interior_error(self):
        """Error for sin should be < 1e-5 on interior points."""
        n = 50
        x = np.linspace(0, 2 * np.pi, n)
        dx = x[1] - x[0]
        dy = hf.deriv14_const_dx(np.sin(x), dx=dx)
        err = np.max(np.abs(dy[5:-5] - np.cos(x[5:-5])))
        assert err < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 4.  deriv23 — 3rd-order accurate 2nd derivative
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriv23:

    def test_sin_second_deriv(self):
        """
        Baseline: deriv23(sin, x) interior max error vs -sin ≈ 2.998e-06.
        Tolerance: < 5e-6.
        """
        n = 50
        x = np.linspace(0, 2 * np.pi, n)
        d2y = hf.deriv23(np.sin(x), x)
        err = np.max(np.abs(d2y[3:-3] + np.sin(x[3:-3])))
        assert err < 5e-6

    def test_quadratic_exact(self):
        """d²(x²)/dx² = 2 should be exact to machine precision."""
        x = np.linspace(0, 5, 50)
        d2y = hf.deriv23(x**2, x)
        assert np.max(np.abs(d2y[4:-4] - 2.0)) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Nbspl — B-spline basis functions
# ─────────────────────────────────────────────────────────────────────────────

class TestNbspl:
    """
    Nbspl(t, x, k=3) → array of shape (len(x), len(t)-k-1).
    Clamped cubic B-spline with repeated endpoints.
    """

    @pytest.fixture
    def clamped_cubic_knots(self):
        """k=3 clamped knot vector with 4 repeated boundary knots."""
        return np.array([0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1.])

    def test_output_shape(self, clamped_cubic_knots):
        """Output shape should be (len(x), n_basis)."""
        knots = clamped_cubic_knots
        x = np.linspace(0.01, 0.99, 100)
        B = hf.Nbspl(knots, x, k=3)
        n_basis = len(knots) - 3 - 1   # len(t) - k - 1
        assert B.shape == (len(x), n_basis)

    def test_partition_of_unity(self, clamped_cubic_knots):
        """
        Baseline: sum of all basis functions at each point = 1.
        Machine-precision result: max error ≈ 2.22e-16.
        """
        x = np.linspace(0.01, 0.99, 200)
        B = hf.Nbspl(clamped_cubic_knots, x, k=3)
        row_sums = B.sum(axis=1)
        assert np.max(np.abs(row_sums - 1.0)) < 1e-12

    def test_non_negative(self, clamped_cubic_knots):
        """All B-spline basis function values should be >= 0."""
        x = np.linspace(0.01, 0.99, 200)
        B = hf.Nbspl(clamped_cubic_knots, x, k=3)
        assert np.all(B >= -1e-14)  # allow tiny floating point noise

    def test_bad_k_raises(self, clamped_cubic_knots):
        """k > len(t)-2 should raise an exception."""
        x = np.linspace(0.1, 0.9, 10)
        with pytest.raises(Exception):
            hf.Nbspl(clamped_cubic_knots, x, k=100)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  cubicInterpFunction — Hermite cubic interpolation
# ─────────────────────────────────────────────────────────────────────────────

class TestCubicInterpFunction:
    """
    cubicInterpFunction(y0, dy0, y1, dy1) → cubic Bezier interpolant.

    API (discovered during probe):
      - Takes the values and *first derivatives* at two endpoints
      - Returns a callable f(t) where t ∈ [0, 1] maps from y0 to y1
      - This is a single-interval interpolant, NOT a grid interpolant
    """

    def test_returns_callable(self):
        """Constructor must return a callable."""
        f = hf.cubicInterpFunction(0.0, 1.0, 1.0, 1.0)
        assert callable(f)

    def test_endpoint_values(self):
        """f(0) = y0 and f(1) = y1 (Bezier endpoint interpolation)."""
        y0, dy0, y1, dy1 = 0.5, 2.0, 3.0, -1.0
        f = hf.cubicInterpFunction(y0, dy0, y1, dy1)
        assert f(0.0) == pytest.approx(y0, abs=1e-12)
        assert f(1.0) == pytest.approx(y1, abs=1e-12)

    def test_linear_case(self):
        """With matched linear derivatives, f(t) should be nearly linear."""
        # Linear: y = 2t, y(0)=0, dy=2, y(1)=2, dy=2
        f = hf.cubicInterpFunction(0.0, 2.0, 2.0, 2.0)
        # At t=0.5: linear value is 1.0
        assert f(0.5) == pytest.approx(1.0, abs=1e-12)

    def test_output_is_scalar(self):
        """f(t) must return a scalar (or 0-d array) for scalar t input."""
        f = hf.cubicInterpFunction(0.0, 1.0, 1.0, 1.0)
        val = f(0.5)
        assert np.ndim(val) == 0 or isinstance(val, float)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Baseline defect tests (xfail)
# ─────────────────────────────────────────────────────────────────────────────

@DEPRECATED_GETARGSPEC
def test_setDefaultArgs_python312():
    """
    B-06: setDefaultArgs uses inspect.getargspec, removed in Python 3.12.
    This is expected to raise AttributeError on Python 3.12+.
    """
    def foo(bar=1):
        return bar
    hf.setDefaultArgs(foo, bar=2)
    assert foo() == 2


def test_module_imports_without_error():
    """Module must be importable (SyntaxWarnings are suppressed via pytest.ini)."""
    import importlib
    mod = importlib.import_module("cosmoTransitions.helper_functions")
    assert mod is not None
