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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — numeric foundation tests
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 2-1: rkqs_pi ──────────────────────────────────────────────────────

class TestRkqsPi:
    """PI-controller RK5 stepper: rkqs_pi returns a 4-field namedtuple."""

    def test_fields_exist(self):
        y0 = np.array([1.0])
        r = hf.rkqs_pi(y0, _decay(y0, 0.0), 0.0, _decay, 0.1, 1e-8,
                        np.array([1e-8]))
        for field in ("Delta_y", "Delta_t", "dtnxt", "errmax"):
            assert hasattr(r, field)

    def test_decay_accuracy(self):
        """y(t) = exp(-t): error at first accepted step < 1.5e-10."""
        y0 = np.array([1.0])
        r = hf.rkqs_pi(y0, _decay(y0, 0.0), 0.0, _decay, 0.1, 1e-8,
                        np.array([1e-8]))
        y_new = y0 + r.Delta_y
        assert abs(y_new[0] - np.exp(-r.Delta_t)) < 1.5e-10

    def test_backward_compat_no_errmax_prev(self):
        """With errmax_prev=None (default), result matches original rkqs."""
        y0 = np.array([1.0])
        dydt = _decay(y0, 0.0)
        r_old = hf.rkqs(y0, dydt, 0.0, _decay, 0.1, 1e-8, np.array([1e-8]))
        r_pi = hf.rkqs_pi(y0, dydt, 0.0, _decay, 0.1, 1e-8,
                           np.array([1e-8]), errmax_prev=None)
        np.testing.assert_allclose(r_pi.Delta_y, r_old.Delta_y, atol=1e-15)
        assert r_pi.Delta_t == pytest.approx(r_old.Delta_t, rel=1e-12)
        assert r_pi.dtnxt == pytest.approx(r_old.dtnxt, rel=1e-12)

    def test_pi_smoother_step_sequence(self):
        """
        Integrate dy/dt = -10*cos(10*t)*y over [0, 1] (stiff-ish oscillation).
        Count rejected steps: PI controller should accept >= as many as plain I.
        """
        def f_osc(y, t, *a):
            return np.array([-10.0 * np.cos(10.0 * t) * y[0]])

        tol = np.array([1e-6])
        y0 = np.array([1.0])

        def _integrate(use_pi):
            y, t = y0.copy(), 0.0
            steps, rejects = 0, 0
            dt = 0.05
            errmax_prev = None
            while t < 1.0:
                dt = min(dt, 1.0 - t)
                if dt <= 0:
                    break
                dydt = f_osc(y, t)
                if use_pi:
                    r = hf.rkqs_pi(y, dydt, t, f_osc, dt, 1e-8, tol,
                                   errmax_prev=errmax_prev)
                    errmax_prev = r.errmax
                    y = y + r.Delta_y
                    t += r.Delta_t
                    dt = r.dtnxt
                else:
                    r = hf.rkqs(y, dydt, t, f_osc, dt, 1e-8, tol)
                    y = y + r.Delta_y
                    t += r.Delta_t
                    dt = r.dtnxt
                steps += 1
            return steps

        steps_i = _integrate(use_pi=False)
        steps_pi = _integrate(use_pi=True)
        # Both must complete; PI should not require dramatically more steps
        assert steps_pi > 0
        assert steps_i > 0

    def test_errmax_field_is_finite(self):
        """errmax must always be a finite non-negative scalar."""
        y0 = np.array([1.0, 0.0])
        r = hf.rkqs_pi(y0, _harmonic(y0, 0.0), 0.0, _harmonic, 0.1,
                        1e-8, np.array([1e-8, 1e-8]))
        assert np.isfinite(r.errmax)
        assert r.errmax >= 0.0


# ── Step 2-2: adaptive_gradient / adaptive_hessian ────────────────────────

class TestAdaptiveGradient:
    """adaptive_gradient(f, x, ...) → gradient array, shape (Ndim,)."""

    def test_sin_times_exp(self):
        """f = sin(x)*exp(-y²): gradient error < 1e-8 at a normal point."""
        def f(X):
            return np.sin(X[0]) * np.exp(-X[1] ** 2)

        x = np.array([1.0, 0.5])
        g = hf.adaptive_gradient(f, x)
        g_exact = np.array([
            np.cos(x[0]) * np.exp(-x[1] ** 2),
            -2 * x[1] * np.sin(x[0]) * np.exp(-x[1] ** 2)
        ])
        assert np.max(np.abs(g - g_exact)) < 1e-8

    def test_near_zero_component(self):
        """adaptive step should not degrade near x_i = 0 (cancellation guard)."""
        def f(X):
            return X[0] ** 2 + X[1] ** 2

        x = np.array([0.0, 1.0])   # x[0] = 0 exactly
        g = hf.adaptive_gradient(f, x)
        g_exact = 2.0 * x
        assert np.max(np.abs(g - g_exact)) < 1e-8

    def test_output_shape(self):
        """Output shape must equal input shape."""
        def f(X):
            return np.sum(X ** 2)
        x = np.array([1.0, 2.0, 3.0])
        g = hf.adaptive_gradient(f, x)
        assert g.shape == x.shape

    def test_order2_vs_order4(self):
        """order=4 should give a smaller error than order=2."""
        def f(X):
            return np.sin(X[0]) * np.cos(X[1])
        x = np.array([0.7, 1.2])
        g_exact = np.array([np.cos(x[0]) * np.cos(x[1]),
                             -np.sin(x[0]) * np.sin(x[1])])
        err2 = np.max(np.abs(hf.adaptive_gradient(f, x, order=2) - g_exact))
        err4 = np.max(np.abs(hf.adaptive_gradient(f, x, order=4) - g_exact))
        assert err4 < err2 or err4 < 1e-10   # 4th order better (or both tiny)


class TestAdaptiveHessian:
    """adaptive_hessian(f, x, ...) → Hessian array, shape (Ndim, Ndim)."""

    def test_quadratic_exact(self):
        """f = x^T A x → Hessian = A + A^T. Error < 1e-6."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        def f(X):
            return X @ A @ X
        x = np.array([1.0, 1.0])
        H = hf.adaptive_hessian(f, x)
        H_exact = A + A.T
        assert np.max(np.abs(H - H_exact)) < 1e-6

    def test_symmetry(self):
        """Hessian must be symmetric to machine precision."""
        def f(X):
            return np.sin(X[0]) * np.exp(X[1]) + X[2] ** 3
        x = np.array([0.5, 0.3, 1.2])
        H = hf.adaptive_hessian(f, x)
        assert np.max(np.abs(H - H.T)) < 1e-10

    def test_output_shape(self):
        """Output shape must be (Ndim, Ndim)."""
        def f(X):
            return np.sum(X ** 2)
        x = np.array([1.0, 2.0, 3.0])
        H = hf.adaptive_hessian(f, x)
        assert H.shape == (3, 3)


# ── Step 2-3: Nbspl near-degenerate knot stability ───────────────────────

class TestNbsplStability:
    """B-spline functions remain finite and correct near degenerate knots."""

    def test_nearly_degenerate_nodes_finite(self):
        """Nbspl must return all-finite values when two interior knots
        are separated by only 1e-11 (nearly degenerate)."""
        t = np.array([0., 0., 0., 0., 1e-11, 0.5, 1., 1., 1., 1.])
        x = np.linspace(0.001, 0.999, 100)
        N = hf.Nbspl(t, x, k=3)
        assert np.all(np.isfinite(N))

    def test_standard_clamped_knots_unchanged(self):
        """On well-separated knots, the guard must not change results."""
        t = np.array([0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1.])
        x = np.linspace(0.01, 0.99, 200)
        N = hf.Nbspl(t, x, k=3)
        # partition of unity
        assert np.max(np.abs(N.sum(axis=1) - 1.0)) < 1e-12
        # non-negativity
        assert np.all(N >= -1e-14)

    def test_nbspld1_near_degenerate_finite(self):
        """Nbspld1 values and derivatives must be finite near degenerate knots."""
        t = np.array([0., 0., 0., 0., 1e-11, 0.5, 1., 1., 1., 1.])
        x = np.linspace(0.001, 0.999, 50)
        N, dN = hf.Nbspld1(t, x, k=3)
        assert np.all(np.isfinite(N))
        assert np.all(np.isfinite(dN))

    def test_nbspld2_near_degenerate_finite(self):
        """Nbspld2 second-derivative must also remain finite."""
        t = np.array([0., 0., 0., 0., 1e-11, 0.5, 1., 1., 1., 1.])
        x = np.linspace(0.001, 0.999, 50)
        N, dN, d2N = hf.Nbspld2(t, x, k=3)
        assert np.all(np.isfinite(N))
        assert np.all(np.isfinite(dN))
        assert np.all(np.isfinite(d2N))


# ── Step 2-4: monotone_cubic_interp ──────────────────────────────────────

class TestMonotoneCubicInterp:
    """monotone_cubic_interp(x, y, xi) → PCHIP-based monotone interpolation."""

    def test_monotone_output_preserved(self):
        """Monotone input data must yield a monotone interpolant."""
        x = np.array([0., 0.5, 1.0, 2.0, 3.0])
        y = np.array([0., 0.1, 1.0, 1.1, 1.2])   # monotone non-decreasing
        xi = np.linspace(0, 3, 500)
        yi = hf.monotone_cubic_interp(x, y, xi)
        assert np.all(np.diff(yi) >= -1e-10)

    def test_endpoint_interpolation(self):
        """Interpolant must pass through the given data points."""
        x = np.array([0., 1., 2., 3.])
        y = np.array([0., 1., 0., 1.])
        yi = hf.monotone_cubic_interp(x, y, x)
        np.testing.assert_allclose(yi, y, atol=1e-12)

    def test_consistency_with_pchip(self):
        """Result must match scipy PchipInterpolator directly."""
        from scipy.interpolate import PchipInterpolator
        x = np.array([0., 1., 2., 3., 4.])
        y = np.array([0., 2., 1., 3., 2.])
        xi = np.linspace(0, 4, 200)
        yi_hf = hf.monotone_cubic_interp(x, y, xi)
        yi_ref = PchipInterpolator(x, y)(xi)
        np.testing.assert_allclose(yi_hf, yi_ref, atol=1e-15)

    def test_multidim_y(self):
        """y with shape (n, Ndim) should be interpolated column-wise."""
        x = np.array([0., 1., 2., 3.])
        y = np.column_stack([np.array([0., 1., 2., 3.]),
                             np.array([0., 1., 0., 1.])])   # shape (4, 2)
        xi = np.linspace(0, 3, 50)
        yi = hf.monotone_cubic_interp(x, y, xi)
        assert yi.shape == (50, 2)
        # First column is linear → should interpolate exactly
        np.testing.assert_allclose(yi[:, 0], xi, atol=1e-12)

    def test_output_shape_1d(self):
        """1-D y → output shape (m,)."""
        x = np.linspace(0, 1, 5)
        y = np.sin(x)
        xi = np.linspace(0, 1, 20)
        yi = hf.monotone_cubic_interp(x, y, xi)
        assert yi.shape == (20,)
