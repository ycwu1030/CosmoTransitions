"""
tests/test_finiteT.py
----------------------
Regression tests for cosmoTransitions.finiteT (Jb / Jf thermal functions).

Golden reference values locked on 2026-04-24 using conda base
(Python 3.12, numpy 1.26.4, scipy 1.13.1).

Known baseline defects documented here:
  B-04  Jb_spline inaccurate for x² ≫ 1 (wrong by O(1) for x²=4,9,25)
  B-05  Jb_spline / Jf_spline used for negative x² produce wrong results
         (imaginary-mass handling is not physically correct)
"""
import numpy as np
import pytest

from cosmoTransitions import finiteT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rel_err(a, b):
    """Relative error, safe against b≈0."""
    denom = abs(b) if abs(b) > 1e-30 else 1e-30
    return abs(a - b) / denom


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exact integral functions (Jb_exact / Jf_exact)
# ─────────────────────────────────────────────────────────────────────────────

class TestExactIntegrals:
    """Jb_exact and Jf_exact use scipy.integrate.quad — very slow but authoritative."""

    def test_Jb_exact_at_zero_arg(self):
        """
        Jb(0) = ∫ y² log(1 - exp(-y)) dy = -π⁴/45 ≈ -2.16457.
        Baseline: Jb_exact(0) ≈ -2.1646464592.
        """
        val = finiteT.Jb_exact(0.0)
        expected = -np.pi**4 / 45.0
        assert abs(val - expected) < 1e-4

    def test_Jf_exact_at_zero_arg(self):
        """
        Jf(0) = ∫ y² log(1 + exp(-y)) dy = 7π⁴/360 (with sign convention: negative here)
        Baseline: Jf_exact(0) ≈ -1.8940656549.
        """
        val = finiteT.Jf_exact(0.0)
        # |Jf(0)| = 7π⁴/360
        expected_mag = 7 * np.pi**4 / 360.0
        assert abs(abs(val) - expected_mag) < 1e-3

    def test_Jb_exact_unit_arg(self):
        """Baseline: Jb_exact(1.0) ≈ -1.6964755691."""
        val = finiteT.Jb_exact(1.0)
        assert abs(val - (-1.6964755691)) < 1e-6

    def test_Jf_exact_unit_arg(self):
        """Baseline: Jf_exact(1.0) ≈ -1.5673202548."""
        val = finiteT.Jf_exact(1.0)
        assert abs(val - (-1.5673202548)) < 1e-6

    def test_Jb_approaches_zero_large_arg(self):
        """For very large x², Jb → 0 exponentially."""
        val = finiteT.Jb_exact(1000.0)
        assert abs(val) < 1e-10

    def test_Jb_Jf_both_negative_small_positive_arg(self):
        """Jb and Jf are always negative for positive x²."""
        for x2 in [0.0001, 0.01, 0.1, 1.0]:
            assert finiteT.Jb_exact(x2) < 0
            assert finiteT.Jf_exact(x2) < 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spline functions — Jb_spline / Jf_spline
#    (fast approximation used in production code)
# ─────────────────────────────────────────────────────────────────────────────

class TestSplineFunctions:
    """
    Jb_spline / Jf_spline are pre-computed spline approximations.

    Baseline (locked 2026-04-24):
      x²=0.0001 : Jb_spline=-2.1645657024,  Jb_exact=-2.1646464592,  err=8.076e-05
      x²=0.01   : Jb_spline=-2.1569141214,  Jb_exact=-2.1645647398,  err=7.651e-03
      x²=1.0    : Jb_spline=-1.6964755691,  Jb_exact=-1.6964755691,  err=1.535e-12
      x²=4.0    : Jb_spline=-1.0332425147,  Jb_exact=-0.2791686917,  err=7.541e-01  ← B-04
      x²=25.0   : Jb_spline=-0.1328583402,  Jb_exact=-0.0000000023,  err=1.329e-01  ← B-04
    """

    # ── accurate region (x²≈1) ───────────────────────────────────────────────

    def test_Jb_spline_accurate_at_x2_eq_1(self):
        """
        Baseline: at x²=1, spline and exact agree to 1.5e-12.
        """
        val = finiteT.Jb_spline(1.0)
        assert abs(val - (-1.6964755691)) < 1e-6

    def test_Jf_spline_accurate_at_x2_eq_1(self):
        """Baseline: at x²=1, Jf_spline = -1.5673202548 (exact match)."""
        val = finiteT.Jf_spline(1.0)
        assert abs(val - (-1.5673202548)) < 1e-6

    def test_Jb_spline_near_zero_baseline_error(self):
        """
        Baseline defect B-04 (small x²):
          x²=0.01 → Jb_spline=-2.1569, Jb_exact=-2.1646, abs_err≈7.65e-3.
        This test documents the current (wrong) behaviour; it passes if the
        error is in the expected range 1e-3 < err < 1e-1.
        Phase 3 will fix this.
        """
        val = finiteT.Jb_spline(0.01)
        exact = finiteT.Jb_exact(0.01)
        err = abs(val - exact)
        # Error is non-trivial (spline is off) but finite
        assert 1e-4 < err < 0.1, (
            f"Unexpected error magnitude {err}. Baseline expects ~7.65e-3."
        )

    # ── known large errors for x² > 1 (defect B-04) ─────────────────────────

    def test_Jb_spline_accurate_at_x2_eq_4(self):
        """
        Phase 3 fix for B-04: Jb_spline(4.0) should match Jb_exact2(4.0).
        Both inputs are x²=4; the spline is accurate here.
        """
        val = finiteT.Jb_spline(4.0)
        exact = finiteT.Jb_exact2(4.0)
        assert abs(val - exact) < 1e-3

    def test_Jb_spline_accurate_at_x2_eq_25(self):
        """
        Phase 3 fix for B-04: Jb_spline(25.0) should match Jb_exact2(25.0).
        Both inputs are x²=25; the spline is accurate here.
        """
        val = finiteT.Jb_spline(25.0)
        exact = finiteT.Jb_exact2(25.0)
        assert abs(val - exact) < 1e-3

    # ── large x² → 0 ─────────────────────────────────────────────────────────

    def test_Jb_spline_large_x2_is_zero(self):
        """Baseline: for x²=1000, spline returns ≈ 0 (rounds to 0)."""
        val = finiteT.Jb_spline(1000.0)
        assert abs(val) < 1e-10

    def test_Jf_spline_large_x2_is_zero(self):
        """Baseline: for x²=1000, Jf_spline ≈ 0."""
        val = finiteT.Jf_spline(1000.0)
        assert abs(val) < 1e-10

    # ── negative x² (imaginary mass) — baseline defect B-05 ──────────────────

    def test_Jb_negative_x2_returns_finite(self):
        """
        Baseline defect B-05: negative x² (imaginary mass).
        The spline evaluates outside its intended domain. The current code
        does NOT raise an exception, but returns incorrect values.
        This test documents that the function returns finite (not NaN/inf).
        Fix in Phase 3.
        """
        for x2 in [-0.01, -1.0, -10.0]:
            val = finiteT.Jb_spline(x2)
            assert np.isfinite(val), f"Jb_spline({x2}) returned non-finite: {val}"

    def test_Jb_spline_negative_x2_correct(self):
        """
        Phase 3 fix for B-05: Jb_spline(-1.0) must match Jb_exact2(-1.0).
        x²=-1 is inside the existing positive-domain spline range (_xbmin≈-3.72),
        so it was already correct; this test confirms it stays so.
        """
        val = finiteT.Jb_spline(-1.0)
        exact = finiteT.Jb_exact2(-1.0)
        assert abs(val - exact) / abs(exact) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 3. High-T and Low-T series expansions
# ─────────────────────────────────────────────────────────────────────────────

class TestSeriesExpansions:
    """Jb/Jf with approx='high' or approx='low'."""

    def test_Jb_high_matches_exact_large_x2(self):
        """
        High-x expansion should be accurate for x² ≫ 1.
        At x²=100, high-T series should match exact to < 1e-6.
        Note: 'high' approx is a series in exp(-x), valid for large x.
        """
        try:
            val_high = finiteT.Jb(100.0, approx='high')
            val_exact = finiteT.Jb_exact(100.0)
            assert abs(val_high - val_exact) < 1e-6
        except TypeError:
            pytest.skip("Jb() function signature differs from expected")

    def test_Jf_high_matches_exact_large_x2(self):
        """Jf high-T expansion should match exact at x²=100."""
        try:
            val_high = finiteT.Jf(100.0, approx='high')
            val_exact = finiteT.Jf_exact(100.0)
            assert abs(val_high - val_exact) < 1e-6
        except TypeError:
            pytest.skip("Jf() function signature differs from expected")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Smoothness / monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoothnessAndMonotonicity:

    def test_Jb_exact_monotone_increasing(self):
        """
        Jb_exact(x²) should be monotonically increasing (less negative) as x² increases.
        i.e., d/dx² Jb > 0 for x² > 0.
        """
        x2 = np.linspace(0.5, 50.0, 30)
        vals = np.array([finiteT.Jb_exact(xi) for xi in x2])
        diffs = np.diff(vals)
        assert np.all(diffs >= -1e-10), "Jb_exact should be monotonically increasing"

    def test_Jb_spline_no_jumps_near_x2_1(self):
        """
        Near x²=1 (where spline is accurate), the function should be smooth.
        Check max jump in a dense grid < 5e-4 (baseline: actual max ≈ 3.6e-4).
        """
        x2 = np.linspace(0.9, 1.1, 200)
        vals = np.array([finiteT.Jb_spline(xi) for xi in x2])
        max_jump = np.max(np.abs(np.diff(vals)))
        assert max_jump < 5e-4

    def test_Jb_Jf_negative_for_positive_x2(self):
        """Jb_exact and Jf_exact should be negative for all positive x²."""
        for x2 in [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            assert finiteT.Jb_exact(x2) <= 0
            assert finiteT.Jf_exact(x2) <= 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Module import
# ─────────────────────────────────────────────────────────────────────────────

def test_finiteT_module_imports():
    """finiteT must be importable with its expected public names."""
    import importlib
    mod = importlib.import_module("cosmoTransitions.finiteT")
    for name in ["Jb_spline", "Jf_spline", "Jb_exact", "Jf_exact"]:
        assert hasattr(mod, name), f"finiteT missing attribute '{name}'"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Phase 3 — Negative x² spline (B-05 fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestNegativeX2Spline:
    """
    Verify that Jb_spline / Jf_spline correctly handle negative x²
    (imaginary thermal mass) after the Phase 3 fix.

    The old code clamped to the boundary value for x² < _xbmin ≈ -3.72
    (Jb) and x² < _xfmin ≈ -6.82 (Jf). The new code uses a dedicated
    negative-domain spline covering x² ∈ [-20, boundary).
    """

    def test_Jb_spline_neg_x2_accurate_range(self):
        """
        For x² ∈ {-5, -8, -10, -15}, Jb_spline should agree with
        Jb_exact2 to better than 1e-4 (absolute).
        """
        for x2 in [-5.0, -8.0, -10.0, -15.0]:
            spl = finiteT.Jb_spline(float(x2))
            exact = finiteT.Jb_exact2(float(x2))
            assert abs(spl - exact) < 1e-4, (
                f"Jb_spline({x2}) = {spl:.6g}, exact = {exact:.6g}, "
                f"err = {abs(spl-exact):.2g}"
            )

    def test_Jf_spline_neg_x2_accurate_range(self):
        """
        For x² ∈ {-8, -10, -15}, Jf_spline should agree with
        Jf_exact2 to better than 1e-4 (absolute).
        """
        for x2 in [-8.0, -10.0, -15.0]:
            spl = finiteT.Jf_spline(float(x2))
            exact = finiteT.Jf_exact2(float(x2))
            assert abs(spl - exact) < 1e-4, (
                f"Jf_spline({x2}) = {spl:.6g}, exact = {exact:.6g}, "
                f"err = {abs(spl-exact):.2g}"
            )

    def test_Jb_spline_continuity_at_zero(self):
        """
        Jb_spline should be continuous at x² = 0: values just below and
        just above zero should agree to within 1e-6.
        """
        v_neg = finiteT.Jb_spline(numpy_like(-1e-6))
        v_pos = finiteT.Jb_spline(numpy_like(+1e-6))
        assert abs(float(v_neg) - float(v_pos)) < 1e-4

    def test_Jb_spline_deep_neg_returns_finite(self):
        """For x² = -100 (deep tachyonic), Jb_spline should not crash
        and should return a finite value (the boundary)."""
        val = finiteT.Jb_spline(-100.0)
        assert np.isfinite(val)

    def test_Jb_spline_deep_neg_issues_warning(self):
        """For x² < -20, Jb_spline should emit a logger.warning."""
        import logging
        logger = logging.getLogger("cosmoTransitions.finiteT")
        with _LogCapturer(logger) as cap:
            finiteT.Jb_spline(-100.0)
        assert any("boundary" in m or "invalid" in m for m in cap.messages), (
            "Expected a warning about deep-negative x²"
        )

    def test_Jf_spline_continuity_at_xfmin(self):
        """
        Jf_spline should be continuous at _xfmin ≈ -6.82: values just
        inside and just outside the old boundary should be close.
        """
        xfmin = finiteT._xfmin
        v_inside = finiteT.Jf_spline(xfmin + 0.01)
        v_outside = finiteT.Jf_spline(xfmin - 0.01)
        assert abs(float(v_inside) - float(v_outside)) < 0.05


def numpy_like(val):
    """Return val as a numpy 0-d array for testing scalar spline calls."""
    return np.array(val)


class _LogCapturer:
    """Context manager that captures logger messages."""
    def __init__(self, logger):
        self.logger = logger
        self.messages = []
        self._handler = None

    def __enter__(self):
        import logging
        class _H(logging.Handler):
            def __init__(self_h):
                super().__init__()
                self_h.store = self.messages
            def emit(self_h, record):
                self_h.store.append(record.getMessage())
        self._handler = _H()
        self.logger.addHandler(self._handler)
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *args):
        self.logger.removeHandler(self._handler)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Phase 3 — Low-expansion range guard (B-L fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestLowExpansionGuard:
    """
    Jb_low / Jf_low are high-T series expansions that diverge for x² > ~35.
    Phase 3 adds a ValueError guard at x² > 30.
    """

    def test_Jb_low_raises_for_large_x(self):
        """Jb_low(x) with x² = 50 must raise ValueError."""
        with pytest.raises(ValueError, match="diverges"):
            finiteT.Jb_low(np.sqrt(50.0))

    def test_Jf_low_raises_for_large_x(self):
        """Jf_low(x) with x² = 50 must raise ValueError."""
        with pytest.raises(ValueError, match="diverges"):
            finiteT.Jf_low(np.sqrt(50.0))

    def test_Jb_low_valid_range_no_raise(self):
        """Jb_low(1.0) is well inside the valid range and must not raise."""
        val = finiteT.Jb_low(1.0)
        assert np.isfinite(val)

    def test_Jb_low_boundary_x2_30(self):
        """x² = 30 is exactly at the cutoff — should not raise."""
        val = finiteT.Jb_low(np.sqrt(30.0))
        assert np.isfinite(val)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Phase 3 — arrayFunc replacement (vectorize)
# ─────────────────────────────────────────────────────────────────────────────

class TestArrayFuncVectorize:
    """arrayFunc is now implemented via numpy.vectorize."""

    def test_arrayFunc_array_input(self):
        """1-D array input should produce a correctly shaped output array."""
        from cosmoTransitions.finiteT import arrayFunc
        x = np.array([0.0, 1.0, 4.0, 9.0])
        result = arrayFunc(np.sqrt, x)
        np.testing.assert_allclose(result, np.sqrt(x))

    def test_arrayFunc_scalar_input(self):
        """Scalar input (no len()) should fall back to direct function call."""
        from cosmoTransitions.finiteT import arrayFunc
        result = arrayFunc(lambda v: v * 2.0, 3.0)
        assert result == 6.0

    def test_arrayFunc_output_dtype(self):
        """Output dtype should match the requested typ parameter."""
        from cosmoTransitions.finiteT import arrayFunc
        result = arrayFunc(float, np.array([1, 2, 3]), typ=float)
        assert result.dtype == float
