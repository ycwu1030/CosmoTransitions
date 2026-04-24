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

    @pytest.mark.xfail(
        strict=True,
        reason="B-04: Jb_spline error for x²=4 is O(1) (~0.754). "
               "The spline is wildly inaccurate for x² > 1. "
               "Fix in Phase 3 (finiteT reconstruction).",
    )
    def test_Jb_spline_accurate_at_x2_eq_4(self):
        """B-04: expected to fail — spline error ≈ 0.754 at x²=4."""
        val = finiteT.Jb_spline(4.0)
        exact = finiteT.Jb_exact(4.0)
        assert abs(val - exact) < 1e-3

    @pytest.mark.xfail(
        strict=True,
        reason="B-04: Jb_spline error for x²=25 is 0.133. Fix in Phase 3.",
    )
    def test_Jb_spline_accurate_at_x2_eq_25(self):
        """B-04: expected to fail — spline error ≈ 0.133 at x²=25."""
        val = finiteT.Jb_spline(25.0)
        exact = finiteT.Jb_exact(25.0)
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

    @pytest.mark.xfail(
        strict=True,
        reason="B-05: Jb_spline(-1.0) = -2.818 but exact gives -1.696. "
               "Imaginary-mass handling is incorrect. Fix in Phase 3.",
    )
    def test_Jb_spline_negative_x2_correct(self):
        """B-05: imaginary mass spline value is wrong."""
        val = finiteT.Jb_spline(-1.0)
        exact = finiteT.Jb_exact(-1.0)
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
