"""
tests/test_transitionFinder.py
-------------------------------
Regression tests for cosmoTransitions.transitionFinder and the
traceMinimum / findAllTransitions pipeline as exercised through model1.

Golden reference values locked on 2026-04-24 using conda base
(Python 3.12, numpy 1.26.4, scipy 1.13.1).

model1 baseline:
  Number of phases: 3
    Phase 0: T=[1.0,   117.2],   X_start=[295.56, 406.39]
    Phase 1: T=[77.6,  222.7],   X_start=[234.30,-111.49]
    Phase 2: T=[223.2, 1000.0],  X_start=[-0.090,  0.068]
  Number of Tc transitions: 2
    Tc[0] = 222.94942912744762   (symmetry-restoring, degenerate vev)
    Tc[1] = 109.40840756818058   (electroweak-like, large vev change)
"""
import numpy as np
import pytest

from cosmoTransitions import transitionFinder


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module API
# ─────────────────────────────────────────────────────────────────────────────

def test_module_imports():
    """transitionFinder must import and expose traceMinimum, findAllTransitions."""
    import importlib
    mod = importlib.import_module("cosmoTransitions.transitionFinder")
    for name in ["traceMinimum", "findAllTransitions", "findCriticalTemperatures"]:
        assert hasattr(mod, name), f"transitionFinder missing '{name}'"


def test_Phase_class_exists():
    """Phase class must exist with the expected attributes."""
    ph = transitionFinder.Phase
    assert ph is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. traceMinimum — direct unit test on a simple analytic potential
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceMinimum:
    """
    Simple test potential:  f(x, T) = (x - x0(T))² + const
    where x0(T) = 1 - T/200.  Minimum at x = x0(T).
    The minimum disappears at T = 200.
    """

    @staticmethod
    def _f(x, T):
        x0 = 1.0 - T / 200.0
        return (x - x0)**2

    @staticmethod
    def _d2f_dx2(x, T):
        return np.array([[2.0]])

    @staticmethod
    def _d2f_dxdt(x, T):
        return np.array([-1.0 / 100.0])

    def test_traceMinimum_tracks_analytic_minimum(self):
        """
        Trace minimum from T=0 to T=100.
        At T=100: x_min = 1 - 100/200 = 0.5.
        """
        x0 = np.array([1.0])   # starting point at T=0
        result = transitionFinder.traceMinimum(
            self._f, self._d2f_dxdt, self._d2f_dx2,
            x0, t0=0.0, tstop=100.0,
            dtstart=5.0, deltaX_target=0.5,
        )
        # Final tracked x should be near 0.5
        assert abs(result.X[-1][0] - 0.5) < 0.05

    def test_traceMinimum_T_coverage(self):
        """T array in the result should span from t0 to approximately tstop."""
        x0 = np.array([1.0])
        result = transitionFinder.traceMinimum(
            self._f, self._d2f_dxdt, self._d2f_dx2,
            x0, t0=0.0, tstop=100.0,
            dtstart=5.0, deltaX_target=0.5,
        )
        assert result.T[0] == pytest.approx(0.0, abs=0.01)
        assert result.T[-1] >= 90.0

    def test_traceMinimum_returns_named_fields(self):
        """Return value must have X, T, dXdT, overX, overT."""
        x0 = np.array([1.0])
        result = transitionFinder.traceMinimum(
            self._f, self._d2f_dxdt, self._d2f_dx2,
            x0, t0=0.0, tstop=50.0,
            dtstart=5.0, deltaX_target=0.5,
        )
        for field in ("X", "T", "dXdT", "overX", "overT"):
            assert hasattr(result, field), f"traceMinimum result missing '{field}'"


# ─────────────────────────────────────────────────────────────────────────────
# 3. getPhases — phase structure of model1
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestModel1Phases:
    """All tests in this class use the session-cached model1 phases fixture."""

    def test_num_phases(self, model1_phases):
        """Baseline: model1 has exactly 3 phases."""
        assert len(model1_phases) == 3

    def test_phase_objects_have_required_attributes(self, model1_phases):
        """Each Phase object must expose T, X, dXdT, valAt."""
        for ph in model1_phases.values():
            assert hasattr(ph, "T"),    "Phase missing attribute 'T'"
            assert hasattr(ph, "X"),    "Phase missing attribute 'X'"
            assert hasattr(ph, "dXdT"), "Phase missing attribute 'dXdT'"
            assert hasattr(ph, "valAt"), "Phase missing attribute 'valAt'"

    def test_phase_T_arrays_nonempty(self, model1_phases):
        """Each Phase must cover a non-trivial temperature range."""
        for ph in model1_phases.values():
            assert len(ph.T) >= 5
            assert ph.T.max() > ph.T.min()

    def test_phase_X_shape(self, model1_phases):
        """X array must have shape (n_steps, 2) for a 2-field model."""
        for ph in model1_phases.values():
            assert ph.X.ndim == 2
            assert ph.X.shape[1] == 2

    def test_phase0_T_range(self, model1_phases):
        """Baseline: Phase 0 covers T=[0.0, 117.2]."""
        ph = model1_phases[0]
        assert ph.T.min() == pytest.approx(0.0, abs=0.5)
        assert ph.T.max() == pytest.approx(117.2, abs=2.0)

    def test_phase2_high_T_near_zero_vev(self, model1_phases):
        """Baseline: Phase 2 (high-T symmetric) X_start ≈ (-0.090, 0.068)."""
        ph = model1_phases[2]
        assert np.max(np.abs(ph.X[0])) < 1.0  # near-zero vev

    def test_valAt_returns_array(self, model1_phases):
        """valAt(T) should return a 1D field array."""
        ph = list(model1_phases.values())[0]
        T_mid = 0.5 * (ph.T.min() + ph.T.max())
        X = ph.valAt(T_mid)
        assert hasattr(X, "__len__")
        assert len(X) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. calcTcTrans — critical temperatures for model1
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestModel1TcTransitions:
    """Critical temperature regression tests for model1."""

    def test_num_transitions(self, model1_tc_transitions):
        """Baseline: model1 has exactly 2 Tc transitions."""
        assert len(model1_tc_transitions) == 2

    def test_Tc0_reference_value(self, model1_tc_transitions):
        """
        Baseline: Tc[0] = 222.94942912744762.
        Tolerance: 0.01% relative (0.1 K on ~200 K scale is easily achievable).
        """
        Tc0 = model1_tc_transitions[0]["Tcrit"]
        assert Tc0 == pytest.approx(222.9494291, rel=1e-4)

    def test_Tc1_reference_value(self, model1_tc_transitions):
        """Baseline: Tc[1] = 109.40840756818058."""
        Tc1 = model1_tc_transitions[1]["Tcrit"]
        assert Tc1 == pytest.approx(109.4084076, rel=1e-4)

    def test_transition_dict_keys(self, model1_tc_transitions):
        """Each transition dict must contain Tcrit, low_vev, high_vev."""
        for tr in model1_tc_transitions:
            for key in ("Tcrit", "low_vev", "high_vev"):
                assert key in tr, f"Transition dict missing key '{key}'"

    def test_Tc_ordering(self, model1_tc_transitions):
        """Tc transitions should be returned in descending temperature order."""
        Tcs = [tr["Tcrit"] for tr in model1_tc_transitions]
        assert Tcs[0] > Tcs[1]

    def test_low_vev_Tc1_is_large(self, model1_tc_transitions):
        """
        Baseline: at Tc[1], the low-vev phase has large field values
        ([263.49, 314.65]), indicating a strong first-order EW-like transition.
        """
        low_vev = model1_tc_transitions[1]["low_vev"]
        assert np.max(np.abs(low_vev)) > 100.0

    def test_vev_arrays_have_correct_dim(self, model1_tc_transitions):
        """low_vev and high_vev must be 2-component arrays (2-field model)."""
        for tr in model1_tc_transitions:
            assert len(tr["low_vev"]) == 2
            assert len(tr["high_vev"]) == 2
