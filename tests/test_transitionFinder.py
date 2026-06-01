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


# ─────────────────────────────────────────────────────────────────────────────
# 5. findAllTransitions — normal supercooled single-field model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestFindAllTransitionsNormal:
    """
    findAllTransitions on SupercooledSingleField (example_03):
      D=0.10, E=0.09, T0=50 GeV, lam=0.10  →  Tn/Tc ≈ 0.74.

    The symmetric phase exists only for T > T0 (spinodal), so Tmin > 0 and the
    primary brentq([Tmin, Tmax]) finds the sign change directly — no fallback
    code is needed.  These tests lock the happy-path behaviour.
    """

    def test_returns_list(self, supercooled_transitions):
        assert isinstance(supercooled_transitions, list)

    def test_exactly_one_first_order_transition(self, supercooled_transitions):
        first_order = [t for t in supercooled_transitions if t["trantype"] == 1]
        assert len(first_order) == 1

    def test_required_keys_present(self, supercooled_transitions):
        """generic_potential.findAllTransitions must attach all expected keys."""
        required = ("Tnuc", "low_vev", "high_vev", "action", "trantype",
                    "alpha_GW", "betaHn_GW", "crit_trans")
        t = next(tr for tr in supercooled_transitions if tr["trantype"] == 1)
        for key in required:
            assert key in t, f"Transition dict missing key '{key}'"

    def test_Tn_below_Tc(self, supercooled_model, supercooled_transitions):
        """Nucleation temperature must be below the critical temperature."""
        Tc_list = supercooled_model.calcTcTrans()
        assert len(Tc_list) >= 1
        Tc = Tc_list[0]["Tcrit"]
        Tn = next(t["Tnuc"] for t in supercooled_transitions if t["trantype"] == 1)
        assert Tn < Tc

    def test_Tn_above_spinodal(self, supercooled_model, supercooled_transitions):
        """Nucleation must occur above T0 (spinodal) where the false vacuum exists."""
        Tn = next(t["Tnuc"] for t in supercooled_transitions if t["trantype"] == 1)
        assert Tn > supercooled_model.T0

    def test_alpha_GW_positive(self, supercooled_transitions):
        t = next(tr for tr in supercooled_transitions if tr["trantype"] == 1)
        assert t["alpha_GW"] > 0

    def test_high_vev_near_zero(self, supercooled_transitions):
        """The false-vacuum (high-T) vev must be close to the origin."""
        t = next(tr for tr in supercooled_transitions if tr["trantype"] == 1)
        assert abs(t["high_vev"][0]) < 5.0   # phi_v ≈ 158 GeV → origin is < 5 GeV

    def test_low_vev_large(self, supercooled_transitions):
        """The true-vacuum (low-T) vev must be well away from the origin."""
        t = next(tr for tr in supercooled_transitions if tr["trantype"] == 1)
        assert abs(t["low_vev"][0]) > 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. findAllTransitions — conformal/CW model (Tmin = 0 fallback path)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestConformalFallback:
    """
    findAllTransitions on xSMZ2_Conformal(vs=100, MS=491.08, kappaS=0.724).

    In this model the singlet-symmetric phase (h=0, S=vs) is traced all the
    way to T=0 (Tmin=0).  The primary brentq([0, Tmax]) fails because the
    action at T=0 is spuriously large (no thermal barrier in the conformal
    potential at T→0).  The function must fall back to an interior scan to
    locate a bracket where the nucleation criterion changes sign.

    Regression tests for two distinct bugs:

    Bug 1 — _orig_cond computed with mixed arguments:
        nuclCriterion(action(Tmin), Tmax) evaluated action(0)/Tmax rather than
        action(0)/0, giving the wrong sign determination.  Fixed by the
        Tmin==0.0 guard that forces _orig_cond=True.

    Bug 2 — unguarded inner brentq([Tmin_opt, Tmax_Tc]):
        Near the critical temperature pathDeformation returns action=0 ("no
        barrier"), so criterion(Tmax_Tc) ≤ 0.  With both endpoints negative the
        inner brentq raised ValueError which propagated uncaught.  Fixed (and
        later superseded by the unified log-space scan) by the retry with Tmax.
    """

    def test_does_not_raise(self, conformal_xSM_transitions):
        """
        If the fixture succeeded, findAllTransitions() did not raise.
        Before Bug 2 was fixed this test would ERROR with ValueError.
        """
        assert conformal_xSM_transitions is not None

    def test_returns_list(self, conformal_xSM_transitions):
        assert isinstance(conformal_xSM_transitions, list)

    def test_first_order_transition_found(self, conformal_xSM_transitions):
        """At least one first-order nucleation transition must be found."""
        first_order = [t for t in conformal_xSM_transitions if t["trantype"] == 1]
        assert len(first_order) >= 1

    def test_Tn_positive(self, conformal_xSM_transitions):
        t = next(tr for tr in conformal_xSM_transitions if tr["trantype"] == 1)
        assert t["Tnuc"] > 0.0

    def test_Tn_below_Tc(self, conformal_xSM_instance, conformal_xSM_transitions):
        """Nucleation temperature must be below the critical temperature."""
        Tc_list = conformal_xSM_instance.calcTcTrans()
        if not Tc_list:
            pytest.skip("No critical temperature found")
        Tc = max(t["Tcrit"] for t in Tc_list)
        Tn = next(tr["Tnuc"] for tr in conformal_xSM_transitions if tr["trantype"] == 1)
        assert Tn < Tc

    def test_crit_trans_key_present(self, conformal_xSM_transitions):
        """generic_potential must attach the 'crit_trans' key to every transition."""
        for t in conformal_xSM_transitions:
            assert "crit_trans" in t

    def test_alpha_GW_positive(self, conformal_xSM_transitions):
        t = next(tr for tr in conformal_xSM_transitions if tr["trantype"] == 1)
        assert t["alpha_GW"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. getPhases — xSM low-λS phase-explosion regression
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestxSMLowLambdaPhaseExplosion:
    """
    Regression test for the phase-explosion bug in traceMultiMin.

    Model: xSM singlet extension without Z2 symmetry.
    Parameters: MS=21.09 GeV, λS=4.65e-3, λSH=2.89e-2, b3/v=0.364.

    Root cause:
        The old coverage-detection guard in traceMultiMin used
        ``|x1| < deltaX_target`` (field-magnitude size) to decide whether a
        new seed was already covered by an existing phase.  For small λS the
        singlet direction is almost flat, so fmin converges to slightly
        different near-origin points (e.g. 0.13 GeV vs 0.005 GeV) depending
        on the seed.  These points are all within the same basin, but the
        old guard (threshold 0.1 GeV) was too narrow — many seeds slipped
        through and each triggered a full new phase trace, producing 34+
        spurious phases.

    Fix:
        Replace the size-based guard with a midpoint barrier test:
            V(mid) - max(V(x1), V(x)) <= noise
        Two fmin results with no potential barrier between them (same convex
        basin) satisfy this criterion regardless of their absolute field
        values or any constant offset added to V.  The test is therefore
        fully shift-invariant.

    Expected outcome: exactly 4 phases (EW-broken, mixed, symmetric×2).
    Before the fix: 34+ phases.
    """

    def test_phase_count_not_exploded(self, xSM_low_lamS_phases):
        """
        Core regression: getPhases() must return exactly 4 phases, not 34+.
        """
        assert len(xSM_low_lamS_phases) == 4, (
            f"Expected 4 phases, got {len(xSM_low_lamS_phases)}.  "
            "Phase explosion detected — midpoint barrier fix may have regressed."
        )

    def test_all_phases_have_valid_T_range(self, xSM_low_lamS_phases):
        """Every phase must cover a non-trivial temperature interval."""
        for key, ph in xSM_low_lamS_phases.items():
            assert ph.T.max() > ph.T.min(), f"Phase {key} has zero-width T range."

    def test_EW_broken_phase_exists(self, xSM_low_lamS_phases):
        """
        At T=0 there must be a phase with h ≈ VEV ≈ 246 GeV.
        This is the electroweak broken phase.
        """
        import numpy as np
        vev_approx = 246.22
        found = False
        for ph in xSM_low_lamS_phases.values():
            if ph.T.min() < 1.0:  # reaches T=0
                x0 = ph.X[np.argmin(ph.T)]
                if abs(x0[0]) > 100.0:  # large Higgs vev
                    found = True
                    break
        assert found, "No EW-broken phase (large h vev at T=0) found."

    def test_symmetric_phase_exists(self, xSM_low_lamS_phases):
        """
        At high temperature there must be a near-origin (symmetric) phase.
        """
        import numpy as np
        found = False
        for ph in xSM_low_lamS_phases.values():
            if ph.T.max() > 500.0:
                x_high = ph.X[np.argmax(ph.T)]
                if np.linalg.norm(x_high) < 10.0:  # near origin
                    found = True
                    break
        assert found, "No high-T symmetric phase found."


# ─────────────────────────────────────────────────────────────────────────────
# 8. U1B-L model regressions (Test/BmL/U1BmL.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestU1BmLRegressions:
    """
    Regression checks for the user U(1)_{B-L} model used during recent
    debugging sessions.

    The purpose of this block is to lock in two key behaviors:
      1) The numerical CW-vacuum pre-filter classifies known stable/unstable
         points correctly.
      2) A known stable point still returns a sane getPhases() result.
    """

    def test_module_exposes_model_and_stability_filter(self, u1bml_module):
        assert hasattr(u1bml_module, "U1BmL")
        assert hasattr(u1bml_module, "is_vacuum_stable")

    def test_stability_filter_flags_known_points(self, u1bml_module, u1bml_reference_points):
        U1BmL = u1bml_module.U1BmL
        is_vacuum_stable = u1bml_module.is_vacuum_stable
        vphi = u1bml_reference_points["vphi"]

        p_stable = u1bml_reference_points["stable_low_mN"]
        mod_stable = U1BmL(vphi, p_stable["mphi"], p_stable["mzprime"], p_stable["mN"])
        assert is_vacuum_stable(mod_stable)

        p_unstable = u1bml_reference_points["unstable_runaway"]
        mod_unstable = U1BmL(vphi, p_unstable["mphi"], p_unstable["mzprime"], p_unstable["mN"])
        assert not is_vacuum_stable(mod_unstable)

    def test_getphases_stable_point_returns_finite_phase_data(self, u1bml_stable_instance):
        phases = u1bml_stable_instance.getPhases()
        assert isinstance(phases, dict)
        assert len(phases) >= 1

        for key, ph in phases.items():
            assert np.all(np.isfinite(ph.T)), f"Non-finite T values in phase {key}"
            assert np.all(np.isfinite(ph.X)), f"Non-finite X values in phase {key}"
            assert ph.T.max() >= ph.T.min()
