"""
Phase 6 tests: TunnelingConfig, ε auto-derivation, and TunnelingConfig
integration into transitionFinder.

Covers:
  - config._epsilon_to_params tier selection
  - config.TunnelingConfig.get_nucl_criterion
  - SingleFieldInstanton._estimate_epsilon
  - findProfile thinCutoff/rmin=None auto-resolution (smoke test)
  - tunnelFromPhase / findAllTransitions accept tunneling_config
"""

import math
import numpy as np
import pytest

from cosmoTransitions import tunneling1D as t1
from cosmoTransitions.config import (
    TunnelingConfig,
    _epsilon_to_params,
    _PROFILE_PARAM_TIERS,
    fixed_140_nucl_criterion,
    cosmological_nucl_criterion,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _quartic_well(asymmetry: float):
    """
    Asymmetric double-well  V(φ) = φ⁴/4 - φ²/2 + asymmetry*φ

    For positive asymmetry the negative-φ minimum is deeper (true vacuum)
    and the positive-φ minimum is metastable.
    """
    from scipy.optimize import minimize_scalar

    def V(phi):
        return phi**4 / 4 - phi**2 / 2 + asymmetry * phi

    def dV(phi):
        return phi**3 - phi + asymmetry

    # absMin = global minimum (deeper, negative-φ side for positive asymmetry)
    phi_absMin = minimize_scalar(V, bounds=(-2.0, -0.5), method="bounded").x
    # metaMin = local (metastable) minimum (higher energy, positive-φ side)
    phi_metaMin = minimize_scalar(V, bounds=(0.5, 2.0), method="bounded").x
    return dict(V=V, dV=dV, phi_absMin=phi_absMin, phi_metaMin=phi_metaMin)


def _make_instanton(asymmetry: float) -> t1.SingleFieldInstanton:
    w = _quartic_well(asymmetry)
    return t1.SingleFieldInstanton(
        w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"]
    )


# -----------------------------------------------------------------------
# 1. _epsilon_to_params tier mapping
# -----------------------------------------------------------------------

class TestEpsilonToParams:
    def test_large_epsilon_gives_first_tier(self):
        """ε ≥ first-tier threshold → loose params (thick-wall default)."""
        eps_thresh_0 = _PROFILE_PARAM_TIERS[0][0]
        tc, rm = _epsilon_to_params(eps_thresh_0 * 2)
        assert tc == _PROFILE_PARAM_TIERS[0][1]
        assert rm == _PROFILE_PARAM_TIERS[0][2]

    def test_small_epsilon_gives_last_tier(self):
        """ε → 0 → tightest tier."""
        tc, rm = _epsilon_to_params(0.0)
        assert tc == _PROFILE_PARAM_TIERS[-1][1]
        assert rm == _PROFILE_PARAM_TIERS[-1][2]

    def test_intermediate_epsilon_selects_middle_tier(self):
        """An ε in the middle range selects a stricter-than-default tier."""
        if len(_PROFILE_PARAM_TIERS) < 2:
            pytest.skip("Need at least 2 tiers")
        eps_mid = (_PROFILE_PARAM_TIERS[0][0] + _PROFILE_PARAM_TIERS[1][0]) / 2
        tc, rm = _epsilon_to_params(eps_mid)
        # thinCutoff should be stricter than the first tier
        assert tc <= _PROFILE_PARAM_TIERS[0][1]

    def test_tier_params_strictly_decreasing(self):
        """Each successive tier must have tighter (smaller) thinCutoff and rmin."""
        for i in range(len(_PROFILE_PARAM_TIERS) - 1):
            tc_curr = _PROFILE_PARAM_TIERS[i][1]
            tc_next = _PROFILE_PARAM_TIERS[i + 1][1]
            rm_curr = _PROFILE_PARAM_TIERS[i][2]
            rm_next = _PROFILE_PARAM_TIERS[i + 1][2]
            assert tc_next < tc_curr, "thinCutoff must decrease tier-to-tier"
            assert rm_next < rm_curr, "rmin must decrease tier-to-tier"


# -----------------------------------------------------------------------
# 2. TunnelingConfig nucleation criteria
# -----------------------------------------------------------------------

class TestNuclCriterion:
    def test_fixed_140_default(self):
        cfg = TunnelingConfig()
        fn = cfg.get_nucl_criterion()
        S, T = 140.0 * 100.0, 100.0
        assert abs(fn(S, T)) < 1e-6

    def test_fixed_140_function_directly(self):
        assert fixed_140_nucl_criterion(140.0, 1.0) == pytest.approx(0.0, abs=1e-9)
        assert fixed_140_nucl_criterion(141.0, 1.0) > 0
        assert fixed_140_nucl_criterion(139.0, 1.0) < 0

    def test_cosmological_criterion_at_100GeV(self):
        """At T=100 GeV the threshold should be > 140 (≈143)."""
        val = cosmological_nucl_criterion(0.0, 100.0)  # should be < 0 (S=0)
        threshold = 4 * math.log(1.22e19 / 100.0)
        assert abs(-val - threshold) < 1e-6

    def test_cosmological_from_config(self):
        cfg = TunnelingConfig(nuclCriterion="cosmological")
        fn = cfg.get_nucl_criterion()
        # at S=0, T=1, fn < 0 (always nucleated)
        assert fn(0.0, 1.0) < 0

    def test_custom_callable(self):
        custom = lambda S, T: S / T - 150.0
        cfg = TunnelingConfig(nuclCriterion=custom)
        fn = cfg.get_nucl_criterion()
        assert fn(150.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_invalid_string_raises(self):
        cfg = TunnelingConfig(nuclCriterion="unknown_criterion")
        with pytest.raises(ValueError):
            cfg.get_nucl_criterion()


# -----------------------------------------------------------------------
# 3. SingleFieldInstanton._estimate_epsilon
# -----------------------------------------------------------------------

class TestEstimateEpsilon:
    def test_thick_wall_epsilon_near_one(self):
        """Quartic well with small asymmetry → thick-wall, ε should be O(1)."""
        inst = _make_instanton(0.1)
        eps = inst._estimate_epsilon()
        assert eps > 0.01, f"Expected ε > 0.01 for thick-wall, got {eps}"
        assert eps < 100.0, f"Expected ε < 100 for quartic well, got {eps}"

    def test_thin_wall_epsilon_small(self):
        """Very small asymmetry → near-degenerate vacua → thin-wall, ε ≪ 1."""
        inst = _make_instanton(0.01)
        eps = inst._estimate_epsilon()
        # In the thin-wall limit ε = ΔV / ΔV_barrier should be small
        thick_eps = _make_instanton(0.1)._estimate_epsilon()
        assert eps < thick_eps, "Smaller asymmetry should give smaller ε"

    def test_phi_bar_top_cached(self):
        """After _estimate_epsilon(), phi_bar_top should be set."""
        inst = _make_instanton(0.2)
        assert inst.phi_bar_top is not None
        phi_bt = inst.phi_bar_top
        inst._estimate_epsilon()  # call again — should use cached value
        assert inst.phi_bar_top == phi_bt

    def test_epsilon_always_positive(self):
        """ε must be > 0 for any valid double-well potential."""
        for asym in [0.05, 0.15, 0.3]:
            inst = _make_instanton(asym)
            eps = inst._estimate_epsilon()
            assert eps > 0, f"ε must be positive, got {eps} for asymmetry={asym}"


# -----------------------------------------------------------------------
# 4. findProfile None → auto-resolution (regression)
# -----------------------------------------------------------------------

class TestFindProfileAutoParams:
    def test_default_none_produces_valid_profile(self):
        """findProfile(thinCutoff=None, rmin=None) should produce a valid profile."""
        inst = _make_instanton(0.3)
        profile = inst.findProfile()  # defaults are now None
        assert hasattr(profile, "R")
        assert hasattr(profile, "Phi")
        assert profile.R.shape == profile.Phi.shape
        assert len(profile.R) > 0

    def test_explicit_params_match_auto_for_thick_wall(self):
        """
        For a clearly thick-wall potential, explicit old defaults and auto
        should give compatible actions (within 0.5%).
        """
        inst_auto = _make_instanton(0.3)
        inst_expl = _make_instanton(0.3)
        p_auto = inst_auto.findProfile()       # thinCutoff=None → auto
        p_expl = inst_expl.findProfile(thinCutoff=0.01, rmin=1e-4)
        S_auto = inst_auto.findAction(p_auto)
        S_expl = inst_expl.findAction(p_expl)
        # Should agree to within 0.5%
        assert abs(S_auto - S_expl) / S_expl < 0.005, (
            f"Auto params give S={S_auto:.4f}, explicit give S={S_expl:.4f}"
        )

    def test_profile_R_increasing(self):
        """Profile R must be strictly increasing."""
        profile = _make_instanton(0.2).findProfile()
        assert np.all(np.diff(profile.R) > 0)

    def test_profile_Phi_monotone(self):
        """For a standard double-well, Phi should be monotone along R."""
        profile = _make_instanton(0.2).findProfile()
        diffs = np.diff(profile.Phi)
        assert np.all(diffs >= 0) or np.all(diffs <= 0), \
            "Profile Phi should be monotone"


# -----------------------------------------------------------------------
# 5. TunnelingConfig integration in transitionFinder
# -----------------------------------------------------------------------

class TestTransitionFinderConfig:
    """Smoke tests: tunnelFromPhase and findAllTransitions accept tunneling_config."""

    @pytest.fixture
    def simple_model_phases(self):
        """Use the testModel1 example to get phases."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
        try:
            from testModel1 import model1
        except ImportError:
            pytest.skip("testModel1 not available")
        m = model1()
        m.getPhases()
        return m

    def test_findAllTransitions_accepts_config(self, simple_model_phases):
        m = simple_model_phases
        cfg = TunnelingConfig()
        # Should not raise; config forwarded through the call chain
        result = m.findAllTransitions(tunneling_config=cfg)
        assert isinstance(result, list)

    def test_findAllTransitions_cosmological_criterion(self, simple_model_phases):
        m = simple_model_phases
        cfg_cos = TunnelingConfig(nuclCriterion="cosmological")
        cfg_def = TunnelingConfig(nuclCriterion="fixed_140")
        # Both should produce a transition list (may differ in Tnuc)
        m.TnTrans = None
        result_cos = m.findAllTransitions(tunneling_config=cfg_cos)
        m.TnTrans = None
        result_def = m.findAllTransitions(tunneling_config=cfg_def)
        assert isinstance(result_cos, list)
        assert isinstance(result_def, list)

    def test_supercooling_preset_is_dataclass(self):
        cfg = TunnelingConfig.supercooling_preset()
        assert isinstance(cfg, TunnelingConfig)
        assert cfg.thinCutoff == 1e-4
        assert cfg.rmin == 1e-7


# -----------------------------------------------------------------------
# 6. TunnelingConfig.get_findProfile_kwargs
# -----------------------------------------------------------------------

class TestGetFindProfileKwargs:
    def test_auto_with_epsilon(self):
        cfg = TunnelingConfig()  # thinCutoff="auto", rmin="auto"
        kwargs = cfg.get_findProfile_kwargs(epsilon=0.5)
        assert "thinCutoff" in kwargs
        assert "rmin" in kwargs
        assert isinstance(kwargs["thinCutoff"], float)
        assert isinstance(kwargs["rmin"], float)

    def test_explicit_values_not_overridden(self):
        cfg = TunnelingConfig(thinCutoff=1e-3, rmin=1e-6)
        kwargs = cfg.get_findProfile_kwargs(epsilon=0.001)
        assert kwargs["thinCutoff"] == 1e-3
        assert kwargs["rmin"] == 1e-6

    def test_no_epsilon_falls_back_to_default(self):
        cfg = TunnelingConfig()  # "auto"
        kwargs = cfg.get_findProfile_kwargs(epsilon=None)
        # Should fall back to original defaults
        assert kwargs["thinCutoff"] == 0.01
        assert kwargs["rmin"] == 1e-4
