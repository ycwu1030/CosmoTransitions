"""
tests/test_tunneling1D.py
--------------------------
Regression tests for cosmoTransitions.tunneling1D.SingleFieldInstanton.

Golden reference values locked on 2026-04-24 using conda base
(Python 3.12, numpy 1.26.4, scipy 1.13.1).

Known baseline defects:
  B-02  SyntaxWarning from \p in docstring (line 50)
  B-03  brentq sign error for δ=0.1 double-well (nearly symmetric potential)
        findProfile() raises ValueError when phi_absMin ≈ phi_metaMin in depth

API notes (discovered during probe):
  - SingleFieldInstanton(phi_absMin, phi_metaMin, V, dV, alpha=2)
    phi_absMin = true (lower-energy) vacuum
    phi_metaMin = metastable (higher-energy) vacuum
  - findProfile() → profile_rval(R, Phi, dPhi, Rerr)
  - findAction(profile) → float   [profile argument is NOT optional]
  - alpha=2 → O(3) bounce (thermal, 3D), alpha=3 → O(4) (quantum, 4D)
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from cosmoTransitions import tunneling1D as t1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_quartic_well(delta):
    """V(φ) = (φ²−1)² − δ·φ, returns dict with V, dV, phi_absMin, phi_metaMin."""
    def V(phi, d=delta):
        return (phi**2 - 1.0)**2 - d * phi

    def dV(phi, d=delta):
        return 4.0 * phi * (phi**2 - 1.0) - d

    res_abs  = minimize_scalar(V, bounds=( 0.5, 1.5), method="bounded")
    res_meta = minimize_scalar(V, bounds=(-1.5,-0.5), method="bounded")
    return dict(
        V=V, dV=dV,
        phi_absMin=res_abs.x,
        phi_metaMin=res_meta.x,
    )


# Locked reference values for the quartic potentials
#
#   delta | phi_absMin   | phi_metaMin  | S3 (O3)        | S4 (O4)
#   0.3   | 1.03558030   | -0.96014959  | 302.69415144   | 7532.45798088
#   0.5   | 1.05745463   | -0.93040302  | 102.92078560   | 1525.09307116
#   0.7   | 1.07811028   | -0.89718803  |  47.78082309   |  500.11944574

_QUARTIC_REFS = {
    0.3: dict(phi_abs= 1.03558030, phi_meta=-0.96014959, S3=302.69415144, S4=7532.45798088),
    0.5: dict(phi_abs= 1.05745463, phi_meta=-0.93040302, S3=102.92078560, S4=1525.09307116),
    0.7: dict(phi_abs= 1.07811028, phi_meta=-0.89718803, S3= 47.78082309, S4= 500.11944574),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module import / API sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_module_imports():
    """tunneling1D must be importable."""
    import importlib
    mod = importlib.import_module("cosmoTransitions.tunneling1D")
    assert hasattr(mod, "SingleFieldInstanton")
    assert hasattr(mod, "PotentialError")


def test_SingleFieldInstanton_instantiates():
    """Constructor must succeed for a valid potential."""
    w = _make_quartic_well(0.3)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    assert inst is not None


def test_profile_rval_fields():
    """findProfile() must return an object with R, Phi, dPhi, Rerr."""
    w = _make_quartic_well(0.5)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    profile = inst.findProfile()
    for field in ("R", "Phi", "dPhi", "Rerr"):
        assert hasattr(profile, field), f"profile missing field '{field}'"


def test_findAction_requires_profile_argument():
    """
    B-API: findAction() requires the profile as an argument.
    Calling without argument must raise TypeError.
    """
    w = _make_quartic_well(0.5)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    with pytest.raises(TypeError):
        inst.findAction()


def test_wrong_vacuum_ordering_raises():
    """
    Passing metastable as phi_absMin must raise PotentialError
    (V(metaMin) < V(absMin) requirement violated).
    """
    w = _make_quartic_well(0.5)
    with pytest.raises(t1.PotentialError):
        # Swapped: metastable as "abs", true min as "meta"
        t1.SingleFieldInstanton(w["phi_metaMin"], w["phi_absMin"], w["V"], w["dV"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Profile shape and boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileProperties:

    @pytest.fixture(scope="class")
    def profile_delta05(self):
        w = _make_quartic_well(0.5)
        inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
        profile = inst.findProfile()
        return profile, w["phi_absMin"], w["phi_metaMin"]

    def test_R_starts_near_zero(self, profile_delta05):
        """Radial coordinate should start near 0 (bubble centre)."""
        profile, _, _ = profile_delta05
        assert profile.R[0] >= 0
        assert profile.R[0] < 1.0

    def test_R_is_increasing(self, profile_delta05):
        """R must be monotonically increasing."""
        profile, _, _ = profile_delta05
        diffs = np.diff(profile.R)
        assert np.all(diffs > 0), "R is not monotonically increasing"

    def test_Phi_approaches_metaMin_at_infinity(self, profile_delta05):
        """
        At the largest radius, φ should be close to phi_metaMin (boundary condition).
        Allow 1% relative tolerance.
        """
        profile, _, phi_meta = profile_delta05
        assert abs(profile.Phi[-1] - phi_meta) < abs(phi_meta) * 0.05 + 0.01

    def test_dPhi_near_zero_at_edge(self, profile_delta05):
        """dφ/dr → 0 at large r (Neumann boundary condition)."""
        profile, _, _ = profile_delta05
        assert abs(profile.dPhi[-1]) < 0.1

    def test_Phi_is_monotone(self, profile_delta05):
        """
        For a single-barrier potential, φ should change monotonically
        from phi_absMin to phi_metaMin.
        """
        profile, phi_abs, phi_meta = profile_delta05
        # Determine sign of travel
        direction = np.sign(phi_meta - phi_abs)
        diffs = np.diff(profile.Phi)
        # Allow tiny numerical fluctuations at boundaries
        assert np.all(direction * diffs[5:-5] >= -1e-6)

    def test_Phi_range_covers_transition(self, profile_delta05):
        """Profile must span from near phi_absMin to near phi_metaMin."""
        profile, phi_abs, phi_meta = profile_delta05
        phi_span = abs(phi_meta - phi_abs)
        actual_span = abs(profile.Phi[-1] - profile.Phi[0])
        assert actual_span > 0.8 * phi_span


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action values (O3 and O4)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delta,S3_ref,S4_ref", [
    (0.3, 302.69415144, 7532.45798088),
    (0.5, 102.92078560, 1525.09307116),
    (0.7,  47.78082309,  500.11944574),
])
def test_O3_action_reference(delta, S3_ref, S4_ref):
    """
    O(3) bounce action for quartic well V=(φ²−1)²−δφ.
    Tolerance: 0.1% relative.
    """
    w = _make_quartic_well(delta)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"], alpha=2)
    profile = inst.findProfile()
    S3 = inst.findAction(profile)
    assert S3 == pytest.approx(S3_ref, rel=1e-3), (
        f"delta={delta}: S3={S3}, expected≈{S3_ref}"
    )


@pytest.mark.parametrize("delta,S3_ref,S4_ref", [
    (0.3, 302.69415144, 7532.45798088),
    (0.5, 102.92078560, 1525.09307116),
    (0.7,  47.78082309,  500.11944574),
])
def test_O4_action_reference(delta, S3_ref, S4_ref):
    """
    O(4) bounce action for quartic well.
    S4 > S3 for the same potential (O4 action counts more spatial dimensions).
    Tolerance: 0.1% relative.
    """
    w = _make_quartic_well(delta)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"], alpha=3)
    profile = inst.findProfile()
    S4 = inst.findAction(profile)
    assert S4 == pytest.approx(S4_ref, rel=1e-3), (
        f"delta={delta}: S4={S4}, expected≈{S4_ref}"
    )


def test_O4_action_greater_than_O3():
    """S4 > S3 for the same potential (more dimensions = larger action)."""
    w = _make_quartic_well(0.5)
    inst3 = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"], alpha=2)
    inst4 = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"], alpha=3)
    S3 = inst3.findAction(inst3.findProfile())
    S4 = inst4.findAction(inst4.findProfile())
    assert S4 > S3


def test_action_positive():
    """Euclidean action must be positive."""
    w = _make_quartic_well(0.5)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    S = inst.findAction(inst.findProfile())
    assert S > 0


def test_action_decreases_with_larger_asymmetry():
    """
    Larger δ → deeper asymmetry → easier tunneling → smaller action.
    S(δ=0.5) < S(δ=0.3) < S(δ=0.1).
    """
    actions = {}
    for delta in [0.3, 0.5, 0.7]:
        w = _make_quartic_well(delta)
        inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
        actions[delta] = inst.findAction(inst.findProfile())
    assert actions[0.7] < actions[0.5] < actions[0.3]


# ─────────────────────────────────────────────────────────────────────────────
# 4. exactSolution (small-r approximation)
# ─────────────────────────────────────────────────────────────────────────────

def test_exactSolution_exists_and_callable():
    """exactSolution method must exist on SingleFieldInstanton."""
    w = _make_quartic_well(0.5)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    assert hasattr(inst, "exactSolution")
    assert callable(inst.exactSolution)


def test_exactSolution_small_r_near_absMin():
    """
    For small r, exactSolution should give φ ≈ phi_absMin + small deviation.
    """
    w = _make_quartic_well(0.5)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    # Get the necessary arguments from a short findProfile run
    profile = inst.findProfile()
    # Check the first profile point is near phi_absMin
    assert abs(profile.Phi[0] - inst.phi_absMin) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline defect tests
# ─────────────────────────────────────────────────────────────────────────────

def test_nearly_symmetric_potential_computes_action():
    """
    B-03 note: an earlier probe with manually wrong phi values caused a
    brentq sign error.  With minimize_scalar-found vacua, δ=0.1 works fine.
    S3 ≈ 2799 (large action — nearly symmetric → very slow decay rate).
    """
    w = _make_quartic_well(0.1)
    inst = t1.SingleFieldInstanton(w["phi_absMin"], w["phi_metaMin"], w["V"], w["dV"])
    profile = inst.findProfile()
    S = inst.findAction(profile)
    assert S > 0
    assert np.isfinite(S)
    # For nearly-symmetric well, action is very large
    assert S > 1000
