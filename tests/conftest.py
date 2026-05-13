"""
tests/conftest.py
-----------------
Shared pytest fixtures for CosmoTransitions regression tests (Phase 0 baseline).

All session-scoped fixtures are computed once per test run and cached, to avoid
re-running expensive phase-tracing computations multiple times.
"""
import sys
import os
import numpy as np
import pytest

# Ensure project root is on sys.path regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Model fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model1_instance():
    """examples/testModel1.py with default parameters."""
    from examples.testModel1 import model1
    return model1()


@pytest.fixture(scope="session")
def model1_tc_transitions(model1_instance):
    """
    model1.calcTcTrans() result, session-cached (~30 s on first run).

    Baseline values (locked 2026-04-24):
      Tc[0] = 222.94942912744762
      Tc[1] = 109.40840756818058
    """
    return model1_instance.calcTcTrans()


@pytest.fixture(scope="session")
def model1_phases(model1_instance):
    """model1.getPhases() result, session-cached."""
    return model1_instance.getPhases()


# ── Simple 1D potentials ──────────────────────────────────────────────────────

@pytest.fixture
def quartic_well_delta03():
    """
    Asymmetric double-well:  V(φ) = (φ²−1)² − 0.3·φ

    Baseline values (locked 2026-04-24, conda base / scipy 1.13.1):
      phi_absMin  ≈  1.03558030   (true minimum, lower energy)
      phi_metaMin ≈ -0.96014959   (metastable minimum)
      S3 (O3, α=2) ≈ 302.69415144
      S4 (O4, α=3) ≈ 7532.45798088
    """
    delta = 0.3
    from scipy.optimize import minimize_scalar

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
        V_absMin=res_abs.fun,
        V_metaMin=res_meta.fun,
        S3_ref=302.69415144,
        S4_ref=7532.45798088,
    )


@pytest.fixture
def quartic_well_delta05():
    """
    Asymmetric double-well:  V(φ) = (φ²−1)² − 0.5·φ

    Baseline values (locked 2026-04-24):
      S3 (O3) ≈ 102.92078560
      S4 (O4) ≈ 1525.09307116
    """
    delta = 0.5
    from scipy.optimize import minimize_scalar

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
        S3_ref=102.92078560,
        S4_ref=1525.09307116,
    )


# ── 2D potential (pathDeformation) ───────────────────────────────────────────

@pytest.fixture
def full_tunneling_potential():
    """
    examples/fullTunneling.py Potential with default params (c=5, fx=10, fy=10).

    Minima (locked 2026-04-24):
      true min  ≈ (1.0, 1.0),   V ≈ -1.66666667
      false min ≈ (0.0, 0.0),   V ≈  0.0
    """
    from examples.fullTunneling import Potential
    return Potential(c=5., fx=10., fy=10.)


# ── findAllTransitions fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def supercooled_model():
    """
    SupercooledSingleField from example_03 with default parameters
    (D=0.10, E=0.09, T0=50, lam=0.10).  Tn/Tc ≈ 0.74 (26% supercooling).
    Exercises the happy path: primary brentq([Tmin, Tmax]) finds a sign change
    directly and returns Tnuc without any fallback.
    """
    from examples.example_03_supercooled_ewpt import SupercooledSingleField
    return SupercooledSingleField()


@pytest.fixture(scope="session")
def supercooled_transitions(supercooled_model):
    """findAllTransitions() for the supercooled model, session-cached."""
    return supercooled_model.findAllTransitions()


@pytest.fixture(scope="session")
def conformal_xSM_instance():
    """
    xSMZ2_Conformal model (Z2-symmetric conformal singlet-SM extension) with
    vs=100, MS=491.081377, kappaS=0.724345.

    In this model the symmetric phase (h=0, S=vs) traces all the way to T=0
    (Tmin=0), so the primary brentq([0, Tmax]) fails because the action at T=0
    is spuriously large (no thermal barrier at T=0 for a conformal potential).
    This exercises the fallback path in tunnelFromPhase.

    Regression fixture for the crash:
        'f(a) and f(b) must have different signs'
    that previously propagated from an unguarded inner brentq([Tmin_opt, Tmax_Tc]).
    """
    from cosmoTransitions import generic_potential

    _MHL = 125.09
    _GF = 1.1663787e-5
    _MT = 173.5
    _VEV = (np.sqrt(2) * _GF) ** (-0.5)
    _GWEAK = 0.65742
    _GHYPER = 0.34123
    _YT = np.sqrt(2) * _MT / _VEV

    class xSMZ2_Conformal(generic_potential.generic_potential):
        def init(self, vs, MS, kappaS):
            self.Ndim = 2
            self.renormScaleSq = _VEV ** 2
            self.kappaH = _MHL ** 2 / _VEV ** 2
            self.kappaS = kappaS
            self.lamSH = 2 * MS ** 2 / _VEV ** 2
            self.wh0 = _VEV
            self.ws0 = vs
            self.ch = (self.lamSH / 12.0
                       + (3.0 * _GWEAK ** 2 + _GHYPER ** 2) / 16.0
                       + _YT ** 2 / 4.0)
            self.cs = self.lamSH / 12.0

        def forbidPhaseCrit(self, X):
            X = np.asanyarray(X)
            return (X[..., 0] < -5.0).any() or (X[..., 1] < -5.0).any()

        def V0(self, X):
            X = np.asanyarray(X)
            h, s = X[..., 0], X[..., 1]
            r = (self.kappaH / 4.0 * h ** 4
                 * (np.log(h ** 2 / self.wh0 ** 2 + 1e-10) / 2.0 - 0.25))
            r += (self.kappaS / 4.0 * s ** 4
                  * (np.log(s ** 2 / self.ws0 ** 2 + 1e-10) / 2.0 - 0.25))
            r += self.lamSH / 4.0 * h ** 2 * s ** 2
            return r

        def V1T_from_X(self, X, T, include_radiation=True):
            T = np.asanyarray(T)
            X = np.asanyarray(X)
            h, s = X[..., 0], X[..., 1]
            return (self.ch / 2.0 * T ** 2 * h ** 2
                    + self.cs / 2.0 * T ** 2 * s ** 2)

        def Vtot(self, X, T, include_radiation=True):
            return self.V0(X) + self.V1T_from_X(X, T)

        def approxZeroTMin(self):
            return [np.array([0.0, self.ws0]), np.array([self.wh0, 0.0])]

    return xSMZ2_Conformal(100, 491.081377, 0.724345)


@pytest.fixture(scope="session")
def conformal_xSM_transitions(conformal_xSM_instance):
    """
    findAllTransitions() for the conformal xSM model, session-cached.

    If findAllTransitions() raises, the fixture itself errors out and every
    test that depends on it is marked ERROR — which is the intended signal for
    the regression case before the fix is applied.
    """
    return conformal_xSM_instance.findAllTransitions()
