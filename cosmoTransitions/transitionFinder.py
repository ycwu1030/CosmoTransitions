"""
The transitionFinder module is used to calculate finite temperature
cosmological phase transitions: it contains functions to find the phase
structure as a function of temperature, and functions to find the transition
(bubble nucleation) temperature for each phase.
In contrast, :mod:`.pathDefomration` is useful for finding the tunneling
solution for a fixed potential or a potential at a fixed temperature.

The most directly used functions in this module will likely be
:func:`traceMultiMin` for finding the phase structure, and
:func:`findAllTransitions` and :func:`findCriticalTemperatures` for calculating
properties of the phase transitions.
"""


from collections import namedtuple
from collections.abc import Callable

import numpy as np
from scipy import linalg, interpolate, optimize


from . import pathDeformation
from . import tunneling1D
from .config import TunnelingConfig

import logging
logger = logging.getLogger(__name__)


_traceMinimum_rval = namedtuple("traceMinimum_rval", "X T dXdT overX overT")
def traceMinimum(
        f: Callable[[np.ndarray, float], float],
        d2f_dxdt: Callable[[np.ndarray, float], np.ndarray],
        d2f_dx2: Callable[[np.ndarray, float], np.ndarray],
        x0: np.ndarray,
        t0: float,
        tstop: float,
        dtstart: float,
        deltaX_target: float,
        dtabsMax: float = 20.0,
        dtfracMax: float = .25,
        dtmin: float = 1e-3,
        deltaX_tol: float = 1.2,
        minratio: float = 1e-2,
) -> _traceMinimum_rval:
    """
    Trace the minimum `xmin(t)` of the function `f(x,t)`, starting at `x0, t0`.

    Parameters
    ----------
    f : callable
        The scalar function `f(x,t)` which needs to be minimized. The input will
        be of the same type as `(x0,t0)`.
    d2f_dxdt, d2f_dx2 : callable
        Functions which return returns derivatives of `f(x)`. `d2f_dxdt` should
        return the derivative of the gradient of `f(x)` with respect to `t`, and
        `d2f_dx2` should return the Hessian matrix of `f(x)` evaluated at `t`.
        Both should take as inputs `(x,t)`.
    x0 : array_like
        The initial starting point. Must be an array even if the potential is
        one-dimensional (in which case the array should have length 1).
    t0 : float
        The initial starting parameter `t`.
    tstop : float
        Stop the trace when `t` reaches `tstop`.
    dtstart : float
        Initial stepsize.
    deltaX_target : float
        The target error in x at each step. Determines the
        stepsize in t by extrapolation from last error.
    dtabsMax : float, optional
    dtfracMax : float, optional
        The largest stepsize in t will be the LARGEST of
        ``abs(dtstart)*dtabsMax`` and ``t*dtfracMax``.
    dtmin : float, optional
        The smallest stepsize we'll allow before assuming the transition ends,
        relative to `dtstart`
    deltaX_tol : float, optional
        ``deltaX_tol*deltaX_target`` gives the maximum error in x
        before we want to shrink the stepsize and recalculate the minimum.
    minratio : float, optional
        The smallest ratio between smallest and largest eigenvalues in the
        Hessian matrix before treating the smallest eigenvalue as zero (and
        thus signaling a saddle point and the end of the minimum).

    Returns
    -------
      X, T, dXdT : array_like
        Arrays of the minimum at different values of t, and
        its derivative with respect to t.
      overX : array_like
        The point beyond which the phase seems to disappear.
      overT : float
        The t-value beyond which the phase seems to disappear.

    Notes
    -----
    In prior versions, `d2f_dx2` was optional and called `d2f`, while `d2f_dxdt`
    was calculated from an optional parameter `df` using finite differences. If
    Neither of these were supplied, they would be calculated directly from
    `f(x,t)` using finite differences. This lead to a messier calling signature,
    since additional parameters were needed to find the finite differences. By
    instead requiring that the derivatives be supplied, the task of creating the
    derivative functions can be delegated to more general purpose routines
    (see e.g. :class:`helper_functions.gradientFunction` and
    :class:`helper_functions.hessianFunction`).

    Also new in this version, `dtmin` and `dtabsMax` are now relative to
    `dtstart`. The idea here is that there should be some required parameter
    that sets the scale, and then optional parameters can set the tolerances
    relative to this scale. `deltaX_target` is now not optional for the same
    reasoning.
    """
    logger.debug("traceMinimum t0 = %0.6g", t0)
    Ndim = len(x0)
    M0 = d2f_dx2(x0,t0)
    _eigs0 = linalg.eigvalsh(M0)
    _eig_ratio_0 = min(abs(_eigs0)) / max(abs(_eigs0))
    # S-6.8: apply the initial eigenvalue ratio but never let the effective
    # threshold drop below minratio * 0.01, so a badly-conditioned starting
    # Hessian doesn't make the spinodal detector blind.
    minratio = max(minratio * _eig_ratio_0, minratio * 0.01)

    def dxmindt(x,t):
        M = d2f_dx2(x,t)
        if abs(linalg.det(M)) < (1e-3*np.max(abs(M)))**Ndim:
            # Assume matrix is singular
            return None, False
        b = -d2f_dxdt(x,t)
        eigs = linalg.eigvalsh(M)
        try:
            dxdt = linalg.solve(M,b, overwrite_a=False, overwrite_b=False)
            # dxdt = linalg.solve(M,b, overwrite_a=True, overwrite_b=True)
            isneg = ((eigs <= 0).any() or min(eigs)/max(eigs) < minratio)
        except:
            dxdt = None
            isneg = False
        return dxdt, isneg
    xeps = deltaX_target * 1e-2

    def fmin(x,t):
        return optimize.fmin(f, x, args=(t,), xtol=xeps, ftol=np.inf,
                             disp=False)
    deltaX_tol = deltaX_tol * deltaX_target
    tscale = abs(dtstart)
    dtabsMax = dtabsMax * tscale
    dtmin = dtmin * tscale

    x,t,dt,xerr = x0,t0,dtstart,0.0
    dxdt, negeig = dxmindt(x,t)
    X,T,dXdT = [x],[t],[dxdt]
    overX = overT = None
    while dxdt is not None:
        logger.debug("traceMinimum step")
        # Get the values at the next step
        tnext = t+dt
        xnext = fmin(x+dxdt*dt, tnext)
        dxdt_next, negeig = dxmindt(xnext,tnext)
        if dxdt_next is None or negeig == True:
            # We got stuck on a saddle, so there must be a phase transition
            # there.
            dt *= .5
            overX, overT = xnext, tnext
        else:
            # The step might still be too big if it's outside of our error
            # tolerance.
            xerr = max(np.sum((x+dxdt*dt - xnext)**2),
                       np.sum((xnext-dxdt_next*dt - x)**2))**.5
            if xerr < deltaX_tol:  # Normal step, error is small
                T.append(tnext)
                X.append(xnext)
                dXdT.append(dxdt_next)
                if overT is None:
                    # change the stepsize only if the last step wasn't
                    # troublesome
                    dt *= deltaX_target/(xerr+1e-100)
                x,t,dxdt = xnext, tnext, dxdt_next
                overX = overT = None
            else:
                # Either stepsize was too big, or we hit a transition.
                # Just cut the step in half.
                dt *= .5
                overX, overT = xnext, tnext
        # Now do some checks on dt.
        # S-6.6: near the spinodal the Hessian becomes ill-conditioned, so
        # allow the step to be much smaller before declaring a transition.
        _M = d2f_dx2(x, t)
        _eigs = abs(linalg.eigvalsh(_M))
        _eig_max = _eigs.max() + 1e-100
        _eig_min = _eigs.min()
        _cond_ratio = _eig_min / _eig_max
        if _cond_ratio < 1e-4:
            # Flatten the floor: allow steps down to dtmin * max(ratio/1e-4, 1e-3)
            effective_dtmin = dtmin * max(_cond_ratio / 1e-4, 1e-3)
        else:
            effective_dtmin = dtmin
        if abs(dt) < abs(effective_dtmin):
            # Found a transition! Or at least a point where the step is really
            # small.
            break
        if dt > 0 and t >= tstop or dt < 0 and t <= tstop:
            # Reached tstop, but we want to make sure we stop right at tstop.
            dt = tstop-t
            x = fmin(x+dxdt*dt, tstop)
            dxdt,negeig = dxmindt(x,tstop)
            t = tstop
            X[-1], T[-1], dXdT[-1] = x,t,dxdt
            break
        dtmax = max(t*dtfracMax, dtabsMax)
        if abs(dt) > dtmax:
            dt = np.sign(dt)*dtmax
    if overT is None:
        overX, overT = X[-1], T[-1]
    X = np.array(X)
    T = np.array(T)
    dXdT = np.array(dXdT)
    return _traceMinimum_rval(X, T, dXdT, overX, overT)


class Phase:
    """
    Describes a temperature-dependent minimum, plus second-order transitions
    to and from that minimum.

    Attributes
    ----------
    key : hashable
        A unique identifier for the phase (usually an int).
    X, T, dXdT : array_like
        The minima and its derivative at different temperatures.
    tck : tuple
        Spline knots and coefficients, used in `interpolate.splev`.
    low_trans : set
        Phases (identified by keys) which are joined by a second-order
        transition to this phase.
    high_trans : set
        Phases (identified by keys) which are joined by a second-order
        transition to this phase.
    """
    def __init__(self, key, X, T, dXdT):
        self.key = key
        # We shouldn't ever really need to sort the array, but there must be
        # some bug in the above code that makes it so that occasionally the last
        # step goes backwards. This should fix that.
        i = np.argsort(T)
        T, X, dXdT = T[i], X[i], dXdT[i]
        self.X = X
        self.T = T
        self.dXdT = dXdT
        # PCHIP interpolant: monotone-preserving cubic Hermite — no spurious
        # oscillations between nodes, even with unevenly-spaced temperature data.
        self._pchip = interpolate.PchipInterpolator(T, X, extrapolate=True)
        self.tck = None  # deprecated; kept for backward compatibility
        # Make default connections
        self.low_trans = set()
        self.high_trans = set()

    def valAt(self, T, deriv=0):
        """
        Find the minimum at the value `T` using a spline.

        Parameters
        ----------
        T : float or array_like
        deriv : int
            If deriv > 0, instead return the derivative of the minimum with
            respect to `T`. Can return up to the third derivative.
        """
        T = np.asanyarray(T)
        if deriv == 0:
            return self._pchip(T)
        return self._pchip.derivative(deriv)(T)

    def addLinkFrom(self, other_phase):
        """
        Add a link from `other_phase` to this phase, checking to see if there
        is a second-order transition.
        """
        if np.min(self.T) >= np.max(other_phase.T):
            self.low_trans.add(other_phase.key)
            other_phase.high_trans.add(self.key)
        if np.max(self.T) <= np.min(other_phase.T):
            self.high_trans.add(other_phase.key)
            other_phase.low_trans.add(self.key)

    def __repr__(self):
        popts = np.get_printoptions()
        np.set_printoptions(formatter={'float': lambda x: "%0.4g" % x})
        if len(self.X) > 1:
            Xstr = "[%s, ..., %s]" % (self.X[0], self.X[-1])
        else:
            Xstr = "[%s]" % self.X[0]
        if len(self.T) > 1:
            Tstr = "[%0.4g, ..., %0.4g]" % (self.T[0], self.T[-1])
        else:
            Tstr = "[%0.4g]" % self.T[0]
        if len(self.dXdT) > 1:
            dXdTstr = "[%s, ..., %s]" % (self.dXdT[0], self.dXdT[-1])
        else:
            dXdTstr = "[%s]" % self.dXdT[0]
        s = "Phase(key=%s, X=%s, T=%s, dXdT=%s)" % (
            self.key, Xstr, Tstr, dXdTstr)
        np.set_printoptions(**popts)
        return s


def traceMultiMin(
        f: Callable[[np.ndarray, float], float],
        d2f_dxdt: Callable[[np.ndarray, float], np.ndarray],
        d2f_dx2: Callable[[np.ndarray, float], np.ndarray],
        points: list,
        tLow: float,
        tHigh: float,
        deltaX_target: float,
        dtstart: float = 1e-3,
        tjump: float = 1e-3,
        forbidCrit: Callable[[np.ndarray], bool] | None = None,
        single_trace_args: dict = {},
        local_min_args: dict = {},
) -> dict[int, "Phase"]:
    """
    Trace multiple minima `xmin(t)` of the function `f(x,t)`.

    This function will trace the minima starting from the initial `(x,t)` values
    given in `points`. When a phase disappears, the function will search for
    new nearby minima, and trace them as well. In this way, if each minimum
    corresponds to a different phase, this function can find the (possibly)
    complete phase structure of the potential.

    Parameters
    ----------
    f : callable
        The scalar function `f(x,t)` which needs to be minimized. The input will
        be of the same type as each entry in the `points` parameter.
    d2f_dxdt, d2f_dx2 : callable
        Functions which return returns derivatives of `f(x)`. `d2f_dxdt` should
        return the derivative of the gradient of `f(x)` with respect to `t`, and
        `d2f_dx2` should return the Hessian matrix of `f(x)` evaluated at `t`.
        Both should take as inputs `(x,t)`.
    points : list
        A list of points [(x1,t1), (x2,t2),...] that we want to trace, where
        `x1`, `x2`, etc. are each a one-dimensional array.
    tLow, tHigh : float
        Lowest and highest temperatures between which to trace.
    deltaX_target : float
        Passed to :func:`traceMinimum` and used to set the tolerance in
        minimization.
    dtstart : float, optional
        The starting stepsize, relative to ``tHigh-tLow``.
    tjump : float, optional
        The jump in `t` from the end of one phase to the initial tracing point
        in another. If this is too large, intermediate phases may be skipped.
        Relative to ``tHigh-tLow``.
    forbidCrit : callable or None, optional
        A function that determines whether or not to forbid a phase with a given
        starting point. Should take a point `x` as input, and return True (if
        the phase should be discarded) or False (if the phase should be kept).
    single_trace_args : dict, optional
        Arguments to pass to :func:`traceMinimum`.
    local_min_args : dict, optoinal
        Arguments to pass to :func:`findApproxLocalMinima`.

    Returns
    -------
    phases : dict
        A dictionary of :class:`Phase` instances. The keys in the dictionary
        are integers corresponding to the order in which the phases were
        constructed.
    """
    # We want the minimization here to be very accurate so that we don't get
    # stuck on a saddle or something. This isn't much of a bottle neck.
    xeps = deltaX_target*1e-2

    def fmin(x,t):
        return optimize.fmin(f, x+xeps, args=(t,), xtol=xeps*1e-3,
                             ftol=np.inf, disp=False)
    dtstart = dtstart * (tHigh-tLow)
    tjump = tjump * (tHigh-tLow)
    phases = {}
    nextPoint = []
    for p in points:
        x,t = p
        nextPoint.append([t,dtstart,fmin(x,t),None])

    while len(nextPoint) != 0:
        t1,dt1,x1,linkedFrom = nextPoint.pop()
        x1 = fmin(x1, t1)  # make sure we start as accurately as possible.
        # Check to see if this point is outside the bounds
        if t1 < tLow or (t1 == tLow and dt1 < 0):
            continue
        if t1 > tHigh or (t1 == tHigh and dt1 > 0):
            continue
        if forbidCrit is not None and forbidCrit(x1) == True:
            continue
        # Check to see if it's redudant with another phase
        for i in phases.keys():
            phase = phases[i]
            if (t1 < min(phase.T[0], phase.T[-1]) or
                t1 > max(phase.T[0], phase.T[-1])):
                continue
            x = fmin(phase.valAt(t1), t1)
            # Guard: if x1 is very near the field origin (|x1| < deltaX_target),
            # it may be the symmetric phase starting point — never mark it as
            # "already covered" by a broken phase far from zero (Issue #1).
            if np.max(np.abs(x1)) < deltaX_target:
                continue  # always trace points that are at/near φ = 0
            if np.sum((x-x1)**2)**.5 < 2*deltaX_target:
                # The point is already covered
                # Skip this phase and change the linkage.
                if linkedFrom != i and linkedFrom is not None:
                    phase.addLinkFrom(phases[linkedFrom])
                break
        else:
            # The point is not already covered. Trace the phase.
            # print("Tracing phase starting at x =", x1, "; t =", t1)
            phase_key = len(phases)
            oldNumPoints = len(nextPoint)
            if (t1 > tLow):
                # print("Tracing minimum down")
                down_trace = traceMinimum(f, d2f_dxdt, d2f_dx2, x1,
                                          t1, tLow, -dt1, deltaX_target,
                                          **single_trace_args)
                X_down, T_down, dXdT_down, nX, nT = down_trace
                t2,dt2 = nT-tjump, .1*tjump
                x2 = fmin(nX,t2)
                nextPoint.append([t2,dt2,x2,phase_key])
                if np.sum((X_down[-1]-x2)**2) > deltaX_target**2:
                    for point in findApproxLocalMin(f,X_down[-1],x2,(t2,)):
                        nextPoint.append([t2,dt2,fmin(point,t2),phase_key])
                X_down = X_down[::-1]
                T_down = T_down[::-1]
                dXdT_down = dXdT_down[::-1]
            if (t1 < tHigh):
                # print("Tracing minimum up")
                up_trace = traceMinimum(f, d2f_dxdt, d2f_dx2, x1,
                                        t1, tHigh, +dt1, deltaX_target,
                                        **single_trace_args)
                X_up, T_up, dXdT_up, nX, nT = up_trace
                t2,dt2 = nT+tjump, .1*tjump
                x2 = fmin(nX,t2)
                nextPoint.append([t2,dt2,x2,phase_key])
                if np.sum((X_up[-1]-x2)**2) > deltaX_target**2:
                    for point in findApproxLocalMin(f,X_up[-1],x2,(t2,)):
                        nextPoint.append([t2,dt2,fmin(point,t2),phase_key])
            # Then join the two together
            if (t1 <= tLow):
                X,T,dXdT = X_up, T_up, dXdT_up
            elif (t1 >= tHigh):
                X,T,dXdT = X_down, T_down, dXdT_down
            else:
                X = np.append(X_down, X_up[1:], 0)
                T = np.append(T_down, T_up[1:], 0)
                dXdT = np.append(dXdT_down, dXdT_up[1:], 0)
            if forbidCrit is not None and (forbidCrit(X[0]) or
                                           forbidCrit(X[-1])):
                # The phase is forbidden.
                # Don't add it, and make it a dead-end.
                nextPoint = nextPoint[:oldNumPoints]
            elif len(X) > 1:
                newphase = Phase(phase_key, X,T,dXdT)
                if linkedFrom is not None:
                    newphase.addLinkFrom(phases[linkedFrom])
                phases[phase_key] = newphase
                logger.info("traceMultiMin: added phase %d, T=[%.4g, %.4g], X=[%.4g...%.4g]",
                            phase_key, newphase.T[0], newphase.T[-1],
                            newphase.X[0].flat[0], newphase.X[-1].flat[0])
            else:
                # The phase is just a single point.
                # Don't add it, and make it a dead-end.
                nextPoint = nextPoint[:oldNumPoints]

    return phases


def findApproxLocalMin(f, x1, x2, args=(), n=100, edge=.05):
    """
    Find minima on a straight line between two points.

    When jumping between phases, we want to make sure that we
    don't jump over an intermediate phase. This function does a rough
    calculation to find any such intermediate phases.

    Parameters
    ----------
    f : callable
        The function `f(x)` to minimize.
    x1, x2 : array_like
        The points between which to find minima.
    args : tuple, optional
        Extra arguments to pass to `f`.
    n : int, optional
        Number of points to test for local minima.
    edge : float, optional
        Don't test for minima directly next to the input points. If ``edge==0``,
        the minima potentially go all the way to input points. If ``edge==0.5``,
        the range of tested minima shrinks to a single point at the center of
        the two points.

    Returns
    -------
    list
        A list of approximate minima, with each minimum having the same shape
        as `x1` and `x2`.
    """
    x1,x2 = np.array(x1), np.array(x2)
    dx = np.sum((x1-x2)**2)**.5
    #if dx < mindeltax:
    #	return np.array([]).reshape(0,len(x1))
    x = x1 + (x2-x1)*np.linspace(edge,1-edge,n).reshape(n,1)
    y = f(x,*args)
    i = (y[2:] > y[1:-1]) & (y[:-2] > y[1:-1])
    return x[1:-1][i]


def _removeRedundantPhase(phases, removed_phase, redundant_with_phase):
    for key in removed_phase.low_trans:
        if key != redundant_with_phase.key:
            p = phases[key]
            p.high_trans.discard(removed_phase.key)
            redundant_with_phase.addLinkFrom(p)
    for key in removed_phase.high_trans:
        if key != redundant_with_phase.key:
            p = phases[key]
            p.low_trans.discard(removed_phase.key)
            redundant_with_phase.addLinkFrom(p)
    del phases[removed_phase.key]


def removeRedundantPhases(f, phases, xeps=1e-5, diftol=1e-2):
    """
    Remove redundant phases from a dictionary output by :func:`traceMultiMin`.

    Although :func:`traceMultiMin` attempts to only trace each phase once, there
    are still instances where a single phase gets traced twice. If a phase is
    included twice, the routines for finding transition regions and tunneling
    get very confused. This attempts to avoid that problem.

    Parameters
    ----------
    f : callable
        The function `f(x,t)` which was passed to :func:`traceMultiMin`.
    phases : dict
        The output of :func:`traceMultiMin`.
    xeps : float, optional
        Error tolerance in minimization.
    diftol : float, optional
        Maximum separation between two phases before they are considered to be
        coincident.

    Returns
    -------
    None

    Notes
    -----
    If two phases are merged to get rid of redundancy, the resulting phase has
    a key that is a string combination of the two prior keys.

    .. todo:: Make sure to test removeRedundantPhases().
    .. todo::
        Possibly add extra logic to account for phases which coinincide
        at one end but not the other.

    Warning
    -------
    This hasn't been thoroughly tested yet.
    """
    # I want to make the logic extremely simple at the cost of checking the
    # same thing multiple times.
    # There's just no way this function is going to be the bottle neck.
    def fmin(x,t):
        return np.array(optimize.fmin(f, x, args=(t,),
                        xtol=xeps, ftol=np.inf, disp=False))
    def _energy_same(x1, x2, t):
        """True when x1 and x2 have nearly equal potential energy (relative tol)."""
        V1 = float(np.ravel(f(x1, t))[0])
        V2 = float(np.ravel(f(x2, t))[0])
        scale = max(abs(V1), abs(V2), 1e-30)
        return abs(V1 - V2) / scale < 1e-4
    has_redundant_phase = True
    while has_redundant_phase:
        has_redundant_phase = False
        for i in phases.keys():
            for j in phases.keys():
                if i == j:
                    continue
                phase1, phase2 = phases[i], phases[j]
                tmax = min(phase1.T[-1], phase2.T[-1])
                tmin = max(phase1.T[0], phase2.T[0])
                if tmin > tmax:  # no overlap in the phases
                    continue
                if tmax == phase1.T[-1]:
                    x1 = phase1.X[-1]
                else:
                    x1 = fmin(phase1.valAt(tmax), tmax)
                if tmax == phase2.T[-1]:
                    x2 = phase2.X[-1]
                else:
                    x2 = fmin(phase2.valAt(tmax), tmax)
                dif = np.sum((x1-x2)**2)**.5
                same_at_tmax = (dif < diftol) or _energy_same(x1, x2, tmax)
                if tmin == phase1.T[0]:
                    x1 = phase1.X[0]
                else:
                    x1 = fmin(phase1.valAt(tmin), tmin)
                if tmin == phase2.T[0]:
                    x2 = phase2.X[0]
                else:
                    x2 = fmin(phase2.valAt(tmin), tmin)
                dif = np.sum((x1-x2)**2)**.5
                same_at_tmin = (dif < diftol) or _energy_same(x1, x2, tmin)
                if same_at_tmin and same_at_tmax:
                    # Phases are redundant
                    has_redundant_phase = True
                    p_low = phase1 if phase1.T[0] < phase2.T[0] else phase2
                    p_high = phase1 if phase1.T[-1] > phase2.T[-1] else phase2
                    if p_low is p_high:
                        p_reject = phase1 if p_low is phase2 else phase2
                        _removeRedundantPhase(phases, p_reject, p_low)
                    else:
                        i = p_low.T <= tmax
                        T_low = p_low.T[i]
                        X_low = p_low.X[i]
                        dXdT_low = p_low.dXdT[i]
                        i = p_high.T > tmax
                        T_high = p_high.T[i]
                        X_high = p_high.X[i]
                        dXdT_high = p_high.dXdT[i]
                        T = np.append(T_low, T_high, axis=0)
                        X = np.append(X_low, X_high, axis=0)
                        dXdT = np.append(dXdT_low, dXdT_high, axis=0)
                        newkey = str(p_low.key) + "_" + str(p_high.key)
                        newphase = Phase(newkey, X, T, dXdT)
                        phases[newkey] = newphase
                        _removeRedundantPhase(phases, p_low, newphase)
                        _removeRedundantPhase(phases, p_high, newphase)
                    break
                elif same_at_tmin or same_at_tmax:
                    raise NotImplementedError(
                        "Two phases have been found to coincide at one end "
                        "but not the other. Ideally, this function would "
                        "find where the two diverge, make a cut, and join them "
                        "such there are no more phase redundancies.\n"
                        "Instead, just raise an exception."
                    )
            if has_redundant_phase:
                break


def getStartPhase(phases, V=None):
    """
    Find the key for the high-T phase.

    Parameters
    ----------
    phases : dict
        Output from :func:`traceMultiMin`.
    V : callable
        The potential V(x,T). Only necessary if there are
        multiple phases with the same Tmax.
    """
    startPhases = []
    startPhase = None
    Tmax = None
    assert len(phases) > 0
    for i in phases.keys():
        if phases[i].T[-1] == Tmax:
            # add this to the startPhases list.
            startPhases.append(i)
        elif Tmax is None or phases[i].T[-1] > Tmax:
            startPhases = [i]
            Tmax = phases[i].T[-1]
    if len(startPhases) == 1 or V is None:
        startPhase = startPhases[0]
    else:
        # more than one phase have the same maximum temperature
        # Pick the stable one at high temp.
        Vmin = None
        for i in startPhases:
            V_ = V(phases[i].X[-1], phases[i].T[-1])
            if Vmin is None or V_ < Vmin:
                Vmin = V_
                startPhase = i
    assert startPhase in phases
    return startPhase


def _tunnelFromPhaseAtT(T, phases, start_phase, V, dV,
                        phitol, overlapAngle, nuclCriterion,
                        fullTunneling_params, outdict):
    """
    Find the lowest action tunneling solution.

    Return ``nuclCriterion(S,T)``, and store a dictionary describing the
    transition in outdict for key `T`.
    """
    try:
        T = T[0]  # need this when the function is run from optimize.fmin
    except:
        pass
    if T in outdict:
        return nuclCriterion(outdict[T]['action'], T)

    def fmin(x):
        """Refine a minimum using L-BFGS-B (falls back to Nelder-Mead)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        try:
            res = optimize.minimize(
                lambda p: float(np.ravel(V(p, T))[0]),
                x,
                jac=lambda p: np.atleast_1d(np.asarray(dV(p, T), dtype=float).ravel()),
                method='L-BFGS-B',
                options={'ftol': 1e-15, 'gtol': phitol, 'maxiter': 200},
            )
            if res.success or res.fun < float(np.ravel(V(x, T))[0]):
                return res.x
        except Exception:
            pass
        # Fallback: Nelder-Mead
        return optimize.fmin(V, x, args=(T,),
                             xtol=phitol, ftol=np.inf, disp=False)

    # Loop through all the phases, adding acceptable minima
    x0 = fmin(start_phase.valAt(T))
    V0 = V(x0, T)
    tunnel_list = []
    for key in phases.keys():
        if key == start_phase.key:
            continue
        p = phases[key]
        if (p.T[0] > T or p.T[-1] < T):
            continue
        x1 = fmin(p.valAt(T))
        V1 = V(x1, T)
        if V1 >= V0:
            continue
        tdict = dict(low_vev=x1, high_vev=x0, Tnuc=T,
                     low_phase=key, high_phase=start_phase.key)
        tunnel_list.append(tdict)
    # Check for overlap
    if overlapAngle > 0:
        excluded = []
        cos_overlap = np.cos(overlapAngle * np.pi/180)
        for i in range(1, len(tunnel_list)):
            for j in range(i):
                xi = tunnel_list[i]['low_vev']
                xj = tunnel_list[j]['low_vev']
                xi2 = np.sum((xi-x0)**2)
                xj2 = np.sum((xj-x0)**2)
                dotij = np.sum((xj-x0)*(xi-x0))
                if dotij >= np.sqrt(xi2*xj2) * cos_overlap:
                    excluded.append(i if xi2 > xj2 else j)
        excluded=list(set(excluded))
        for i in sorted(excluded)[::-1]:
            del tunnel_list[i]
    # Get rid of the T parameter for V and dV
    def V_(x,T=T,V=V): return V(x,T)
    def dV_(x,T=T,dV=dV): return dV(x,T)
    # For each item in tunnel_list, try tunneling
    lowest_action = np.inf
    lowest_tdict = dict(action=np.inf)
    for tdict in tunnel_list:
        x1 = tdict['low_vev']
        try:
            # print("Tunneling from phase %s to phase %s at T=%0.7g"
                #   % (tdict['high_phase'], tdict['low_phase'], T))
            # print("high_vev =", tdict['high_vev'])
            # print("low_vev =", tdict['low_vev'])
            tobj = pathDeformation.fullTunneling(
                [x1,x0], V_, dV_, callback_data=T,
                **fullTunneling_params)
            tdict['instanton'] = tobj
            tdict['action'] = tobj.action
            tdict['trantype'] = 1
        except tunneling1D.PotentialError as err:
            if err.args[1] == "no barrier":
                tdict['trantype'] = 0
                tdict['action'] = 0.0
            elif err.args[1] == "stable, not metastable":
                tdict['trantype'] = 0
                tdict['action'] = np.inf
            else:
                logger.error("Unexpected error message.")
                raise
        if tdict['action'] <= lowest_action:
            lowest_action = tdict['action']
            lowest_tdict = tdict
    outdict[T] = lowest_tdict
    crit_val = nuclCriterion(lowest_action, T)
    logger.debug("_tunnelFromPhaseAtT: T=%.4g, action=%.4g, S3/T=%.4g, criterion=%.4g",
                 T, lowest_action, lowest_action / (T + 1e-100), crit_val)
    return crit_val


def _potentialDiffForPhase(T, start_phase, other_phases, V):
    """
    Returns the maximum difference between the other phases and `start_phase`.

    Return value is positive/negative when `start_phase` is stable/unstable.
    """
    V0 = V(start_phase.valAt(T),T)
    delta_V = np.inf
    for phase in other_phases:
        V1 = V(phase.valAt(T),T)
        if V1-V0 < delta_V:
            delta_V = V1-V0
    return delta_V


def _maxTCritForPhase(phases, start_phase, V, Ttol):
    """
    Find the maximum temperature at which `start_phase` is degenerate with one
    of the other phases.
    """
    other_phases = []
    for phase in phases.values():
        if phase.key != start_phase.key:
            other_phases.append(phase)
    if len(other_phases) == 0:
        # No other phases, just return the lowest temperature
        return start_phase.T[0]
    Tmin = min([phase.T[0] for phase in other_phases])
    Tmax = max([phase.T[-1] for phase in other_phases])
    Tmin = max(Tmin, start_phase.T[0])
    Tmax = min(Tmax, start_phase.T[-1])
    DV_Tmin = _potentialDiffForPhase(Tmin, start_phase, other_phases, V)
    DV_Tmax = _potentialDiffForPhase(Tmax, start_phase, other_phases, V)
    if DV_Tmin >= 0: return Tmin  # stable at Tmin
    if DV_Tmax <= 0: return Tmax  # unstable at Tmax
    return optimize.brentq(
        _potentialDiffForPhase, Tmin, Tmax,
        args=(start_phase, other_phases, V),
        xtol=Ttol, maxiter=200, disp=False)


def tunnelFromPhase(
        phases: "dict[int, Phase]",
        start_phase: "Phase",
        V: Callable[[np.ndarray, float], float],
        dV: Callable[[np.ndarray, float], np.ndarray],
        Tmax: float,
        Ttol: float = 1e-3,
        maxiter: int = 100,
        phitol: float = 1e-8,
        overlapAngle: float = 45.0,
        nuclCriterion: "Callable[[float, float], float] | None" = None,
        fullTunneling_params: dict = {},
        tunneling_config: "TunnelingConfig | None" = None,
) -> "dict | None":
    """
    Find the instanton and nucleation temeprature for tunneling from
    `start_phase`.

    Parameters
    ----------
    phases : dict
        Output from :func:`traceMultiMin`.
    start_phase : Phase object
        The metastable phase from which tunneling occurs.
    V, dV : callable
        The potential V(x,T) and its gradient.
    Tmax : float
        The highest temperature at which to try tunneling.
    Ttol : float, optional
        Tolerance for finding the nucleation temperature.
    maxiter : int, optional
        Maximum number of times to try tunneling.
    phitol : float, optional
        Tolerance for finding the minima.
    overlapAngle : float, optional
        If two phases are in the same direction, only try tunneling to the
        closer one. Set to zero to always try tunneling to all available phases.
    nuclCriterion : callable or None, optional
        Function of the action *S* and temperature *T*. Should return 0 for the
        correct nucleation rate, > 0 for a low rate and < 0 for a high rate.
        Defaults to ``S/T - 140`` (or ``tunneling_config.get_nucl_criterion()``
        if *tunneling_config* is provided and *nuclCriterion* is ``None``).
    fullTunneling_params : dict
        Parameters to pass to :func:`pathDeformation.fullTunneling`.
    tunneling_config : TunnelingConfig or None, optional
        Configuration object.  When provided and *nuclCriterion* is ``None``,
        the nucleation criterion is taken from
        ``tunneling_config.get_nucl_criterion()``.

    Returns
    -------
    dict or None
        A description of the tunneling solution at the nucleation temperature,
        or None if there is no found solution. Has the following keys:

        - *Tnuc* : the nucleation temperature
        - *low_vev, high_vev* : vevs for the low-T phase (the phase that the
          instanton tunnels to) and high-T phase (the phase that the instanton
          tunnels from).
        - *low_phase, high_phase* : identifier keys for the low-T and high-T
          phases.
        - *action* : The Euclidean action of the instanton.
        - *instanton* : Output from :func:`pathDeformation.fullTunneling`, or
          None for a second-order transition.
        - *trantype* : 1 or 2 for first or second-order transitions.
    """
    outdict = {}  # keys are T values
    # --- Resolve nucleation criterion ---
    if nuclCriterion is None:
        if tunneling_config is not None:
            nuclCriterion = tunneling_config.get_nucl_criterion()
        else:
            nuclCriterion = lambda S, T: S / (T + 1e-100) - 140.0
    # --- Wire V_spline_samples from config into fullTunneling_params ---
    if tunneling_config is not None and 'V_spline_samples' not in fullTunneling_params:
        fullTunneling_params = {
            'V_spline_samples': tunneling_config.V_spline_samples,
            **fullTunneling_params,
        }
    # --- Wire deformation params from config ---
    if (tunneling_config is not None
            and 'deformation_deform_params' not in fullTunneling_params):
        fullTunneling_params = {
            'deformation_deform_params': tunneling_config.get_deform_params(),
            **fullTunneling_params,
        }
    args = (phases, start_phase, V, dV,
            phitol, overlapAngle, nuclCriterion,
            fullTunneling_params, outdict)
    Tmin = start_phase.T[0]
    T_highest_other = Tmin
    for phase in phases.values():
        T_highest_other = max(T_highest_other, phase.T[-1])
    Tmax = min(Tmax, T_highest_other)
    assert Tmax >= Tmin
    logger.info("tunnelFromPhase: phase %s, searching Tn in [%.4g, %.4g]",
                start_phase.key, Tmin, Tmax)
    try:
        Tnuc = optimize.brentq(_tunnelFromPhaseAtT, Tmin, Tmax, args=args,
                               xtol=Ttol, maxiter=maxiter, disp=False)
    except ValueError as err:
        if err.args[0] != "f(a) and f(b) must have different signs":
            raise
        if nuclCriterion(outdict[Tmax]['action'], Tmax) > 0:
            # brentq failed: criterion > 0 at both endpoints (both > threshold).
            # Check the original condition: if action(Tmin)/Tmax is small enough
            # it means a sign change exists between Tmin and some interior T.
            _orig_cond = nuclCriterion(outdict[Tmin]['action'], Tmax) < 0
            # Extended-scan fallback: when the original condition fails but
            # Tmin is near 0 (conformal-CW / zero-T quantum barrier case),
            # use fmin with a log-space initial guess to find the S3/T minimum.
            _do_extended = (
                not _orig_cond
                and tunneling_config is not None
                and tunneling_config.T_scan_extension
                and Tmin < 1e-4 * Tmax
            )
            if _orig_cond or _do_extended:
                # tunneling *may* be possible. Find the minimum of S3/T.
                Tmax_Tc = _maxTCritForPhase(phases, start_phase, V, Ttol)
                if _do_extended:
                    # At very low T the symmetric phase may not be a stable
                    # local minimum → fmin starting near T=0 always returns
                    # action=inf.  Use a log-scan starting at 10% of Tmax_Tc,
                    # descending decade-by-decade, to find a T where criterion<0.
                    n_extend = tunneling_config.T_scan_max_extend
                    Tneg = None
                    for k in range(1, n_extend + 1):
                        T_try = Tmax_Tc * 10.0**(-k)
                        if T_try < Ttol:
                            break
                        crit_try = _tunnelFromPhaseAtT(
                            T_try, phases, start_phase, V, dV,
                            phitol, overlapAngle, nuclCriterion,
                            fullTunneling_params, outdict)
                        logger.info(
                            "tunnelFromPhase: T_scan_extension k=%d T=%.4g "
                            "S3/T=%.4g", k, T_try,
                            outdict[T_try]['action'] / (T_try + 1e-100))
                        if np.isfinite(crit_try) and crit_try < 0:
                            Tneg = T_try
                            break
                    if Tneg is None:
                        logger.info("tunnelFromPhase: no nucleation found "
                                    "(T_scan_extension: no T with S3/T < threshold)")
                        return None
                    Tnuc = optimize.brentq(
                        _tunnelFromPhaseAtT, Tneg, Tmax_Tc,
                        args=args, xtol=Ttol, maxiter=maxiter, disp=False)
                else:
                    _Tstart = 0.5 * (Tmin + Tmax_Tc)

                    def abort_fmin(T, outdict=outdict, nc=nuclCriterion):
                        T = T[0]  # T is an array of size 1
                        if nc(outdict[T]['action'], T) <= 0:
                            raise StopIteration(T)

                    try:
                        Tmin_opt = optimize.fmin(_tunnelFromPhaseAtT, _Tstart,
                                                 args=args, xtol=Ttol*10,
                                                 ftol=1.0, maxiter=maxiter,
                                                 disp=0, callback=abort_fmin)[0]
                    except StopIteration as se:
                        Tmin_opt = se.args[0]
                    if nuclCriterion(outdict[Tmin_opt]['action'], Tmin_opt) > 0:
                        # no tunneling possible
                        logger.info("tunnelFromPhase: no nucleation found (S/T always > threshold)")
                        return None
                    Tnuc = optimize.brentq(
                        _tunnelFromPhaseAtT, Tmin_opt, Tmax_Tc,
                        args=args, xtol=Ttol, maxiter=maxiter, disp=False)
            else:
                # no tunneling possible
                logger.info("tunnelFromPhase: no nucleation found (S/T < threshold at all T)")
                return None
        else:
            # tunneling happens right away at Tmax
            Tnuc = Tmax
    rdict = outdict[Tnuc]
    logger.info("tunnelFromPhase: found Tn=%.6g, S3/T=%.4g (trantype=%d)",
                Tnuc, rdict.get('action', float('nan')) / (Tnuc + 1e-100),
                rdict.get('trantype', 0))
    rdict['betaHn_GW'] = 0.0
    if rdict['trantype'] == 1:
        outdict_tmp = {}
        low_phase_key = rdict['low_phase']
        low_phase_dic = {low_phase_key:phases[low_phase_key]}
        args = (low_phase_dic, start_phase, V, dV,
                phitol, overlapAngle, lambda S,T: S/(T+1e-100),
                fullTunneling_params, outdict_tmp)
        # try:
        def for_derivative(T):
            try:
                return _tunnelFromPhaseAtT(T, *args)
            except Exception:
                return float('nan')

        def _numeric_derivative(func, x0, rel_step=1e-3, abs_step=1e-6, max_attempts=8):
            """Robust central-difference derivative of func at x0.

            Tries progressively smaller step sizes if evaluations fail or
            return non-finite values. Returns (df, success).
            """
            # coerce x0 to scalar float if it's an array-like
            try:
                x0 = float(np.asarray(x0).ravel()[0])
            except Exception:
                try:
                    x0 = float(x0)
                except Exception:
                    return 0.0, False

            h = max(abs_step, abs(x0) * rel_step)
            for _ in range(max_attempts):
                xm = x0 - h
                xp = x0 + h
                try:
                    fm = func(xm)
                    fp = func(xp)
                except Exception:
                    fm = fp = float('nan')

                # Try to coerce to floats
                try:
                    fm = float(fm)
                except Exception:
                    fm = float('nan')
                try:
                    fp = float(fp)
                except Exception:
                    fp = float('nan')

                if np.isfinite(fm) and np.isfinite(fp):
                    return ((fp - fm) / (2.0 * h), True)

                # try smaller step size
                h /= 10.0

            return (0.0, False)

        df, ok = _numeric_derivative(for_derivative, Tnuc)
        if ok and np.isfinite(df):
            rdict['betaHn_GW'] = float(Tnuc) * float(df)
            logger.debug("Computed betaHn_GW = %g -> %g at Tnuc = %g", df, rdict['betaHn_GW'], Tnuc)
        else:
            rdict['betaHn_GW'] = 0.0
            logger.warning("Could not compute betaHn_GW at Tnuc = %g; defaulting to 0.0", Tnuc)
        # except Exception as e:
            # print("Could not compute betaHn_GW due to exception:", e)
            # rdict['betaHn_GW'] = 0.0
    return rdict if rdict['trantype'] > 0 else None


def secondOrderTrans(high_phase, low_phase, Tstr='Tnuc'):
    """
    Assemble a dictionary describing a second-order phase transition.
    """
    rdict = {}
    rdict[Tstr] = 0.5*(high_phase.T[0] + low_phase.T[-1])
    rdict['low_vev'] = rdict['high_vev'] = high_phase.X[0]
    rdict['low_phase'] = low_phase.key
    rdict['high_phase'] = high_phase.key
    rdict['action'] = 0.0
    rdict['instanton'] = None
    rdict['trantype'] = 2
    rdict['betaHn_GW'] = 0.0
    return rdict


def findAllTransitions(
        phases: "dict[int, Phase]",
        V: Callable[[np.ndarray, float], float],
        dV: Callable[[np.ndarray, float], np.ndarray],
        tunnelFromPhase_args: dict = {},
        tunneling_config: "TunnelingConfig | None" = None,
) -> list[dict]:
    """
    Find the complete phase transition history for the potential `V`.

    This functions uses :func:`tunnelFromPhase` to find the transition
    temperature and instanton for each phase, starting at the highest phase
    in the potential. Note that if there are multiple transitions that could
    occur at the same minimum (if, for example, there is a Z2 symmetry or
    a second-order transition breaks in multiple directions), only one of the
    transitions will be used.

    Parameters
    ----------
    phases : dict
        Output from :func:`traceMultiMin`.
    V, dV : callable
        The potential function and its gradient, each a function of field
        value (which should be an array, not a scalar) and a temperature.
    tunnelFromPhase_args : dict
        Parameters to pass to :func:`tunnelFromPhase`.
    tunneling_config : TunnelingConfig or None, optional
        Configuration object forwarded to :func:`tunnelFromPhase`.

    Returns
    -------
    list of transitions
        Each item is a dictionary describing the transition (see
        :func:`tunnelFromPhase` for keys). The first transition is the one at
        the highest temperature.
    """
    phases = phases.copy()
    start_phase = phases[getStartPhase(phases, V)]
    Tmax = start_phase.T[-1]
    transitions = []
    # Merge config-provided tunnelFromPhase defaults.
    # Explicit tunnelFromPhase_args take precedence over config.
    if tunneling_config is not None:
        _cfg_kwargs = tunneling_config.get_tunnelFromPhase_kwargs()
        _merged_args = {**_cfg_kwargs, **tunnelFromPhase_args}
        tunneling_config.apply_log_level()
    else:
        _merged_args = tunnelFromPhase_args
    logger.info("findAllTransitions: starting phase %s, Tmax=%.4g",
                start_phase.key, Tmax)
    while start_phase is not None:
        del phases[start_phase.key]
        trans = tunnelFromPhase(phases, start_phase, V, dV, Tmax,
                                tunneling_config=tunneling_config,
                                **_merged_args)
        if trans is None and not start_phase.low_trans:
            logger.info("findAllTransitions: no transition from phase %s",
                        start_phase.key)
            start_phase = None
        elif trans is None:
            low_key = None
            for key in start_phase.low_trans:
                if key in phases:
                    low_key = key
                    break
            if low_key is not None:
                low_phase = phases[low_key]
                transitions.append(secondOrderTrans(start_phase, low_phase))
                start_phase = low_phase
                Tmax = low_phase.T[-1]
            else:
                start_phase = None
        else:
            transitions.append(trans)
            logger.info("findAllTransitions: %s-order transition at Tn=%.4g, "
                        "S3/T=%.4g, %s->%s",
                        'first' if trans['trantype'] == 1 else 'second',
                        trans['Tnuc'], trans['action'] / (trans['Tnuc'] + 1e-100),
                        trans['high_phase'], trans['low_phase'])
            start_phase = phases[trans['low_phase']]
            Tmax = trans['Tnuc']
    logger.info("findAllTransitions: done, %d transition(s) found",
                len(transitions))
    return transitions


def findCriticalTemperatures(
        phases: "dict[int, Phase]",
        V: Callable[[np.ndarray, float], float],
        start_high: bool = False,
) -> list[dict]:
    """
    Find all temperatures `Tcrit` such that there is degeneracy between any
    two phases.

    Parameters
    ----------
    phases : dict
        Output from :func:`traceMultiMin`.
    V : callable
        The potential function `V(x,T)`, where `x` is the field value (which
        should be an array, not a scalar) and `T` is the temperature.
    start_high : bool, optional
        If True, only include those transitions which could be reached starting
        from the high-T phase. NOT IMPLEMENTED YET.

    Returns
    -------
    list of transitions
        Transitions are sorted in decreasing temperature. Each transition is a
        dictionary with the following keys:

        - *Tcrit* : the critical temperature
        - *low_vev, high_vev* : vevs for the low-T phase (the phase that the
          model transitions to) and high-T phase (the phase that the model
          transitions from).
        - *low_phase, high_phase* : identifier keys for the low-T and high-T
          phases.
        - *trantype* : 1 or 2 for first or second-order transitions.
    """
    transitions = []
    for i in phases.keys():
        for j in phases.keys():
            if i == j:
                continue
            # Try going from i to j (phase1 -> phase2)
            phase1, phase2 = phases[i], phases[j]
            tmax = min(phase1.T[-1], phase2.T[-1])
            tmin = max(phase1.T[0], phase2.T[0])
            if tmin >= tmax:
                # No overlap. Try for second-order.
                if phase2.key in phase1.low_trans:
                    transitions.append(
                        secondOrderTrans(phase1, phase2, 'Tcrit'))
                continue

            def DV(T):
                return V(phase1.valAt(T), T) - V(phase2.valAt(T), T)

            if DV(tmin) < 0:
                # phase1 is lower at tmin, no tunneling
                continue
            if DV(tmax) > 0:
                # phase1 is higher even at tmax, no critical temperature
                continue
            Tcrit = optimize.brentq(DV, tmin, tmax, disp=False)
            tdict = {}
            tdict['Tcrit'] = Tcrit
            tdict['high_vev'] = phase1.valAt(Tcrit)
            tdict['high_phase'] = phase1.key
            tdict['low_vev'] = phase2.valAt(Tcrit)
            tdict['low_phase'] = phase2.key
            tdict['trantype'] = 1
            transitions.append(tdict)
    if not start_high:
        return sorted(transitions, key=lambda x: x['Tcrit'])[::-1]
    start_phase = getStartPhase(phases, V)
    raise NotImplementedError("start_high=True not yet supported")


def addCritTempsForFullTransitions(phases, crit_trans, full_trans):
    """
    For each transition dictionary in `full_trans`, find the corresponding
    transition in `crit_trans` and add it to the dictionary for the key
    `crit_trans`, or add None if no corresponding transition is found.

    Notes
    -----
    The phases in the supercooled transitions might not be exactly
    the same as the phases in the critical temperature transitions. This would
    be the case, for example, if in `full_trans` the phase transitions go like
    1 -> 2 -> 3, but in `crit_trans` they go like 1 -> (2 or 3).

    Parameters
    ----------
    phases : dict
    crit_trans : list
    full_trans : list
    """
    parents_dict = {}
    for i in phases.keys():
        parents = [i]
        for tcdict in crit_trans[::-1]:
            j = tcdict['high_phase']
            if tcdict['low_phase'] in parents and j not in parents:
                parents.append(j)
        parents_dict[i] = parents
    for tdict in full_trans:
        low_parents = parents_dict[tdict['low_phase']]
        high_parents = parents_dict[tdict['high_phase']]
        common_parents = set.intersection(
            set(low_parents), set(high_parents))
        for p in common_parents:
            # exclude the common parents
            try:
                k = low_parents.index(p)
                low_parents = low_parents[:k]
            except: pass
            try:
                k = high_parents.index(p)
                high_parents = high_parents[:k+1]
            except: pass
        for tcdict in crit_trans[::-1]:  # start at low-T
            if tcdict['Tcrit'] < tdict['Tnuc']:
                continue
            if (tcdict['low_phase'] in low_parents
                    and tcdict['high_phase'] in high_parents):
                tdict['crit_trans'] = tcdict
                break
        else:
            tdict['crit_trans'] = None
