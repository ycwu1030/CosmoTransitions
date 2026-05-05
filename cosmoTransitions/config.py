"""TunnelingConfig — centralized numerical configuration.

Holds numerical parameters that are independent of the model physics and
are propagated through the tunneling pipeline (``findAllTransitions``,
``tunnelFromPhase``, etc.). Use a ``TunnelingConfig`` instance to control
integration tolerances, deformation settings, logging, and other runtime
parameters without changing model code.

This module also provides small helpers used by the tunneling machinery,
notably the built-in nucleation-criterion functions and the
``enable_logging`` helper.
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


def fixed_140_nucl_criterion(S: float, T: float) -> float:
    """Nucleation criterion with a fixed threshold of 140.

    Returns ``S/T - 140``; a negative value indicates nucleation has occurred.
    This is the historical default used in cosmoTransitions.

    Parameters
    ----------
    S : float
        Euclidean action S3 (units GeV).
    T : float
        Temperature (GeV).

    Returns
    -------
    float
        Negative values indicate nucleation (criterion satisfied).
    """
    return S / (T + 1e-100) - 140.0


def cosmological_nucl_criterion(S: float, T: float, M_Pl: float = 1.22e19) -> float:
    r"""Cosmological nucleation criterion for a radiation-dominated Universe.

    The criterion solves

    .. math::
        \frac{S_3}{T} - 4 \ln\left(\frac{M_{\rm Pl}}{T}\right) = 0

    where :math:`M_{\rm Pl} \approx 1.22\times10^{19}` GeV. The threshold is
    about 143 at T=100 GeV (vs fixed 140).

    Parameters
    ----------
    S : float
        Euclidean action S3 (units GeV).
    T : float
        Temperature (GeV).
    M_Pl : float, optional
        Planck mass (GeV), default 1.22e19.

    Returns
    -------
    float
        Negative values indicate nucleation (criterion satisfied).
    """
    if T <= 0:
        return np.inf
    threshold = 4.0 * np.log(M_Pl / T)
    return S / (T + 1e-100) - threshold


# ---------------------------------------------------------------------------
# TunnelingConfig
# ---------------------------------------------------------------------------

@dataclass
class TunnelingConfig:
    r"""
    Configuration object for tunneling/transition searches.

    This dataclass centralizes numerical parameters that control the
    tunneling pipeline (phase tracing, profile finding, path deformation,
    and nucleation searches). Populate a `TunnelingConfig` and pass it to
    `model.findAllTransitions(tunneling_config=cfg)` to customize behavior.

    The most relevant fields are documented near their declarations below.

    Notes
    -----
    A ``TunnelingConfig`` instance may be passed optionally to
    ``tunnelFromPhase``, ``findAllTransitions`` (transitionFinder), and
    ``generic_potential.findAllTransitions``. If ``None`` is passed the default
    ``TunnelingConfig()`` is used, preserving historical behavior.

    Backwards-compatibility note: when ``thinCutoff``/``rmin`` are set to
    "auto", the derived values for thick-wall cases (\epsilon ~ 1) match the
    original defaults (0.01 / 1e-4), so thick-wall model results are unchanged.
    """

    # findProfile parameters
    thinCutoff: Union[float, str] = "auto"
    rmin: Union[float, str] = "auto"
    xtol: float = 1e-6
    phitol: float = 1e-6
    rmax: float = 1e4
    npoints: int = 500

    # Temperature scan
    T_scan_extension: bool = True
    T_scan_max_extend: int = 3

    # Nucleation criterion (default 'fixed_140' for backward compatibility)
    nuclCriterion: Union[str, Callable[[float, float], float]] = "fixed_140"

    # findProfile adaptive retries
    enable_profile_retry: bool = True
    max_profile_retries: int = 3

    # Adaptive finite-difference switch (read by generic_potential)
    use_adaptive_grad: bool = True

    # --- fullTunneling / SplinePath ---

    V_spline_samples: Optional[int] = 100
    r"""Controls whether `SplinePath` pre-samples the potential V(x).

    ``100`` (default): uniformly sample 100 points on [\phi_sym, \phi_broken]
    and construct a PCHIP spline for V(x). This reduces calls to full
    ``Vtot`` while maintaining good accuracy for typical electroweak-scale
    models.

    ``None`` disables pre-sampling so that ``SplinePath.V(x)`` calls
    ``Vtot`` directly. For extreme supercooling (T_n/T_c ≲ 0.05) pre-sampling
    can miss a narrow thermal barrier located near \phi ~ T_n; in such
    cases set `V_spline_samples = None`. The helper ``supercooling_preset()``
    sets this field to ``None`` automatically.
    """

    maxiter_fullTunneling: int = 20
    """Maximum outer iterations for ``fullTunneling`` (path-deform + 1D tunneling).

    Each iteration: (1) fit a SplinePath to the current path points, (2) perform
    a 1D tunneling along the path, (3) deform the path via ``deformPath``,
    (4) check convergence. Typically 10 iterations suffice.
    """

    # --- Path deformation (pathDeformation) ---

    deform_fRatioConv: float = 0.02
    """Convergence tolerance for path deformation: stop when
    max(normal_force)/max(|gradV|) < deform_fRatioConv.

    Smaller values increase path accuracy at the cost of more deformation
    steps. Typical value for FOPTs: 0.02. Use ~0.005 for supercooling-sensitive cases.
    """

    deform_maxiter: int = 500
    """Maximum deformation iterations for ``deformPath``. A warning is
    emitted if exceeded and the current best path is returned."""

    # --- tunnelFromPhase parameters ---

    Ttol: float = 1e-3
    """Temperature tolerance for locating T_n (brentq xtol), units: GeV.

    Smaller values increase T_n accuracy but increase the number of
    `_tunnelFromPhaseAtT` calls. For T_n ~ O(100) GeV, `1e-3` is sufficient;
    for extreme cases relax to `1.0` to speed up scanning.
    """

    maxiter_tunnel: int = 100
    """Maximum iterations for brentq / fmin in ``tunnelFromPhase``."""

    phitol_tunnel: float = 1e-8
    """Field-value tolerance passed as L-BFGS-B `gtol` when refining the
    vacuum location inside `_tunnelFromPhaseAtT`.

    This differs from `phitol` used by the ODE solver. Too large a value
    yields the action evaluated at an inaccurate field value; too small
    increases optimization runtime.
    """

    overlapAngle: float = 45.0
    """When starting from the false vacuum, if two candidate target vacua
    have an angular separation (in field-space) smaller than this threshold
    (degrees), only the nearer direction is attempted for tunneling.

    Set to `0.0` to force attempts for all targets (more complete but slower).
    45° is the historical default.
    """

    # --- Phase-tracing parameters (traceMultiMin / traceMinimum) ---

    dtstart: float = 1e-3
    """Initial temperature step for `traceMultiMin`, relative to
    (tHigh - tLow). Large values may skip narrow phases; small values
    increase computation. Typical default: 1e-3. For very wide temperature
    spans, consider 1e-4.
    """

    tjump: float = 1e-3
    """Temperature jump used when searching for the next phase after
    finishing one phase, relative to (tHigh - tLow). Reduce to lower the
    chance of skipping narrow intermediate phases at the cost of more
    starting points.
    """

    # --- Logging level ---

    log_level: Optional[int] = None
    """If not ``None``, automatically enable cosmoTransitions logging at this
    level when :meth:`apply_log_level` is called.

    Example: ``log_level=logging.INFO`` is equivalent to calling
    ``cosmoTransitions.enable_logging(logging.INFO)``.
    ``None`` leaves the logging configuration unchanged (default).
    """

    log_file: Optional[str] = None
    """Optional path to a log file.  When set together with :attr:`log_level`,
    log records are written to this file (append mode) instead of ``stderr``.

    Example::

        cfg = TunnelingConfig(log_level=logging.DEBUG, log_file='run.log')
        cfg.apply_log_level()  # writes all DEBUG messages to run.log
    """

    def get_nucl_criterion(self) -> Callable[[float, float], float]:
        """Return the nucleation-criterion callable.

        Returns
        -------
        callable
            A function with signature ``(S: float, T: float) -> float``; a
            negative return value indicates nucleation.
        """
        if callable(self.nuclCriterion):
            return self.nuclCriterion
        if self.nuclCriterion == "fixed_140":
            return fixed_140_nucl_criterion
        if self.nuclCriterion == "cosmological":
            return cosmological_nucl_criterion
        raise ValueError(
            f"Unknown nuclCriterion: {self.nuclCriterion!r}. "
            "Use 'fixed_140', 'cosmological', or a callable."
        )

    def get_fullTunneling_kwargs(self) -> dict:
        """Return kwargs to pass to ``pathDeformation.fullTunneling``.

        Includes ``V_spline_samples`` and ``maxiter`` and is safe to
        ``**``-unpack. Deformation-specific params come from
        :meth:`get_deform_params`.
        """
        return dict(
            V_spline_samples=self.V_spline_samples,
            maxiter=self.maxiter_fullTunneling,
        )

    def get_deform_params(self) -> dict:
        """Return the deformation parameters for ``deformPath`` as a dict."""
        return dict(
            fRatioConv=self.deform_fRatioConv,
            maxiter=self.deform_maxiter,
        )

    def get_tunnelFromPhase_kwargs(self) -> dict:
        """Return kwargs for ``tunnelFromPhase`` (Ttol, maxiter, phitol, overlapAngle)."""
        return dict(
            Ttol=self.Ttol,
            maxiter=self.maxiter_tunnel,
            phitol=self.phitol_tunnel,
            overlapAngle=self.overlapAngle,
        )

    def get_traceMultiMin_kwargs(self) -> dict:
        """Return kwargs for ``traceMultiMin`` (dtstart, tjump)."""
        return dict(
            dtstart=self.dtstart,
            tjump=self.tjump,
        )

    def apply_log_level(self) -> None:
        """Enable cosmoTransitions logging if :attr:`log_level` is not ``None``.

        Writes to :attr:`log_file` when set, otherwise to ``stderr``.
        """
        if self.log_level is not None:
            enable_logging(self.log_level, log_file=self.log_file)

    def get_findProfile_kwargs(self, epsilon: float | None = None) -> dict:
        """Return a kwargs dict suitable for ``findProfile``.

        Parameters
        ----------
        epsilon : float or None
            Thin-wall ratio estimated by ``SingleFieldInstanton._estimate_epsilon()``.
            When ``thinCutoff``/``rmin`` are "auto", this value is used to
            select recommended tiers.

        Returns
        -------
        dict
            Dict with keys ``thinCutoff``, ``rmin``, ``xtol``, ``phitol``, ``rmax``,
            ``npoints`` that can be ``**``-unpacked into ``findProfile``.
        """
        tc = self.thinCutoff
        rm = self.rmin
        if tc == "auto" or rm == "auto":
            if epsilon is None:
                # Unable to derive recommended tier; fall back to original defaults
                if tc == "auto":
                    tc = 0.01
                if rm == "auto":
                    rm = 1e-4
            else:
                auto_tc, auto_rm = _epsilon_to_params(epsilon)
                if tc == "auto":
                    tc = auto_tc
                if rm == "auto":
                    rm = auto_rm
        return dict(
            thinCutoff=tc,
            rmin=rm,
            xtol=self.xtol,
            phitol=self.phitol,
            rmax=self.rmax,
            npoints=self.npoints,
        )

    # ------------------------------------------------------------------
    # Preset factory methods
    # ------------------------------------------------------------------

    @classmethod
    def supercooling_preset(cls) -> "TunnelingConfig":
        r"""Preset tuned for extreme supercooling (T_n/T_c ≲ 0.05).

        Key features:

        * ``V_spline_samples=None`` — required for extreme supercooling so that
          narrow thermal barriers near \phi ~ T_n are resolved by direct calls
          to ``Vtot`` rather than coarse pre-sampling.
        * Strict integration tiers: ``thinCutoff=1e-4``, ``rmin=1e-7`` for
          thin-wall instantons.
        * Enable temperature-scan extension (up to 5 extensions, each lowering
          the lower bound by ×0.1) to find very low T_n values.
        * ``Ttol=1.0``: for T_n typically O(1e4) GeV, GeV-level precision is adequate.
        """
        return cls(
            V_spline_samples=None,
            thinCutoff=1e-4,
            rmin=1e-7,
            T_scan_extension=True,
            T_scan_max_extend=5,
            Ttol=1.0,
        )

    @classmethod
    def write_default(cls, path: str = "cosmoTransitions_config.toml") -> None:
        """
        Write the built-in default configuration template to a TOML file.

        The generated file contains all parameters with detailed comments.
        Users may copy and edit it, then load with :meth:`from_file`.

        Parameters
        ----------
        path : str
            Output path, default ``"cosmoTransitions_config.toml"``.

        Examples
        --------
        ::

            TunnelingConfig.write_default("my_scan.toml")   # write template
            # edit my_scan.toml, then:
            cfg = TunnelingConfig.from_file("my_scan.toml")
        """
        import shutil
        import os
        src = os.path.join(os.path.dirname(__file__), "default_config.toml")
        shutil.copy2(src, path)
        logger.info("Default config written to %s", path)

    @classmethod
    def from_file(cls, path: str) -> "TunnelingConfig":
        """
        Load configuration from the ``[tunneling]`` section of a TOML file.

        Parameters
        ----------
        path : str
            Path to the TOML file (e.g. ``scan_config.toml``).

        Returns
        -------
        TunnelingConfig
        """
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "Reading TOML files requires Python 3.11+ or the 'tomli' package "
                    "(`pip install tomli`)."
                )
        with open(path, "rb") as f:
            data = tomllib.load(f)
        section = data.get("tunneling", {})

        # --- post-process special sentinel values ---
        # TOML has no native null; strings like "none" are interpreted as None
        for key in ("V_spline_samples", "log_level"):
            if isinstance(section.get(key), str) and section[key].lower() == "none":
                section[key] = None

        # log_level may also be specified as a level name, e.g. "INFO" → logging.INFO
        if isinstance(section.get("log_level"), str):
            lvl = getattr(logging, section["log_level"].upper(), None)
            if lvl is None:
                raise ValueError(
                    f"Unknown log_level string: {section['log_level']!r}. "
                    "Use an integer (10/20/30/40) or a level name "
                    "('DEBUG'/'INFO'/'WARNING'/'ERROR')."
                )
            section["log_level"] = lvl

        return cls(**section)


# ---------------------------------------------------------------------------
# Tier mapping (module-level, used by tunneling1D)
# ---------------------------------------------------------------------------

# (epsilon_lower_bound, thinCutoff, rmin)
# Apply the first tier for which epsilon > epsilon_lower_bound
_PROFILE_PARAM_TIERS: list[tuple[float, float, float]] = [
    (0.1,   0.01,  1e-4),   # thick-wall tier: ε > 0.1 (historical default)
    (1e-3,  1e-3,  1e-6),   # middle tier: 1e-3 < ε ≤ 0.1
    (0.0,   1e-4,  1e-7),   # thin-wall tier: ε ≤ 1e-3
]


def _epsilon_to_params(epsilon: float) -> tuple[float, float]:
    r"""
    Return the `(thinCutoff, rmin)` tier parameters selected by the thin-wall ratio \epsilon.

    Parameters
    ----------
    epsilon : float
        Thin-wall ratio estimated by ``SingleFieldInstanton._estimate_epsilon()``.
        \epsilon = ΔV / ΔV_barrier: \epsilon ~ 1 indicates thick-wall; \epsilon ≪ 1 indicates thin-wall.

    Returns
    -------
    thinCutoff, rmin : float, float
    """
    for eps_thresh, thinCutoff, rmin in _PROFILE_PARAM_TIERS:
        if epsilon > eps_thresh:
            return thinCutoff, rmin
    return 1e-4, 1e-7  # fallback (theoretical epsilon=0 case)


# ---------------------------------------------------------------------------
# enable_logging — package-level logging helper
# ---------------------------------------------------------------------------

def enable_logging(
        level: int = logging.DEBUG,
        fmt: str | None = None,
        stream=None,
        log_file: str | None = None,
) -> logging.Logger:
    """
    Enable logging for all cosmoTransitions modules.

    Attaches a handler to the ``cosmoTransitions`` root logger so that every
    module's ``logging.getLogger(__name__)`` output becomes visible.
    Safe to call multiple times — duplicate handlers on the same target are
    not added.

    Parameters
    ----------
    level : int
        Logging level, e.g. ``logging.DEBUG`` (default) or ``logging.INFO``.
    fmt : str or None
        Format string.  Default: ``'[%(levelname)s %(name)s] %(message)s'``.
    stream : file-like or None
        Output stream for a ``StreamHandler``.  Default: ``sys.stderr``.
        Ignored when *log_file* is given.
    log_file : str or None
        If provided, write logs to this file path instead of a stream.
        The file is opened in append mode (``'a'``) so successive runs
        accumulate in the same file.  The containing directory must exist.
        When both *stream* and *log_file* are given, *log_file* takes
        precedence.

    Returns
    -------
    logging.Logger
        The configured ``cosmoTransitions`` root logger.

    Examples
    --------
    ::

        import logging
        from cosmoTransitions import enable_logging

        enable_logging()                              # DEBUG  → stderr
        enable_logging(logging.INFO)                  # INFO   → stderr
        enable_logging(logging.DEBUG, log_file='ct.log')  # DEBUG → file

    Can also be triggered automatically via ``TunnelingConfig``::

        cfg = TunnelingConfig(log_level=logging.INFO, log_file='run.log')
        cfg.apply_log_level()  # equivalent to enable_logging(logging.INFO, log_file='run.log')
    """
    if fmt is None:
        fmt = '[%(levelname)s %(name)s] %(message)s'
    formatter = logging.Formatter(fmt)
    pkg_logger = logging.getLogger('cosmoTransitions')
    pkg_logger.setLevel(level)

    if log_file is not None:
        # FileHandler — avoid attaching a second handler to the same path.
        resolved = os.path.abspath(log_file)
        _already_has = any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(h.baseFilename) == resolved
            for h in pkg_logger.handlers
        )
        if not _already_has:
            fh = logging.FileHandler(resolved, mode='a', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            pkg_logger.addHandler(fh)
    else:
        # StreamHandler — avoid attaching a second handler to the same stream.
        target_stream = stream
        _already_has = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)  # FileHandler is a subclass
            and getattr(h, 'stream', None) is (
                target_stream if target_stream is not None else sys.stderr
            )
            for h in pkg_logger.handlers
        )
        if not _already_has:
            sh = logging.StreamHandler(target_stream)
            sh.setLevel(level)
            sh.setFormatter(formatter)
            pkg_logger.addHandler(sh)
    return pkg_logger
