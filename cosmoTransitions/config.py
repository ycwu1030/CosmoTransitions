"""
TunnelingConfig — 统一的数值参数配置对象。

封装与模型无关的数值参数，通过 ``findAllTransitions`` / ``tunnelFromPhase``
向整个计算链传递，避免逐层手动传参。

典型用法::

    from cosmoTransitions.config import TunnelingConfig

    # 使用深度过冷预设
    cfg = TunnelingConfig.supercooling_preset()
    model.findAllTransitions(tunneling_config=cfg)

    # 从 TOML 文件读取
    cfg = TunnelingConfig.from_file("scan_config.toml")  # [tunneling] 节
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核成判据函数
# ---------------------------------------------------------------------------

def fixed_140_nucl_criterion(S: float, T: float) -> float:
    """经典硬编码判据：S3/T - 140。与原始 CosmoTransitions 默认行为完全一致。"""
    return S / (T + 1e-100) - 140.0


def cosmological_nucl_criterion(
        S: float,
        T: float,
        M_Pl: float = 1.22e19,
) -> float:
    R"""
    辐射主导宇宙的精确宇宙学核成判据：

    .. math::
        \frac{S_3}{T} - 4 \ln\!\left(\frac{M_{\rm Pl}}{T}\right) = 0

    其中 :math:`M_{\rm Pl} \approx 1.22 \times 10^{19}` GeV（Planck 质量，单位
    需与 `T` 一致）。

    在 T = 100 GeV 时阈值约 143（vs. 固定 140），T = 1 GeV 时约 171，
    T = 1 MeV 时约 199。

    Parameters
    ----------
    S : float
        Euclidean 作用量 S3（单位 GeV）。
    T : float
        温度（单位 GeV）。
    M_Pl : float, optional
        Planck 质量，默认 1.22e19 GeV。

    Returns
    -------
    float
        < 0 表示已核化；> 0 表示未核化。
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
    """
    封装与模型无关的数值参数，统一传递至整个隧穿计算链。

    Parameters
    ----------
    thinCutoff : float or ``"auto"``
        传递至 ``SingleFieldInstanton.findProfile`` 的 ``thinCutoff`` 参数。
        ``"auto"`` 时根据薄壁比 ε 自动推导。
    rmin : float or ``"auto"``
        传递至 ``findProfile`` 的 ``rmin`` 参数（相对 ``rscale``）。
        ``"auto"`` 时由 ε 自动推导。
    xtol : float
        ``findProfile`` 的 ``xtol``（二分精度）。
    phitol : float
        ``findProfile`` 的 ``phitol``（ODE 积分精度）。
    rmax : float
        ``findProfile`` 的 ``rmax``（最大积分半径，相对 ``rscale``）。
    npoints : int
        ``findProfile`` 的 ``npoints``（输出点数）。
    T_scan_extension : bool
        是否允许 ``tunnelFromPhase`` 向低温延伸 T 扫描区间。
    T_scan_max_extend : int
        最多延伸次数（每次将扫描下界 ×0.1）。
    nuclCriterion : ``"fixed_140"`` | ``"cosmological"`` | callable
        核成判据。``"fixed_140"`` 保持原始默认行为（S/T−140）；
        ``"cosmological"`` 使用 4ln(M_Pl/T) 物理阈值；
        callable 应满足 ``nuclCriterion(S, T) -> float``，< 0 代表已核化。
    enable_profile_retry : bool
        当 ``findProfile`` 返回 ``Rerr is not None``（步长崩溃）时，
        是否自动用更严格的参数档位重试。
    max_profile_retries : int
        最多重试次数。
    use_adaptive_grad : bool
        是否在 ``generic_potential.gradV``/``d2V`` 中使用
        ``helper_functions.adaptive_gradient/adaptive_hessian``。
        （由 generic_potential 读取；保留在此供集中配置。）

    Notes
    -----
    ``TunnelingConfig`` 对象作为可选参数传入 ``tunnelFromPhase``、
    ``findAllTransitions``（transitionFinder）、以及
    ``generic_potential.findAllTransitions``。``None`` 时内部使用
    ``TunnelingConfig()`` 默认实例，行为与历史版本完全一致。

    向后兼容保证：``thinCutoff/rmin = "auto"`` 在厚壁势（ε≈1）下
    推导出与原始默认值（0.01 / 1e-4）相同的参数，不改变厚壁模型结果。
    """

    # findProfile 参数
    thinCutoff: Union[float, str] = "auto"
    rmin: Union[float, str] = "auto"
    xtol: float = 1e-6
    phitol: float = 1e-6
    rmax: float = 1e4
    npoints: int = 500

    # T 扫描
    T_scan_extension: bool = True
    T_scan_max_extend: int = 3

    # 核成判据（默认 fixed_140 保持向后兼容）
    nuclCriterion: Union[str, Callable[[float, float], float]] = "fixed_140"

    # findProfile 自适应重试
    enable_profile_retry: bool = True
    max_profile_retries: int = 3

    # adaptive FD 开关（由 generic_potential 读取）
    use_adaptive_grad: bool = True

    def get_nucl_criterion(self) -> Callable[[float, float], float]:
        """
        返回核成判据 callable。

        Returns
        -------
        callable
            签名 ``(S: float, T: float) -> float``，< 0 表示已核化。
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

    def get_findProfile_kwargs(self, epsilon: float | None = None) -> dict:
        """
        返回应传给 ``findProfile`` 的关键字参数字典。

        Parameters
        ----------
        epsilon : float or None
            薄壁比（由 ``SingleFieldInstanton._estimate_epsilon()`` 计算）。
            若 ``thinCutoff``/``rmin`` 为 ``"auto"``，由此推导实际值。

        Returns
        -------
        dict
            包含 ``thinCutoff``、``rmin``、``xtol``、``phitol``、``rmax``、
            ``npoints`` 的字典，可直接 ``**`` 解包传入 ``findProfile``。
        """
        tc = self.thinCutoff
        rm = self.rmin
        if tc == "auto" or rm == "auto":
            if epsilon is None:
                # 无法推导，使用原始默认值
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
    # 预设工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def supercooling_preset(cls) -> "TunnelingConfig":
        """
        深度过冷预设，对应 ``Main.py`` 风格参数。

        适用于极度过冷的强一阶相变（薄壁泡泡），使用严格的
        ``thinCutoff=1e-4``、``rmin=1e-7``，并启用 T 扫描延伸。
        """
        return cls(
            thinCutoff=1e-4,
            rmin=1e-7,
            T_scan_extension=True,
            T_scan_max_extend=5,
        )

    @classmethod
    def from_file(cls, path: str) -> "TunnelingConfig":
        """
        从 TOML 文件的 ``[tunneling]`` 节读取配置。

        Parameters
        ----------
        path : str
            TOML 文件路径（例如 ``scan_config.toml``）。

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
        # nuclCriterion 在 TOML 中只能是字符串
        return cls(**section)


# ---------------------------------------------------------------------------
# 档位映射（模块级，供 tunneling1D 使用）
# ---------------------------------------------------------------------------

# (epsilon_lower_bound, thinCutoff, rmin)
# 按 epsilon 从大到小：epsilon > 档位的下界时使用该档
_PROFILE_PARAM_TIERS: list[tuple[float, float, float]] = [
    (0.1,   0.01,  1e-4),   # 厚壁档：ε > 0.1（原始默认行为）
    (1e-3,  1e-3,  1e-6),   # 中档：  1e-3 < ε ≤ 0.1
    (0.0,   1e-4,  1e-7),   # 薄壁档：ε ≤ 1e-3（对应 Main.py 参数）
]


def _epsilon_to_params(epsilon: float) -> tuple[float, float]:
    """
    根据薄壁比 ε 返回 ``(thinCutoff, rmin)`` 档位参数。

    Parameters
    ----------
    epsilon : float
        薄壁比，由 ``SingleFieldInstanton._estimate_epsilon()`` 计算。
        ε = ΔV / ΔV_barrier；ε≈1 为厚壁，ε≪1 为极薄壁。

    Returns
    -------
    thinCutoff, rmin : float, float
    """
    for eps_thresh, thinCutoff, rmin in _PROFILE_PARAM_TIERS:
        if epsilon > eps_thresh:
            return thinCutoff, rmin
    return 1e-4, 1e-7  # 兜底（ε=0 理论情形）
