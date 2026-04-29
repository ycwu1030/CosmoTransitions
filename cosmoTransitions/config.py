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
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

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

    # --- fullTunneling / SplinePath ---

    V_spline_samples: Optional[int] = 100
    """控制 ``SplinePath`` 是否对势能预采样。

    ``100``（默认）在[φ_sym, φ_broken]上均匀取100个点，用PCHIP样条逼近V(x)。
    对典型EW尺度模型（势垒占场值范围的较大比例）效果良好。

    设为 ``None`` 时，``SplinePath.V(x)`` 直接调用 ``Vtot``，不做预采样。
    **对极度过冷模型（Tn/Tc ≲ 0.05）必须使用 ``None``**：均匀100点的间距
    约为 w/99，而热势垒位于 φ~Tn ≪ w/99，预采样完全错过势垒，
    导致 ``SingleFieldInstanton`` 找不到势垒或计算出错误的作用量。

    ``supercooling_preset()`` 自动将此项设为 ``None``。
    """

    maxiter_fullTunneling: int = 20
    """``fullTunneling`` 最大外层迭代数（路径形变 + 1D隧穿交替的轮数）。

    每次迭代依次执行：(1) 用当前路径点拟合 SplinePath，(2) 沿路径做1D隧穿，
    (3) 用 deformPath 修正路径，(4) 检查收敛。通常10次即收敛；
    极薄壁泡泡可能需要更多。
    """

    # --- 路径形变（pathDeformation）---

    deform_fRatioConv: float = 0.02
    """路径形变收敛判据：路径上最大法向力 / 最大势能梯度 < 此值时停止。

    越小越精确但越慢。对一般FOPT取 0.02；若过冷模型作用量对路径敏感，
    可减小至 0.005。
    """

    deform_maxiter: int = 500
    """``deformPath`` 最大形变步数。超过此值时打印警告并返回当前最优路径。"""

    # --- tunnelFromPhase 参数 ---

    Ttol: float = 1e-3
    """寻找核成温度 Tn 的温度精度（brentq 的 ``xtol``），单位与 T 一致（GeV）。

    减小此值可得到更精确的 Tn，但会增加 ``_tunnelFromPhaseAtT`` 调用次数。
    对 Tn~O(100) GeV 的典型EW模型，``1e-3`` 已足够；极度过冷模型
    （Tn~O(1e4) GeV）可放宽至 ``1.0``。
    """

    maxiter_tunnel: int = 100
    """``tunnelFromPhase`` 中 brentq / fmin 的最大迭代数。"""

    phitol_tunnel: float = 1e-8
    """``_tunnelFromPhaseAtT`` 中精化真空位置时的场值精度（传给 L-BFGS-B ``gtol``）。

    注意：这与 ``phitol``（``findProfile`` 的ODE精度）是不同的参数。
    此值过大会导致在错误的场值处计算作用量；过小会显著增加优化迭代次数。
    """

    overlapAngle: float = 45.0
    """从假真空出发，若两个目标相的方向夹角小于此角度（度），仅对较近的那个做隧穿计算。

    设为 ``0.0`` 则始终对所有可能的目标相尝试隧穿（多相方向相近时更完整但更慢）。
    45° 是原始CosmoTransitions默认值。
    """

    # --- 相追踪参数（traceMultiMin / traceMinimum）---

    dtstart: float = 1e-3
    """``traceMultiMin`` 初始温度步长，相对于 ``tHigh - tLow`` 的比值。

    过大可能跳过窄相；过小增加计算量。典型值 1e-3（即全温度范围的0.1%）。
    对温度范围跨度大（如超过3个量级）的模型，适当减小至 1e-4。
    """

    tjump: float = 1e-3
    """``traceMultiMin`` 中一个相结束后、搜索下一个相时的温度跳跃，相对于
    ``tHigh - tLow`` 的比值。

    若某个中间相的温度宽度小于 ``tjump * (tHigh - tLow)``，该相可能被跳过。
    减小此值可减少漏相风险，但会增加不必要的追踪起始点。
    """

    # --- 日志级别 ---

    log_level: Optional[int] = None
    """若不为 ``None``，在此 ``TunnelingConfig`` 实例首次使用时自动启用
    cosmoTransitions 日志到该级别。

    例如：``log_level=logging.INFO`` 等价于在代码中调用
    ``cosmoTransitions.enable_logging(logging.INFO)``。
    ``None`` 表示不修改任何日志配置（默认行为）。
    """

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

    def get_fullTunneling_kwargs(self) -> dict:
        """返回应传给 ``pathDeformation.fullTunneling`` 的关键字参数字典。

        包含 ``V_spline_samples`` 和 ``maxiter``，可直接 ``**`` 解包。
        不包含 ``deformation_deform_params``（由 :meth:`get_deform_params` 管理）。

        Returns
        -------
        dict
        """
        return dict(
            V_spline_samples=self.V_spline_samples,
            maxiter=self.maxiter_fullTunneling,
        )

    def get_deform_params(self) -> dict:
        """返回应传给 ``deformPath`` 的 ``deformation_deform_params`` 字典。

        Returns
        -------
        dict
        """
        return dict(
            fRatioConv=self.deform_fRatioConv,
            maxiter=self.deform_maxiter,
        )

    def get_tunnelFromPhase_kwargs(self) -> dict:
        """返回应传给 ``tunnelFromPhase`` 的关键字参数字典。

        包含 ``Ttol``、``maxiter``、``phitol``、``overlapAngle``，
        不包含 ``fullTunneling_params``（由 :meth:`get_fullTunneling_kwargs` 管理）。

        Returns
        -------
        dict
        """
        return dict(
            Ttol=self.Ttol,
            maxiter=self.maxiter_tunnel,
            phitol=self.phitol_tunnel,
            overlapAngle=self.overlapAngle,
        )

    def get_traceMultiMin_kwargs(self) -> dict:
        """返回应传给 ``traceMultiMin`` 的关键字参数字典。

        包含 ``dtstart``、``tjump``。

        Returns
        -------
        dict
        """
        return dict(
            dtstart=self.dtstart,
            tjump=self.tjump,
        )

    def apply_log_level(self) -> None:
        """若 :attr:`log_level` 不为 ``None``，自动启用 cosmoTransitions 日志。"""
        if self.log_level is not None:
            enable_logging(self.log_level)

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
        """极度过冷预设（Tn/Tc ≲ 0.05）。

        关键特性：

        * ``V_spline_samples=None`` — **必须**：极度过冷模型的热势垒
          位于 φ~Tn ≪ w，均匀100点采样（间距~w/100）完全错过势垒。
          设为 None 后 SplinePath 直接调用 Vtot 求值，势垒分辨率无限制。
        * 严格的 ``thinCutoff=1e-4``、``rmin=1e-7``：薄壁泡泡的数值积分参数。
        * 启用 T 扫描延伸（最多5次，每次将下界×0.1）以追踪极低 Tn。
        * ``Ttol=1.0``：Tn 通常在 O(1e4) GeV，GeV 级精度已足够。
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
        将内置的默认配置模板写出为 TOML 文件。

        生成的文件包含所有参数的默认值和详细注释，
        用户可将其复制后修改，再通过 :meth:`from_file` 加载。

        Parameters
        ----------
        path : str
            输出路径，默认 ``"cosmoTransitions_config.toml"``。

        Examples
        --------
        ::

            TunnelingConfig.write_default("my_scan.toml")   # 生成模板
            # 编辑 my_scan.toml 后：
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

        # --- 后处理特殊值 ---
        # TOML 无 null 类型；用字符串 "none" 表示 Python None
        for key in ("V_spline_samples", "log_level"):
            if isinstance(section.get(key), str) and section[key].lower() == "none":
                section[key] = None

        # log_level 允许写日志级别名称，如 "INFO" → logging.INFO
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


# ---------------------------------------------------------------------------
# enable_logging — 包级日志启用工具
# ---------------------------------------------------------------------------

def enable_logging(
        level: int = logging.DEBUG,
        fmt: str | None = None,
        stream=None,
) -> logging.Logger:
    """
    为所有 cosmoTransitions 模块启用日志输出。

    在 ``cosmoTransitions`` 根 logger 上添加一个 ``StreamHandler``，
    使包内所有模块的 ``logging.getLogger(__name__)`` 日志可见。
    多次调用安全（不会重复添加相同 handler）。

    Parameters
    ----------
    level : int
        日志级别，如 ``logging.DEBUG``（默认）或 ``logging.INFO``。
    fmt : str or None
        格式字符串。默认：``'[%(levelname)s %(name)s] %(message)s'``。
    stream : file-like or None
        输出流。默认：``sys.stderr``。

    Returns
    -------
    logging.Logger
        已配置好的 ``cosmoTransitions`` 根 logger。

    Examples
    --------
    ::

        import logging
        from cosmoTransitions import enable_logging

        enable_logging()                              # DEBUG → stderr
        enable_logging(logging.INFO)                  # INFO  → stderr
        enable_logging(logging.DEBUG, stream=open('ct.log', 'w'))

    也可通过 ``TunnelingConfig.log_level`` 字段自动触发::

        cfg = TunnelingConfig(log_level=logging.INFO)
        cfg.apply_log_level()  # 等价于 enable_logging(logging.INFO)
    """
    if fmt is None:
        fmt = '[%(levelname)s %(name)s] %(message)s'
    target_stream = stream  # keep reference for duplicate check
    pkg_logger = logging.getLogger('cosmoTransitions')
    pkg_logger.setLevel(level)
    # Avoid adding a duplicate StreamHandler to the same stream.
    _already_has = any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, 'stream', None) is (
            target_stream if target_stream is not None else sys.stderr
        )
        for h in pkg_logger.handlers
    )
    if not _already_has:
        handler = logging.StreamHandler(target_stream)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        pkg_logger.addHandler(handler)
    return pkg_logger
