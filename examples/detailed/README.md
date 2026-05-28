# CosmoTransitions — Examples

This folder contains seven self-contained example scripts that build progressively
from a simple weak first-order phase transition to advanced usage patterns
(extreme supercooling, parameter scans, configuration tuning, and diagnostics).

Each script can be run directly from the repository root:

```bash
python examples/example_XX_....py
```

---

## Quick-reference table

| # | Script | Model | Key result | Run time |
|---|--------|-------|-----------|----------|
| 01 | `example_01_single_field_ewpt.py` | Single field, weak FOPT | Tc = 81.65 GeV, Tn = 81.52 GeV | ~5 s |
| 02 | `example_02_multifield_ewpt.py` | Two fields (testModel1) | Tc = 222.95 GeV, Tn = 84.24 GeV | ~30 s |
| 03 | `example_03_supercooled_ewpt.py` | Single field, strong FOPT | Tc = 114.71 GeV, Tn = 84.56 GeV, Tn/Tc = 0.74 | ~15 s |
| 04 | `example_04_extreme_supercooled_ewpt.py` | Conformal U(1)X (KPX) | Tc = 3.31×10⁶ GeV, Tn = 8.84×10⁴ GeV, Tn/Tc = 0.027 | ~5 min |
| 05 | `example_05_config_guide.py` | Single field (inline) | Full `TunnelingConfig` parameter reference | ~60 s |
| 06 | `example_06_parameter_scan.py` | Single field, E scan | Tn/Tc vs E; E_crit = √(Dλ) = 0.10 | ~30 s |
| 07 | `example_07_logging_debug.py` | Single field (inline) | Logging levels: 0 / 22 / 984 lines | ~60 s |

---

## Suggested reading order

```
01  →  02  →  03  →  04          (physics: weak → strong → extreme supercooling)
                 ↓
                05  →  06  →  07  (tools: config tuning → scans → debugging)
```

Examples 05–07 are independent tool-focused guides; they can be read in any order
after example 01.

---

## Example descriptions

### example_01 — Single-field weak first-order EWPT

**Model**
$$V(\phi, T) = D(T^2 - T_0^2)\,\phi^2 - E\,T\,\phi^3 + \tfrac{\lambda}{4}\,\phi^4$$
with $D=0.10$, $E=0.02$, $T_0=80\,\text{GeV}$, $\lambda=0.10$.

**What it demonstrates**

- The canonical three-step CosmoTransitions pipeline:
  `getPhases()` → `calcTcTrans()` → `findAllTransitions()`
- How to define a model by subclassing `generic_potential.generic_potential`
  with an `init()` method and `Vtot()`.
- Reading back transition output: `Tc`, `Tn`, `S3/Tn`, order parameter
  $\phi_n / T_n$, and phase-transition strength $\alpha$.
- Default `TunnelingConfig()` settings are sufficient here because
  Tn/Tc ≈ 0.9984 (very weak transition).

**Key numbers**: Tc = 81.65 GeV, Tn = 81.52 GeV, S₃/Tn = 139.89,
φn/Tn ≈ 0.16, Tn/Tc = 0.9984.

**Output**: terminal summary + `example_01_output.png`
(potential at Tc, S₃/T curve, instanton profile).

---

### example_02 — Two-field EWPT with path deformation

**Model**
`examples.testModel1.model1` — a built-in two-field demo with extra bosons.
Fields: $(\phi_1, \phi_2)$; approximate zero-T minimum near $(246, 246)$ GeV.

**What it demonstrates**

- Multi-field phase structure: three distinct phases (high-T symmetric,
  and two degenerate low-T broken-phase branches).
- How the path-deformation algorithm finds the optimal tunnelling path
  through the two-dimensional field space.
- The role of `TunnelingConfig.deform_fRatioConv` (convergence threshold for
  the bounce path): smaller values → more accurate but slower.
- Visualizing the bounce path overlaid on the 2-D potential contour.

**Key numbers**: Tc = 222.95 GeV, Tn = 84.24 GeV, Tn/Tc = 0.378.

**Output**: terminal summary + `example_02_output.png`
(2-D potential surface, phase traces, bounce path at Tn).

---

### example_03 — Moderately supercooled single-field EWPT

**Model**
Same potential as example_01, but with $E = 0.09$ (larger cubic coefficient)
and $T_0 = 50\,\text{GeV}$, $\lambda = 0.10$, $D = 0.10$.

**What it demonstrates**

- How a larger cubic coupling $E$ deepens the barrier and causes the
  transition to nucleate well below $T_c$ (here Tn/Tc ≈ 0.74, 26%
  supercooling).
- The thin-wall parameter
  $\varepsilon = \Delta V / \Delta V_\text{barrier}$ at $T_n$:
  large $\varepsilon \gg 1$ (thick-wall) here, so the default config suffices.
- When and why to switch to `TunnelingConfig.supercooling_preset()`:
  the preset is needed when Tn/Tc ≲ 0.5 (thick-wall + large barrier).
- The $S_3(T)/T$ profile computed by `SingleFieldInstanton`, showing how
  the action curve crosses the nucleation threshold.

**Key numbers**: Tc = 114.71 GeV, Tn = 84.56 GeV, Tn/Tc = 0.737,
φc/Tc ≈ 1.8 (strongly first-order).

**Output**: terminal summary + `example_03_output.png`
(S₃/T curve, potential at Tc and Tn, instanton profile).

---

### example_04 — Extreme supercooling with a conformal U(1)X model

**Model**
Dark U(1)X scalar $h$ with Coleman-Weinberg potential + thermal corrections
(KPX model). No tree-level mass (conformally symmetric):
$$V_0 = 0,\quad
  V_\text{CW}(h) = \frac{3 g_X^4}{32\pi^2} h^4
  \!\left(\ln\frac{h}{w} - \frac{1}{4}\right),\quad
  V_T(h,T) = \frac{3T^4}{2\pi^2} J_b\!\!\left(\frac{g_X^2 h^2}{T^2}\right)
             - \frac{g_X^3 T}{12\pi}\!\left[(h^2+T^2)^{3/2}-h^3\right]$$
Parameters: $g_X = 0.65$, $m_X = 10^7\,\text{GeV}$, $w = m_X/g_X$.

**What it demonstrates**

1. **Why the default config fails for conformal models.**
   With `V_spline_samples=100` (default), the spline misses the narrow
   thermal barrier near $T_c$, causing `tunneling1D` to underestimate
   the action $S_3/T$ and report spuriously high nucleation temperatures.
   Default gives Tn ≈ 1.01×10⁵ GeV (1.14× too high).

2. **How `TunnelingConfig.supercooling_preset()` fixes this.**
   Sets `V_spline_samples=None` (exact potential), `thinCutoff=1e-4`,
   `rmin=1e-7`, `T_scan_extension=True`. Result: Tn = 8.84×10⁴ GeV.

3. **Nucleation criterion comparison: `fixed_140` vs `cosmological`.**
   - `fixed_140`: threshold constant at S₃/T = 140.
   - `cosmological`: threshold = $4\ln(M_\text{Pl}/T)$, which is ≈ 157 at low T
     (looser, higher Tn) and ≈ 130 at high T (stricter, lower Tn).
   - Crossover temperature: $4\ln(M_\text{Pl}/T) = 140 \Rightarrow T \approx 7700\,\text{GeV}$.
   - For this model (Tn ~ 10⁴–10⁵ GeV), cosmological criterion gives
     Tn = 6.64×10⁴ GeV (−24.9% vs fixed_140).

**Key numbers**: Tc = 3.305×10⁶ GeV, Tn (preset) = 8.840×10⁴ GeV,
Tn/Tc = 0.0267 (97.3% supercooling).

**Output**: terminal summary + `example_04_output.png`
(S₃/T curve with both thresholds, comparison table of all three runs).

---

### example_05 — TunnelingConfig parameter reference guide

**Purpose**
A comprehensive reference guide for every numerical parameter in
`TunnelingConfig`. Runs live demonstrations on a simple single-field model.

**Sections**

| § | Topic |
|---|-------|
| A | Full parameter table (8 groups: findProfile, SplinePath, deformation, nucleation, T-scan, phase tracing, retries, logging) |
| B | Adaptive ε-tier selection — how `_epsilon_to_params()` auto-chooses `thinCutoff` and `rmin` based on the thin-wall ratio ε |
| C | `write_default()` + `from_file()` — TOML round-trip (config serialization) |
| D | Nucleation criteria comparison — `fixed_140` vs `cosmological` vs custom callable; threshold table; crossover at T ≈ 7700 GeV |
| E | `enable_logging()` demo — how to capture logs to a file and read back the first 15 lines |
| F | `supercooling_preset()` vs default — side-by-side comparison of all parameter values |
| G | Visualization — threshold curves and ε-tier step chart |

**Output**: terminal reference guide + `example_05_output.png`

---

### example_06 — Parameter scan: cubic coupling E vs transition strength

**Purpose**
Shows how to systematically scan a model parameter and collect
$T_c$, $T_n$, $T_n/T_c$, $S_3/T_n$ at each scan point.

**Model**
Same single-field potential as example_01, with $D=0.10$, $T_0=80\,\text{GeV}$,
$\lambda=0.10$ fixed while $E$ is varied over `[0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]`.

**Physics covered**

- For $E < E_\text{crit} = \sqrt{D\lambda} = 0.10$: standard first-order
  transition with finite $T_c = T_0 / \sqrt{1 - E^2/(D\lambda)}$. As E increases,
  the barrier grows, Tc rises, and Tn/Tc decreases.
- At $E = E_\text{crit}$: $T_c \to \infty$ (critical point of the phase diagram).
- For $E > E_\text{crit}$: no finite $T_c$; `findAllTransitions` cannot bound
  its temperature search and returns no result. A different strategy
  (spinodal temperature, fixed Tmax) is needed.

**Scan results**

| E | Tc (GeV) | Tn (GeV) | Tn/Tc | S₃/Tn |
|---|----------|----------|-------|-------|
| 0.01 | 80.40 | 80.38 | 0.9997 | 137.51 |
| 0.02 | 81.65 | 81.52 | 0.9984 | 139.89 |
| 0.04 | 87.29 | 86.32 | 0.9889 | 139.98 |
| 0.06 | 100.00 | 95.97 | 0.9597 | 140.00 |
| 0.08 | 133.33 | 115.82 | 0.8687 | 140.00 |
| ≥0.10 | ∞ (no Tc) | N/A | — | — |

**Patterns demonstrated**

- `ScanPoint` dataclass for structured result collection.
- `adaptive_rescan()`: automatically retry failed scan points with
  `TunnelingConfig.supercooling_preset()`.
- `concurrent.futures.ProcessPoolExecutor` pattern for parallel scans
  (requires model class to be importable from a module, not `__main__`).
- `TunnelingConfig` selection guide based on Tn/Tc:
  - Tn/Tc > 0.8 → defaults
  - 0.5 < Tn/Tc ≤ 0.8 → `T_scan_extension=True`
  - Tn/Tc ≤ 0.5 → `supercooling_preset()`

**Output**: terminal results table + `example_06_output.png`
(three-panel: Tc/Tn vs E, Tn/Tc vs E, S₃/Tn vs E).

---

### example_07 — Logging and debugging guide

**Purpose**
Shows how to use the CosmoTransitions logging system for diagnosing
numerical issues, tracking calculation progress, and writing audit logs to file.

**Sections**

| § | Topic |
|---|-------|
| A | Logging architecture — module hierarchy, which modules emit at which levels |
| B | Log level comparison: `WARNING` (0 lines), `INFO` (22 lines), `DEBUG` (984 lines), captured to an in-memory buffer for display |
| C | Writing DEBUG logs to a file via `enable_logging(level=DEBUG, log_file=...)` |
| D | Per-module filtering — silence `tunneling1D` and `pathDeformation` while keeping `transitionFinder` output |
| E | `TunnelingConfig(log_level=INFO)` + `cfg.apply_log_level()` integration pattern |

**Module log volume summary**

| Module | Default verbosity | What it emits |
|--------|------------------|---------------|
| `transitionFinder` | INFO | Phase tracing milestones, Tc/Tn results |
| `pathDeformation` | INFO/DEBUG | Path convergence steps and fRatio values |
| `tunneling1D` | DEBUG | Instanton ODE steps, profile details |
| `generic_potential` | INFO | Phase seeding, phase count |
| `config` | INFO | Config load/save events |

**Key insight**: the `cosmoTransitions` root logger aggregates all child loggers.
`enable_logging()` attaches a single handler there; per-module silencing uses
`logging.getLogger('cosmoTransitions.X').setLevel(logging.WARNING)`.

**Output**: terminal guide (log output to stderr, structured text to stdout)

---

## Common model pattern

All examples that define a custom model follow this structure:

```python
from cosmoTransitions import generic_potential
import numpy as np

class MyModel(generic_potential.generic_potential):
    def init(self, param1=..., param2=...):
        self.Ndim = 1          # number of scalar fields
        self.Tmax = ...        # upper temperature for phase tracing
        self.x_eps = 0.001     # field-space step for numerical gradients

    def approxZeroTMin(self):
        # Seed the broken-phase minimum at T=0 (required for large barriers)
        return [np.array([phi_vev])]

    def Vtot(self, X, T, include_radiation=False):
        phi = np.asanyarray(X, dtype=float)[..., 0]
        T   = np.asanyarray(T,   dtype=float)
        return ...   # finite-T effective potential

m = MyModel(param1=..., param2=...)
results = m.findAllTransitions(tunneling_config=TunnelingConfig())
```

> **Note:** use `init()`, not `__init__()`. The base class `__init__` calls
> `init()` internally after wiring up internal state.

---

## TunnelingConfig quick selection guide

| Scenario | Recommended config |
|----------|--------------------|
| Tn/Tc > 0.80 (weak transition) | `TunnelingConfig()` defaults |
| 0.50 < Tn/Tc ≤ 0.80 | `TunnelingConfig(T_scan_extension=True)` |
| 0.10 < Tn/Tc ≤ 0.50 | `TunnelingConfig.supercooling_preset()` |
| Tn/Tc ≤ 0.10 (conformal / extreme) | `supercooling_preset()` + `V_spline_samples=None` |
| Debugging a failed scan | `TunnelingConfig(log_level=logging.DEBUG)` + `cfg.apply_log_level()` |
| Reproducible runs | `cfg.write_default('params.toml')` + `TunnelingConfig.from_file(...)` |

---

## Output files

Each script saves a figure to the same directory as the script:

| Script | Figure |
|--------|--------|
| example_01 | `example_01_output.png` |
| example_02 | `example_02_output.png` |
| example_03 | `example_03_output.png` |
| example_04 | `example_04_output.png` |
| example_05 | `example_05_output.png` |
| example_06 | `example_06_output.png` |
| example_07 | *(no figure — logging demo only)* |
