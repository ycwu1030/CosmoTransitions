# CosmoTransitions — Slim Examples

This folder contains the same seven example scripts as `../detailed/`, but with
all pedagogical commentary stripped out.  The core physics code and pipeline
calls are identical; only the extras are removed:

| What was removed | Why |
|-----------------|-----|
| `V0()` / `V1T_from_X()` helper methods | Not required by the pipeline; provided in detailed for teaching |
| `compute_epsilon()` analysis (ex03) | ε diagnostics; useful for learning, not needed to run |
| Long parameter docstrings | Condensed to one-line summaries |
| Parallel scan code (ex06) | Sequential-only; see detailed for `ProcessPoolExecutor` pattern |
| ε-tier commentary (ex03) | Thin-wall regime explanation; see detailed or example_05 |

Use the **detailed** versions when learning the library.  Use the **slim**
versions as copy-paste starting points for your own models.

Each script can be run directly from the repository root:

```bash
python examples/slim/example_XX_....py
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

## Example descriptions

### example_01 — Single-field weak first-order EWPT

**Model**
$$V(\phi, T) = D(T^2 - T_0^2)\,\phi^2 - E\,T\,\phi^3 + \tfrac{\lambda}{4}\,\phi^4$$
with $D=0.10$, $E=0.02$, $T_0=80\,\text{GeV}$, $\lambda=0.10$.

**What it shows**
The canonical three-step pipeline — `getPhases()` → `calcTcTrans()` →
`findAllTransitions()` — plus how to read back Tc, Tn, S₃/Tn and the
order parameter φn/Tn.  Default `TunnelingConfig()` is sufficient (Tn/Tc ≈ 0.9984).

**Slim vs detailed**: no `V0()`, no `V1T_from_X()`, condensed `init()`.

**Key numbers**: Tc = 81.65 GeV, Tn = 81.52 GeV, S₃/Tn = 139.89, Tn/Tc = 0.9984.

**Output**: terminal summary + `example_01_output.png`

---

### example_02 — Two-field EWPT with path deformation

**Model**
`examples.testModel1.model1` — built-in two-field demo.
Fields: $(\phi_1, \phi_2)$; zero-T minimum near $(246, 246)$ GeV.

**What it shows**
Multi-field phase structure (three phases), path-deformation convergence
(`TunnelingConfig.deform_fRatioConv`), and bounce-path visualization on the
2-D potential surface.

**Slim vs detailed**: condensed docstrings; same code structure.

**Key numbers**: Tc = 222.95 GeV, Tn = 84.24 GeV, Tn/Tc = 0.378.

**Output**: terminal summary + `example_02_output.png`

---

### example_03 — Moderately supercooled single-field EWPT

**Model**
Same potential as example_01 with $E=0.09$, $T_0=50\,\text{GeV}$.

**What it shows**
How a larger cubic coupling deepens the barrier and pushes Tn below Tc
(26% supercooling).  Shows the S₃(T)/T profile and when to switch to
`TunnelingConfig.supercooling_preset()`.

**Slim vs detailed**: removed `compute_epsilon()` helper and ε-tier discussion;
removed `V1T_from_X()`.  The ε analysis is still covered in `example_05` (§B).

**Key numbers**: Tc = 114.71 GeV, Tn = 84.56 GeV, Tn/Tc = 0.737, φc/Tc ≈ 1.8.

**Output**: terminal summary + `example_03_output.png`

---

### example_04 — Extreme supercooling with a conformal U(1)X model

**Model**
Dark U(1)X scalar with Coleman-Weinberg potential + thermal corrections (KPX
model); no tree-level mass.  Parameters: $g_X = 0.65$, $m_X = 10^7\,\text{GeV}$.

**What it shows**

1. Why `V_spline_samples=100` (default) misses the narrow thermal barrier and
   reports a spuriously high Tn.
2. How `supercooling_preset()` (sets `V_spline_samples=None`) fixes this.
3. Nucleation criterion comparison: `fixed_140` vs `cosmological`
   ($S_3/T < 4\ln(M_\text{Pl}/T)$); crossover at $T \approx 7700\,\text{GeV}$.

**Slim vs detailed**: condensed docstring; same core steps.

**Key numbers**: Tc = 3.305×10⁶ GeV, Tn = 8.840×10⁴ GeV, Tn/Tc = 0.0267.

**Output**: terminal summary + `example_04_output.png`

---

### example_05 — TunnelingConfig parameter reference guide

**What it shows**
The same seven-section tour of `TunnelingConfig` as in detailed:

| § | Topic |
|---|-------|
| A | Full parameter table (8 groups) |
| B | Adaptive ε-tier selection |
| C | TOML round-trip (`write_default` / `from_file`) |
| D | Nucleation criteria comparison |
| E | `enable_logging()` demo |
| F | `supercooling_preset()` vs default |
| G | Visualization |

**Slim vs detailed**: minor code simplification in the model class.

**Output**: terminal reference guide + `example_05_output.png`

---

### example_06 — Parameter scan: cubic coupling E vs transition strength

**Model**
Single-field potential, $D=0.10$, $T_0=80\,\text{GeV}$, $\lambda=0.10$;
$E$ scanned over `[0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]`.

**What it shows**
How to loop over model parameters, collect Tc/Tn/S₃ at each point, handle
missing transitions gracefully, and auto-retry with `supercooling_preset()`.
Includes the analytical result $T_c = T_0/\sqrt{1 - E^2/(D\lambda)}$ and the
critical point at $E_\text{crit} = \sqrt{D\lambda} = 0.10$.

**Slim vs detailed**: `run_scan()` is sequential-only (no `ProcessPoolExecutor`
branch); `V1T_from_X()` removed; docstring condensed.

**Scan results**

| E | Tc (GeV) | Tn (GeV) | Tn/Tc |
|---|----------|----------|-------|
| 0.01 | 80.40 | 80.38 | 0.9997 |
| 0.02 | 81.65 | 81.52 | 0.9984 |
| 0.04 | 87.29 | 86.32 | 0.9889 |
| 0.06 | 100.00 | 95.97 | 0.9597 |
| 0.08 | 133.33 | 115.82 | 0.8687 |
| ≥0.10 | ∞ (no Tc) | N/A | — |

**Output**: terminal results table + `example_06_output.png`

---

### example_07 — Logging and debugging guide

**What it shows**
The same five-section logging guide as in detailed — identical content,
same code.

| § | Topic |
|---|-------|
| A | Module hierarchy and log volumes |
| B | Level comparison: WARNING / INFO / DEBUG (0 / 22 / 984 lines) |
| C | Writing to a log file |
| D | Per-module silencing |
| E | `TunnelingConfig.log_level` + `apply_log_level()` |

**Slim vs detailed**: no differences.

**Output**: terminal guide (log output to stderr, structured text to stdout)

---

## Minimal model template

```python
from cosmoTransitions import generic_potential
import numpy as np

class MyModel(generic_potential.generic_potential):
    def init(self, D=0.10, E=0.02, T0=80.0, lam=0.10):
        self.Ndim = 1
        self.D = D; self.E = E; self.T0 = T0; self.lam = lam
        self.Tmax = 2.5 * T0
        self.x_eps = 0.001
        self.phi_v = T0 * np.sqrt(2.0 * D / lam)

    def approxZeroTMin(self):
        return [np.array([self.phi_v])]

    def Vtot(self, X, T, include_radiation=False):
        phi = np.asanyarray(X, dtype=float)[..., 0]
        T   = np.asanyarray(T,   dtype=float)
        return (self.D * (T**2 - T0**2) * phi**2
                - self.E * T * phi**3
                + self.lam / 4.0 * phi**4)

from cosmoTransitions.config import TunnelingConfig
m = MyModel()
results = m.findAllTransitions(tunneling_config=TunnelingConfig())
```

> **Note:** use `init()`, not `__init__()`.

---

## TunnelingConfig quick selection guide

| Scenario | Recommended config |
|----------|--------------------|
| Tn/Tc > 0.80 | `TunnelingConfig()` defaults |
| 0.50 < Tn/Tc ≤ 0.80 | `TunnelingConfig(T_scan_extension=True)` |
| 0.10 < Tn/Tc ≤ 0.50 | `TunnelingConfig.supercooling_preset()` |
| Tn/Tc ≤ 0.10 (conformal) | `supercooling_preset()` + `V_spline_samples=None` |
| Debugging | `TunnelingConfig(log_level=logging.DEBUG)` + `cfg.apply_log_level()` |

---

## Output files

| Script | Figure |
|--------|--------|
| example_01 | `example_01_output.png` |
| example_02 | `example_02_output.png` |
| example_03 | `example_03_output.png` |
| example_04 | `example_04_output.png` |
| example_05 | `example_05_output.png` |
| example_06 | `example_06_output.png` |
| example_07 | *(no figure)* |
