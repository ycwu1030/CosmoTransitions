"""
example_07_logging_debug.py
============================
Demonstrates the CosmoTransitions logging system.

Topics covered
--------------
  A. Which modules emit what (logging architecture overview)
  B. Comparing log levels: no logging / INFO / DEBUG
  C. Writing logs to a file (log_file parameter)
  D. Per-module filtering (silence verbose sub-modules)
  E. TunnelingConfig integration (log_level + apply_log_level)

Uses the same minimal single-field model as examples 05 and 06.
Expected run time: ~60 s.
"""

import logging
import os
import sys
import tempfile
import io

import numpy as np

# ------------------------------------------------------------------
# Add parent dir to path so the local cosmoTransitions package is found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmoTransitions import generic_potential
from cosmoTransitions.config import TunnelingConfig, enable_logging

# Silence cosmoTransitions logging at import time — sections will control it.
import logging as _logging
_logging.getLogger('cosmoTransitions').setLevel(_logging.WARNING)
for _h in list(_logging.getLogger('cosmoTransitions').handlers):
    _logging.getLogger('cosmoTransitions').removeHandler(_h)
    _h.close()
del _logging

# ══════════════════════════════════════════════════════════════════════════════
#  Minimal model (same as example_05 / example_06)
# ══════════════════════════════════════════════════════════════════════════════

class FiniteT_SingleField(generic_potential.generic_potential):
    """V(φ,T) = D(T²−T₀²)φ² − E·T·φ³ + (λ/4)φ⁴"""

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
        T   = np.asanyarray(T, dtype=float)
        return (self.D * (T**2 - self.T0**2) * phi**2
                - self.E * T * phi**3
                + self.lam / 4.0 * phi**4)


def _make_model():
    return FiniteT_SingleField(D=0.10, E=0.06, T0=80.0, lam=0.10)


def _reset_ct_logging():
    """Remove all handlers from the cosmoTransitions root logger."""
    pkg = logging.getLogger('cosmoTransitions')
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
        h.close()
    pkg.setLevel(logging.WARNING)   # effectively silent


sep = "=" * 70

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION A — Architecture overview
# ══════════════════════════════════════════════════════════════════════════════

def section_A():
    print(sep)
    print("  SECTION A — CosmoTransitions logging architecture")
    print(sep)
    print()
    print("  Every CosmoTransitions module obtains its own logger via:")
    print("    logger = logging.getLogger(__name__)")
    print()
    print("  All loggers are children of the 'cosmoTransitions' root logger.")
    print("  enable_logging() attaches a handler to that root logger so all")
    print("  child loggers are captured automatically.")
    print()
    print("  Module hierarchy and typical log volume:")
    print()
    modules = [
        ("cosmoTransitions.transitionFinder", "INFO",  "Phase finding, Tc search, nucleation temperature"),
        ("cosmoTransitions.pathDeformation",  "DEBUG", "Path deformation iterations, convergence details"),
        ("cosmoTransitions.tunneling1D",      "DEBUG", "1-D instanton ODE steps, profile details"),
        ("cosmoTransitions.finiteT",          "INFO",  "Finite-T potential integrals"),
        ("cosmoTransitions.config",           "INFO",  "Config loading / saving"),
    ]
    print(f"  {'Module':<42}  {'Level':<6}  Description")
    print(f"  {'─'*42}  {'─'*6}  {'─'*42}")
    for mod, lvl, desc in modules:
        print(f"  {mod:<42}  {lvl:<6}  {desc}")
    print()
    print("  Typical message counts per findAllTransitions() call (rough):")
    print("    INFO  level: ~10–30 messages (phase/transition summary)")
    print("    DEBUG level: ~100–500 messages (full numerical detail)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION B — Log levels compared
# ══════════════════════════════════════════════════════════════════════════════

def _capture_logs(level):
    """Run findAllTransitions capturing logs to a StringIO buffer.
    Returns (Tn_str, log_lines_list)."""
    _reset_ct_logging()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter('[%(levelname)s %(name)s] %(message)s'))
    pkg = logging.getLogger('cosmoTransitions')
    pkg.setLevel(level)
    pkg.addHandler(handler)

    m = _make_model()
    cfg = TunnelingConfig()
    res = m.findAllTransitions(tunneling_config=cfg)
    Tn_vals = [t['Tnuc'] for t in res if t.get('Tnuc') is not None]
    Tn_str = f"{Tn_vals[0]:.4f} GeV" if Tn_vals else "N/A"

    _reset_ct_logging()
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    return Tn_str, lines


def section_B():
    print(sep)
    print("  SECTION B — Comparing log levels")
    print(sep)
    print()
    print("  Running findAllTransitions() three times with different log levels.")
    print("  (E=0.06 → weakly first-order, Tn/Tc ~ 0.96)")
    print()

    # No logging (WARNING only → essentially silent)
    print("  [1/3] No logging (level=WARNING) ...")
    Tn_none, lines_none = _capture_logs(logging.WARNING)
    print(f"        Tn = {Tn_none}")
    print(f"        Log lines captured: {len(lines_none)}  (none shown — silent)")
    print()

    # INFO
    print("  [2/3] INFO level ...")
    Tn_info, lines_info = _capture_logs(logging.INFO)
    print(f"        Tn = {Tn_info}")
    print(f"        Log lines captured: {len(lines_info)}")
    if lines_info:
        show = lines_info[:12]
        print()
        print("        First 12 INFO lines:")
        for ln in show:
            print(f"          {ln}")
    print()

    # DEBUG
    print("  [3/3] DEBUG level ...")
    Tn_debug, lines_debug = _capture_logs(logging.DEBUG)
    print(f"        Tn = {Tn_debug}")
    print(f"        Log lines captured: {len(lines_debug)}")
    if lines_debug:
        # Show first 15 lines
        show = lines_debug[:15]
        print()
        print("        First 15 DEBUG lines:")
        for ln in show:
            print(f"          {ln}")
        if len(lines_debug) > 15:
            print(f"          ... ({len(lines_debug) - 15} more lines omitted)")
    print()

    print("  Summary:")
    print(f"    WARNING:  {len(lines_none):>4} lines  (silent — no output unless something goes wrong)")
    print(f"    INFO:     {len(lines_info):>4} lines  (transition milestones, Tn result)")
    print(f"    DEBUG:    {len(lines_debug):>4} lines  (full numerical trace)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION C — Writing logs to a file
# ══════════════════════════════════════════════════════════════════════════════

def section_C():
    print(sep)
    print("  SECTION C — Writing DEBUG logs to a file")
    print(sep)
    print()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.log',
                                     delete=False, prefix='ct_example07_') as f:
        log_path = f.name

    try:
        _reset_ct_logging()
        enable_logging(level=logging.DEBUG, log_file=log_path)

        m = _make_model()
        cfg = TunnelingConfig()
        res = m.findAllTransitions(tunneling_config=cfg)
        Tn_vals = [t['Tnuc'] for t in res if t.get('Tnuc') is not None]
        Tn = Tn_vals[0] if Tn_vals else None

        _reset_ct_logging()

        with open(log_path, 'r', encoding='utf-8') as fh:
            all_lines = fh.readlines()

        print(f"  Log file: {log_path}")
        print(f"  Total lines written: {len(all_lines)}")
        print(f"  Tn = {Tn:.4f} GeV" if Tn else "  Tn = N/A")
        print()
        show_n = min(20, len(all_lines))
        print(f"  First {show_n} lines of log file:")
        for ln in all_lines[:show_n]:
            print(f"    {ln.rstrip()}")
        if len(all_lines) > show_n:
            print(f"    ... ({len(all_lines) - show_n} more lines in file)")
        print()

        # Count by level
        level_counts = {}
        for ln in all_lines:
            for lvl in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
                if f'[{lvl}' in ln:
                    level_counts[lvl] = level_counts.get(lvl, 0) + 1
                    break
        print("  Line counts by level:")
        for lvl, cnt in sorted(level_counts.items()):
            print(f"    {lvl:<10} {cnt}")
        print()

    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass

    print("  Code pattern:")
    print()
    print("    from cosmoTransitions import enable_logging")
    print("    import logging")
    print("    enable_logging(level=logging.DEBUG, log_file='ct_run.log')")
    print("    # ... run your model ...")
    print("    # Logs appended to ct_run.log (mode='a')")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION D — Per-module filtering
# ══════════════════════════════════════════════════════════════════════════════

def section_D():
    print(sep)
    print("  SECTION D — Per-module log filtering")
    print(sep)
    print()
    print("  Sometimes you want INFO from transitionFinder but not the verbose")
    print("  DEBUG chatter from tunneling1D or pathDeformation.")
    print()
    print("  Strategy: enable_logging(DEBUG) first, then raise the level for")
    print("  noisy sub-modules.")
    print()

    _reset_ct_logging()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter('[%(levelname)s %(name)s] %(message)s'))
    pkg = logging.getLogger('cosmoTransitions')
    pkg.setLevel(logging.DEBUG)
    pkg.addHandler(handler)

    # Silence the two most verbose sub-modules
    logging.getLogger('cosmoTransitions.tunneling1D').setLevel(logging.WARNING)
    logging.getLogger('cosmoTransitions.pathDeformation').setLevel(logging.WARNING)

    m = _make_model()
    cfg = TunnelingConfig()
    res = m.findAllTransitions(tunneling_config=cfg)
    Tn_vals = [t['Tnuc'] for t in res if t.get('Tnuc') is not None]
    Tn = Tn_vals[0] if Tn_vals else None

    _reset_ct_logging()

    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    print(f"  With tunneling1D + pathDeformation silenced:")
    print(f"    Log lines captured: {len(lines)}")
    print(f"    Tn = {Tn:.4f} GeV" if Tn else "    Tn = N/A")
    print()
    if lines:
        show = lines[:15]
        print(f"  First {len(show)} remaining lines (transitionFinder / config only):")
        for ln in show:
            print(f"    {ln}")
    print()

    print("  Code pattern:")
    print()
    print("    import logging")
    print("    from cosmoTransitions import enable_logging")
    print()
    print("    enable_logging(logging.DEBUG)  # root: all DEBUG")
    print()
    print("    # Suppress verbose sub-modules:")
    print("    logging.getLogger('cosmoTransitions.tunneling1D').setLevel(logging.WARNING)")
    print("    logging.getLogger('cosmoTransitions.pathDeformation').setLevel(logging.WARNING)")
    print()
    print("    # Now only transitionFinder (and others) will emit at DEBUG/INFO.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION E — TunnelingConfig integration
# ══════════════════════════════════════════════════════════════════════════════

def section_E():
    print(sep)
    print("  SECTION E — TunnelingConfig.log_level + apply_log_level()")
    print(sep)
    print()
    print("  TunnelingConfig carries logging settings alongside numerical params.")
    print("  Call cfg.apply_log_level() once to activate them, or let")
    print("  findAllTransitions activate them automatically when cfg.log_level")
    print("  is not None.")
    print()

    print("  TunnelingConfig logging fields:")
    print()
    print(f"  {'Field':<20}  {'Default':<18}  Description")
    print(f"  {'─'*20}  {'─'*18}  {'─'*40}")
    print(f"  {'log_level':<20}  {'None':<18}  logging level int (None = no change)")
    print(f"  {'log_file':<20}  {'None':<18}  path to write logs (None = stderr)")
    print()

    print("  Demo: cfg with log_level=logging.INFO (captured to buffer)")
    print()

    _reset_ct_logging()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter('[%(levelname)s %(name)s] %(message)s'))

    # We manually attach handler first, then use cfg.apply_log_level() which
    # sets the pkg logger level but won't add a duplicate stream handler.
    pkg = logging.getLogger('cosmoTransitions')
    pkg.addHandler(handler)

    cfg = TunnelingConfig(log_level=logging.INFO)
    cfg.apply_log_level()   # sets level on the pkg logger

    m = _make_model()
    res = m.findAllTransitions(tunneling_config=cfg)
    Tn_vals = [t['Tnuc'] for t in res if t.get('Tnuc') is not None]
    Tn = Tn_vals[0] if Tn_vals else None

    _reset_ct_logging()

    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    print(f"    cfg.log_level = logging.INFO = {logging.INFO}")
    print(f"    Tn = {Tn:.4f} GeV" if Tn else "    Tn = N/A")
    print(f"    Log lines captured: {len(lines)}")
    print()
    if lines:
        print("    Log output:")
        for ln in lines:
            print(f"      {ln}")
    print()

    print("  Code pattern:")
    print()
    print("    import logging")
    print("    from cosmoTransitions.config import TunnelingConfig")
    print("    from cosmoTransitions import transitionFinder")
    print()
    print("    cfg = TunnelingConfig(")
    print("        log_level=logging.INFO,")
    print("        log_file='my_run.log',   # omit to use stderr")
    print("        T_scan_extension=True,")
    print("    )")
    print("    cfg.apply_log_level()        # activates logging immediately")
    print()
    print("    m = MyModel()")
    print("    results = m.findAllTransitions(tunneling_config=cfg)")
    print()
    print("  Tip: store cfg in a TOML file for reproducible runs:")
    print("    cfg.write_default('params.toml')")
    print("    cfg2 = TunnelingConfig.from_file('params.toml')")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _reset_ct_logging()  # ensure clean slate at start
    print()
    print(sep)
    print("  example_07 — CosmoTransitions logging and debugging guide")
    print(sep)
    print()
    print("  Model:  V(φ,T) = D(T²−T₀²)φ² − E·T·φ³ + (λ/4)φ⁴")
    print("  Fixed:  D=0.10, E=0.06, T0=80 GeV, λ=0.10  (weak 1st-order)")
    print()

    section_A()
    section_B()
    section_C()
    section_D()
    section_E()

    print(sep)
    print("  example_07 complete.")
    print(sep)
    print()


if __name__ == '__main__':
    main()
