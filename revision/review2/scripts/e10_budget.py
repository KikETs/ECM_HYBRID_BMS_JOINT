"""E10 (R1-6): derived pack budget from the measured per-cell frame times (review1 followup wcet/timing_summary.csv).
Per-cell worst = forced path with 4 inversions x 24 iterations (the iteration cap), observed maximum; not a static WCET.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import write_csv, out_dir
T = list(csv.DictReader(open(os.path.join(REV1, 'followup', 'wcet', 'timing_summary.csv'))))
per = {}
for r in T:
    n = int(r['n_cells'])
    per.setdefault((r['cache'], r['path']), []).append(float(r['max_us']) / n)
rows = []
for (cache, path), v in per.items():
    t = max(v) / 1000.0     # ms per cell, worst over pack sizes
    for N in (1, 12, 48, 96, 192):
        for period_ms in (10, 100, 1000):
            rows.append(dict(cache=cache, path=path, per_cell_worst_ms=round(t, 4), n_cells=N, refresh_ms=period_ms,
                             cpu_load_pct=round(100 * N * t / period_ms, 2),
                             staggered_10ms_tick_ms=round(math.ceil(N / (period_ms / 10)) * t, 3) if period_ms >= 10 else None))
write_csv(os.path.join(out_dir('E10'), 'E10_budget.csv'), rows)
for r in rows:
    if r['path'] == 'full_24_stress' and r['refresh_ms'] in (100, 1000):
        print(r)
