"""Build the voltage RMSE table — values that until now were hard-coded into
the figures and the documents.

Why it is separate: `fig_ladder.py` held six voltage RMSEs x two directions as
constants.  The figure would not change when the table did, and there was no
way to check that the figure's numbers were right.  Pulling them into a table
lets verify.py see them.

The convention matches the trim tables — **per-cell RMSE first, then the
mean** (not a pooled RMSE).  Get that wrong and A0 comes out as 87.24 instead
of 85.36.

    python3 repro/run_voltage.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')
sys.path.insert(0, ANALYSIS)

# (name used in the table, discharge run dir, charge run dir)
# A0 lives in the directory's pred_A0_*.npz (the k=1 version).
RUNS = [
    ('A0', 'runs_trim_a8', 'runs_trim_a8_chg'),
    ('direct', 'runs_trim_direct', 'runs_trim_direct_chg'),
    ('shrink', 'runs_trim_shrink', 'runs_trim_shrink_chg'),
    ('A8', 'runs_trim_a8', 'runs_trim_a8_chg'),
    ('A3', 'runs_trim_v2', 'runs_trim_chg_v2'),
    ('RLS', 'runs_trim_rls', 'runs_trim_rls_chg'),
]


def per_cell_rmse(run_dir, rung):
    """Per-cell voltage RMSE (mV).  rung='A0' is the uncorrected version."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ANALYSIS, run_dir,
                                            'pred_A*_*.npz'))):
        base = os.path.basename(f)
        if base.startswith('pred_A0_'):
            continue
        cell = base.split('_', 2)[2][:-4]
        z = np.load(f, allow_pickle=True)
        if rung == 'A0':
            # The uncorrected version was already saved at training time
            # (`base`).  Recomputing it here could drift slightly from
            # sop_trim.py, so use the stored one.
            p = z['base']
        elif 'pred' in z:
            p = z['pred']
        else:
            I, NOM = z['I'].astype(float), z['NOM'].astype(float)
            kf, ks = z['k_f'].astype(float), z['k_s'].astype(float)
            p = np.stack([I * (kf * NOM[:, 0] + ks * NOM[:, 1]),
                          I * (kf * NOM[:, 2] + ks * NOM[:, 3])], 1)
        out[cell] = float(np.sqrt(np.mean((p - z['Y']) ** 2)) * 1000)
    return out


def main():
    rows = []
    for name, dis, chg in RUNS:
        for direction, run in (('discharge', dis), ('charge', chg)):
            if not os.path.isdir(os.path.join(ANALYSIS, run)):
                continue
            try:
                per = per_cell_rmse(run, 'A0' if name == 'A0' else name)
            except Exception as e:                       # noqa: BLE001
                print(f'  skipped {name}/{direction}: {e}', flush=True)
                continue
            if not per:
                continue
            mean = float(np.mean(list(per.values())))
            rows.append([direction, name, len(per), f'{mean:.2f}']
                        + [f'{per[c]:.2f}' for c in sorted(per)])
            print(f'  {direction:<10}{name:<8}{mean:>9.2f} mV   '
                  f'({len(per)} cells)', flush=True)

    cells = sorted(per_cell_rmse('runs_trim_a8', 'A8'))
    os.makedirs(TABLES, exist_ok=True)
    out = os.path.join(TABLES, 'voltage.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'method', 'n_cells', 'rmse_mV'] + cells)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(out, ROOT)}  ({len(rows)} rows)',
          flush=True)
    print('  convention: mean of per-cell RMSE (not a pooled RMSE — same as '
          'sop_trim.py)', flush=True)


if __name__ == '__main__':
    main()
