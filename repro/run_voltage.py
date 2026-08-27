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


def per_cell_rmse(run_dir, rung, horizon=None):
    """Per-cell voltage RMSE (mV).  rung='A0' is the uncorrected version.

    horizon=None pools both columns of Y, which is what the shipped table
    did.  horizon=0 is tau = 2 s and horizon=1 is tau = 10 s.

    Pooling matters: the figure that compares "ranked by voltage" against
    "ranked by usable current" drew the *same* pooled voltage number in the
    tau = 2 s and tau = 10 s panels, so a horizon-specific current metric
    was being set against a horizon-agnostic voltage metric.  Any ranking
    difference between the two panels on the left axis was structurally
    impossible.
    """
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
        Y = z['Y']
        if horizon is None:
            e = p - Y
        else:
            e = p[:, horizon] - Y[:, horizon]
        out[cell] = float(np.sqrt(np.mean(e ** 2)) * 1000)
    return out


def main():
    rows = []
    failures = []
    for name, dis, chg in RUNS:
        for direction, run in (('discharge', dis), ('charge', chg)):
            if not os.path.isdir(os.path.join(ANALYSIS, run)):
                failures.append(f'{name}/{direction}: no {run}/')
                continue
            for tau, hz in (('pooled', None), ('2.0', 0), ('10.0', 1)):
                try:
                    per = per_cell_rmse(run, 'A0' if name == 'A0' else name,
                                        hz)
                except Exception as e:                   # noqa: BLE001
                    failures.append(f'{name}/{direction}/tau={tau}: {e}')
                    continue
                if not per:
                    failures.append(f'{name}/{direction}/tau={tau}: no cells')
                    continue
                mean = float(np.mean(list(per.values())))
                rows.append([direction, tau, name, len(per), f'{mean:.2f}']
                            + [f'{per[c]:.2f}' for c in sorted(per)])
                if tau == 'pooled':
                    print(f'  {direction:<10}{name:<8}{mean:>9.2f} mV   '
                          f'({len(per)} cells)', flush=True)

    cells = sorted(per_cell_rmse('runs_trim_a8', 'A8'))
    os.makedirs(TABLES, exist_ok=True)
    out = os.path.join(TABLES, 'voltage.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'tau_s', 'method', 'n_cells', 'rmse_mV']
                   + cells)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(out, ROOT)}  ({len(rows)} rows)',
          flush=True)
    print('  convention: mean of per-cell RMSE (not a pooled RMSE — same as '
          'sop_trim.py)', flush=True)
    print("  tau_s='pooled' reproduces the previous horizon-agnostic number; "
          "2.0 and 10.0 are the horizon-specific ones.", flush=True)
    if failures:
        print(f'\n  {len(failures)} incomplete:', flush=True)
        for f in failures:
            print(f'    {f}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
