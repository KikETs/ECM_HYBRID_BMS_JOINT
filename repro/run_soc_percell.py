"""Give the SOC arm the same treatment the other two already get.

The SOP arm reports a per-cell held-out lambda, a worst cell and a
cell-cluster bootstrap interval.  The SOH arm reports a per-cell RMSE and a
worst cell.  The SOC arm reported one pooled mean over six disturbances and
nothing else -- no spread, no worst cell, no interval -- so a reader could not
tell whether 2.14 %p is what every cell does or an average over one good cell
and one bad one.  The per-run errors were being computed and then averaged
away.

Also worth stating plainly, because the pooled number hides it: every filter
here reads its OWN cell's characterisation surface.  This is a per-cell
calibrated deployment, not a leave-one-cell-out transfer, and the spread below
is a spread over operating conditions rather than over unseen cells.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

OUT = os.path.join(ANALYSIS, 'results', 'tables', 'soc_percell.csv')
N_BOOT, BOOT_SEED = 2000, 12345

PERTURBATIONS = [
    ('initial SOC +10 %p', dict(dsoc=+0.10)),
    ('initial SOC -10 %p', dict(dsoc=-0.10)),
    ('current offset +0.10 A', dict(ibias=+0.10)),
    ('current offset -0.10 A', dict(ibias=-0.10)),
    ('current gain +1 %', dict(igain=+0.01)),
    ('current gain -1 %', dict(igain=-0.01)),
]


def bootstrap_cell_mean(per_cell, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cell-cluster bootstrap on the mean, same method as the SOP arm."""
    cells = sorted(per_cell)
    if len(cells) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    vals = np.array([per_cell[c] for c in cells])
    stat = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                     for _ in range(n_boot)])
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    os.chdir(ANALYSIS)
    import soc_perturb_bench as B
    if not os.path.exists('results/soc_runs.pkl'):
        print('  missing results/soc_runs.pkl — run build_soc_runs.py',
              file=sys.stderr)
        return 1
    import pickle
    B.RUNS = pickle.load(open('results/soc_runs.pkl', 'rb'))
    cells = [r['cell'] for r in B.RUNS]
    order = sorted(set(cells))

    rows = []
    print(f"  {'perturbation':<24}{'mean':>8}{'worst cell':>22}"
          f"{'worst':>8}{'spread':>9}")
    print('  ' + '-' * 74)
    per_pert = {}
    for name, kw in PERTURBATIONS:
        _ci, _pi, full, _tail = B.run_case(**kw)
        by = collections.defaultdict(list)
        for c, v in zip(cells, full):
            by[c].append(v * 100)
        per_cell = {c: float(np.mean(v)) for c, v in by.items()}
        per_pert[name] = per_cell
        worst = max(per_cell, key=per_cell.get)
        lo, hi = bootstrap_cell_mean(per_cell)
        rows.append([name, f'{np.mean(list(per_cell.values())):.3f}',
                     worst, f'{per_cell[worst]:.3f}',
                     f'{min(per_cell.values()):.3f}',
                     f'{lo:.3f}', f'{hi:.3f}']
                    + [f'{per_cell[c]:.3f}' for c in order])
        print(f'  {name:<24}{np.mean(list(per_cell.values())):>8.3f}'
              f'{worst:>22}{per_cell[worst]:>8.3f}'
              f'{per_cell[worst] - min(per_cell.values()):>9.3f}')

    # The headline: mean over the six disturbances, per cell.
    agg = {c: float(np.mean([per_pert[n][c] for n, _ in PERTURBATIONS]))
           for c in order}
    worst = max(agg, key=agg.get)
    lo, hi = bootstrap_cell_mean(agg)
    rows.append(['mean of the six', f'{np.mean(list(agg.values())):.3f}',
                 worst, f'{agg[worst]:.3f}', f'{min(agg.values()):.3f}',
                 f'{lo:.3f}', f'{hi:.3f}']
                + [f'{agg[c]:.3f}' for c in order])
    print('  ' + '-' * 74)
    print(f"  {'mean of the six':<24}{np.mean(list(agg.values())):>8.3f}"
          f"{worst:>22}{agg[worst]:>8.3f}"
          f"{agg[worst] - min(agg.values()):>9.3f}")
    print(f'  cell-cluster bootstrap on that mean: [{lo:.3f}, {hi:.3f}] %p')
    print(f'  worst cell is {agg[worst] / min(agg.values()):.2f}x the best.  '
          f'Every filter reads its own cell\'s surface, so this is spread '
          f'over conditions, not over unseen cells.')

    HEADER = (['perturbation', 'mean_pct', 'worst_cell', 'worst_pct',
               'best_pct', 'boot_lo_pct', 'boot_hi_pct'] + list(order))
    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(a.out, HEADER, rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
