"""Give the SOC arm the same treatment the other two already get.

The SOP arm reports a per-cell held-out lambda, a worst cell and a
cell-cluster bootstrap interval.  The SOH arm reports a per-cell RMSE and a
worst cell.  The SOC arm reported one pooled mean over six disturbances and
nothing else -- no spread, no worst cell, no interval -- so a reader could not
tell whether 2.14 %p is what every cell does or the average of one good cell
and one bad one.  The per-run errors were computed and then averaged away.

Nothing is re-simulated here.  soc_perturb_bench.py already writes every
run's RMSE to results/soc_perturb.npz; this reads that array and the cell
label of each run from soc_runs.pkl, so the per-cell numbers are the pooled
number taken apart, not a second experiment that might disagree with it.
The index mapping is asserted against soc_headline.csv on every run.

Two things the pooled number hides, and they belong next to it:

  * Every filter reads its OWN cell's characterisation surface.  This is a
    per-cell calibrated deployment, not a leave-one-cell-out transfer like
    the SOP and SOH arms, so the spread below is over operating conditions
    within a cell -- it is not evidence that the filter transfers to a cell
    it has never seen.
  * The six cells each carry a different aging protocol (DATA.md), so a
    per-cell spread here is a cell-and-protocol spread.  Only CC and
    CC_CELL2 share a protocol, which makes that pair the one honest read on
    cell-to-cell variation alone.
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

RESULTS = os.path.join(ANALYSIS, 'results')
OUT = os.path.join(RESULTS, 'tables', 'soc_percell.csv')
ADOPTED = 'EKF adopted (gate)'
N_BOOT, BOOT_SEED = 2000, 12345

HEADER = ['cell', 'n_runs', 'undisturbed_pct', 'mean_6_pct', 'worst_of_6_pct',
          'worst_perturbation']


def load():
    from soc_perturb_bench import CONFIGS, PERTURB
    z = np.load(os.path.join(RESULTS, 'soc_perturb.npz'))
    runs = pickle.load(open(os.path.join(RESULTS, 'soc_runs.pkl'), 'rb'))
    ci = [n for n, _ in CONFIGS].index(ADOPTED)
    nper = len(PERTURB)
    full = z['full']
    if full.shape[0] != len(CONFIGS) * nper:
        raise SystemExit(
            f'soc_perturb.npz holds {full.shape[0]} rows, expected '
            f'{len(CONFIGS) * nper} = {len(CONFIGS)} configs x {nper} '
            f'perturbations.  Re-run the soc stage.')
    if full.shape[1] != len(runs):
        raise SystemExit(
            f'soc_perturb.npz has {full.shape[1]} runs per row but '
            f'soc_runs.pkl holds {len(runs)}.  They are out of step; re-run '
            f'the soc_runs and soc stages together.')
    # The row order is config-major (see soc_perturb_bench.main).  Getting it
    # wrong would silently report a different filter's numbers as the adopted
    # one, so it is checked against the published table rather than trusted.
    block = full[ci * nper:(ci + 1) * nper]
    published = {r['config']: r for r in csv.DictReader(
        open(os.path.join(RESULTS, 'tables', 'soc_headline.csv'),
             encoding='utf-8'))}[ADOPTED]
    for label, mine, col in (
            ('undisturbed', block[0].mean() * 100, 'undisturbed_pct'),
            ('mean of 6', np.mean([block[p].mean() for p in range(1, nper)])
             * 100, 'mean_6_disturbances_pct'),
            ('worst of 6', max(block[p].mean() for p in range(1, nper)) * 100,
             'worst_of_6_pct')):
        theirs = float(published[col])
        if abs(mine - theirs) > 5e-3:
            raise SystemExit(
                f'index mapping is wrong: this script reads {label} = '
                f'{mine:.3f} %p from soc_perturb.npz, soc_headline.csv '
                f'publishes {theirs:.3f}.')
    cells = [r['cell'] for r in runs]
    return block, cells, [n for n, _ in PERTURB]


def rows(block, cells, pert_names):
    out, order = [], []
    for c in sorted(set(cells)):
        idx = [i for i, x in enumerate(cells) if x == c]
        und = block[0][idx].mean() * 100
        per = [block[p][idx].mean() * 100 for p in range(1, block.shape[0])]
        worst_p = int(np.argmax(per))
        out.append([c, len(idx), f'{und:.3f}', f'{np.mean(per):.3f}',
                    f'{per[worst_p]:.3f}', pert_names[1 + worst_p]])
        order.append(np.mean(per))
    return out, np.array(order), sorted(set(cells))


def bootstrap(block, cells):
    """Cell-cluster bootstrap on the mean-of-six, same recipe as the SOP arm.

    Six clusters is few, so the interval is wide -- which is the honest
    output, not a defect to tune away.
    """
    names = sorted(set(cells))
    per_cell = []
    for c in names:
        idx = [i for i, x in enumerate(cells) if x == c]
        per_cell.append(np.mean([block[p][idx].mean()
                                 for p in range(1, block.shape[0])]) * 100)
    per_cell = np.array(per_cell)
    rng = np.random.default_rng(BOOT_SEED)
    draws = per_cell[rng.integers(0, len(per_cell), size=(N_BOOT,
                                                          len(per_cell)))]
    m = draws.mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='fail if the table on disk differs')
    a = ap.parse_args()

    block, cells, pert_names = load()
    body, means, names = rows(block, cells, pert_names)
    lo, hi = bootstrap(block, cells)

    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(OUT, HEADER, body, 'soc_percell')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(body)
    print(f'  -> {os.path.relpath(OUT, ROOT)}  ({len(body)} rows)\n')
    print(f"  {'cell':<20}{'undist':>9}{'mean6':>9}{'worst6':>9}"
          f"  worst perturbation")
    print('  ' + '-' * 76)
    for r in body:
        print(f'  {r[0]:<20}{r[2]:>9}{r[3]:>9}{r[4]:>9}  {r[5]}')
    w_i = int(np.argmax(means))
    print(f'\n  pooled mean of 6      {means.mean():.3f} %p')
    print(f'  worst cell            {names[w_i]} at {means[w_i]:.3f} %p, '
          f'{means[w_i] / means.min():.2f}x the best ({names[int(np.argmin(means))]})')
    print(f'  cell-cluster 95 % CI  [{lo:.3f}, {hi:.3f}] %p over 6 clusters')
    print('\n  Read it as a per-cell calibrated deployment: every filter uses '
          'its own\n  cell\'s surface, so this is a spread over conditions '
          'within a cell, not\n  evidence of transfer to an unseen cell.  '
          'The six cells also differ by\n  aging protocol, so cell and '
          'protocol are confounded except for the\n  CC / CC_CELL2 pair.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
