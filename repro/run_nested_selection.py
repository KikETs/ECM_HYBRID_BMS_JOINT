"""Nested grouped selection: choose the configuration without seeing the test cell.

    python3 repro/run_nested_selection.py --train      # leave-two-out fits
    python3 repro/run_nested_selection.py --select     # inner pick + outer score

Why this exists
    The brief for this audit forbids choosing model, feature, aggregation or
    tolerance after seeing the test set.  Three decisions in this project
    break that, and lambda was only the first of them:

      rung        sec 29.7 adopts A8 over A3 by comparing usable current ON
                  THE LEAVE-ONE-CELL-OUT EVALUATION - the same rows the paper
                  then reports.
      aggregation --trim-agg max was picked from {last, median, q75, q90,
                  max}; sec 31.1 describes pinning the settings down by
                  finding the version that reproduces sec 16's lambda, which
                  is reverse-engineering from results.
      tolerance   the 0.5 A charge tolerance is sec 25's "knee", read off the
                  same evaluation.

    So the LOCO evaluation is simultaneously the selection set and the
    reported test set.  That inflates whatever is reported by however much
    the selection was worth.

What this does instead
    outer   leave cell i out.  Cell i is touched only to report.
    inner   leave-one-out again over the five training cells, using models
            trained on the remaining FOUR (leave-two-out).  The whole grid
            of {rung x aggregation x tolerance} is scored there and the
            winner is chosen there.
    report  the winning configuration is applied to cell i using the
            ordinary five-cell model - the one that would ship - and scored.

    Selection therefore never sees cell i, and the outer numbers are what
    the configuration is worth on data that played no part in choosing it.

Cost
    6 outer x 5 inner x 2 rungs x 2 directions = 120 model fits, about three
    hours.  Aggregation and tolerance need no retraining; they are applied
    to the stored per-row predictions.
"""
import argparse
import collections
import csv
import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

OUTDIR = os.path.join(ANALYSIS, 'runs_nested')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')

RUNGS = {'A8': list(range(1, 12)), 'A3': None}     # ablate mask per rung
AGGS = ['last', 'median', 'q75', 'q90', 'max']
TOLS = {'discharge': [0.0, 0.25, 0.5], 'charge': [0.0, 0.25, 0.5, 1.0]}
DATA = {'discharge': 'cache/trim', 'charge': 'cache/trim_chg'}
SHIPPED = {('discharge', 'A8'): 'runs_trim_a8',
           ('discharge', 'A3'): 'runs_trim_v2',
           ('charge', 'A8'): 'runs_trim_a8_chg',
           ('charge', 'A3'): 'runs_trim_chg_v2'}
TAUS = (10.0, 2.0)
EXTRAP_MAX = 1.5


def train_all(epochs, lr, seeds):
    """Leave-two-out fits: for each (outer, inner) pair, train on the rest."""
    import torch
    import sop_trim
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for direction, data in DATA.items():
        cells = sop_trim.load_cells(os.path.join(ANALYSIS, data))
        names = sorted(cells)
        for outer in names:
            for inner in [c for c in names if c != outer]:
                keep = [c for c in names if c not in (outer, inner)]
                for rung, ablate in RUNGS.items():
                    d = os.path.join(OUTDIR, direction, f'outer_{outer}',
                                     rung)
                    os.makedirs(d, exist_ok=True)
                    out = os.path.join(d, f'pred_{rung}_{inner}.npz')
                    if os.path.exists(out):
                        continue
                    tr = {k: np.concatenate([cells[c][k] for c in keep])
                          for k in ('X', 'Y', 'NOM', 'I')}
                    te = cells[inner]
                    ps, kfs, kss = [], [], []
                    for s in range(seeds):
                        p, kf, ks, _, _ = sop_trim.train_fold(
                            sop_trim.TrimLinear, tr, te, epochs, lr, s, dev,
                            ablate=ablate)
                        ps.append(p); kfs.append(kf); kss.append(ks)
                    np.savez(out, k_f=np.mean(kfs, 0), k_s=np.mean(kss, 0),
                             cycle=te['cycle'], SOC=te['SOC'], SOH=te['SOH'],
                             rank=te['rank'], exc=te['exc'], I=te['I'],
                             NOM=te['NOM'], Y=te['Y'])
                    print(f'  {direction:<10}outer {outer:<20}inner '
                          f'{inner:<20}{rung}', flush=True)


def eval_dirs(direction, outer, rung, agg):
    """Score one (rung, agg) on the five inner cells of this outer fold."""
    import subprocess
    d = os.path.join(OUTDIR, direction, f'outer_{outer}', rung)
    out = os.path.join(d, f'eval_{agg}.csv')
    if not os.path.exists(out):
        cmd = [sys.executable, 'eval_sop_amps.py', '--direction', direction,
               '--trim', os.path.relpath(d, ANALYSIS), '--trim-agg', agg,
               '--out', out]
        p = subprocess.run(cmd, cwd=ANALYSIS, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(out):
            return None
    return out


def score(path, tau, tol, min_train=25):
    """Strict per-cell held-out lambda on whatever cells this file holds."""
    from run_safety import load, keep
    from run_safety_strict import fit_lambda
    d = load(path)
    m = keep(d, tau)
    cells = sorted(set(d['cell'][m]))
    us, ns, exc = [], [], 0
    for c in cells:
        tr, te = m & (d['cell'] != c), m & (d['cell'] == c)
        if tr.sum() < min_train or te.sum() == 0:
            continue
        lam = fit_lambda(d['hyb'][tr], d['meas'][tr], tol)
        p, y = d['hyb'][te], d['meas'][te]
        exc += int((lam * p - y > tol).sum())
        us.append(float(np.median(lam * p / y) * 100))
        ns.append(int(te.sum()))
    if not us:
        return None
    return dict(usable=float(np.average(us, weights=ns)),
                worst=float(min(us)), exceed=exc, n=int(sum(ns)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    if not (a.train or a.select):
        a.train = a.select = True

    if a.train:
        print('  leave-two-out fits (6 outer x 5 inner x 2 rungs x 2 dirs)')
        train_all(a.epochs, a.lr, a.seeds)

    if not a.select:
        return 0

    import sop_trim
    rows, picks = [], []
    for direction in ('discharge', 'charge'):
        names = sorted(sop_trim.load_cells(os.path.join(ANALYSIS,
                                                        DATA[direction])))
        for tau in TAUS:
            for outer in names:
                # ---- inner: score the whole grid on the five other cells --
                best = None
                for rung, agg in itertools.product(RUNGS, AGGS):
                    ev = eval_dirs(direction, outer, rung, agg)
                    if ev is None:
                        continue
                    for tol in TOLS[direction]:
                        r = score(ev, tau, tol)
                        if r is None or r['exceed'] > 0:
                            continue
                        key = (r['usable'], r['worst'])
                        if best is None or key > best[0]:
                            best = (key, rung, agg, tol, r)
                if best is None:
                    continue
                _, rung, agg, tol, inner_r = best
                # ---- outer: apply the winner to the untouched cell --------
                from run_safety import load, keep
                from run_safety_strict import fit_lambda
                import subprocess
                shipped = SHIPPED[(direction, rung)]
                ev_out = os.path.join(OUTDIR, direction,
                                      f'shipped_{rung}_{agg}.csv')
                if not os.path.exists(ev_out):
                    subprocess.run(
                        [sys.executable, 'eval_sop_amps.py', '--direction',
                         direction, '--trim', shipped, '--trim-agg', agg,
                         '--out', ev_out], cwd=ANALYSIS,
                        capture_output=True, text=True)
                if not os.path.exists(ev_out):
                    continue
                d = load(ev_out)
                m = keep(d, tau)
                tr, te = m & (d['cell'] != outer), m & (d['cell'] == outer)
                if tr.sum() < 25 or te.sum() == 0:
                    continue
                lam = fit_lambda(d['hyb'][tr], d['meas'][tr], tol)
                p, y = d['hyb'][te], d['meas'][te]
                ex = int((lam * p - y > tol).sum())
                us = float(np.median(lam * p / y) * 100)
                rows.append([direction, f'{tau:.1f}', outer, rung, agg,
                             f'{tol:.2f}', f'{lam:.4f}', int(te.sum()), ex,
                             f'{us:.2f}', f'{inner_r["usable"]:.2f}'])
                picks.append((direction, tau, rung, agg, tol))
                print(f'  {direction:<10}tau {tau:>4.0f}  outer {outer:<20}'
                      f'picked {rung}/{agg}/tol {tol:.2f}   '
                      f'inner {inner_r["usable"]:.2f} %  ->  outer {us:.2f} %'
                      f'  exc {ex}', flush=True)

    os.makedirs(TABLES, exist_ok=True)
    out = os.path.join(TABLES, 'nested_selection.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'tau_s', 'outer_cell', 'picked_rung',
                    'picked_agg', 'picked_tol_A', 'lambda', 'n_rows',
                    'exceed', 'outer_usable_pct', 'inner_usable_pct'])
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(out, ROOT)}  ({len(rows)} rows)')

    c = collections.Counter((r[3], r[4], r[5]) for r in rows)
    print('\n  configurations the inner splits chose:')
    for (rung, agg, tol), n in c.most_common():
        print(f'    {rung}/{agg}/tol {tol}   chosen in {n} of {len(rows)} '
              f'outer folds')
    for direction in ('discharge', 'charge'):
        for tau in TAUS:
            sel = [r for r in rows
                   if r[0] == direction and r[1] == f'{tau:.1f}']
            if not sel:
                continue
            u = np.array([float(r[9]) for r in sel])
            n = np.array([int(r[7]) for r in sel], float)
            print(f'  {direction:<10}tau {tau:>4.0f}  nested outer usable '
                  f'{np.average(u, weights=n):.2f} %  worst cell '
                  f'{u.min():.2f} %  exceed '
                  f'{sum(int(r[8]) for r in sel)}/{int(n.sum())}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
