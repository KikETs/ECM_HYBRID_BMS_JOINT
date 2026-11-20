"""E3 (R2-2): configuration selection without the outer cell, on the current strict-nested assets.

grid: rung {A3, A8} x aggregation {last, median, q75, q90, max}; tolerance declared (discharge 0 A, charge 0.5 A;
charge 0 A run separately as exploratory).
outer c: inner score of each configuration = leave-one-cell-out over the five inner cells j, using the stored inner
assets built without c (trims trained without {c, j}, pools without {c, j}, inner SOH without c); lambda_j from the
other inner cells' rows. Winner = max (-inner exceedance rate, inner usable). The winner is then applied to c with
the submitted nested protocol (lambda_c on the inner rows, outer rows from the shipped five-cell assets).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, pickle, itertools, multiprocessing as mp
from collections import Counter
import numpy as np

os.environ.update(RCSOP_REPO=REPO_ROOT, RCSOP_BUNDLE=os.path.join(REPO_ROOT, 'nested'),
                  SOLVER='hull', OUTDIR=os.path.join(R2, 'out', 'E3'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'nested', 'scripts', 'nested'))
import score_methods_nested as SM          # noqa: E402
sys.path.insert(0, os.path.join(R2, 'scripts'))
from r2common import fit_lambda, write_csv, cell_boot   # noqa: E402

OUT = os.path.join(R2, 'out', 'E3'); os.makedirs(OUT, exist_ok=True)
CELLS = SM.CELLS
RUNGS = {'A8': 'a8', 'A3': 'a3'}
AGGS = ['last', 'median', 'q75', 'q90', 'max']


def task(rung, agg, level, c, j, d):
    SM.AGG = agg
    _, rows = SM.rows_for(RUNGS[rung], level, c, j, d)
    return (rung, agg, level, c, j, d), rows


def keep(rows, tau):
    return [r for r in rows if r['tau'] == tau and r['meas'] > .5 and r['extrap'] <= 1.5 and np.isfinite(r['hyb'])]


def score_loco(rowsets, tol, min_train=25):
    """rowsets: {cell: rows}. per-cell held-out lambda from the other cells."""
    us, ns, exc = [], [], 0
    for c, te in rowsets.items():
        tr = [r for k, v in rowsets.items() if k != c for r in v]
        if len(tr) < min_train or not te:
            continue
        lam = fit_lambda(np.array([r['hyb'] for r in tr]), np.array([r['meas'] for r in tr]), tol)
        p = lam * np.array([r['hyb'] for r in te]); y = np.array([r['meas'] for r in te])
        exc += int((p - y > tol).sum()); us.append(float(np.median(p / y) * 100)); ns.append(len(te))
    if not us:
        return None
    return dict(usable=float(np.average(us, weights=ns)), exceed=exc, n=int(sum(ns)))


if __name__ == '__main__':
    cache = os.path.join(OUT, 'grid_rows.pkl')
    if os.path.exists(cache):
        res = pickle.load(open(cache, 'rb'))
    else:
        tasks = [(rg, ag, lvl, c, j, d) for rg in RUNGS for ag in AGGS for d in ('discharge', 'charge') for c in CELLS
                 for (lvl, j) in [('outer', c)] + [('inner', j) for j in CELLS if j != c]]
        with mp.Pool(int(os.environ.get('NPROC', '16'))) as pool:
            res = dict(pool.starmap(task, tasks))
        pickle.dump(res, open(cache, 'wb'))
    picks, summ = [], []
    for d, tols in (('discharge', (0.0,)), ('charge', (0.5, 0.0))):
        for tol in tols:
            for tau in (10.0, 2.0):
                fold = []
                for c in CELLS:
                    best = None; grid = []
                    for rg, ag in itertools.product(RUNGS, AGGS):
                        rs = {j: keep(res[(rg, ag, 'inner', c, j, d)], tau) for j in CELLS if j != c}
                        s = score_loco(rs, tol)
                        if s is None:
                            continue
                        grid.append((rg, ag, s))
                        key = (-s['exceed'] / max(s['n'], 1), s['usable'])
                        if best is None or key > best[0]:
                            best = (key, rg, ag, s)
                    if best is None:
                        continue
                    _, rg, ag, s_in = best
                    tr = [r for j in CELLS if j != c for r in keep(res[(rg, ag, 'inner', c, j, d)], tau)]
                    te = keep(res[(rg, ag, 'outer', c, c, d)], tau)
                    if len(tr) < 25 or not te:
                        continue
                    lam = fit_lambda(np.array([r['hyb'] for r in tr]), np.array([r['meas'] for r in tr]), tol)
                    p = lam * np.array([r['hyb'] for r in te]); y = np.array([r['meas'] for r in te])
                    rec = dict(direction=d, tau_s=tau, tol=tol, outer=c, picked_rung=rg, picked_agg=ag, lam=lam, n=len(te),
                               exceed=int((p - y > tol).sum()), outer_usable=float(np.median(p / y) * 100), inner_usable=s_in['usable'],
                               inner_exceed=s_in['exceed'],
                               inner_rank_of_A8_max=1 + sorted([(-g[2]['exceed'] / max(g[2]['n'], 1), g[2]['usable']) for g in grid], reverse=True)
                               .index(next((-g[2]['exceed'] / max(g[2]['n'], 1), g[2]['usable']) for g in grid if g[0] == 'A8' and g[1] == 'max')))
                    # submitted configuration on the same fold, for the paired comparison
                    tr0 = [r for j in CELLS if j != c for r in keep(res[('A8', 'max', 'inner', c, j, d)], tau)]
                    te0 = keep(res[('A8', 'max', 'outer', c, c, d)], tau)
                    lam0 = fit_lambda(np.array([r['hyb'] for r in tr0]), np.array([r['meas'] for r in tr0]), tol)
                    p0 = lam0 * np.array([r['hyb'] for r in te0]); y0 = np.array([r['meas'] for r in te0])
                    rec.update(submitted_usable=float(np.median(p0 / y0) * 100), submitted_exceed=int((p0 - y0 > tol).sum()), submitted_n=len(te0))
                    picks.append(rec); fold.append(rec)
                if not fold:
                    continue
                n = np.array([r['n'] for r in fold]); u = np.array([r['outer_usable'] for r in fold])
                n0 = np.array([r['submitted_n'] for r in fold]); u0 = np.array([r['submitted_usable'] for r in fold])
                lo, hi = cell_boot(u, n)
                summ.append(dict(direction=d, tau_s=tau, tol=tol, folds=len(fold),
                                 picks=';'.join(f'{k[0]}/{k[1]}:{v}' for k, v in Counter((r['picked_rung'], r['picked_agg']) for r in fold).most_common()),
                                 nested_usable=float(np.average(u, weights=n)), nested_boot_lo=lo, nested_boot_hi=hi,
                                 nested_exceed=int(sum(r['exceed'] for r in fold)), nested_n=int(n.sum()),
                                 submitted_usable=float(np.average(u0, weights=n0)), submitted_exceed=int(sum(r['submitted_exceed'] for r in fold)),
                                 submitted_n=int(n0.sum())))
    write_csv(os.path.join(OUT, 'E3_picks.csv'), picks)
    write_csv(os.path.join(OUT, 'E3_summary.csv'), summ)
    allp = Counter((r['picked_rung'], r['picked_agg']) for r in picks if r['tol'] == (0.0 if r['direction'] == 'discharge' else 0.5))
    print('declared-tolerance picks:', allp.most_common())
    print('E3 done')
