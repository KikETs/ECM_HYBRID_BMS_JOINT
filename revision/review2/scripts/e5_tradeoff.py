"""E5 step 2 (R2-6): usable current vs exceedance for all methods, on the rows every method solved.

(a) calibrated: lambda_c from inner rows of the other cells (k-th smallest (I_ref + tol)/I_pred, k = floor(alpha (n+1)),
    alpha = 0 -> max-safe, the submitted rule); scored on the outer cell. Out-of-sample.
(b) frontier (descriptive, uses the outer rows): one pooled lambda per method swept over a grid; usable at the largest
    lambda whose outer exceedance rate <= target. Shows predictor quality independent of the calibration rule.
Usable = manuscript metric (per-cell median of lambda |I_pred| / |I_ref|, row-weighted); cell bootstrap CI.
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import CELLS, cell_boot, write_csv, out_dir

OUT = out_dir('E5')
res = pickle.load(open(os.path.join(OUT, 'method_rows.pkl'), 'rb'))
METHODS = ['a0', 'a8', 'a3', 'shrink', 'lstm', 'gru', 'ffrls', 'hppc']
SRC = {'a0': ('a8', 'ecm'), 'a8': ('a8', 'hyb'), 'a3': ('a3', 'hyb'), 'shrink': ('shrink', 'hyb'), 'lstm': ('lstm', 'hyb'),
       'gru': ('gru', 'hyb'), 'ffrls': ('ffrls', 'hyb'), 'hppc': ('hppc', 'hyb')}


def ident(r):
    return (r['cell'], round(r['meas'], 3), r['tau'], round(r['extrap'], 3))


def rows(method, level, c, j, d, tau):
    m, k = SRC[method]
    out = []
    for r in res[(m, level, c, j, d)]:
        if r['tau'] == tau and r['meas'] > .5 and r['extrap'] <= 1.5 and np.isfinite(r[k]) and r[k] > 0:
            out.append((ident(r), r[k], r['meas']))
    return out


def common_ids(level, c, j, d, tau):
    s = None
    for m in METHODS:
        ids = {x[0] for x in rows(m, level, c, j, d, tau)}
        s = ids if s is None else s & ids
    return s


cal, front, curves, pairs = [], [], [], []
for d, tols in (('discharge', (0.0,)), ('charge', (0.5, 0.0))):
    for tau in (2.0, 10.0):
        co = {c: common_ids('outer', c, c, d, tau) for c in CELLS}
        ci = {(c, j): common_ids('inner', c, j, d, tau) for c in CELLS for j in CELLS if j != c}
        for tol in tols:
            percell = {}
            for m in METHODS:
                for alpha in (0.0, 0.005, 0.01, 0.02):
                    pcs, ex, n_all, maxpos, recs = [], 0, 0, 0., []
                    for c in CELLS:
                        out = [x for x in rows(m, 'outer', c, c, d, tau) if x[0] in co[c]]
                        inn = [x for j in CELLS if j != c for x in rows(m, 'inner', c, j, d, tau) if x[0] in ci[c, j]]
                        if not out or len(inn) < 25:
                            continue
                        pi = np.array([x[1] for x in inn]); yi = np.array([x[2] for x in inn])
                        rr = np.sort((yi + tol) / pi); k = max(int(np.floor(alpha * (len(rr) + 1))), 1)
                        lam = rr[k - 1]
                        po = lam * np.array([x[1] for x in out]); yo = np.array([x[2] for x in out])
                        u = float(np.median(po / yo) * 100)
                        pcs.append((u, len(po))); ex += int((po - yo > tol).sum()); n_all += len(po); maxpos = max(maxpos, float((po - yo).max()))
                        recs.append((c, u, len(po), lam))
                    if not pcs:
                        continue
                    lo, hi = cell_boot([x[0] for x in pcs], [x[1] for x in pcs])
                    cal.append(dict(direction=d, tau_s=tau, tol=tol, method=m, alpha=alpha, n_cells=len(pcs), n_rows=n_all,
                                    exceed=ex, exceed_rate_pct=100 * ex / n_all, max_pos_A=max(0., maxpos),
                                    usable_pct=float(np.average([x[0] for x in pcs], weights=[x[1] for x in pcs])), boot_lo=lo, boot_hi=hi))
                    percell[m, alpha] = {c: (u, n) for c, u, n, _ in recs}
            # paired differences vs a8 at the same alpha (cell bootstrap)
            for alpha in (0.0, 0.005, 0.01, 0.02):
                ref = percell.get(('a8', alpha))
                for m in METHODS:
                    if m == 'a8' or (m, alpha) not in percell or ref is None:
                        continue
                    cs = sorted(set(ref) & set(percell[m, alpha]))
                    dd = np.array([percell[m, alpha][c][0] - ref[c][0] for c in cs]); w = np.array([ref[c][1] for c in cs])
                    lo, hi = cell_boot(dd, w)
                    pairs.append(dict(direction=d, tau_s=tau, tol=tol, alpha=alpha, method=m, n_cells=len(cs),
                                      diff_vs_a8_pp=float(np.average(dd, weights=w)), ci_lo=lo, ci_hi=hi, cells_method_higher=int((dd > 0).sum())))
            # frontier with one pooled lambda per method (outer rows, common set)
            grid = np.linspace(0.05, 1.6, 1551)
            for m in METHODS:
                allr = [(c, x) for c in CELLS for x in rows(m, 'outer', c, c, d, tau) if x[0] in co[c]]
                if not allr:
                    continue
                cellv = np.array([c for c, _ in allr]); p = np.array([x[1] for _, x in allr]); y = np.array([x[2] for _, x in allr])
                rate = np.array([np.mean(l * p - y > tol) * 100 for l in grid])
                def usable(l):
                    us, ns = [], []
                    for c in np.unique(cellv):
                        mm = cellv == c; us.append(np.median(l * p[mm] / y[mm]) * 100); ns.append(mm.sum())
                    return float(np.average(us, weights=ns))
                for i in range(0, len(grid), 10):
                    curves.append(dict(direction=d, tau_s=tau, tol=tol, method=m, lam=grid[i], exceed_rate_pct=rate[i], usable_pct=usable(grid[i])))
                for target in (0.0, 0.5, 1.0, 2.0, 5.0):
                    okl = grid[rate <= target]
                    l = okl.max() if len(okl) else np.nan
                    front.append(dict(direction=d, tau_s=tau, tol=tol, method=m, target_rate_pct=target, n_rows=len(p),
                                      lam=l, usable_pct=usable(l) if np.isfinite(l) else np.nan))
write_csv(os.path.join(OUT, 'E5_calibrated.csv'), cal)
write_csv(os.path.join(OUT, 'E5_paired_vs_a8.csv'), pairs)
write_csv(os.path.join(OUT, 'E5_frontier.csv'), front)
write_csv(os.path.join(OUT, 'E5_curves.csv'), curves)

# figure: frontier curves
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
C = pd.DataFrame(curves)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for ax, (d, tau, tol) in zip(axs.flat, [('discharge', 10.0, 0.0), ('discharge', 2.0, 0.0), ('charge', 10.0, 0.0), ('charge', 2.0, 0.0)]):
    for m in METHODS:
        g = C[(C.direction == d) & (C.tau_s == tau) & (C.tol == tol) & (C.method == m)]
        g = g[g.exceed_rate_pct <= 10]
        ax.plot(g.exceed_rate_pct, g.usable_pct, label=m.upper() if m != 'shrink' else 'Shrink', lw=2.2 if m == 'a8' else 1.2)
    ax.set_title(f'{d}, {tau:.0f} s, tolerance {tol:.1f} A'); ax.set_xlabel('exceedance rate (%)'); ax.set_ylabel('usable current (%)')
    ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'E5_frontier.png'), dpi=200)
print('E5 tradeoff done')
