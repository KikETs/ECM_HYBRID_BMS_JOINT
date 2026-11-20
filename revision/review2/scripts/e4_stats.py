"""E4 (R2-3, R2-4, R1-5): fold dependence and statistics for the primary comparison (estimated SOH + reference SOC).

1. leave-protocol-out (LPO): CC and CC_CELL2 share a protocol. For target CC (CC_CELL2), the outer predictions come from
   the models trained without both cells (stored inner level of the other one) and lambda is calibrated without the
   sibling's rows. Other cells are unchanged (they already hold out their own protocol).
2. two-level cluster bootstrap (cell -> cycle) for usable current and the A8 - A0 paired difference.
3. row-weighted (manuscript) vs cell-equal means.
4. per-cell table: lambda_c, n, exceedance, usable for A0/A8, both inner-asset versions (submitted, strict inner).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import *

OUT = out_dir('E4')
arms = load_arms()
STRICT = os.path.join(REV1, 'strict_inner', 'scored', 'strict_results_inner.pkl')
strict_inner = pickle.load(open(STRICT, 'rb')) if os.path.exists(STRICT) else None
SIB = {'CC': 'CC_CELL2', 'CC_CELL2': 'CC'}


def rows_for(direction, tau, c, mode, inner_src):
    keep = keep_factory(tau)
    if mode == 'lpo' and c in SIB:
        s = SIB[c]
        outer = [r for r in arms['inner', s, c, direction, 'eo', 'replace'] if keep(r)]
        inner = [r for j in CELLS if j not in (c, s) for r in inner_src['inner', c, j, direction, 'eo', 'replace'] if keep(r)]
    else:
        outer = [r for r in arms['outer', 'NONE', c, direction, 'eo', 'replace'] if keep(r)]
        inner = [r for j in CELLS if j != c for r in inner_src['inner', c, j, direction, 'eo', 'replace'] if keep(r)]
    return inner, outer


def per_cell(direction, tau, tol, mode, inner_src):
    recs = []
    for c in CELLS:
        inner, outer = rows_for(direction, tau, c, mode, inner_src)
        if not outer or len(inner) < 25:
            continue
        rec = dict(direction=direction, tau_s=tau, tol=tol, mode=mode, cell=c, n_inner=len(inner), n_outer=len(outer))
        for m in ('ecm', 'hyb'):
            lam = fit_lambda(np.array([r[m] for r in inner]), np.array([r['meas'] for r in inner]), tol)
            p = lam * np.array([r[m] for r in outer]); y = np.array([r['meas'] for r in outer])
            rec[f'lam_{m}'] = lam; rec[f'exceed_{m}'] = int((p - y > tol).sum())
            rec[f'maxpos_{m}'] = max(0., float((p - y).max())); rec[f'usable_{m}'] = float(np.median(p / y) * 100)
            rec[f'_ratio_{m}'] = p / y
        rec['_cycle'] = np.array([r['cycle'] for r in outer])
        recs.append(rec)
    return recs


def cluster_boot(recs, B=4000, seed=7):
    """resample cells, then cycles within cell; statistic = row-weighted mean of per-cell medians"""
    g = np.random.default_rng(seed)
    a8, a0, dd = [], [], []
    for _ in range(B):
        cells = g.integers(0, len(recs), len(recs))
        num8 = num0 = den = 0.0
        for i in cells:
            r = recs[i]; cyc = r['_cycle']; uc = np.unique(cyc)
            pick = g.choice(uc, len(uc), replace=True)
            idx = np.concatenate([np.flatnonzero(cyc == k) for k in pick])
            n = len(idx)
            num8 += n * np.median(r['_ratio_hyb'][idx]) * 100; num0 += n * np.median(r['_ratio_ecm'][idx]) * 100; den += n
        a8.append(num8 / den); a0.append(num0 / den); dd.append((num8 - num0) / den)
    q = lambda v: np.quantile(v, [.025, .975])
    return q(a8), q(a0), q(dd)


percell, summary = [], []
versions = [('submitted', arms)] + ([('strict_inner', strict_inner)] if strict_inner else [])
for vname, src in versions:
    for d in DIRECTIONS:
        for tau in TAUS:
            for tol in ([TOL[d]] + ([0.0] if d == 'charge' else [])):
                for mode in ('loco', 'lpo'):
                    recs = per_cell(d, tau, tol, mode, src)
                    if not recs:
                        continue
                    n = np.array([r['n_outer'] for r in recs])
                    u8 = np.array([r['usable_hyb'] for r in recs]); u0 = np.array([r['usable_ecm'] for r in recs])
                    c8 = cell_boot(u8, n); cd = cell_boot(u8 - u0, n)
                    k8, k0, kd = cluster_boot(recs)
                    diff = u8 - u0
                    summary.append(dict(version=vname, direction=d, tau_s=tau, tol=tol, mode=mode, n_cells=len(recs), n_rows=int(n.sum()),
                                        n_cycles=int(sum(len(np.unique(r['_cycle'])) for r in recs)),
                                        a8_usable_rowweighted=float(np.average(u8, weights=n)), a8_usable_cellequal=float(u8.mean()),
                                        a0_usable_rowweighted=float(np.average(u0, weights=n)),
                                        a8_exceed=int(sum(r['exceed_hyb'] for r in recs)), a0_exceed=int(sum(r['exceed_ecm'] for r in recs)),
                                        a8_cellboot_lo=c8[0], a8_cellboot_hi=c8[1], a8_clusterboot_lo=k8[0], a8_clusterboot_hi=k8[1],
                                        diff_pp=float(np.average(diff, weights=n)), diff_cellequal_pp=float(diff.mean()),
                                        diff_cellboot_lo=cd[0], diff_cellboot_hi=cd[1], diff_clusterboot_lo=kd[0], diff_clusterboot_hi=kd[1],
                                        cells_a8_higher=int((diff > 0).sum()),
                                        diff_dz=float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 and diff.std(ddof=1) > 0 else np.nan))
                    for r in recs:
                        percell.append({k: v for k, v in r.items() if not k.startswith('_')} | {'version': vname})
write_csv(os.path.join(OUT, 'E4_summary.csv'), summary)
write_csv(os.path.join(OUT, 'E4_percell.csv'), percell)
print('E4 done', len(summary), len(percell))
