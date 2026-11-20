"""E6 (R2-7): retained current, influential observations, and quantile / split-conformal calibration.

lambda_max(c) = min over inner rows of (I_ref + tol) / I_pred  (the submitted max-safe rule, identical to bisection)
alternatives on the same inner rows:
  quantile / split-conformal: lambda = k-th smallest ratio, k = floor(alpha (n+1)) (k >= 1; k = 1 is the max-safe rule)
  drop-top-m: remove the m most binding inner rows (influence of single observations)
  leave-one-cycle-out: lambda recomputed with each inner (cell, cycle) removed -> spread
retained ratio = lambda |I_pred| / |I_ref| on outer rows, by SOC / SOH / cell.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import *

OUT = out_dir('E6')
arms = load_arms()

def lam_rule(pred, meas, tol, k=1):
    r = np.sort((meas + tol) / pred)
    return float(r[min(max(k, 1), len(r)) - 1])

conds = [('discharge', 2.0, 0.0), ('discharge', 10.0, 0.0), ('charge', 2.0, 0.5), ('charge', 10.0, 0.5), ('charge', 2.0, 0.0), ('charge', 10.0, 0.0)]
cal, infl, retained, loco_cyc = [], [], [], []
for d, tau, tol in conds:
    data = {}
    for c in CELLS:
        inner, outer = split_rows(arms, d, tau, c)
        if not outer or len(inner) < 25:
            continue
        data[c] = (inner, outer)
    # calibration rules
    rules = [('max-safe (submitted)', 'k', 1)] + [(f'conformal alpha={a}', 'alpha', a) for a in (0.001, 0.005, 0.01, 0.02, 0.05)] + \
            [(f'drop top-{m} binding rows', 'k', m + 1) for m in (1, 2, 5)]
    for name, kind, v in rules:
        pcs, ex, ex0, n_all, maxpos = [], 0, 0, 0, 0.
        for c, (inner, outer) in data.items():
            pi = np.array([r['hyb'] for r in inner]); yi = np.array([r['meas'] for r in inner])
            k = int(np.floor(v * (len(pi) + 1))) if kind == 'alpha' else v
            lam = lam_rule(pi, yi, tol, k)
            po = lam * np.array([r['hyb'] for r in outer]); yo = np.array([r['meas'] for r in outer])
            pcs.append((float(np.median(po / yo) * 100), len(po)))
            ex += int((po - yo > tol).sum()); ex0 += int((po - yo > 0).sum()); n_all += len(po); maxpos = max(maxpos, float((po - yo).max()))
        lo, hi = cell_boot([x[0] for x in pcs], [x[1] for x in pcs])
        cal.append(dict(direction=d, tau_s=tau, tol=tol, rule=name, n_rows=n_all, exceed=ex, exceed_rate_pct=100 * ex / n_all,
                        exceed_gt0=ex0, max_pos_A=max(0., maxpos), usable_pct=paper_usable(pcs), boot_lo=lo, boot_hi=hi))
    # influence: binding rows, leave-one-cycle-out spread
    for c, (inner, outer) in data.items():
        pi = np.array([r['hyb'] for r in inner]); yi = np.array([r['meas'] for r in inner])
        ratio = (yi + tol) / pi
        order = np.argsort(ratio)
        lam0 = ratio[order[0]]
        for rank in range(3):
            r = inner[order[rank]]
            infl.append(dict(direction=d, tau_s=tau, tol=tol, target=c, rank=rank + 1, lam_if_this_is_binding=float(ratio[order[rank]]),
                             src_cell=r['cell'], cycle=r['cycle'], soc=r['soc_ref'], soh=r['soh_ref'], extrap=r['extrap'],
                             I_ref=r['meas'], I_pred=r['hyb']))
        keys = sorted({(r['cell'], r['cycle']) for r in inner})
        lams = []
        for kc in keys:
            m = np.array([(r['cell'], r['cycle']) != kc for r in inner])
            lams.append(ratio[m].min())
        lams = np.array(lams)
        loco_cyc.append(dict(direction=d, tau_s=tau, tol=tol, target=c, n_inner_cycles=len(keys), lam=float(lam0),
                             lam_max_after_drop=float(lams.max()), lam_change_max_pct=float(100 * (lams.max() / lam0 - 1)),
                             n_cycles_that_change_lam=int((lams > lam0 + 1e-12).sum())))
        # retained ratio by state (submitted rule)
        po = lam0 * np.array([r['hyb'] for r in outer]); yo = np.array([r['meas'] for r in outer])
        for r, q in zip(outer, po / yo):
            retained.append(dict(direction=d, tau_s=tau, tol=tol, cell=c, cycle=r['cycle'], soc=r['soc_ref'], soh=r['soh_ref'],
                                 soh_in=r['soh_in'], extrap=r['extrap'], retained=float(q)))
write_csv(os.path.join(OUT, 'E6_calibration_rules.csv'), cal)
write_csv(os.path.join(OUT, 'E6_binding_rows.csv'), infl)
write_csv(os.path.join(OUT, 'E6_leave_one_cycle_out.csv'), loco_cyc)
write_csv(os.path.join(OUT, 'E6_retained_rows.csv'), retained)

import pandas as pd
R = pd.DataFrame(retained)
R['soc_bin'] = pd.cut(R.soc, [0, .2, .4, .6, .8, 1.01], right=False).astype(str)
R['soh_bin'] = pd.cut(R.soh, [0, .8, .85, .9, .95, 1.01], right=False).astype(str)
tabs = []
for by in ('cell', 'soc_bin', 'soh_bin'):
    t = R.groupby(['direction', 'tau_s', 'tol', by]).retained.agg(n='size', median='median', p05=lambda s: s.quantile(.05), p95=lambda s: s.quantile(.95)).reset_index()
    t = t.rename(columns={by: 'level'}); t['by'] = by
    tabs.append(t)
pd.concat(tabs).to_csv(os.path.join(OUT, 'E6_retained_by_state.csv'), index=False)
print('E6 done')
