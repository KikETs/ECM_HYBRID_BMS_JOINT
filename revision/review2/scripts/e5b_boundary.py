"""E5b (R1-1, R2-1, R2-6): score every method against MEASURED near-cutoff pulses instead of the fitted HPPC reference.

Observable boundary: a recorded fixed-current HPPC pulse whose endpoint voltage at tau lies within `band` mV of the
cutoff (2.5 V discharge / 4.2 V charge). I_b = applied current moved to the exact cutoff with the local V-I slope of
the same pulse group (correction <= band / slope). Rows matched to the evaluation rows by cell, cycle, SOC group,
tau, direction, with |dV_pre| <= 10 mV and |dSOC| <= 0.005.
For each method: allowed current = lambda_c |I_pred| (lambda_c: nested max-safe on the method's own inner rows vs the
fitted reference, i.e. exactly what each method would ship). Reported: allowed / I_b, count allowed > I_b, and the
fitted reference itself (I_ref / I_b) to show how far the reference is from the measured boundary.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, csv, pickle
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import CELLS, REPO, read_csv, write_csv, out_dir, fit_lambda

OUT = out_dir('E5b')
A = os.path.join(REPO, 'analysis')
F = os.path.join(REV1, 'followup', 'boundary', 'held_out_raw_pulses.csv')
res = pickle.load(open(os.path.join(os.path.dirname(OUT), 'E5', 'method_rows.pkl'), 'rb'))
SRC = {'a0': ('a8', 'ecm'), 'a8': ('a8', 'hyb'), 'a3': ('a3', 'hyb'), 'shrink': ('shrink', 'hyb'), 'lstm': ('lstm', 'hyb'),
       'gru': ('gru', 'hyb'), 'ffrls': ('ffrls', 'hyb'), 'hppc': ('hppc', 'hyb')}

# local slope per pulse group (all ranks)
grp = defaultdict(list)
for r in read_csv(os.path.join(A, 'uypydj_hppc_resistance.csv')):
    grp[r['direction'], r['protocol'], int(r['cycle']), int(r['soc_group']), float(r['tau_s'])].append((float(r['I_A']), float(r['V_tau_V'])))
labels = {}
for d, name in (('charge', 'sop_label_charge.csv'), ('discharge', 'sop_label_measured.csv')):
    for r in read_csv(os.path.join(A, name)):
        labels[r['cell'], int(r['cycle']), int(r['soc_group']), float(r['tau_s']), d] = r


def ident(cell, tau, meas, extrap):
    return (cell, tau, round(abs(meas), 3), round(float(extrap), 3))


def lam_for(method, d, tau, c, tol):
    m, k = SRC[method]
    inn = [r for j in CELLS if j != c for r in res[(m, 'inner', c, j, d)]
           if r['tau'] == tau and r['meas'] > .5 and r['extrap'] <= 1.5 and np.isfinite(r[k]) and r[k] > 0]
    if len(inn) < 25:
        return None
    return fit_lambda(np.array([r[k] for r in inn]), np.array([r['meas'] for r in inn]), tol)


pred = {}
for method, (m, k) in SRC.items():
    for c in CELLS:
        for d in ('discharge', 'charge'):
            for r in res[(m, 'outer', c, c, d)]:
                if np.isfinite(r[k]) and r[k] > 0:
                    pred[method, d, ident(c, r['tau'], r['meas'], r['extrap'])] = r[k]
lams = {(mt, d, tau, c, tol): lam_for(mt, d, tau, c, tol) for mt in SRC for d in ('discharge', 'charge') for tau in (2.0, 10.0)
        for c in CELLS for tol in ((0.0,) if d == 'discharge' else (0.0, 0.5))}

rows = []
for h in read_csv(F):
    d, tau = h['direction'], float(h['tau_s'])
    cut = 2.5 if d == 'discharge' else 4.2
    key = (h['cell'], int(h['cycle']), int(h['group']), tau, d)
    lab = labels.get(key)
    if lab is None or not lab.get('I_star_lin4_A'):
        continue
    if abs(float(h['V_pre']) - float(lab['V_pre_V'])) * 1e3 > 10.00001 or abs(float(h['SOC']) - float(lab['SOC'])) > .00501:
        continue
    dist = abs(float(h['V_V']) - cut) * 1e3
    if dist > 20.000001:
        continue
    pts = grp[d, h['cell'], int(h['cycle']), int(h['group']), tau]
    I = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts])
    b = np.polyfit(I, V, 1)[0] if len(pts) >= 2 else np.nan
    Iobs = float(h['I_A']); Ib = abs(Iobs + (cut - float(h['V_V'])) / b) if np.isfinite(b) and b > 0 else np.nan
    if float(lab['extrap']) > 1.5:
        continue
    iid = ident(h['cell'], tau, float(lab['I_star_lin4_A']), lab['extrap'])
    base = dict(cell=h['cell'], cycle=int(h['cycle']), group=int(h['group']), direction=d, tau_s=tau, dist_mV=dist,
                inside=int(float(h['V_V']) >= cut if d == 'discharge' else float(h['V_V']) <= cut),
                I_obs=abs(Iobs), I_b=Ib, I_ref=abs(float(lab['I_star_lin4_A'])), extrap=float(lab['extrap']),
                soc=float(lab['SOC']))
    for mt in SRC:
        p = pred.get((mt, d, iid))
        if p is None:
            continue
        for tol in ((0.0,) if d == 'discharge' else (0.0, 0.5)):
            lam = lams[mt, d, tau, h['cell'], tol]
            if lam is None:
                continue
            rows.append(dict(base, method=mt, tol=tol, lam=lam, allowed=lam * p, I_pred=p))
write_csv(os.path.join(OUT, 'E5b_rows.csv'), rows)

import pandas as pd
D = pd.DataFrame(rows)
summ = []
for (d, tau, tol, band), _ in pd.DataFrame([(d, t, tl, b) for d in ('discharge', 'charge') for t in (2.0, 10.0)
                                             for tl in ((0.0,) if d == 'discharge' else (0.0, 0.5)) for b in (10.0, 20.0)],
                                            columns=['d', 't', 'tl', 'b']).groupby(['d', 't', 'tl', 'b']):
    g = D[(D.direction == d) & (D.tau_s == tau) & (D.tol == tol) & (D.dist_mV <= band)]
    # common pulses: every method present
    cnt = g.groupby(['cell', 'cycle', 'group']).method.nunique()
    keep = cnt[cnt == len(SRC)].index
    g = g.set_index(['cell', 'cycle', 'group']).loc[keep].reset_index()
    if g.empty:
        continue
    ref = g[g.method == 'a8']
    summ.append(dict(direction=d, tau_s=tau, tol=tol, band_mV=band, method='fitted reference I_ref', n=len(ref), cells=ref.cell.nunique(),
                     median_ratio_to_Ib=float((ref.I_ref / ref.I_b).median()), n_above_Ib=int((ref.I_ref > ref.I_b).sum()),
                     max_above_A=float(max(0, (ref.I_ref - ref.I_b).max()))))
    for mt in SRC:
        x = g[g.method == mt]
        r = x.allowed / x.I_b
        cell_med = x.assign(r=r).groupby('cell').r.median()
        summ.append(dict(direction=d, tau_s=tau, tol=tol, band_mV=band, method=mt, n=len(x), cells=x.cell.nunique(),
                         median_ratio_to_Ib=float(r.median()), cell_equal_ratio=float(cell_med.mean()),
                         n_above_Ib=int((x.allowed > x.I_b).sum()), max_above_A=float(max(0, (x.allowed - x.I_b).max())),
                         n_above_Iobs_outside=int(((x.allowed >= x.I_obs) & (x.inside == 0)).sum())))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(OUT, 'E5b_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(S.round(3).to_string(index=False))
