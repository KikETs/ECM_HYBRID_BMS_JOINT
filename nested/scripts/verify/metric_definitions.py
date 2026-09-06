"""Row-level metric definitions applied to the final (level 2, hull) assets.

For every scored row:  d = lambda_c * |I_pred| - |I_ref|
  exceedance                      = d > tolerance
  maximum positive difference     = max(0, max d)
  maximum excess beyond tolerance = max(0, max (d - tolerance))
Both size metrics are reported side by side; the tables of the previous
follow-up (score()['worst']) report the FIRST one under the name
'worst_overshoot_A'.  Nothing is fitted here: lambda_c is read from the
level-2 per-cell tables.
"""
import csv
import os
import pickle

import numpy as np
import sys
sys.path.insert(0, os.path.join(os.environ['RCSOP_REPO'], 'repro'))
from run_safety_strict import fit_lambda, MIN_TRAIN   # noqa: E402  (same bisection as the tables)

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLW = os.environ['RCSOP_WORK']
SN = os.path.join(FLW, 'arms')   # outputs of reproduce.sh score
TOL = {'discharge': 0.0, 'charge': 0.5}
EXTRAP_MAX = 1.5
ARMS = {'oo': 'oracle SOH + oracle SOC', 'eo': 'estimated SOH + oracle SOC',
        'oe': 'oracle SOH + estimated SOC', 'ee': 'estimated SOH + estimated SOC'}
CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']

D = pickle.load(open(os.path.join(SN, 'arm_rows_replace_level2_hull2.pkl'), 'rb'))
lam = {}
for key in ('hyb', 'ecm'):
    for r in csv.DictReader(open(os.path.join(SN, f'e2e_nested_percell_replace_level2_hull2_{key}.csv'))):
        if r['protocol'] == 'nested':
            lam[(key, r['condition'], r['direction'], float(r['tau_s']), r['cell'])] = float(r['lambda'])
prev = {(r['condition'], r['direction'], r['tau_s']): r
        for r in csv.DictReader(open(os.path.join(SN, 'e2e_nested_full_replace_level2_hull2_hyb.csv')))
        if r['protocol'] == 'nested'}

rows_out, per_row = [], []
for key, model in (('hyb', 'A8'), ('ecm', 'A0')):
    for a, label in ARMS.items():
        for d in ('discharge', 'charge'):
            for tau in (10.0, 2.0):
                tol = TOL[d]
                dd, cells_ex, n = [], set(), 0
                cellrows = {}
                for c in CELLS:
                    L4 = lam.get((key, label, d, tau, c))
                    if L4 is None:
                        continue
                    # refit lambda_c at full precision from the same inner rows (tables store 4 decimals)
                    tr = [r for j in CELLS if j != c for r in D[('inner', c, j, d, a, 'replace')]
                          if r['status'] == 'ok' and r['tau'] == tau and r['meas'] > 0.5
                          and r['extrap'] <= EXTRAP_MAX and np.isfinite(r[key])]
                    L = fit_lambda(np.array([r[key] for r in tr]), np.array([r['meas'] for r in tr]), tol)
                    assert abs(L - L4) < 6e-5, (c, L, L4)
                    rows = [r for r in D[('outer', 'NONE', c, d, a, 'replace')]
                            if r['status'] == 'ok' and r['tau'] == tau and r['meas'] > 0.5
                            and r['extrap'] <= EXTRAP_MAX and np.isfinite(r[key])]
                    for r in rows:
                        dv = L * r[key] - r['meas']
                        dd.append(dv); n += 1
                        if dv > tol:
                            cells_ex.add(c)
                        per_row.append(dict(model=model, arm=label, direction=d, tau_s=tau, cell=c, key=r['key'],
                                            lambda_c=L, I_pred_abs=r[key], I_ref_abs=r['meas'], d=dv,
                                            tolerance=tol, exceed=int(dv > tol)))
                    cellrows[c] = len(rows)
                if not dd:
                    continue
                dd = np.array(dd)
                mpd = float(max(0.0, dd.max()))
                mex = float(max(0.0, (dd - tol).max()))
                rec = dict(model=model, arm=label, direction=d, tau_s=tau, tolerance_A=tol, n_rows=n,
                           exceed=int((dd > tol).sum()), cells_with_exceed=len(cells_ex),
                           max_positive_difference_A=round(mpd, 3), max_excess_beyond_tolerance_A=round(mex, 3),
                           cells_exceeding=' '.join(sorted(cells_ex)))
                if key == 'hyb':
                    p = prev.get((label, d, f'{tau:.1f}'))
                    rec['prev_table_worst_overshoot_A'] = p['worst_overshoot_A'] if p else ''
                    rec['prev_table_matches'] = ('max_positive_difference' if p and abs(float(p['worst_overshoot_A']) - mpd) < 5e-4
                                                 else ('max_excess' if p and abs(float(p['worst_overshoot_A']) - mex) < 5e-4 else 'NO'))
                rows_out.append(rec)
os.makedirs(V + '/metrics', exist_ok=True)
with open(os.path.join(V, 'metrics', 'metric_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
with open(os.path.join(V, 'metrics', 'row_level_d.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_row[0].keys())); w.writeheader(); w.writerows(per_row)
for r in rows_out:
    if r['model'] == 'A8' and r['arm'] == ARMS['eo']:
        print(r)
print(len(rows_out), 'condition rows;', len(per_row), 'row-level rows')
