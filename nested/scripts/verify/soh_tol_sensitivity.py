"""SOH-input and tolerance sensitivity on the FINAL assets (level 2: every
inner asset built without the outer target) with the hull-aware solver.

Primary arm only (estimated SOH, reference SOC), RC-SOP = A8 ('hyb') and the
nominal A0 ('ecm') from the same rows.

Analyses
  soh_fixed_lambda   SOH handed to the surface displaced by delta on the
                     TARGET rows only; lambda_c kept at the final baseline
                     (fitted with delta = 0, tol_fit = TOL[direction]).
  soh_refit_lambda   the same displacement applied to the inner calibration
                     rows too, lambda_c refitted with tol_fit = TOL[direction].
  tol_eval_only      baseline rows and baseline lambda (tol_fit = TOL);
                     only the exceedance decision threshold tol_eval varies.
  tol_joint          tol_fit = tol_eval = t: lambda refitted at t on the
                     baseline inner rows and rows judged at t.  This is a
                     calibration-policy sensitivity, not a threshold one.
Row sets: 'own' = the arm's eligible rows at that setting; 'common' = the
intersection of eligible keys over all deltas of the same analysis.
Eligibility changes only through the solver (meas, extrap are fixed); the
added/removed counts against delta = 0 are reported with that reason.
A uniform SOH displacement is a sensitivity of the surface axis; it is not a
conversion between the two SOH definitions.
"""
import csv
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
FLW = os.environ['RCSOP_WORK']
X = os.path.join(os.environ['RCSOP_BUNDLE'], 'assets')   # innerpool / innertrim
SN = os.path.join(FLW, 'arms')   # outputs of reproduce.sh score
sys.path.insert(0, A); sys.path.insert(0, os.path.join(os.environ['RCSOP_BUNDLE'], 'scripts')); sys.path.insert(0, os.path.join(REPO, 'repro'))
import eval_sop_amps as EV                                   # noqa: E402
import ecm_pool                                              # noqa: E402
import hull_solver                                           # noqa: E402
from run_safety_strict import fit_lambda, bootstrap_usable, MIN_TRAIN   # noqa: E402

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
TOL = {'discharge': 0.0, 'charge': 0.5}
EXTRAP_MAX = 1.5
DELTAS = [0.0, -0.0037, 0.0037, -0.0074, 0.0074]
TOLS = [0.0, 0.25, 0.5, 1.0]
TAUS = (10.0, 2.0)
D = pickle.load(open(os.path.join(SN, 'arm_rows_replace_level2_hull2.pkl'), 'rb'))


def keep(r, tau, key):
    return (r['status'] == 'ok' and r['tau'] == tau and r['meas'] > 0.5
            and r['extrap'] <= EXTRAP_MAX and np.isfinite(r[key]))


def resolve(args):
    """re-solve one block (level, outer, cell, direction) with SOH + delta."""
    level, outer, cell, direction, delta = args
    charge = direction == 'charge'
    if level == 'outer':
        sd, sc = ecm_pool.surfaces(cell)
    else:
        pd_ = os.path.join(X, 'innerpool', f'no{outer}')
        sd, sc = ecm_pool.surfaces(cell, outdir=os.path.join(pd_, 'pool'),
                                   ecm_csv=os.path.join(pd_, 'uypydj_ecm.csv'),
                                   ocv_csv=os.path.join(pd_, 'uypydj_ocv.csv'))
    surf = sc if charge else sd
    SUB = {'discharge': 'trim', 'charge': 'trim_chg'}
    tdir = (os.path.join(A, 'runs_trim_a8' if not charge else 'runs_trim_a8_chg') if level == 'outer'
            else os.path.join(X, 'innertrim', f'{SUB[direction]}_no{outer}'))
    K = EV.trim_k_agg(cell, tdir, 'max')
    src = D[(level, outer, cell, direction, 'eo', 'replace')]
    out = []
    v_lim = 4.2 if charge else 2.5
    for r in src:
        if r['status'] in ('no_k',) or r['cycle'] not in K:
            continue
        kf, ks = K[r['cycle']]
        rec = {k: r[k] for k in ('key', 'cell', 'tau', 'extrap', 'I_meas')}
        rec['meas'] = abs(r['I_meas'])
        soc, soh = r['soc_in'], r['soh_in'] + delta
        v_pre = float(r['V_pre'])
        i0, n0, s0 = hull_solver.solve(surf, soc, soh, v_pre, v_lim, r['tau'], 1.0, 1.0, charge, EV.r_eff)
        i3, n3, s3 = hull_solver.solve(surf, soc, soh, v_pre, v_lim, r['tau'], kf, ks, charge, EV.r_eff)
        if not (np.isfinite(i0) and np.isfinite(i3)):
            rec.update(status='hull', hyb=np.nan, ecm=np.nan)
        else:
            rec.update(status='ok', hyb=abs(i3), ecm=abs(i0))
        out.append(rec)
    return (level, outer, cell, direction, delta), out


def score_rows(rows, lam_by_cell, tol_eval, key):
    recs = []
    for c in CELLS:
        rr = [r for r in rows if r['cell'] == c]
        if not rr or c not in lam_by_cell:
            continue
        L = lam_by_cell[c]
        p = np.array([r[key] for r in rr]); y = np.array([r['meas'] for r in rr])
        dv = L * p - y
        recs.append(dict(cell=c, n=len(rr), lam=L, exceed=int((dv > tol_eval).sum()),
                         mpd=float(max(0.0, dv.max())), mex=float(max(0.0, (dv - tol_eval).max())),
                         usable=float(np.median(L * p / y) * 100)))
    if not recs:
        return None
    n = sum(r['n'] for r in recs); us = np.array([r['usable'] for r in recs]); w = np.array([r['n'] for r in recs], float)
    lo, hi = bootstrap_usable(recs)
    return dict(n_cells=len(recs), n_rows=n, exceed=sum(r['exceed'] for r in recs),
                cells_with_exceed=sum(1 for r in recs if r['exceed'] > 0),
                cells_exceeding=' '.join(r['cell'] for r in recs if r['exceed'] > 0),
                max_positive_difference_A=round(max(r['mpd'] for r in recs), 3),
                max_excess_beyond_tol_A=round(max(r['mex'] for r in recs), 3),
                usable_mean_pct=round(float(np.average(us, weights=w)), 2), usable_worst_pct=round(float(us.min()), 2),
                worst_cell=recs[int(np.argmin(us))]['cell'], boot_lo=round(lo, 2), boot_hi=round(hi, 2),
                lambda_min=round(min(r['lam'] for r in recs), 4), lambda_median=round(float(np.median([r['lam'] for r in recs])), 4),
                lambda_per_cell=' '.join(f"{r['cell']}:{r['lam']:.4f}" for r in recs))


def main():
    tasks = []
    for delta in DELTAS:
        for d in ('discharge', 'charge'):
            for c in CELLS:
                tasks.append(('outer', 'NONE', c, d, delta))
                for j in CELLS:
                    if j != c:
                        tasks.append(('inner', c, j, d, delta))
    print(f'  {len(tasks)} re-solve tasks (hull solver)', flush=True)
    with mp.Pool(int(os.environ.get('NPROC', '12'))) as pool:
        R = dict(pool.map(resolve, tasks))
    pickle.dump(R, open(os.path.join(V, 'sensitivity', 'resolved_rows.pkl'), 'wb'))
    out = []
    for key, model in (('hyb', 'A8'), ('ecm', 'A0')):
        for d in ('discharge', 'charge'):
            for tau in TAUS:
                tol0 = TOL[d]
                inner0 = {c: [r for j in CELLS if j != c for r in R[('inner', c, j, d, 0.0)] if keep(r, tau, key)] for c in CELLS}
                outer = {delta: {c: [r for r in R[('outer', 'NONE', c, d, delta)] if keep(r, tau, key)] for c in CELLS} for delta in DELTAS}
                lam0 = {c: fit_lambda(np.array([r[key] for r in inner0[c]]), np.array([r['meas'] for r in inner0[c]]), tol0)
                        for c in CELLS if len(inner0[c]) >= MIN_TRAIN}
                keys0 = {r['key'] for c in CELLS for r in outer[0.0][c]}
                common = set.intersection(*[{r['key'] for c in CELLS for r in outer[dl][c]} for dl in DELTAS])
                # ---- SOH: fixed lambda / refit lambda
                for delta in DELTAS:
                    rows_all = [r for c in CELLS for r in outer[delta][c]]
                    keys = {r['key'] for r in rows_all}
                    added, removed = len(keys - keys0), len(keys0 - keys)
                    inner_d = {c: [r for j in CELLS if j != c for r in R[('inner', c, j, d, delta)] if keep(r, tau, key)] for c in CELLS}
                    lam_d = {c: fit_lambda(np.array([r[key] for r in inner_d[c]]), np.array([r['meas'] for r in inner_d[c]]), tol0)
                             for c in CELLS if len(inner_d[c]) >= MIN_TRAIN}
                    for analysis, lam in (('soh_fixed_lambda', lam0), ('soh_refit_lambda', lam_d)):
                        for rs, rows in (('own', rows_all), ('common', [r for r in rows_all if r['key'] in common])):
                            s = score_rows(rows, lam, tol0, key)
                            if s:
                                out.append(dict(analysis=analysis, model=model, direction=d, tau_s=tau, soh_delta=delta,
                                                tol_fit_A=tol0, tol_eval_A=tol0, row_set=rs,
                                                protocol='level2 nested lambda_c on inner rows; ' + ('baseline lambda (delta=0)' if analysis == 'soh_fixed_lambda' else 'inner rows displaced by delta, lambda refitted'),
                                                rows_added_vs_delta0=added, rows_removed_vs_delta0=removed,
                                                n_unscored=0, change_reason='solver eligibility (hull/no root) only; meas and extrap unchanged',
                                                **s))
                # ---- tolerance
                rows0 = [r for c in CELLS for r in outer[0.0][c]]
                for t in TOLS:
                    lam_t = {c: fit_lambda(np.array([r[key] for r in inner0[c]]), np.array([r['meas'] for r in inner0[c]]), t)
                             for c in CELLS if len(inner0[c]) >= MIN_TRAIN}
                    for analysis, lam, tf in (('tol_eval_only', lam0, tol0), ('tol_joint', lam_t, t)):
                        s = score_rows(rows0, lam, t, key)
                        if s:
                            out.append(dict(analysis=analysis, model=model, direction=d, tau_s=tau, soh_delta=0.0,
                                            tol_fit_A=tf, tol_eval_A=t, row_set='own',
                                            protocol=('baseline lambda (tol_fit=TOL), decision threshold varied' if analysis == 'tol_eval_only'
                                                      else 'lambda refitted with tol_fit=t and judged at t (joint policy change)'),
                                            rows_added_vs_delta0=0, rows_removed_vs_delta0=0, n_unscored=0,
                                            change_reason='rows unchanged', **s))
    with open(os.path.join(V, 'sensitivity_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(len(out), 'rows -> sensitivity_results.csv', flush=True)


if __name__ == '__main__':
    main()
