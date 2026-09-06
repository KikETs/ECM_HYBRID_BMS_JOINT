"""Six-method comparison (Fig 4 / L189) under the nested calibration protocol.

For outer target c and method m:
  calibration  inner rows of the five cells j != c, each solved with
               - trim multipliers from the inner run of m trained without c
                 (INNERTRIM_A8/<sub>_no<c> for a8, BASETREES/no<c>/analysis/
                 runs_trim_<tag><sfx> for the others),
               - ECM/OCV surfaces pooled without {c, j} (INNERPOOL/no<c>/pool),
               - SOH from the inner ridge trained without c (INNERSOH),
               reference SOC (the primary arm).
  evaluation   c's own rows solved with the shipped five-cell assets (repo
               runs_trim_<tag>, cache/pool holdout=c, results/soh_pred.npz).
SOLVER = fp (shipped damped fixed point started at the reference current) or
hull (reference-independent hull-aware search, smallest-|I| root, explicit
failure).  Eligibility: |meas| > 0.5, extrap <= 1.5, both models solved.
Output: safety_strict_<m>_est.csv / safety_strict_percell_<m>_est.csv in the
repository's own format, plus method_comparison_est.csv (rank by mean usable,
interval overlap against a8).  Nothing in the repository is written.
"""
import csv
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
RA = os.path.join(os.environ['RCSOP_BUNDLE'], 'assets')
HERE = os.environ.get('OUTDIR', os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(os.environ['RCSOP_BUNDLE'], 'scripts')
INNERPOOL = os.environ.get('INNERPOOL', os.path.join(RA, 'innerpool'))
INNERTRIM_A8 = os.environ.get('INNERTRIM', os.path.join(RA, 'innertrim'))
BASETREES = os.environ.get('BASETREES', os.path.join(RA, 'basetrees'))
INNERSOH = os.environ.get('INNERSOH', os.path.join(RA, 'innersoh'))
SOLVER = os.environ.get('SOLVER', 'fp')
OUT = os.environ.get('OUT', os.path.join(HERE, f'level1_{SOLVER}'))
NPROC = int(os.environ.get('NPROC', '8'))
sys.path.insert(0, F); sys.path.insert(0, A); sys.path.insert(0, os.path.join(REPO, 'repro'))
import hull_solver                                            # noqa: E402
import eval_sop_amps as EV                                    # noqa: E402
import ecm_pool                                               # noqa: E402
from run_safety import TOL, EXTRAP_MAX                        # noqa: E402
from run_safety_strict import (fit_lambda, clopper_pearson_upper,  # noqa: E402
                               bootstrap_usable, MIN_TRAIN)

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
AGG = 'max'
TAUS = (10.0, 2.0)
TAG = {'a8': 'runs_trim_a8', 'a3': 'runs_trim_v2', 'shrink': 'runs_trim_shrink',
       'lstm': 'runs_trim_lstm', 'gru': 'runs_trim_gru', 'ffrls': 'runs_trim_ffrls'}
SUB = {'discharge': 'trim', 'charge': 'trim_chg'}
SFX = {'discharge': '', 'charge': '_chg'}


OUTER_CHG = {'a8': 'runs_trim_a8_chg', 'a3': 'runs_trim_chg_v2', 'shrink': 'runs_trim_shrink_chg',
             'lstm': 'runs_trim_lstm_chg', 'gru': 'runs_trim_gru_chg', 'ffrls': 'runs_trim_ffrls_chg'}


def trim_dir(method, level, c, direction):
    if level == 'outer':   # shipped directories (repro/run_evals.py TRIMS); untracked seq-baseline predictions come from the bundle
        d = OUTER_CHG[method] if direction == 'charge' else TAG[method]
        p = os.path.join(A, d)
        if not glob.glob(os.path.join(p, 'pred_A*_*.npz')):
            p = os.path.join(os.environ['RCSOP_BUNDLE'], 'assets', 'outer_trims', d)
        return p
    if method == 'a8':
        return os.path.join(INNERTRIM_A8, f'{SUB[direction]}_no{c}')
    return os.path.join(BASETREES, f'no{c}', 'analysis', TAG[method] + SFX[direction])


def rows_for(method, level, c, j, direction):
    """rows of cell j; level outer -> j == c with shipped assets."""
    charge = direction == 'charge'
    lab = EV.LABEL_CHG if charge else EV.LABEL
    rows = [r for r in csv.DictReader(open(lab, encoding='utf-8')) if r['cell'] == j]
    if level == 'outer':
        sd, sc = ecm_pool.surfaces(j)
        soh_map = dict(np.load(os.path.join(A, 'results', 'soh_pred.npz')))
    else:
        pd_ = os.path.join(INNERPOOL, f'no{c}')
        sd, sc = ecm_pool.surfaces(j, outdir=os.path.join(pd_, 'pool'),
                                   ecm_csv=os.path.join(pd_, 'uypydj_ecm.csv'),
                                   ocv_csv=os.path.join(pd_, 'uypydj_ocv.csv'))
        soh_map = dict(np.load(os.path.join(INNERSOH, f'soh_pred_no{c}.npz')))
    surf = sc if charge else sd
    K = EV.trim_k_agg(j, trim_dir(method, level, c, direction), AGG)
    out = []
    for r in rows:
        cyc = int(r['cycle'])
        if r.get('I_star_lin4_A') in (None, '', 'nan') or cyc not in K:
            continue
        soc = float(r['SOC'])
        cc, pp = soh_map[f'{j}_cycle'], soh_map[f'{j}_pred']
        prior = cc <= cyc
        soh = float(pp[prior][int(np.argmax(cc[prior]))]) if prior.any() else float(pp[int(np.argmin(cc))])
        meas = float(r['I_star_lin4_A'])
        kf, ks = K[cyc]
        v_pre, v_min, tau = float(r['V_pre_V']), float(r['V_min_V']), float(r['tau_s'])
        if SOLVER == 'hull':
            i0, n0, s0 = hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, 1.0, 1.0, charge, EV.r_eff)
            i3, n3, s3 = hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, kf, ks, charge, EV.r_eff)
        else:
            i0 = EV.solve_I(surf, soc, soh, v_pre, v_min, tau, 1.0, 1.0, I0=meas, charge=charge)
            i3 = EV.solve_I(surf, soc, soh, v_pre, v_min, tau, kf, ks, I0=meas, charge=charge)
        if not (np.isfinite(i0) and np.isfinite(i3)):
            continue
        out.append(dict(cell=j, meas=abs(meas), hyb=abs(i3), ecm=abs(i0),
                        extrap=float(r['extrap']), tau=tau))
    return (method, level, c, j, direction), out


def keep(r, tau, key):
    return r['tau'] == tau and r['meas'] > 0.5 and r['extrap'] <= EXTRAP_MAX and np.isfinite(r[key])


def main(methods):
    os.makedirs(OUT, exist_ok=True)
    tasks = []
    for m in methods:
        for d in ('discharge', 'charge'):
            for c in CELLS:
                tasks.append((m, 'outer', c, c, d))
                for j in CELLS:
                    if j != c:
                        tasks.append((m, 'inner', c, j, d))
    print(f'  {len(tasks)} tasks, solver={SOLVER}', flush=True)
    with mp.Pool(NPROC) as pool:
        res = dict(pool.starmap(rows_for, tasks))
    comp = []
    for m in methods:
        keys = ['hyb'] + (['ecm'] if m == 'a8' else [])
        for key in keys:
            name = m if key == 'hyb' else 'a0'
            summary, percell = [], []
            for d in ('discharge', 'charge'):
                tol = TOL[d]
                for tau in TAUS:
                    recs = []
                    for c in CELLS:
                        te = [r for r in res[(m, 'outer', c, c, d)] if keep(r, tau, key)]
                        tr = [r for j in CELLS if j != c for r in res[(m, 'inner', c, j, d)] if keep(r, tau, key)]
                        if not te or len(tr) < MIN_TRAIN:
                            continue
                        lam = fit_lambda(np.array([r[key] for r in tr]), np.array([r['meas'] for r in tr]), tol)
                        p, y = np.array([r[key] for r in te]), np.array([r['meas'] for r in te])
                        over = lam * p - y
                        recs.append(dict(cell=c, n=len(te), lam=lam, n_train=len(tr),
                                         exceed=int((over > tol).sum()), worst=float(max(over.max(), 0.0)),
                                         usable=float(np.median(lam * p / y) * 100),
                                         optimism=float(np.mean(p > y) * 100),
                                         rmse=float(np.sqrt(np.mean((p - y) ** 2)))))
                    if not recs:
                        continue
                    n = sum(r['n'] for r in recs); k = sum(r['exceed'] for r in recs)
                    us = np.array([r['usable'] for r in recs]); ws = np.array([r['n'] for r in recs], float)
                    lam = np.array([r['lam'] for r in recs]); lo, hi = bootstrap_usable(recs); wi = int(np.argmin(us))
                    for r in recs:
                        percell.append([d, f'{tau:.1f}', name, 'est', r['cell'], r['n'], f"{r['lam']:.4f}", r['exceed'],
                                        f"{r['worst']:.3f}", f"{r['usable']:.2f}", f"{r['optimism']:.1f}", f"{r['rmse']:.3f}"])
                    summary.append([d, f'{tau:.1f}', name, 'est', len(recs), n, k, f'{np.average(us, weights=ws):.2f}',
                                    f'{np.median(us):.2f}', f'{us[wi]:.2f}', recs[wi]['cell'], f'{lam.min():.4f}',
                                    f'{np.median(lam):.4f}', f'{lam.max():.4f}', '', f'{clopper_pearson_upper(k, n) * 100:.3f}',
                                    f'{lo:.2f}', f'{hi:.2f}', max(r['worst'] for r in recs)])
                    comp.append(dict(method=name, direction=d, tau=f'{tau:.1f}', n=n, exceed=k,
                                     usable=float(np.average(us, weights=ws)), lo=lo, hi=hi, worst=float(us[wi]),
                                     worst_A=max(r['worst'] for r in recs)))
                    print(f'  {name:<6} {d:<9} tau={tau:>4.0f} n={n:>5} exc={k} usable={np.average(us, weights=ws):.2f} '
                          f'[{lo:.2f},{hi:.2f}] worst={us[wi]:.2f}', flush=True)
            hs = ['direction', 'tau_s', 'method', 'soh_arm', 'n_cells', 'n_rows', 'exceed', 'usable_mean_pct',
                  'usable_median_pct', 'usable_worst_pct', 'worst_cell', 'lambda_min', 'lambda_median', 'lambda_max',
                  'lambda_pooled_shipped', 'exceed_ub95_pct', 'usable_boot_lo_pct', 'usable_boot_hi_pct', 'worst_overshoot_A']
            hp = ['direction', 'tau_s', 'method', 'soh_arm', 'cell', 'n', 'lambda_i', 'exceed', 'worst_A', 'usable_pct',
                  'optimism_pct', 'rmse_A']
            for fn, h, body in ((f'safety_strict_{name}_est.csv', hs, summary),
                                (f'safety_strict_percell_{name}_est.csv', hp, percell)):
                with open(os.path.join(OUT, fn), 'w', newline='') as f:
                    w = csv.writer(f); w.writerow(h); w.writerows(body)
    # ranking (repro/run_method_comparison.py logic: rank by mean usable, overlap of intervals vs a8)
    rows = []
    for d in ('discharge', 'charge'):
        for tau in TAUS:
            grp = [r for r in comp if r['direction'] == d and r['tau'] == f'{tau:.1f}' and r['method'] != 'a0']
            ref = next((r for r in grp if r['method'] == 'a8'), None)
            grp.sort(key=lambda r: -r['usable'])
            for i, r in enumerate(grp, 1):
                if ref is None or r is ref:
                    rel = '—'
                else:
                    overlap = not (r['hi'] < ref['lo'] or r['lo'] > ref['hi'])
                    rel = 'overlaps' if overlap else ('separated, A8 higher' if ref['usable'] > r['usable'] else 'separated, A8 lower')
                rows.append([d, f'{tau:.1f}', r['method'], f"{r['usable']:.2f}", f"{r['lo']:.2f}", f"{r['hi']:.2f}",
                             r['n'], r['exceed'], f"{r['worst']:.2f}", i, rel, f"{r['worst_A']:.3f}"])
    with open(os.path.join(OUT, 'method_comparison_est.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'tau_s', 'method', 'usable_mean_pct', 'boot_lo_pct', 'boot_hi_pct', 'n_rows', 'exceed',
                    'usable_worst_pct', 'rank_in_condition', 'interval_vs_a8', 'worst_overshoot_A'])
        w.writerows(rows)
    print(f'  -> {OUT}/method_comparison_est.csv', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['a8', 'a3', 'shrink', 'lstm', 'gru', 'ffrls'])
