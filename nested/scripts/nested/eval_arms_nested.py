"""Row-level predictions for the four state-input arms, outer and inner level.

Arms (labels as in repro/run_end_to_end.py)
    oo  reference SOH + reference SOC
    eo  estimated SOH + reference SOC          (= the primary arm)
    oe  reference SOH + reference SOC displaced by the EKF terminal error
    ee  estimated SOH + displaced SOC

'estimated SOC' is, as in the shipped code, the reference SOC of the pulse
plus the mean terminal-quarter SOC error of the adopted EKF on the nearest
drive record that ENDED before the pulse cycle, clipped to [0.02, 1].  A
pulse with no preceding record is left unscored (no later record is used).

Levels
    outer  target cell c evaluated with the five-cell assets (repo:
           runs_trim_a8{,_chg}, cache/pool holdout=c, results/soh_pred.npz,
           soc_err_outer_<rule>.npz built here from LOCO surfaces)
    inner  for outer target c, calibration cell j != c evaluated with the
           assets built without c (assets: innertrim, innerpool,
           innersoh; soc_err_inner_no<c>_<rule>.npz built here)

Solver: the shipped damped fixed point started at the reference current
(eval_sop_amps.solve_I), so that the only things that change against the
shipped tables are the calibration lineage and the SOC keys.  Nothing in the
repository is written.
"""
import csv
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
RA = os.path.join(os.environ['RCSOP_BUNDLE'], 'assets')
INNERPOOL = os.environ.get('INNERPOOL', os.path.join(RA, 'innerpool'))
INNERTRIM = os.environ.get('INNERTRIM', os.path.join(RA, 'innertrim'))
INNERSOH = os.environ.get('INNERSOH', os.path.join(RA, 'innersoh'))
SOCTAG = os.environ.get('SOCTAG', '')          # suffix of the soc_err npz files
SOLVER = os.environ.get('SOLVER', 'fp')        # fp | hull
OUTTAG = os.environ.get('OUTTAG', '')
HERE = os.environ.get('OUTDIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ['RCSOP_BUNDLE'], 'scripts'))
import hull_solver                               # noqa: E402
sys.path.insert(0, A)
import eval_sop_amps as EV                       # noqa: E402
import ecm_pool                                  # noqa: E402

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST',
         'CC', 'CC_CELL2']
AGG = 'max'
ARMS = {'oo': (False, False), 'eo': (True, False),
        'oe': (False, True), 'ee': (True, True)}
LABEL = {'oo': 'oracle SOH + oracle SOC', 'eo': 'estimated SOH + oracle SOC',
         'oe': 'oracle SOH + estimated SOC',
         'ee': 'estimated SOH + estimated SOC'}
SUB = {'discharge': 'trim', 'charge': 'trim_chg'}
OUTER_TRIM = {'discharge': 'runs_trim_a8', 'charge': 'runs_trim_a8_chg'}


def nearest_preceding(cc, vals, cycle):
    prior = cc < cycle          # strictly before: end cycle < pulse cycle
    if not prior.any():
        return None, None
    i = int(np.argmax(cc[prior]))
    return float(vals[prior][i]), int(cc[prior][i])


def rows_for(level, outer, cell, direction, arm, rule):
    use_soh, use_soc = ARMS[arm]
    charge = direction == 'charge'
    lab = EV.LABEL_CHG if charge else EV.LABEL
    rows = [r for r in csv.DictReader(open(lab, encoding='utf-8'))
            if r['cell'] == cell]
    if level == 'outer':
        sd, sc = ecm_pool.surfaces(cell)
        tdir = os.path.join(A, OUTER_TRIM[direction])
        soh_map = dict(np.load(os.path.join(A, 'results', 'soh_pred.npz')))
        soc_map = dict(np.load(os.path.join(HERE, f'soc_err_outer_{rule}{SOCTAG}.npz')))
    else:
        pooldir = os.path.join(INNERPOOL, f'no{outer}')
        sd, sc = ecm_pool.surfaces(cell, outdir=os.path.join(pooldir, 'pool'),
                                   ecm_csv=os.path.join(pooldir, 'uypydj_ecm.csv'),
                                   ocv_csv=os.path.join(pooldir, 'uypydj_ocv.csv'))
        tdir = os.path.join(INNERTRIM, f'{SUB[direction]}_no{outer}')
        soh_map = dict(np.load(os.path.join(INNERSOH, f'soh_pred_no{outer}.npz')))
        soc_map = dict(np.load(os.path.join(HERE, f'soc_err_inner_no{outer}_{rule}{SOCTAG}.npz')))
    surf = sc if charge else sd
    K = EV.trim_k_agg(cell, tdir, AGG)
    out = []
    for r in rows:
        cyc = int(r['cycle'])
        tau = float(r['tau_s'])
        if r.get('I_star_lin4_A') in (None, '', 'nan'):
            continue
        meas = float(r['I_star_lin4_A'])
        rec = dict(level=level, outer=outer, cell=cell, direction=direction,
                   arm=arm, cycle=cyc, tau=tau, extrap=float(r['extrap']),
                   V_pre=r['V_pre_V'], I_meas=round(meas, 3),
                   key=f"{cell}|{cyc}|{tau:.1f}|{r['V_pre_V']}|{round(meas, 3)}",
                   soc_ref=float(r['SOC']), soh_ref=float(r['SOH']),
                   status='ok')
        if cyc not in K:
            rec['status'] = 'no_k'; out.append(rec); continue
        soc, soh = float(r['SOC']), float(r['SOH'])
        if use_soh:
            cc, pp = soh_map[f'{cell}_cycle'], soh_map[f'{cell}_pred']
            prior = cc <= cyc
            soh = float(pp[prior][int(np.argmax(cc[prior]))]) if prior.any() \
                else float(pp[int(np.argmin(cc))])
        if use_soc:
            e, src = nearest_preceding(soc_map[f'{cell}_cycle'],
                                       soc_map[f'{cell}_err'], cyc)
            if e is None:
                rec['status'] = 'no_prior_drive'; out.append(rec); continue
            soc = float(np.clip(soc + e, 0.02, 1.0))
            rec['soc_err'] = e; rec['soc_src_cycle'] = src
        rec['soc_in'] = soc; rec['soh_in'] = soh
        v_pre, v_min = float(r['V_pre_V']), float(r['V_min_V'])
        kf, ks = K[cyc]
        if SOLVER == 'hull':
            i0, n0, s0 = hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, 1.0, 1.0, charge, EV.r_eff)
            i3, n3, s3 = hull_solver.solve(surf, soc, soh, v_pre, v_min, tau, kf, ks, charge, EV.r_eff)
            rec.update(n_roots_ecm=n0, n_roots_hyb=n3, st_ecm=s0, st_hyb=s3)
        else:
            i0 = EV.solve_I(surf, soc, soh, v_pre, v_min, tau, 1.0, 1.0, I0=meas, charge=charge)
            i3 = EV.solve_I(surf, soc, soh, v_pre, v_min, tau, kf, ks, I0=meas, charge=charge)
        if not (np.isfinite(i0) and np.isfinite(i3)):
            rec['status'] = 'hull'; out.append(rec); continue
        rec.update(meas=abs(meas), hyb=abs(i3), ecm=abs(i0), kf=kf, ks=ks)
        out.append(rec)
    return (level, outer, cell, direction, arm, rule), out


def main():
    rules = sys.argv[1:] or ['replace']
    tasks = []
    for rule in rules:
        for direction in ('discharge', 'charge'):
            for arm in ARMS:
                for c in CELLS:
                    tasks.append(('outer', 'NONE', c, direction, arm, rule))
                    for j in CELLS:
                        if j != c:
                            tasks.append(('inner', c, j, direction, arm, rule))
    print(f'  {len(tasks)} tasks', flush=True)
    with mp.Pool(int(os.environ.get('NPROC', '14'))) as pool:
        res = pool.starmap(rows_for, tasks)
    for rule in rules:
        D = {k: v for k, v in res if k[5] == rule}
        pickle.dump(D, open(os.path.join(HERE, f'arm_rows_{rule}{OUTTAG}.pkl'), 'wb'))
        n = sum(len(v) for v in D.values())
        print(f'  {rule}: {len(D)} blocks, {n} rows -> arm_rows_{rule}{OUTTAG}.pkl', flush=True)


if __name__ == '__main__':
    main()
