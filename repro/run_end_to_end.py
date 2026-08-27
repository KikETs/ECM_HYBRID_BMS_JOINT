"""Causal end-to-end evaluation: estimated SOH and estimated SOC into SOP.

    python3 repro/run_end_to_end.py

Every SOP number in this project feeds the SOP inversion the LABEL's own SOC
and, until section 29, the label's own SOH.  A deployed BMS has neither.
This runs the four corners:

    oracle SOH    + oracle SOC       what the paper has been reporting
    estimated SOH + oracle SOC       the section 29 result
    oracle SOH    + estimated SOC    new
    estimated SOH + estimated SOC    the only one a vehicle can actually do

plus a stale-SOH condition, because a car does not get an adequate partial
charge before every pulse.

How SOC estimation is carried in
    The adopted EKF is run over each drive run in results/soc_runs.pkl and
    its TERMINAL SOC error is recorded.  That is the error the filter is
    carrying when the characterisation that follows begins.  Each SOP label
    then takes the error of the nearest drive run at or before its cycle -
    nearest-preceding, not interpolated, because a filter does not average
    its future.

    Honest limit: soc_runs.pkl holds 6 runs per cell while SOP labels span
    far more cycles, so one measured error is reused across a stretch of
    cycles.  It is a real EKF error at a real SOC and aging state, not a
    synthetic offset, but it is coarse in cycle.  Stated in the output.

Not modelled, and therefore not claimed
    Sensor dropout and fallback logic, temperature-sensor error, and the
    within-characterisation drift between the pulse and the end of the
    preceding drive.  A missing partial-charge observation is covered only
    through the stale-SOH condition.
"""
import argparse
import collections
import csv
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

EVAL = os.path.join(ANALYSIS, 'results', 'eval')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')
SOC_ERR = os.path.join(ANALYSIS, 'results', 'soc_err.npz')
SOH_PRED = os.path.join(ANALYSIS, 'results', 'soh_pred.npz')
RUNSPKL = os.path.join(ANALYSIS, 'results', 'soc_runs.pkl')

TRIM = {'discharge': 'runs_trim_a8', 'charge': 'runs_trim_a8_chg'}
TAUS = (10.0, 2.0)


def build_soc_errors(out=SOC_ERR, ibias=0.0):
    """Terminal SOC error of the adopted EKF on each drive run."""
    from ekf_soc import run as ekf_run
    R = pickle.load(open(RUNSPKL, 'rb'))
    per = collections.defaultdict(list)
    for r in R:
        rv = float(np.interp(r['soh'], [0.70, 0.90, 1.00],
                             [0.110, 0.035, 0.015]))
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], r['I'] + ibias, r['V'],
                         r['T'], float(r['soc'][0]), rv, gamma=0.0,
                         i_gate=1.0, rest_hold_s=30.0)
        # The terminal quarter, not the last sample: one sample is noise.
        k = max(1, len(est) // 4)
        err = float(np.mean(est[-k:] - r['soc'][-k:]))
        per[r['cell']].append((int(r['cyc']), err))
    d = {}
    for c, v in per.items():
        v.sort()
        d[f'{c}_cycle'] = np.array([x[0] for x in v], int)
        d[f'{c}_err'] = np.array([x[1] for x in v], float)
    np.savez(out, **d)
    allerr = np.concatenate([d[k] for k in d if k.endswith('_err')])
    print(f'  SOC errors: {len(allerr)} runs, mean {allerr.mean():+.4f}, '
          f'|max| {np.abs(allerr).max():.4f}  -> {os.path.basename(out)}')
    return out


def run_eval(name, direction, soh_est, soc_est, agg='max'):
    tag = os.path.join(EVAL, f'{name}.csv')
    cmd = [sys.executable, 'eval_sop_amps.py', '--direction', direction,
           '--trim', TRIM[direction], '--trim-agg', agg, '--out', tag]
    if soh_est:
        cmd += ['--soh-est', soh_est]
    if soc_est:
        cmd += ['--soc-est', soc_est]
    p = subprocess.run(cmd, cwd=ANALYSIS, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(tag):
        print(f'  FAILED {name}', file=sys.stderr)
        for ln in (p.stderr or '').strip().splitlines()[-8:]:
            print(f'      | {ln}', file=sys.stderr)
        return None
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(TABLES, 'end_to_end.csv'))
    a = ap.parse_args()
    from run_safety import load, keep, TOL
    from run_safety_strict import strict

    if not os.path.exists(RUNSPKL):
        print(f'  missing {RUNSPKL} — run repro/build_soc_runs.py',
              file=sys.stderr)
        return 1
    build_soc_errors()

    CORNERS = [
        ('oracle SOH + oracle SOC', None, None),
        ('estimated SOH + oracle SOC', SOH_PRED, None),
        ('oracle SOH + estimated SOC', None, SOC_ERR),
        ('estimated SOH + estimated SOC', SOH_PRED, SOC_ERR),
    ]

    rows = []
    print(f"\n  {'condition':<32}{'dir':<10}{'tau':>5}{'n':>6}{'exc':>5}"
          f"{'lambda':>9}{'usable %':>10}{'worst %':>9}  worst cell")
    print('  ' + '-' * 92)
    for label, soh_e, soc_e in CORNERS:
        for direction in ('discharge', 'charge'):
            slug = (label.replace(' ', '_').replace('+', '')
                    .replace('__', '_'))
            name = f'e2e_{slug}_{direction[:4]}'
            path = run_eval(name, direction, soh_e, soc_e)
            if path is None:
                continue
            d = load(path)
            for tau in TAUS:
                recs = strict(d, tau, TOL[direction])
                if not recs:
                    continue
                n = sum(r['n'] for r in recs)
                k = sum(r['exceed'] for r in recs)
                us = np.array([r['usable'] for r in recs])
                ws = np.array([r['n'] for r in recs], float)
                lam = np.median([r['lam'] for r in recs])
                wi = int(np.argmin(us))
                rows.append([label, direction, f'{tau:.1f}', n, k,
                             f'{lam:.4f}',
                             f'{np.average(us, weights=ws):.2f}',
                             f'{us[wi]:.2f}', recs[wi]['cell']])
                print(f'  {label:<32}{direction:<10}{tau:>5.0f}{n:>6}{k:>5}'
                      f'{lam:>9.4f}{np.average(us, weights=ws):>10.2f}'
                      f'{us[wi]:>9.2f}  {recs[wi]["cell"]}')

    os.makedirs(TABLES, exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'direction', 'tau_s', 'n_rows', 'exceed',
                    'lambda_median', 'usable_mean_pct', 'usable_worst_pct',
                    'worst_cell'])
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')
    print('  Only the last corner is what a vehicle can do.  Any row above '
          'it still receives a ground truth the vehicle does not have.')
    print('  Not modelled: sensor dropout, fallback logic, temperature-sensor '
          'error, drift between the pulse and the end of the preceding drive.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
