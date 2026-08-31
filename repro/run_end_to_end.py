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


def load_keyed(path):
    """load() plus a per-row identity key, so corners can be intersected.

    The key has to name the physical pulse, and it cannot contain SOC: SOC is
    the quantity being perturbed, so keying on it would make every row of the
    estimated-SOC corners a different row by construction.  What does not move
    is what the cell actually did -- the cycle, the horizon, the pre-pulse
    voltage and the measured current.  Those four are unique per row inside
    each corner (checked below), so they identify a pulse across corners.
    """
    from run_safety import load
    d = load(path)
    r = list(csv.DictReader(open(path, encoding='utf-8')))
    k = np.array([f"{x['cell']}|{x['cycle']}|{float(x['tau_s']):.1f}"
                  f"|{x['V_pre_V']}|{x['I_meas_A']}" for x in r])
    if len(set(k)) != len(k):
        raise RuntimeError(
            f'{os.path.basename(path)}: pulse key is not unique '
            f'({len(k)} rows, {len(set(k))} keys) -- the paired comparison '
            f'would silently compare different row counts')
    d['key'] = k
    return d


def subset(d, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v)
            for k, v in d.items()}


def drift(corner_data, direction, tol, out_rows):
    """Score the rows a corner keeps that the intersection does not.

    The paired table answers "what does estimated state cost on the same
    pulse".  It cannot answer the other half, because the filter that decides
    which pulses are evaluated at all -- the trustworthy-label test, which is
    an extrapolation distance measured against SOC -- is itself computed from
    the estimated SOC.  A wrong SOC therefore admits pulses whose label should
    have been rejected.  Those rows are exactly the ones the paired view drops,
    so they have to be reported separately rather than averaged away.

    strict() still needs MIN_TRAIN rows per held-out cell, so a corner whose
    extra rows are spread thin scores none of them; `scored` is printed next
    to `outside` so that case is visible instead of reading as zero risk.
    """
    from run_safety import keep
    from run_safety_strict import strict

    for tau in TAUS:
        masks = {lab: keep(d, tau) for lab, d in corner_data.items()}
        common = None
        for lab, d in corner_data.items():
            ks = set(d['key'][masks[lab]])
            common = ks if common is None else (common & ks)
        for lab, d in corner_data.items():
            kept = set(d['key'][masks[lab]])
            extra = kept - common
            sel = masks[lab] & np.isin(d['key'], list(extra) or [''])
            recs = strict(subset(d, sel), tau, tol) if sel.sum() else []
            n = sum(r['n'] for r in recs)
            k = sum(r['exceed'] for r in recs)
            out_rows.append([
                lab, direction, f'{tau:.1f}', len(kept), len(common),
                int(sel.sum()), n, k,
                f'{100 * k / n:.2f}' if n else ''])
            print(f'  {lab:<32}{direction:<10}{tau:>5.0f}{len(kept):>7}'
                  f'{int(sel.sum()):>9}{n:>8}{k:>8}'
                  f'{(100 * k / n if n else float("nan")):>9.2f}')


def paired_fixed_lambda(corner_data, direction, tol, out_rows,
                        base='oracle SOH + oracle SOC'):
    """Paired, but with lambda frozen at what the oracle corner calibrates.

    The paired table refits lambda inside every corner, which answers "how
    well could this corner be calibrated" and therefore confounds the state
    error with the recalibration that hides it.  A deployed system does not
    get to recalibrate: lambda is fitted offline on the best labels available
    and then the vehicle runs with whatever state its filters produce.  So the
    per-cell lambda from the oracle corner is carried across unchanged and
    every corner is scored under it, on the same rows.

    This is the comparison the safety claim actually needs.  The refitting
    version stays, because the gap between the two is itself the measurement:
    it says how much of the apparent robustness came from re-tuning.
    """
    from run_safety import keep
    from run_safety_strict import strict, clopper_pearson_upper

    for tau in TAUS:
        masks = {lab: keep(d, tau) for lab, d in corner_data.items()}
        common = None
        for lab, d in corner_data.items():
            ks = set(d['key'][masks[lab]])
            common = ks if common is None else (common & ks)
        if not common or base not in corner_data:
            continue
        sel0 = masks[base] & np.isin(corner_data[base]['key'], list(common))
        lam_by_cell = {r['cell']: r['lam']
                       for r in strict(subset(corner_data[base], sel0), tau, tol)}
        if not lam_by_cell:
            continue
        for lab, d in corner_data.items():
            sel = masks[lab] & np.isin(d['key'], list(common))
            sub = subset(d, sel)
            n = k = 0
            worst = 0.0
            us, wts, per = [], [], []
            for c, lam in sorted(lam_by_cell.items()):
                m = sub['cell'] == c
                if not m.any():
                    continue
                pred, meas = sub['hyb'][m], sub['meas'][m]
                over = lam * pred - meas
                ni, ki = int(m.sum()), int((over > tol).sum())
                n += ni
                k += ki
                worst = max(worst, float(over.max()))
                us.append(float(np.median(lam * pred / meas) * 100))
                wts.append(float(ni))
                per.append(f'{c}:{ki}/{ni}')
            if not n:
                continue
            out_rows.append([
                lab, direction, f'{tau:.1f}', n, k,
                f'{clopper_pearson_upper(k, n) * 100:.2f}',
                f'{max(worst, 0.0):.3f}',
                f'{np.average(us, weights=wts):.2f}',
                f'{min(us):.2f}',
                f'{np.median(list(lam_by_cell.values())):.4f}',
                ' '.join(per)])
            print(f'  {lab:<32}{direction:<10}{tau:>5.0f}{n:>6}{k:>5}'
                  f'{clopper_pearson_upper(k, n) * 100:>9.2f}'
                  f'{max(worst, 0.0):>9.3f}'
                  f'{np.average(us, weights=wts):>10.2f}   {" ".join(per)}')


def paired(corner_data, direction, tol, out_rows):
    """Score every corner on the rows all four corners keep.

    keep() drops rows on extrap, on a non-finite prediction and on
    meas <= 0.5 A.  Shifting SOC moves rows across those thresholds, so the
    four corners are not scored on the same rows unless we intersect first.
    Calibration is restricted too: a paired comparison has to hold the row
    set fixed for the lambda fit as well as for the test.
    """
    from run_safety import keep
    from run_safety_strict import strict, clopper_pearson_upper

    for tau in TAUS:
        masks = {lab: keep(d, tau) for lab, d in corner_data.items()}
        common = None
        for lab, d in corner_data.items():
            ks = set(d['key'][masks[lab]])
            common = ks if common is None else (common & ks)
        if not common:
            print(f'  no common rows at tau={tau:.0f}', file=sys.stderr)
            continue
        sizes = set()
        for lab, d in corner_data.items():
            sel = masks[lab] & np.isin(d['key'], list(common))
            sizes.add(int(sel.sum()))
            recs = strict(subset(d, sel), tau, tol)
            if not recs:
                continue
            n = sum(r['n'] for r in recs)
            k = sum(r['exceed'] for r in recs)
            us = np.array([r['usable'] for r in recs])
            w = np.array([r['n'] for r in recs], float)
            worst_over = max(r['worst'] for r in recs)
            per_cell = ' '.join(f"{r['cell']}:{r['exceed']}/{r['n']}"
                                for r in recs)
            out_rows.append([
                lab, direction, f'{tau:.1f}', n, k,
                f'{clopper_pearson_upper(k, n) * 100:.2f}',
                f'{worst_over:.3f}',
                f'{np.average(us, weights=w):.2f}',
                f'{us.min():.2f}',
                f'{np.median([r["lam"] for r in recs]):.4f}', per_cell])
            print(f'  {lab:<32}{direction:<10}{tau:>5.0f}{n:>6}{k:>5}'
                  f'{clopper_pearson_upper(k, n) * 100:>9.2f}'
                  f'{worst_over:>9.3f}'
                  f'{np.average(us, weights=w):>10.2f}   {per_cell}')
        if len(sizes) != 1:
            raise RuntimeError(
                f'paired intersection at tau={tau:.0f} left different row '
                f'counts per corner: {sorted(sizes)} -- the key is not '
                f'identifying the same pulses')


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
    ap.add_argument('--paired-out', default=os.path.join(
        TABLES, 'end_to_end_paired.csv'))
    a = ap.parse_args()
    from run_safety import TOL
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
    held = collections.defaultdict(dict)
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
            d = load_keyed(path)
            held[direction][label] = d
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

    prows = []
    print('\n  PAIRED — every corner scored on the rows all four keep\n')
    print(f"  {'condition':<32}{'dir':<10}{'tau':>5}{'n':>6}{'exc':>5}"
          f"{'CP95 %':>9}{'worst A':>9}{'usable %':>10}   per-cell exceed/n")
    print('  ' + '-' * 110)
    for direction in ('discharge', 'charge'):
        if len(held[direction]) == len(CORNERS):
            paired(held[direction], direction, TOL[direction], prows)
        else:
            print(f'  {direction}: only {len(held[direction])}/{len(CORNERS)} '
                  f'corners available, paired comparison skipped',
                  file=sys.stderr)

    frows = []
    print('\n  PAIRED, LAMBDA FIXED AT THE ORACLE CORNER\'S CALIBRATION\n')
    print(f"  {'condition':<32}{'dir':<10}{'tau':>5}{'n':>6}{'exc':>5}"
          f"{'CP95 %':>9}{'worst A':>9}{'usable %':>10}   per-cell exceed/n")
    print('  ' + '-' * 110)
    for direction in ('discharge', 'charge'):
        if len(held[direction]) == len(CORNERS):
            paired_fixed_lambda(held[direction], direction, TOL[direction],
                                frows)

    drows = []
    print('\n  DRIFT — rows a corner keeps that the intersection does not\n')
    print(f"  {'condition':<32}{'dir':<10}{'tau':>5}{'kept':>7}{'outside':>9}"
          f"{'scored':>8}{'exceed':>8}{'rate %':>9}")
    print('  ' + '-' * 88)
    for direction in ('discharge', 'charge'):
        if len(held[direction]) == len(CORNERS):
            drift(held[direction], direction, TOL[direction], drows)

    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, 'end_to_end_fixed_lambda.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'direction', 'tau_s', 'n_rows', 'exceed',
                    'exceed_rate_upper95_pct', 'worst_overshoot_A',
                    'usable_mean_pct', 'usable_worst_pct',
                    'lambda_median_from_oracle', 'per_cell_exceed'])
        w.writerows(frows)

    with open(os.path.join(TABLES, 'end_to_end_drift.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'direction', 'tau_s', 'n_kept', 'n_common',
                    'n_outside', 'n_scored', 'exceed', 'exceed_rate_pct'])
        w.writerows(drows)

    with open(a.paired_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'direction', 'tau_s', 'n_rows', 'exceed',
                    'exceed_rate_upper95_pct', 'worst_overshoot_A',
                    'usable_mean_pct', 'usable_worst_pct', 'lambda_median',
                    'per_cell_exceed'])
        w.writerows(prows)
    print(f'  -> {os.path.relpath(a.paired_out, ROOT)}  ({len(prows)} rows)')

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
