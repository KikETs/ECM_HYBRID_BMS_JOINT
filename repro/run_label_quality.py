"""SOP label reliability: how much of the target is measured, and does the
model ranking survive dropping the parts that are not.

    python3 repro/run_label_quality.py

**Why.**  The UYPYDJ SOP targets are not direct SOP measurements.  A 30 A
cycler cannot reach the discharge current the cell can take, so I* is a
projection of a fit through four HPPC pulse rates down to the voltage floor.
`extrap = |I*| / max|I_measured|` is how far past the measured fan that
projection reaches.  Calling these "directly measured SOP labels" overstates
them; "pulse-derived current-limit reference" is what they are.

This stratifies the labels by extrap band, reports the fit diagnostics the
label file already carries (spread_A between the lin4 and lin2hi fits, and
fit_r2), and then re-runs the model comparison on progressively cleaner
subsets.  If the ranking of the methods moves when the weakly-supported rows
are removed, the ranking was a property of the extrapolation, not of the
models.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_safety import load, keep, TOL          # noqa: E402
from run_safety_strict import strict, fit_lambda   # noqa: E402

ANALYSIS = os.path.join(ROOT, 'analysis')
EVAL = os.path.join(ANALYSIS, 'results', 'eval')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')

BANDS = [('<=1  interpolated', -np.inf, 1.0),
         ('1-1.5 mild extrap', 1.0, 1.5),
         ('>1.5  strong extrap', 1.5, np.inf)]

LABELS = {'discharge': 'sop_label_measured.csv',
          'charge': 'sop_label_charge.csv'}

METHODS = [('A0  no correction', 'ecm'), ('A8  dR_fast alone', 'hyb')]


def band_stats(direction):
    p = os.path.join(ANALYSIS, LABELS[direction])
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p, encoding='utf-8')))

    def col(k):
        return np.array([float(x[k]) if x[k] not in ('', 'nan') else np.nan
                         for x in r])
    ex, sp, r2 = col('extrap'), col('spread_A'), col('fit_r2')
    i4, i2 = col('I_star_lin4_A'), col('I_star_lin2hi_A')
    tau = col('tau_s')
    out = []
    for name, lo, hi in BANDS:
        m = np.isfinite(ex) & (ex > lo) & (ex <= hi)
        if m.sum() == 0:
            continue
        rel = np.abs(sp[m]) / np.abs(i4[m])
        out.append([direction, name, int(m.sum()),
                    f'{100 * m.sum() / len(ex):.1f}',
                    f'{np.nanmedian(ex[m]):.3f}',
                    f'{np.nanmedian(np.abs(sp[m])):.3f}',
                    f'{np.nanmedian(rel) * 100:.1f}',
                    f'{np.nanmedian(r2[m]):.5f}',
                    f'{np.nanmin(r2[m]):.5f}',
                    int(np.isfinite(i2[m]).sum()),
                    f'{np.nanmedian(np.abs(i4[m])):.1f}'])
    return out


def ranking(direction, tau, extrap_max):
    """Usable current per method under strict per-cell lambda, at a given
    extrap ceiling."""
    tag = 'disc' if direction == 'discharge' else 'char'
    p = os.path.join(EVAL, f'a8_{tag}_oracle.csv')
    if not os.path.exists(p):
        return None
    d = load(p)
    tol = TOL[direction]
    rows = []
    for name, key in METHODS:
        # Reproduce keep() but with a movable extrap ceiling.
        m = (np.isfinite(d['meas']) & np.isfinite(d[key]) & (d['meas'] > 0.5)
             & (d['extrap'] <= extrap_max) & (np.round(d['tau'], 1) == tau))
        if m.sum() < 30:
            continue
        us, ns, ex = [], [], 0
        for c in sorted(set(d['cell'][m])):
            tr = m & (d['cell'] != c)
            te = m & (d['cell'] == c)
            if tr.sum() < 25 or te.sum() == 0:
                continue
            lam = fit_lambda(d[key][tr], d['meas'][tr], tol)
            pr, y = d[key][te], d['meas'][te]
            ex += int((lam * pr - y > tol).sum())
            us.append(float(np.median(lam * pr / y) * 100))
            ns.append(int(te.sum()))
        if not us:
            continue
        rows.append([direction, f'{tau:.1f}', f'{extrap_max:g}', name,
                     int(m.sum()), ex,
                     f'{np.average(us, weights=ns):.2f}',
                     f'{min(us):.2f}'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extrap-sweep', default='1.0,1.5,2.0,999')
    a = ap.parse_args()
    os.makedirs(TABLES, exist_ok=True)

    bands = []
    for direction in ('discharge', 'charge'):
        b = band_stats(direction)
        if b is None:
            print(f'  MISSING {LABELS[direction]} — run the label stage',
                  file=sys.stderr)
            return 1
        bands += b
    hdr = ['direction', 'extrap_band', 'n', 'share_pct', 'extrap_median',
           'spread_A_median', 'spread_rel_pct', 'fit_r2_median',
           'fit_r2_min', 'n_with_lin2hi', 'abs_I_star_median_A']
    p = os.path.join(TABLES, 'label_quality.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(bands)
    print(f'  -> {os.path.relpath(p, ROOT)}  ({len(bands)} rows)')
    print(f"\n  {'dir':<10}{'band':<22}{'n':>6}{'share':>8}{'extrap':>9}"
          f"{'spread A':>10}{'spread %':>10}{'r2 med':>9}{'r2 min':>9}")
    for b in bands:
        print(f'  {b[0]:<10}{b[1]:<22}{b[2]:>6}{b[3]:>7}%{b[4]:>9}'
              f'{b[5]:>10}{b[6]:>9}%{b[7]:>9}{b[8]:>9}')

    rank = []
    for direction in ('discharge', 'charge'):
        for tau in (10.0, 2.0):
            for e in [float(x) for x in a.extrap_sweep.split(',')]:
                r = ranking(direction, tau, e)
                if r:
                    rank += r
    p = os.path.join(TABLES, 'label_sensitivity.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'tau_s', 'extrap_max', 'method', 'n_rows',
                    'exceed', 'usable_mean_pct', 'usable_worst_pct'])
        w.writerows(rank)
    print(f'\n  -> {os.path.relpath(p, ROOT)}  ({len(rank)} rows)')
    print(f"\n  {'dir':<10}{'tau':>5}{'extrap<=':>9}{'method':<20}"
          f"{'n':>6}{'exc':>5}{'usable':>8}{'worst':>8}")
    for r in rank:
        print(f'  {r[0]:<10}{r[1]:>5}{r[2]:>9}{r[3]:<20}{r[4]:>6}{r[5]:>5}'
              f'{r[6]:>8}{r[7]:>8}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
