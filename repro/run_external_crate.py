"""Does the frozen safety factor survive a change of prior load, at 0 C?

RPCWBY Test#8 measures SOP on the same cell at 0 C after discharging at
0, C/3, 1C, 2C, 3C and 4C, over thirteen SOC points from 0.95 down to 0.05.
That is the one axis Test#3 does not sweep, and it matters here for a
specific reason: 0 C is exactly where the frozen lambda stopped having room.
external_temp_envelope.csv puts the margin at 1.396 there, against 1.447 at
40 C and 0.878 at -10 C.  If prior load moves the requirement at all, it
moves it at the temperature with the least to give.

Same caveat as the temperature sweep, and it is the whole scope of the
result: Test#8 carries no paired drive cycle, so the A8 trim cannot be
computed on it.  What is scored is the nominal 2RC layer the trim sits on.

The C-rate column is the rate the cell was discharged at BEFORE the pulse,
not the pulse current.  So this asks whether recent history changes what the
physics layer owes, which is the assumption the trim's exponentially-weighted
features exist to relax.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

LAB = os.path.join(ANALYSIS, 'rpcwby_sop_test8.csv')
OUT = os.path.join(ANALYSIS, 'results', 'tables', 'external_crate_envelope.csv')
LAM_SHIPPED = 0.6832


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='CC')
    ap.add_argument('--tau', type=float, default=10.0)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(LAB):
        print(f'  missing {os.path.relpath(LAB, ROOT)}', file=sys.stderr)
        return 1

    from ecm_surface import ECMSurface
    from run_chen2026_baseline import chen_search, tcell_map, SOH_25C
    from run_safety_strict import clopper_pearson_upper

    surf = ECMSurface(a.holdout, 'discharge')
    T = tcell_map().get(0, 0.0)
    soh = SOH_25C

    def ocv_fn(soc):
        v, _ = surf.ocv(soc, soh)
        m, _ = surf.hyst_M(soc, soh)
        return float(np.atleast_1d(v)[0]), float(np.atleast_1d(m)[0])

    rows = [r for r in csv.DictReader(open(LAB, encoding='utf-8'))
            if r['SOP_disch'] not in ('', 'nan')]
    by_rate = collections.defaultdict(list)
    for r in rows:
        soc = float(r['SOC'])
        meas = abs(float(r['SOP_disch']))
        pred, lim = chen_search(surf, soc, soh, T, a.tau, ocv_fn)
        by_rate[r['rate_label']].append((soc, meas, pred, lim))

    out = []
    print(f'  Test#8, 0 C, tau = {a.tau:g} s, surface {a.holdout}, '
          f'{len(rows)} measured rows')
    print(f"  {'prior rate':>11}{'n':>5}{'in hull':>9}{'SOC range':>13}"
          f"{'lam need':>10}{'margin':>8}{'exceed':>8}{'ub95 %':>9}"
          f"{'worst W':>9}")
    print('  ' + '-' * 84)
    for label in ('0C', 'C/3', '1C', '2C', '3C', '4C'):
        g = by_rate.get(label, [])
        ih = [x for x in g if np.isfinite(x[2]) and x[3] != 'out-of-hull']
        if not ih:
            out.append([label, len(g), 0, '', '', '', '', '', '', ''])
            continue
        socs = sorted(x[0] for x in ih)
        m = np.array([x[1] for x in ih])
        q = np.array([x[2] for x in ih])
        ok = q > 0
        need = float(np.min(m[ok] / q[ok])) if ok.any() else float('nan')
        over = LAM_SHIPPED * q - m
        exceed = int((over > 0).sum())
        ub = clopper_pearson_upper(exceed, len(ih)) * 100
        out.append([label, len(g), len(ih), f'{min(socs):.2f}',
                    f'{max(socs):.2f}', f'{LAM_SHIPPED:.4f}', f'{need:.4f}',
                    f'{need / LAM_SHIPPED:.3f}', exceed, f'{ub:.1f}',
                    f'{max(over.max(), 0.0):.2f}'])
        print(f'  {label:>11}{len(g):>5}{len(ih):>9}'
              f'{f"{min(socs):.2f}-{max(socs):.2f}":>13}{need:>10.4f}'
              f'{need / LAM_SHIPPED:>8.3f}{exceed:>8}{ub:>9.1f}'
              f'{max(over.max(), 0.0):>9.2f}')

    HEADER = ['prior_rate', 'n_points', 'n_in_hull', 'soc_in_hull_min',
              'soc_in_hull_max', 'lambda_shipped', 'lambda_needed', 'margin',
              'exceed', 'exceed_ub95_pct', 'worst_overshoot_W']
    full = [r for r in out if len(r) == len(HEADER)]
    if full:
        n = sum(int(r[2]) for r in full)
        k = sum(int(r[8]) for r in full)
        mins = [float(r[6]) for r in full]
        print(f'\n  pooled over all prior rates: {k} exceedances in {n} '
              f'in-hull points, 95 % upper bound '
              f'{clopper_pearson_upper(k, n) * 100:.1f} %')
        print(f'  lambda_needed spans {min(mins):.4f}-{max(mins):.4f} across '
              f'the six rates; the shipped factor is {LAM_SHIPPED}')
        out.append(['0C..4C pooled', sum(int(r[1]) for r in full), n, '', '',
                    f'{LAM_SHIPPED:.4f}', f'{min(mins):.4f}',
                    f'{min(mins) / LAM_SHIPPED:.3f}', k,
                    f'{clopper_pearson_upper(k, n) * 100:.1f}', ''])

    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(a.out, HEADER, out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(out)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(out)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
