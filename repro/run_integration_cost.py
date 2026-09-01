"""Does adding per-stage timings match timing the stages together?

mcu_cycle.csv reports 339.84 us as the sum of each stage's observed maximum,
and 37.10 relabelled it a derived cycle-budget estimate rather than a WCET
because no integrated loop was ever timed end to end.  That is honest and it
leaves a question open: how wrong is the summation?

The firmware answers part of it already.  SOP_CMD_FULL runs the trim and the
solve inside ONE DWT window; SOP_CMD_TRIM and SOP_CMD_SOLVE time the same two
pieces separately, over the same 500 operating points.  That is the largest
pair the current firmware can integrate, it is 80 of the 340 us, and it is a
measurement rather than an assumption.

What it does NOT do is close the gap.  The EKF and the feature extraction are
still only ever timed alone, an integrated loop over all four would need
firmware this repository cannot build here (no arm-none-eabi toolchain on the
audit host), and cache and pipeline effects across a longer chain are exactly
what a two-stage pair cannot show.  This bounds the summation error where it
can be measured; it does not license calling the total a WCET.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, 'mcu', 'sop_mcu_bench.csv')
OUT = os.path.join(ROOT, 'analysis', 'results', 'tables',
                   'integration_cost.csv')
sys.path.insert(0, HERE)

HEADER = ['quantity', 'integrated_us', 'summed_us', 'delta_us', 'delta_pct']


def rows():
    recs = list(csv.DictReader(open(BENCH, encoding='utf-8')))
    by = collections.defaultdict(dict)
    for x in recs:
        by[(x['soc'], x['soh'], x['tau_s'])][x['cmd']] = float(x['us'])
    pairs = [(v['FULL'], v['TRIM'] + v['SOLVE']) for v in by.values()
             if {'FULL', 'TRIM', 'SOLVE'} <= set(v)]
    if not pairs:
        raise SystemExit(f'{BENCH} has no operating point carrying FULL, TRIM '
                         f'and SOLVE together')
    f = np.array([a for a, _ in pairs])
    s = np.array([b for _, b in pairs])
    d = s - f

    def row(name, i, j):
        return [name, f'{i:.3f}', f'{j:.3f}', f'{j - i:+.3f}',
                f'{100 * (j - i) / i:+.3f}']

    return [
        row('median over 500 points', float(np.median(f)), float(np.median(s))),
        row('maximum over 500 points', float(f.max()), float(s.max())),
        row('sum of the two stage maxima', float(f.max()),
            float(max(x['TRIM'] for x in by.values())
                  + max(x['SOLVE'] for x in by.values()))),
        ['worst single point', f'{f[int(np.argmax(d))]:.3f}',
         f'{s[int(np.argmax(d))]:.3f}', f'{d.max():+.3f}',
         f'{100 * (d / f).max():+.3f}'],
        ['points where summing is the larger', f'{len(f)}',
         f'{int((d > 0).sum())}', '', f'{100 * (d > 0).mean():.1f}'],
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    out = rows()

    print('  SOP_CMD_FULL (trim+solve in one DWT window) against the same two '
          'stages timed apart,\n  500 operating points, paired on '
          '(SOC, SOH, tau).\n')
    print(f"  {'quantity':<34}{'integrated':>12}{'summed':>10}"
          f"{'delta':>10}{'delta %':>10}")
    print('  ' + '-' * 78)
    for r in out:
        print(f'  {r[0]:<34}{r[1]:>12}{r[2]:>10}{r[3]:>10}{r[4]:>10}')
    print('\n  Summing is the larger figure at 88.4 % of operating points, '
          'by 0.21 % at the median\n  and 0.78 % at worst, and the quantity '
          'mcu_cycle.csv actually uses -- the sum of the\n  two stage maxima '
          '-- sits 0.32 % above the integrated maximum.')
    print('  It is NOT conservative pointwise.  At the single worst '
          'integrated point the paired\n  sum is 0.089 % BELOW it: the two '
          'maxima fall at different operating points, so\n  adding stage '
          'figures can under-report where integration is worst.  Small here, '
          'and\n  not a property to rely on.')
    print('  That bounds the summation error where it can be measured.  It '
          'does not extend to\n  the EKF or the feature stage, which are '
          'still only ever timed alone, and it does\n  not make 339.84 us a '
          'WCET.')

    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(OUT, HEADER, out, 'integration_cost')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(out)
    print(f'\n  -> {os.path.relpath(OUT, ROOT)}  ({len(out)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
