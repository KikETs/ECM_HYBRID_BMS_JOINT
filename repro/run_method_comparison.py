"""Is the A8 trim better than the baselines, or only competitive with them?

The paper used to say "equivalence".  That word is not available: an
equivalence claim needs a margin fixed before the data is seen and a formal
noninferiority test, and neither exists here.  Choosing a margin now, after
the numbers are in, would be the same defect as choosing a model on the test
set.

What can be said is descriptive, so this table says exactly that and no more.
For every method scored by run_safety_strict.py it reports the safety-adjusted
usable current with its cell-cluster bootstrap interval, and whether that
interval overlaps A8's.  Overlap is not a test and is not evidence of
equivalence -- it is the weaker statement that the data does not separate the
two, which is all the paper is entitled to.

The ranking column exists because the ranking moves: A8 is not uniformly first
across direction and horizon, and a reader should be able to see that in one
place rather than reconstruct it from six files.
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')
REF = 'a8'


def rows_for(path):
    out = []
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r.get('soh_arm') != 'oracle':
            continue
        out.append(dict(
            method=r.get('method', 'a8'),
            direction=r['direction'], tau=r['tau_s'],
            n=int(r['n_rows']), exceed=int(r['exceed']),
            usable=float(r['usable_mean_pct']),
            lo=float(r['usable_boot_lo_pct']),
            hi=float(r['usable_boot_hi_pct']),
            worst=float(r['usable_worst_pct'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(TABLES,
                                                  'method_comparison.csv'))
    a = ap.parse_args()

    found = []
    for p in sorted(glob.glob(os.path.join(TABLES,
                                           'safety_strict_*oracle.csv'))):
        b = os.path.basename(p)
        if 'percell' in b or 'tolsens' in b:
            continue
        found += rows_for(p)
    if not found:
        print('  no safety_strict_*_oracle.csv tables found', file=sys.stderr)
        return 1

    keys = sorted({(r['direction'], r['tau']) for r in found})
    rows = []
    print(f"  {'dir':<10}{'tau':>5}  {'method':<9}{'usable %':>10}"
          f"{'boot 95 %':>18}{'exceed':>8}{'rank':>6}   vs A8")
    print('  ' + '-' * 84)
    for direction, tau in keys:
        here = [r for r in found if r['direction'] == direction
                and r['tau'] == tau]
        ref = next((r for r in here if r['method'] == REF), None)
        order = sorted(here, key=lambda r: -r['usable'])
        for r in order:
            rank = order.index(r) + 1
            if ref is None or r['method'] == REF:
                rel = '—'
            else:
                ov = not (r['hi'] < ref['lo'] or r['lo'] > ref['hi'])
                rel = ('overlaps' if ov else
                       ('separated, A8 higher' if ref['usable'] > r['usable']
                        else 'separated, A8 lower'))
            rows.append([direction, tau, r['method'], f"{r['usable']:.2f}",
                         f"{r['lo']:.2f}", f"{r['hi']:.2f}", r['n'],
                         r['exceed'], f"{r['worst']:.2f}", rank, rel])
            ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
            print(f"  {direction:<10}{float(tau):>5.0f}  {r['method']:<9}"
                  f"{r['usable']:>10.2f}{ci:>18}"
                  f"{r['exceed']:>8}{rank:>6}   {rel}")

    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'tau_s', 'method', 'usable_mean_pct',
                    'boot_lo_pct', 'boot_hi_pct', 'n_rows', 'exceed',
                    'usable_worst_pct', 'rank_in_condition',
                    'interval_vs_a8'])
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')

    ranks = {}
    for direction, tau in keys:
        here = sorted([r for r in found if r['direction'] == direction
                       and r['tau'] == tau], key=lambda r: -r['usable'])
        for i, r in enumerate(here):
            ranks.setdefault(r['method'], []).append(i + 1)
    a8r = ranks.get(REF, [])
    print(f"  A8's rank across the {len(keys)} conditions: {a8r}")
    sep = [r for r in rows if r[10].startswith('separated')]
    print(f'  intervals that separate from A8: {len(sep)} of '
          f'{len([r for r in rows if r[2] != REF])}')
    print('  Overlap is not equivalence.  It means the data does not separate '
          'the two, which is weaker and is what the paper may say.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
