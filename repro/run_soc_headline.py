"""Make the SOC headline number explicit and checkable.

    python3 repro/run_soc_headline.py

The README quotes "2.05 %p under sensor perturbation" and sec 30.9 describes
it as the average over initial-SOC error, current offset and gain error.  It
is not: 2.05 is the mean over all **seven** rows of soc_perturb.csv, and the
seventh row is the *undisturbed* case.  The mean over the six disturbance
rows is 2.14, which is what sec 30.4's own table prints.

Nothing pinned 2.05 to a table, so verify.py never saw the discrepancy.
This writes every defensible aggregation side by side so the paper can pick
one and be checked against it.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'analysis', 'results', 'tables', 'soc_perturb.csv')
OUT = os.path.join(ROOT, 'analysis', 'results', 'tables', 'soc_headline.csv')

UNDISTURBED = 'no distortion'
PAIRS = {
    'initial SOC': ['initial SOC +10 %p', 'initial SOC -10 %p'],
    'current offset': ['current offset +0.10 A', 'current offset -0.10 A'],
    'current gain': ['current gain +1 %', 'current gain -1 %'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print(f'  missing {a.src} — run the soc stage', file=sys.stderr)
        return 1
    r = list(csv.DictReader(open(a.src, encoding='utf-8')))
    configs = sorted({x['config'] for x in r})
    rows = []
    for cfg in configs:
        f = {x['perturbation']: float(x['rmse_full_pct'])
             for x in r if x['config'] == cfg}
        if UNDISTURBED not in f:
            continue
        dist = [v for k, v in f.items() if k != UNDISTURBED]
        pair = [np.mean([f[k] for k in v]) for v in PAIRS.values()
                if all(k in f for k in v)]
        rows.append([
            cfg, len(f),
            f'{f[UNDISTURBED]:.3f}',
            f'{np.mean(dist):.3f}',
            f'{np.mean(list(f.values())):.3f}',
            f'{np.median(dist):.3f}',
            f'{max(dist):.3f}',
            f'{np.mean(pair):.3f}' if pair else '',
        ])
    hdr = ['config', 'n_rows', 'undisturbed_pct',
           'mean_6_disturbances_pct', 'mean_7_rows_pct',
           'median_6_pct', 'worst_of_6_pct', 'mean_of_3_pair_means_pct']
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')
    print(f"\n  {'config':<26}{'undist':>8}{'mean6':>8}{'mean7':>8}"
          f"{'med6':>8}{'worst6':>8}")
    print('  ' + '-' * 66)
    for x in rows:
        print(f'  {x[0]:<26}{x[2]:>8}{x[3]:>8}{x[4]:>8}{x[5]:>8}{x[6]:>8}')
    print('\n  The published 2.05 %p is mean7 (it includes the undisturbed '
          'row).\n  The six-disturbance mean that sec 30.4 prints is 2.14 %p.'
          '\n  Quote one and label it; they are not the same quantity.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
