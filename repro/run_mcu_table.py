"""Build results/tables/mcu.csv from the board benchmark's per-sample CSV.

    python3 repro/run_mcu_table.py --bench mcu/sop_mcu_bench.csv
    python3 repro/run_mcu_table.py --check

mcu.csv carried four verify.py checks but had no producer: `bench_sop.py`
writes per-sample rows to mcu/sop_mcu_bench.csv and prints an aggregate to
the terminal, and results/tables/mcu.csv was never written by anything.
This closes that gap so the board numbers trace to a command and a file.

The per-sample CSV is the artifact of record.  Re-running this reducer does
not touch the board; to re-measure, run mcu/bench_sop.py against the
hardware first.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, 'mcu', 'sop_mcu_bench.csv')
OUT = os.path.join(ROOT, 'analysis', 'results', 'tables', 'mcu.csv')

# Reporting order, chosen to read cheap -> expensive within each arm.
ORDER = ['FEAT', 'FEAT_A8', 'EKF', 'EKF_P', 'REFF', 'TRIM', 'SOLVE', 'FULL']
HEADER = ['stage', 'n', 'median_us', 'p95_us', 'max_us']


def rows(bench_path=BENCH, order=ORDER):
    r = list(csv.DictReader(open(bench_path, encoding='utf-8')))
    if not r:
        raise ValueError(f'{bench_path} is empty')
    for col in ('cmd', 'us'):
        if col not in r[0]:
            raise KeyError(f'{bench_path} has no {col!r} column '
                           f'(found {list(r[0])})')
    out = []
    for c in order:
        us = np.array([float(x['us']) for x in r if x['cmd'] == c])
        if us.size == 0:
            raise ValueError(f'{bench_path} has no rows for stage {c!r} — '
                             f'the board run was partial, not complete')
        out.append([c, us.size, f'{np.median(us):.2f}',
                    f'{np.percentile(us, 95):.2f}', f'{us.max():.2f}'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default=BENCH)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(a.bench):
        print(f'  missing board benchmark: {a.bench}\n'
              f'  run mcu/bench_sop.py against the hardware first',
              file=sys.stderr)
        return 1
    r = rows(a.bench)

    if a.check:
        if not os.path.exists(a.out):
            print(f'  missing table: {a.out}', file=sys.stderr)
            return 1
        old = list(csv.reader(open(a.out, encoding='utf-8')))
        new = [HEADER] + [[str(x) for x in row] for row in r]
        if old != new:
            print('  MISMATCH between stored mcu.csv and the bench CSV',
                  file=sys.stderr)
            for i, (o, n) in enumerate(zip(old, new)):
                if o != n:
                    print(f'    line {i}: stored={o} recomputed={n}',
                          file=sys.stderr)
            return 1
        print(f'  mcu.csv matches {os.path.relpath(a.bench, ROOT)} '
              f'({len(r)} stages)')
        return 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(r)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(r)} stages)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
