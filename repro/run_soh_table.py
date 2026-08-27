"""Build results/tables/soh.csv from the stored SOH predictions.

    python3 repro/run_soh_table.py
    python3 repro/run_soh_table.py --check   # verify without writing

soh.csv carried two verify.py checks (soh.rmse, soh.bias) but had no
producer in the repository — the numbers could not be traced to a command.
This is that producer.  It reduces results/soh_pred.npz, which the `soh`
training stage writes, into the per-cell table.

Per-cell rows matter: the aggregate RMSE hides that BOOST_REST is roughly
four times the pooled value, and a paper that quotes only the aggregate is
quoting its best case.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, 'analysis', 'results', 'soh_pred.npz')
OUT = os.path.join(ROOT, 'analysis', 'results', 'tables', 'soh.csv')

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST',
         'CC', 'CC_CELL2']
HEADER = ['cell', 'n', 'rmse', 'mae', 'bias']


def rows(pred_path=PRED):
    z = np.load(pred_path)
    missing = [c for c in CELLS
               if f'{c}_y' not in z or f'{c}_pred' not in z]
    if missing:
        raise KeyError(f'{pred_path} has no predictions for: {missing}')
    out, ys, ps = [], [], []
    for c in CELLS:
        y, p = np.asarray(z[f'{c}_y'], float), np.asarray(z[f'{c}_pred'], float)
        if y.shape != p.shape:
            raise ValueError(f'{c}: y {y.shape} vs pred {p.shape}')
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError(f'{c}: non-finite values in predictions')
        e = p - y
        ys.append(y)
        ps.append(p)
        out.append([c, len(y), f'{np.sqrt(np.mean(e ** 2)):.4f}',
                    f'{np.mean(np.abs(e)):.4f}', f'{e.mean():+.4f}'])
    y, p = np.concatenate(ys), np.concatenate(ps)
    e = p - y
    out.append(['ALL', len(y), f'{np.sqrt(np.mean(e ** 2)):.4f}',
                f'{np.mean(np.abs(e)):.4f}', f'{e.mean():+.4f}'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=PRED)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--check', action='store_true',
                    help='compare with the existing table instead of writing')
    a = ap.parse_args()

    if not os.path.exists(a.pred):
        print(f'  missing input: {a.pred}', file=sys.stderr)
        return 1
    r = rows(a.pred)

    if a.check:
        if not os.path.exists(a.out):
            print(f'  missing table: {a.out}', file=sys.stderr)
            return 1
        old = list(csv.reader(open(a.out, encoding='utf-8')))
        new = [HEADER] + [[str(x) for x in row] for row in r]
        if old != new:
            print('  MISMATCH between stored soh.csv and soh_pred.npz',
                  file=sys.stderr)
            for i, (o, n) in enumerate(zip(old, new)):
                if o != n:
                    print(f'    line {i}: stored={o} recomputed={n}',
                          file=sys.stderr)
            return 1
        print(f'  soh.csv matches {os.path.relpath(a.pred, ROOT)} '
              f'({len(r)} rows)')
        return 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(r)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(r)} rows)')
    worst = max(r[:-1], key=lambda x: float(x[2]))
    print(f'  worst cell: {worst[0]}  RMSE {worst[2]}  '
          f'({float(worst[2]) / float(r[-1][2]):.1f}x the pooled value)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
