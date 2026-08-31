"""Nested model-family selection for the SOH arm.

Why this exists.  soh_baselines.csv shows ridge beating the shipped CNN on the
leave-one-cell-out splits.  Adopting ridge on that evidence would be choosing a
model by looking at the held-out cells -- the same selection-on-the-test-set
defect the trim's nested check (run_nested_selection.py) was written to catch.
The reported ridge RMSE would then be an optimistic number for a procedure
nobody ran.

So the family is chosen the way it would have to be chosen in practice: for
each outer held-out cell, every candidate family is scored by leave-one-cell-out
over the FIVE TRAINING CELLS ONLY, the winner is refitted on those five, and
only then does it see the held-out cell.  The pooled error of that procedure is
what an honest paper can report, and the per-fold choices say whether "use
ridge" is a stable conclusion or an artifact of one split.

Outputs
  results/tables/soh_nested.csv       per outer fold: chosen family, inner and
                                      outer error, and every family's score
  results/tables/soh_nested_summary.csv  the procedure's pooled error against
                                      always-ridge and always-CNN
"""
import argparse
import collections
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

DATA = os.path.join(ANALYSIS, 'cache', 'soh_charge.npz')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')

CNN_SEEDS = 3
CNN_EPOCHS = 300
CNN_LR = 3e-3


def sk_families():
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import HistGradientBoostingRegressor
    return [
        ('mean baseline', lambda p: 'mean', [None]),
        ('ridge', lambda p: Ridge(alpha=p),
         [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]),
        ('PLS', lambda p: PLSRegression(n_components=p),
         [1, 2, 3, 4, 6, 8, 12]),
        ('SVR (RBF)', lambda p: SVR(C=p, epsilon=1e-3),
         [1.0, 10.0, 100.0, 1000.0]),
        ('gradient boosting',
         lambda p: HistGradientBoostingRegressor(
             max_iter=300, learning_rate=p, random_state=0),
         [0.02, 0.05, 0.1, 0.2]),
    ]


def fit_predict_sk(make, p, Xtr, ytr, Xte):
    if make(p) == 'mean':
        return np.full(len(Xte), ytr.mean())
    m = make(p)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte)).ravel()


def fit_predict_cnn(Xtr, ytr, Xte, seeds=CNN_SEEDS):
    """The shipped CNN, seed-averaged, same recipe as soh_cnn.py."""
    import torch
    from soh_cnn import train_fold
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds = []
    for s in range(seeds):
        p, _, _ = train_fold(Xtr.astype(np.float32), ytr,
                             Xte.astype(np.float32), None,
                             CNN_EPOCHS, CNN_LR, s, dev)
        preds.append(p)
    return np.mean(preds, 0)


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def inner_score(name, make, grid, X, y, cell, use_cnn):
    """Leave-one-cell-out over the training cells.  Returns (rmse, chosen_p)."""
    cells = sorted(set(cell))
    if use_cnn:
        errs = []
        for c in cells:
            tr, te = cell != c, cell == c
            if te.sum() < 3 or tr.sum() < 10:
                continue
            a, b = standardise(X[tr], X[te])
            errs.append(fit_predict_cnn(a, y[tr], b) - y[te])
        if not errs:
            return np.inf, None
        e = np.concatenate(errs)
        return float(np.sqrt(np.mean(e ** 2))), None
    best, best_p = np.inf, grid[0]
    for p in grid:
        errs = []
        for c in cells:
            tr, te = cell != c, cell == c
            if te.sum() < 3 or tr.sum() < 10:
                continue
            a, b = standardise(X[tr], X[te])
            try:
                errs.append(fit_predict_sk(make, p, a, y[tr], b) - y[te])
            except Exception:                              # noqa: BLE001
                errs = []
                break
        if not errs:
            continue
        e = np.concatenate(errs)
        r = float(np.sqrt(np.mean(e ** 2)))
        if r < best:
            best, best_p = r, p
    return best, best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--no-cnn', action='store_true',
                    help='skip the CNN candidate (fast dry run)')
    ap.add_argument('--out', default=os.path.join(TABLES, 'soh_nested.csv'))
    a = ap.parse_args()

    z = np.load(a.data, allow_pickle=True)
    X, y, cell = z['X'], z['y'], z['cell']
    cells = sorted(set(cell))
    print(f'  {len(y)} curves, {X.shape[1]} inputs, {len(cells)} cells')
    print(f'  outer: leave one cell out.  inner: leave one of the remaining '
          f'five out.  The held-out cell is never seen by the selection.\n')

    fams = sk_families()
    cand = [(n, m, g, False) for n, m, g in fams]
    if not a.no_cnn:
        cand.append(('1D CNN (shipped)', None, [None], True))

    rows, summary_pred = [], {}
    t0 = time.time()
    for c in cells:
        otr, ote = cell != c, cell == c
        if ote.sum() < 3:
            continue
        scores = {}
        for name, make, grid, use_cnn in cand:
            r, p = inner_score(name, make, grid, X[otr], y[otr], cell[otr],
                               use_cnn)
            scores[name] = (r, p)
            print(f'    [{c}] inner {name:<20} {r:.4f}'
                  f'{"" if p is None else f"  (p={p})"}', flush=True)
        win = min(scores, key=lambda k: scores[k][0])
        wr, wp = scores[win]

        Xa, Xb = standardise(X[otr], X[ote])
        if win == '1D CNN (shipped)':
            pred = fit_predict_cnn(Xa, y[otr], Xb)
        else:
            make = [m for n, m, g in fams if n == win][0]
            pred = fit_predict_sk(make, wp, Xa, y[otr], Xb)
        err = pred - y[ote]
        outer = float(np.sqrt(np.mean(err ** 2)))
        summary_pred[c] = (pred, y[ote], err)
        print(f'  {c:<20} chose {win:<20} inner {wr:.4f}  '
              f'-> outer {outer:.4f}  ({time.time() - t0:.0f} s)\n', flush=True)
        rows.append([c, int(ote.sum()), win, '' if wp is None else str(wp),
                     f'{wr:.4f}', f'{outer:.4f}']
                    + [f'{scores[n][0]:.4f}' for n, _, _, _ in cand])

    os.makedirs(TABLES, exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['holdout', 'n', 'chosen_family', 'chosen_param',
                    'inner_rmse', 'outer_rmse']
                   + [f'inner_{n}' for n, _, _, _ in cand])
        w.writerows(rows)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')

    # The procedure's own error, against the two fixed choices it is meant to
    # replace.  always-ridge and always-CNN are what soh_baselines.csv reports.
    e = np.concatenate([v[2] for v in summary_pred.values()])
    per = {c: float(np.sqrt(np.mean(v[2] ** 2)))
           for c, v in summary_pred.items()}
    picked = collections.Counter(r[2] for r in rows)
    srows = [['nested selection', f'{np.sqrt(np.mean(e ** 2)):.4f}',
              f'{np.mean(list(per.values())):.4f}',
              f'{max(per.values()):.4f}', max(per, key=per.get),
              '; '.join(f'{k}={v}' for k, v in picked.most_common())]]
    with open(os.path.join(TABLES, 'soh_nested_summary.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['procedure', 'rmse_pooled', 'rmse_mean_of_cells',
                    'rmse_worst_cell', 'worst_cell', 'families_chosen'])
        w.writerows(srows)
    print(f'\n  nested-selection pooled RMSE {np.sqrt(np.mean(e ** 2)):.4f}, '
          f'worst cell {max(per.values()):.4f} ({max(per, key=per.get)})')
    print(f'  families chosen: '
          f'{", ".join(f"{k} x{v}" for k, v in picked.most_common())}')
    print('  Compare with soh_baselines.csv, whose ridge and CNN rows are '
          'both fixed choices made after seeing these same held-out cells.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
