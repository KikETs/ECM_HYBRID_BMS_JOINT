"""SOH baselines under the same nested cell-held-out protocol.

    python3 repro/run_soh_baselines.py

The paper reports a CNN on partial-charge windows with no baseline beside
it, so "the CNN is needed" is untested.  This runs the obvious alternatives
on the identical features, identical splits and identical target.

Protocol, and it matters:

  outer   leave one cell out.  The held-out cell is never seen.
  inner   GroupKFold over the five training cells only, used to pick each
          model's one hyperparameter.  No hyperparameter is chosen by
          looking at the outer cell.
  scaling fitted on the training cells only, applied to the held-out cell.

Note on naming: analysis/soh_cnn.py is a plain two-layer 1D CNN over the
64-bin partial-charge vector.  There is no physics or residual term in it.
Calling it "physics-aware" in the paper would not be supported by the code.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'analysis', 'cache', 'soh_charge.npz')
OUT = os.path.join(ROOT, 'analysis', 'results', 'tables', 'soh_baselines.csv')
PRED = os.path.join(ROOT, 'analysis', 'results', 'soh_pred.npz')


def models():
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


def fit_predict(make, p, Xtr, ytr, Xte):
    if make(p) == 'mean':
        return np.full(len(Xte), ytr.mean())
    m = make(p)
    m.fit(Xtr, ytr)
    q = m.predict(Xte)
    return np.asarray(q).ravel()


def inner_select(make, grid, X, y, g, seed=0):
    """Pick the hyperparameter on grouped inner folds of the training cells."""
    from sklearn.model_selection import GroupKFold
    if len(grid) == 1:
        return grid[0]
    uniq = np.unique(g)
    k = min(5, len(uniq))
    gkf = GroupKFold(n_splits=k)
    best, best_p = np.inf, grid[0]
    for p in grid:
        errs = []
        for tr, te in gkf.split(X, y, groups=g):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            try:
                q = fit_predict(make, p, (X[tr] - mu) / sd, y[tr],
                                (X[te] - mu) / sd)
            except Exception:                              # noqa: BLE001
                errs = [np.inf]
                break
            errs.append(np.sqrt(np.mean((q - y[te]) ** 2)))
        e = float(np.mean(errs))
        if e < best:
            best, best_p = e, p
    return best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if not os.path.exists(a.data):
        print(f'  missing {a.data}\n'
              f'  build it: python3 repro/run.py soh_data', file=sys.stderr)
        return 1

    z = np.load(a.data, allow_pickle=True)
    X, y, cell = np.asarray(z['X'], float), np.asarray(z['y'], float), z['cell']
    cells = sorted(set(cell))
    print(f'  {len(y)} samples, {X.shape[1]} features, {len(cells)} cells')

    rows = []
    for name, make, grid in models():
        per, ns, chosen = {}, {}, {}
        for c in cells:
            tr, te = cell != c, cell == c
            p = inner_select(make, grid, X[tr], y[tr], cell[tr])
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            q = fit_predict(make, p, (X[tr] - mu) / sd, y[tr],
                            (X[te] - mu) / sd)
            per[c] = float(np.sqrt(np.mean((q - y[te]) ** 2)))
            ns[c] = int(te.sum())
            chosen[c] = p
        pooled = float(np.sqrt(np.average(
            [per[c] ** 2 for c in cells], weights=[ns[c] for c in cells])))
        worst = max(cells, key=lambda c: per[c])
        rows.append([name, f'{pooled:.4f}',
                     f'{np.mean([per[c] for c in cells]):.4f}',
                     f'{per[worst]:.4f}', worst,
                     ';'.join(f'{c}={chosen[c]}' for c in cells)]
                    + [f'{per[c]:.4f}' for c in cells])

    # The shipped CNN, read from its saved predictions (same outer splits).
    if os.path.exists(PRED):
        zp = np.load(PRED)
        per, ns = {}, {}
        for c in cells:
            if f'{c}_y' not in zp:
                per = {}
                break
            e = np.asarray(zp[f'{c}_pred'], float) - np.asarray(zp[f'{c}_y'],
                                                                float)
            per[c] = float(np.sqrt(np.mean(e ** 2)))
            ns[c] = len(e)
        if per:
            pooled = float(np.sqrt(np.average(
                [per[c] ** 2 for c in cells], weights=[ns[c] for c in cells])))
            worst = max(cells, key=lambda c: per[c])
            rows.append(['1D CNN (shipped, 3 seeds)', f'{pooled:.4f}',
                         f'{np.mean([per[c] for c in cells]):.4f}',
                         f'{per[worst]:.4f}', worst,
                         'seeds=0,1,2 (fixed, not tuned)']
                        + [f'{per[c]:.4f}' for c in cells])

    hdr = (['method', 'rmse_pooled', 'rmse_mean_of_cells', 'rmse_worst_cell',
            'worst_cell', 'hyperparameter_per_fold'] + list(cells))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f'  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')
    print(f"\n  {'method':<28}{'pooled':>9}{'mean-cell':>11}{'worst':>9}"
          f"  worst cell")
    print('  ' + '-' * 70)
    for r in rows:
        print(f'  {r[0]:<28}{r[1]:>9}{r[2]:>11}{r[3]:>9}  {r[4]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
