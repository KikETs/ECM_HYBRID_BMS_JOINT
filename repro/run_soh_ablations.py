"""SOH ablations: which part of the partial-charge window carries the signal.

    python3 repro/run_soh_ablations.py

Protocol is the same everywhere: leave one cell out; the ridge penalty is
chosen on GroupKFold inner folds of the training cells only; scaling is
fitted on training cells only.  Nothing is selected on a held-out cell.

Axes
    input block     dQ/dV (the shipped input), time-per-bin, both, and the
                    single 3.6-3.7 V band-crossing time the dataset's own
                    docstring names as its baseline
    window length   the 64-bin grid spans 3.55-4.05 V.  Truncating from the
                    low-voltage end tests the claim in soh_charge_dataset.py
                    that the signal sits at the phase transition and the
                    3.8-3.9 V band carries none of it
    charge coverage the same truncation read the other way round: how much
                    of the window a car must actually deliver
    update cadence  re-estimate only every Nth characterisation and carry the
                    stale value forward, which is what a BMS that only sees
                    an adequate charge occasionally would do

Ridge is the estimator here because it beat the shipped CNN on this data
under the same splits (soh_baselines.csv).  The CNN is run on the input-block
axis as well so the ablation also speaks to the model that shipped.
"""
import argparse
import csv
import os
import sys

import os as _os_early, sys as _sys_early
_sys_early.path.insert(0, _os_early.path.join(
    _os_early.path.dirname(_os_early.path.dirname(
        _os_early.path.abspath(__file__))), 'analysis'))
import determinism            # sets CUBLAS_WORKSPACE_CONFIG before
                              # the lazy `import torch` below
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'analysis', 'cache', 'soh_charge.npz')
OUT = os.path.join(ROOT, 'analysis', 'results', 'tables', 'soh_ablations.csv')

ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def ridge_loco(X, y, cell, cells):
    """Per-cell RMSE under leave-one-cell-out with inner alpha selection."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    per, ns = {}, {}
    for c in cells:
        tr, te = cell != c, cell == c
        if te.sum() == 0 or tr.sum() < 10:
            continue
        g = cell[tr]
        k = min(5, len(np.unique(g)))
        best, best_a = np.inf, ALPHAS[0]
        for al in ALPHAS:
            errs = []
            for i, j in GroupKFold(n_splits=k).split(X[tr], y[tr], groups=g):
                Xi, Xj = X[tr][i], X[tr][j]
                mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
                m = Ridge(alpha=al).fit((Xi - mu) / sd, y[tr][i])
                q = m.predict((Xj - mu) / sd)
                errs.append(np.sqrt(np.mean((q - y[tr][j]) ** 2)))
            if np.mean(errs) < best:
                best, best_a = np.mean(errs), al
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        m = Ridge(alpha=best_a).fit((X[tr] - mu) / sd, y[tr])
        q = m.predict((X[te] - mu) / sd)
        per[c] = float(np.sqrt(np.mean((q - y[te]) ** 2)))
        ns[c] = int(te.sum())
    pooled = float(np.sqrt(np.average([per[c] ** 2 for c in per],
                                      weights=[ns[c] for c in per])))
    worst = max(per, key=per.get)
    return pooled, per[worst], worst, per


def cnn_loco(X, y, cell, cells, seeds=3, epochs=300, lr=3e-3):
    """The shipped architecture on an arbitrary input width."""
    import torch
    import torch.nn as nn
    from soh_cnn import pool_to_8 as _pool_to_8

    class Net(nn.Module):
        def __init__(self, n_in, ch=16, drop=0.1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, ch, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(ch, ch * 2, 5, padding=2), nn.ReLU(),
                _pool_to_8(n_in // 2))
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(drop),
                nn.Linear(ch * 2 * 8, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    per, ns = {}, {}
    for c in cells:
        tr, te = cell != c, cell == c
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        xt = torch.tensor(((X[tr] - mu) / sd), dtype=torch.float32, device=dev)
        xe = torch.tensor(((X[te] - mu) / sd), dtype=torch.float32, device=dev)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
        preds = []
        for s in range(seeds):
            determinism.enable(s)
            m = Net(xt.shape[1]).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
            # Mirror soh_cnn.train_fold exactly: mini-batch 32 and a
            # MultiStepLR at 50 % and 75 %.  A full-batch loop with no
            # scheduler trains a visibly worse model (0.0166 vs the shipped
            # 0.0135) and would have made the CNN column look worse than it
            # is for reasons that have nothing to do with the ablation.
            sch = torch.optim.lr_scheduler.MultiStepLR(
                opt, [int(epochs * f) for f in (0.5, 0.75)], 0.1)
            lossf = nn.MSELoss()
            n = len(xt)
            for _ in range(epochs):
                m.train()
                perm = torch.randperm(n, device=dev)
                for k in range(0, n, 32):
                    b = perm[k:k + 32]
                    opt.zero_grad(set_to_none=True)
                    lossf(m(xt[b]), yt[b]).backward()
                    opt.step()
                sch.step()
            m.eval()
            with torch.no_grad():
                preds.append(m(xe).cpu().numpy())
        q = np.mean(preds, 0)
        per[c] = float(np.sqrt(np.mean((q - y[te]) ** 2)))
        ns[c] = int(te.sum())
    pooled = float(np.sqrt(np.average([per[c] ** 2 for c in per],
                                      weights=[ns[c] for c in per])))
    worst = max(per, key=per.get)
    return pooled, per[worst], worst, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--with-cnn', action='store_true', default=True)
    a = ap.parse_args()
    if not os.path.exists(a.data):
        print(f'  missing {a.data}', file=sys.stderr)
        return 1

    z = np.load(a.data, allow_pickle=True)
    X = np.asarray(z['X'], float)          # dQ/dV on a fixed voltage grid
    Xt = np.asarray(z['Xt'], float)        # time per voltage bin
    y = np.asarray(z['y'], float)
    cell = z['cell']
    band = np.asarray(z['band'], float).reshape(-1, 1)
    cyc = np.asarray(z['cycle'], int)
    v_lo, v_hi = float(z['v_lo']), float(z['v_hi'])
    cells = sorted(set(cell))
    n = X.shape[1]
    print(f'  {len(y)} samples, {n}-bin grid over {v_lo}-{v_hi} V, '
          f'{len(cells)} cells')

    rows = []

    def add(axis, name, pooled, worst, wcell, extra=''):
        rows.append([axis, name, f'{pooled:.4f}', f'{worst:.4f}', wcell,
                     extra])
        print(f'  {axis:<16}{name:<34}{pooled:>9.4f}{worst:>9.4f}  {wcell}')

    print(f"\n  {'axis':<16}{'variant':<34}{'pooled':>9}{'worst':>9}  worst cell")
    print('  ' + '-' * 78)

    # ---- input block ----------------------------------------------------
    for name, M in (('dQ/dV only (shipped)', X),
                    ('time-per-bin only', Xt),
                    ('dQ/dV + time-per-bin', np.hstack([X, Xt])),
                    ('single 3.6-3.7 V band time', band)):
        p, w, wc, _ = ridge_loco(M, y, cell, cells)
        add('input, ridge', name, p, w, wc)

    if a.with_cnn:
        for name, M in (('dQ/dV only (shipped)', X),
                        ('time-per-bin only', Xt),
                        ('dQ/dV + time-per-bin', np.hstack([X, Xt]))):
            p, w, wc, _ = cnn_loco(M, y, cell, cells)
            add('input, CNN', name, p, w, wc)

    # ---- window length / charge coverage --------------------------------
    for k in (8, 16, 24, 32, 48, 64):
        vk = v_lo + (v_hi - v_lo) * k / n
        p, w, wc, _ = ridge_loco(X[:, :k], y, cell, cells)
        add('window (low end)', f'{k}/{n} bins, {v_lo:.2f}-{vk:.2f} V',
            p, w, wc, f'{100 * k / n:.0f}% coverage')
    for k in (8, 16, 24, 32, 48):
        vk = v_hi - (v_hi - v_lo) * k / n
        p, w, wc, _ = ridge_loco(X[:, -k:], y, cell, cells)
        add('window (high end)', f'last {k}/{n} bins, {vk:.2f}-{v_hi:.2f} V',
            p, w, wc, f'{100 * k / n:.0f}% coverage')

    # ---- update cadence -------------------------------------------------
    # Estimate only every Nth characterisation per cell and hold the last
    # value; score every sample against its true SOH.
    order = {c: np.argsort(cyc[cell == c]) for c in cells}
    for every in (1, 2, 4, 8):
        errs, ns_ = [], []
        for c in cells:
            tr, te = cell != c, cell == c
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, y[tr])
            q = m.predict((X[te] - mu) / sd)
            yt = y[te]
            o = order[c]
            q_o, y_o = q[o], yt[o]
            held = np.empty_like(q_o)
            last = q_o[0]
            for i in range(len(q_o)):
                if i % every == 0:
                    last = q_o[i]
                held[i] = last
            errs.append(np.sqrt(np.mean((held - y_o) ** 2)))
            ns_.append(len(y_o))
        pooled = float(np.sqrt(np.average(np.array(errs) ** 2, weights=ns_)))
        add('update cadence', f'refresh every {every} characterisation(s)',
            pooled, float(max(errs)), cells[int(np.argmax(errs))],
            'alpha fixed at 1.0')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w_ = csv.writer(f)
        w_.writerow(['axis', 'variant', 'rmse_pooled', 'rmse_worst_cell',
                     'worst_cell', 'note'])
        w_.writerows(rows)
    print(f'\n  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
