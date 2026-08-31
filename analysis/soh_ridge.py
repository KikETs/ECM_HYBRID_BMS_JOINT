"""Ridge SOH — the model nested selection actually chooses.

Same input, same splits and the same standardisation as soh_cnn.py, so the
predictions it writes drop into every consumer of results/soh_pred.npz without
a schema change.

The penalty is never chosen by looking at the held-out cell.  For an evaluation
fold it is picked by leave-one-cell-out over the five training cells; for the
deployment fit, where no cell is held out, by leave-one-cell-out over all six.
That is the only defensible reading of "choose alpha" when the artifact has to
be trained on everything available.

What the board runs is exactly this, and nothing more:

    soh = b + sum_i w_i * (x_i - mu_i) / sd_i

64 multiply-accumulates against the CNN's three seed-averaged convolutional
passes.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cache", "soh_charge.npz")
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _fit(X, y, alpha):
    """Ridge on standardised inputs.  Returns (w, b, mu, sd) in raw units."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    n, d = Z.shape
    # Centre the target so the penalty never touches the intercept.
    ym = y.mean()
    A = Z.T @ Z + alpha * np.eye(d)
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return w, float(ym), mu, sd


def _predict(w, b, mu, sd, X):
    return ((X - mu) / sd) @ w + b


def pick_alpha(X, y, cell, alphas=ALPHAS):
    """Leave-one-cell-out over whatever cells are given.  Never sees a fold
    the caller is holding out -- that filtering is the caller's job."""
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        errs = []
        for c in sorted(set(cell)):
            tr, te = cell != c, cell == c
            if te.sum() < 3 or tr.sum() < 10:
                continue
            w, b, mu, sd = _fit(X[tr], y[tr], a)
            errs.append(_predict(w, b, mu, sd, X[te]) - y[te])
        if not errs:
            continue
        r = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        if r < best:
            best, best_a = r, a
    return best_a, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--save-model", default=None,
                    help="directory for per-fold weights (and the all-cell "
                         "fit under --deployment)")
    ap.add_argument("--save-pred", default=None,
                    help="write per-curve predictions so a trajectory can be "
                         "drawn without refitting")
    ap.add_argument("--deployment", action="store_true",
                    help="also fit on all six cells and save it as _ALL.  "
                         "That fit is the artifact; it is not evaluated here "
                         "and cannot be -- there is no seventh cell.")
    a = ap.parse_args()

    z = np.load(a.data, allow_pickle=True)
    X, y, cell, cyc = z["X"], z["y"], z["cell"], z["cycle"]
    print(f"{len(y)} curves, {X.shape[1]} inputs, {len(set(cell))} cells")
    print(f"  {'holdout':<20} {'n':>4} {'alpha':>8} {'RMSE':>9} {'MAE':>9} "
          f"{'bias':>9}")

    keep, allerr = {}, []
    for c in sorted(set(cell)):
        tr, te = cell != c, cell == c
        if te.sum() < 3:
            continue
        alpha, inner = pick_alpha(X[tr], y[tr], cell[tr])
        w, b, mu, sd = _fit(X[tr], y[tr], alpha)
        p = _predict(w, b, mu, sd, X[te]).astype(np.float32)
        keep[c] = (p, y[te], cyc[te])
        e = p - y[te]
        allerr.append(e)
        print(f"  {c:<20} {te.sum():>4} {alpha:>8g} "
              f"{np.sqrt(np.mean(e ** 2)):>9.4f} {np.mean(np.abs(e)):>9.4f} "
              f"{np.mean(e):>+9.4f}")
        if a.save_model:
            os.makedirs(a.save_model, exist_ok=True)
            np.savez(os.path.join(a.save_model, f"soh_{c}.npz"),
                     w=w, b=b, mu=mu, sd=sd, alpha=alpha, holdout=c,
                     inner_rmse=inner, n_in=X.shape[1])
    e = np.concatenate(allerr)
    print(f"  {'ALL':<20} {len(e):>4} {'':>8} {np.sqrt(np.mean(e ** 2)):>9.4f} "
          f"{np.mean(np.abs(e)):>9.4f} {np.mean(e):>+9.4f}")
    print(f"\n  {X.shape[1] + 1} coefficients (64 weights + 1 intercept), "
          f"against the CNN's 10,945 parameters")

    if a.deployment:
        alpha, inner = pick_alpha(X, y, cell)
        w, b, mu, sd = _fit(X, y, alpha)
        if not a.save_model:
            raise SystemExit("  --deployment needs --save-model")
        np.savez(os.path.join(a.save_model, "soh_ALL.npz"),
                 w=w, b=b, mu=mu, sd=sd, alpha=alpha, holdout="ALL",
                 inner_rmse=inner, n_in=X.shape[1])
        print(f"  deployment fit on all six cells: alpha {alpha:g}, "
              f"leave-one-cell-out RMSE {inner:.4f} -> soh_ALL.npz")
        print("  That number is the selection's own estimate, not a test "
              "score.  The all-cell fit has no held-out cell by construction.")

    if a.save_pred:
        np.savez(a.save_pred, **{f"{c}_{k}": v for c, t in keep.items()
                                 for k, v in zip(("pred", "y", "cycle"), t)})
        print(f"  -> {a.save_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
