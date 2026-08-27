"""Is the SOH CNN reading the curve's SHAPE, or just counting charge?

THE OBJECTION, STATED PROPERLY
    The input is dQ/dV on a fixed voltage grid. Summing it and multiplying by the
    bin width returns dQ = Q(4.05 V) - Q(3.55 V) - a partial capacity. Partial
    capacity tracks total capacity, and total capacity IS SOH. So the network
    could reach its number by adding up its own input, and the claim that it
    "learned the incremental-capacity curve" would be decoration on a coulomb
    count.

    soh_charge_dataset.py already refused the FULL curve for exactly this reason.
    The window makes the integral partial, not absent, so the same objection
    survives the windowing and has to be measured rather than argued.

HOW IT IS SEPARATED
    Magnitude and shape are pulled apart and each is given to a model alone.

      dQ alone        - one scalar, the summed input, linear on SOH.
                        If this matches the CNN, the CNN is a coulomb counter.
      shape alone     - each curve divided by its own sum, so every input has
                        unit area and dQ is unrecoverable. If this matches the
                        CNN, the shape carries the estimate on its own.
      per-curve z     - each curve standardised by its own mean and sd, removing
                        scale AND offset, a stricter version of the above.

    The published normalisation cannot do this job: X[tr].mean(0) standardises
    each grid point ACROSS curves, which leaves each individual curve's magnitude
    fully intact. That is why the original number cannot answer the question.

WHAT WOULD FALSIFY THE SHAPE CLAIM
    dQ alone landing at or below 0.0128. Nothing else is needed - no threshold,
    no significance test. The baseline either reaches the CNN or it does not.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soh_cnn import DATA, train_fold  # noqa: E402


def loco(X, y, cell, fn):
    """fn(Xtr, ytr, Xte) -> predictions. Returns pooled errors."""
    errs = []
    for c in sorted(set(cell.tolist())):
        tr, te = cell != c, cell == c
        if te.sum() < 3:
            continue
        errs.append(fn(X[tr], y[tr], X[te]) - y[te])
    return np.concatenate(errs)


def lin(Xtr, ytr, Xte):
    A = np.column_stack([Xtr, np.ones(len(Xtr))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([Xte, np.ones(len(Xte))]) @ w


def cnn(epochs, lr, seeds, dev):
    def f(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        a = ((Xtr - mu) / sd).astype(np.float32)
        b = ((Xte - mu) / sd).astype(np.float32)
        ps = [train_fold(a, ytr, b, None, epochs, lr, s, dev)[0]
              for s in range(seeds)]
        return np.mean(ps, 0)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    X, y, cell, band = z["X"], z["y"], z["cell"], z["band"]
    dV = (float(z["v_hi"]) - float(z["v_lo"])) / X.shape[1]
    dQ = X.sum(1) * dV                       # Ah passed over 3.55-4.05 V
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    r = np.corrcoef(dQ, y)[0, 1]
    print(f"  곡선 {len(y)}개   dQ(3.55~4.05 V) 중앙 {np.median(dQ):.4f} Ah   "
          f"SOH 와의 상관 {r:+.4f}")
    print(f"  {'':<38} {'RMSE':>8} {'MAE':>8}")

    def show(name, e):
        print(f"  {name:<38} {np.sqrt(np.mean(e**2)):>8.4f} "
              f"{np.mean(np.abs(e)):>8.4f}")

    show("평균 예측", loco(np.zeros((len(y), 1)), y, cell,
                       lambda a, b, c: np.full(len(c), b.mean())))
    show("대역시간 3.6~3.7 V 1개 (기존 기준선)",
         loco(band.reshape(-1, 1), y, cell, lin))
    show("dQ 1개  <- 크기만",
         loco(dQ.reshape(-1, 1), y, cell, lin))
    show("dQ + 대역시간 2개",
         loco(np.column_stack([dQ, band]), y, cell, lin))

    f = cnn(args.epochs, args.lr, args.seeds, dev)
    show("CNN, 원본 dQ/dV (기존)", loco(X, y, cell, f))
    Xs = X / (X.sum(1, keepdims=True) + 1e-12)
    show("CNN, 면적 정규화  <- 모양만", loco(Xs, y, cell, f))
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    show("CNN, 곡선별 z 정규화  <- 모양만(엄격)", loco(Xz, y, cell, f))
    show("CNN, 면적정규화 + dQ 를 채널로 되돌림",
         loco(np.column_stack([Xs * 64, dQ[:, None] * np.ones((1, 1))]),
              y, cell, f))


if __name__ == "__main__":
    main()
