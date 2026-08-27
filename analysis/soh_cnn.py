"""1D CNN for SOH from a partial charge curve, held out by cell.

WHAT IT HAS TO BEAT
    A single scalar - seconds to cross 3.6-3.7 V - fitted by least squares gives
    RMSE 0.0254 in SOH under the same leave-one-cell-out split
    (soh_charge_dataset.py). Predicting the mean gives 0.0876. A network that
    lands between those two has learned less than one hand-picked number.

WHY THE MODEL IS THIS SMALL
    290 curves, about 50 per cell, so ~240 in training per fold. That is a
    curve-regression problem with a few hundred examples, not an ImageNet. Two
    conv blocks and a small head keep the parameter count near the sample count
    rather than three orders above it.

INPUT
    dQ/dV on a fixed voltage grid over 3.55-4.05 V. The full curve is excluded on
    purpose: integrating it returns the capacity outright, so a model trained on
    it would score well while learning nothing transferable to a partial charge.

SPLIT
    Leave-one-cell-out. Neighbouring cycles of the same cell are near-duplicates;
    a random split would put them either side of the line and report a number no
    new cell would see. This is the same discipline the SOP work used.

NORMALISATION IS FITTED ON TRAINING CELLS ONLY
    Otherwise the held-out cell's range leaks into the transform - small, but the
    entire point of the split is to have no leak at all.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cache", "soh_charge.npz")


class SOHNet(nn.Module):
    def __init__(self, n_in=64, ch=16, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(drop),
            nn.Linear(ch * 2 * 8, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


def train_fold(Xtr, ytr, Xte, yte, epochs, lr, seed, dev):
    torch.manual_seed(seed)
    m = SOHNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(epochs * f) for f in (0.5, 0.75)], 0.1)
    lossf = nn.MSELoss()
    xtr = torch.from_numpy(Xtr).to(dev); ttr = torch.from_numpy(ytr).to(dev)
    xte = torch.from_numpy(Xte).to(dev)
    n = len(xtr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, 32):
            b = perm[k:k + 32]
            opt.zero_grad(set_to_none=True)
            lossf(m(xtr[b]), ttr[b]).backward()
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return (m(xte).cpu().numpy(),
                sum(p.numel() for p in m.parameters()), m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save-model", default=None,
                    help="셀별 최종 가중치를 저장한다. MCU 배포에는 "
                         "가중치가 필요한데 지금까지 예측만 저장했다.")
    ap.add_argument("--save-pred", default=None,
                    help="write per-curve predictions so a trajectory can be "
                         "drawn without retraining")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    X, y, cell = z["X"], z["y"], z["cell"]
    cyc = z["cycle"]
    keep = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{len(y)}개, 입력 {X.shape[1]}차원, 셀 {len(set(cell))}개, {dev}")
    print(f"기준선: 단일 특징 선형회귀 0.0254,  평균 예측 0.0876\n")
    print(f"  {'홀드아웃 셀':<20} {'n':>4} {'RMSE':>9} {'MAE':>9} {'편향':>9}")
    allerr = []
    nparam = 0
    for c in sorted(set(cell)):
        tr, te = cell != c, cell == c
        if te.sum() < 3:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = ((X[tr] - mu) / sd).astype(np.float32)
        Xte = ((X[te] - mu) / sd).astype(np.float32)
        preds, models = [], []
        for s in range(args.seeds):
            p, nparam, mdl = train_fold(Xtr, y[tr], Xte, y[te], args.epochs,
                                        args.lr, s, dev)
            preds.append(p)
            models.append(mdl)
        p = np.mean(preds, 0)
        keep[c] = (p, y[te], cyc[te])
        if args.save_model:
            os.makedirs(args.save_model, exist_ok=True)
            torch.save({"seeds": [mm.state_dict() for mm in models],
                        "mu": mu, "sd": sd, "n_in": X.shape[1],
                        "holdout": c, "nparam": nparam},
                       os.path.join(args.save_model, f"soh_{c}.pt"))
        e = p - y[te]
        allerr.append(e)
        print(f"  {c:<20} {te.sum():>4} {np.sqrt(np.mean(e**2)):>9.4f} "
              f"{np.mean(np.abs(e)):>9.4f} {np.mean(e):>+9.4f}")
    e = np.concatenate(allerr)
    print(f"  {'전체':<20} {len(e):>4} {np.sqrt(np.mean(e**2)):>9.4f} "
          f"{np.mean(np.abs(e)):>9.4f} {np.mean(e):>+9.4f}")
    print(f"\n  파라미터 {nparam:,}개, 시드 {args.seeds}개 평균")
    if args.save_pred:
        np.savez(args.save_pred, **{f"{c}_{k}": v for c, t in keep.items()
                                    for k, v in zip(("pred", "y", "cycle"), t)})
        print(f"  -> {args.save_pred}")


if __name__ == "__main__":
    main()
