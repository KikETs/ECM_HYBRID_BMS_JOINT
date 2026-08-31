"""SOH from an ordinary driving window, held out by cell.

THE NUMBER THAT MATTERS IS PER FILE, NOT PER WINDOW
    The archive holds 28,534 windows but only 558 distinct SOH values - about 51
    windows share every label, because one aging file carries one capacity
    measurement and the windows are cut out of it with 75 % overlap. Scoring per
    window would count each label ~51 times and report a confidence the data does
    not have. Every table below therefore aggregates predictions to one per
    (cell, cycle) before taking the error, and the per-window number is printed
    beside it only to show how large the gap is.

WHAT IT HAS TO BEAT, FIXED BEFORE TRAINING
    Same leave-one-cell-out split, ten hand statistics, ridge: 0.0399 per window.
    Resistance alone: 0.0442. Predicting the mean: 0.0862. These are printed by
    soh_drive_dataset.py so they cannot be chosen after seeing the result.

INPUT IS (V, I) ONLY
    SOC is excluded because the cache's rated SOC is built from the aged capacity
    and a file's minimum SOC predicts its SOH with residual RMSE 0.0025 - the
    channel IS the label. Temperature is excluded because dropping it cost
    nothing (0.0382 -> 0.0377) and it might carry a calendar signal rather than a
    cell one.

THE CHRONOLOGY AUDIT IS PART OF THE RESULT
    Within a cell, cycle count and SOH are nearly collinear, so a model that
    identified WHERE in the record a window sits would score well without reading
    degradation. Leave-one-cell-out is the defence - a held-out cell's ageing
    rate differs (BOOST_REST reaches 80 % at cycle 784, CC at 1497), so a clock
    reader cannot transfer. The audit prints the error against position-in-file
    and cycle so the claim is checked rather than asserted.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cache", "soh_drive.npz")
OUT = os.path.join(HERE, "runs_soh_drive")


class DriveSOHNet(nn.Module):
    """Small on purpose: 558 distinct labels, six cells."""

    def __init__(self, ch_in=2, ch=16, n_feat=0, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch, 7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 5, stride=2, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch * 2, ch * 2, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.n_feat = n_feat
        self.head = nn.Sequential(
            nn.Dropout(drop), nn.Linear(ch * 2 * 8 + n_feat, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, f=None):
        z = self.conv(x.transpose(1, 2)).flatten(1)
        if self.n_feat:
            z = torch.cat([z, f], 1)
        return self.head(z).squeeze(-1)


def per_file(pred, y, cell, cyc):
    """One prediction and one label per (cell, cycle)."""
    key = np.array([f"{a}|{b}" for a, b in zip(cell, cyc)])
    out_p, out_y = [], []
    for k in np.unique(key):
        m = key == k
        out_p.append(np.median(pred[m])); out_y.append(y[m][0])
    return np.array(out_p), np.array(out_y)


def train_fold(Xtr, Ftr, ytr, Xte, Fte, epochs, lr, seed, dev, use_feat, batch):
    torch.manual_seed(seed)
    m = DriveSOHNet(Xtr.shape[2], n_feat=Ftr.shape[1] if use_feat else 0).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(epochs * f) for f in (0.5, 0.75)], 0.1)
    lossf = nn.MSELoss()
    xtr = torch.from_numpy(Xtr).to(dev); ttr = torch.from_numpy(ytr).to(dev)
    ftr = torch.from_numpy(Ftr).to(dev)
    n = len(xtr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, batch):
            b = perm[k:k + batch]
            opt.zero_grad(set_to_none=True)
            lossf(m(xtr[b], ftr[b] if use_feat else None), ttr[b]).backward()
            opt.step()
        sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(Xte), 2048):
            xb = torch.from_numpy(Xte[k:k + 2048]).to(dev)
            fb = torch.from_numpy(Fte[k:k + 2048]).to(dev)
            out.append(m(xb, fb if use_feat else None).cpu().numpy())
    return np.concatenate(out), sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--with-feat", action="store_true",
                    help="concatenate the ten hand statistics at the head")
    ap.add_argument("--tag", default="cnn")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    X, F, y, cell = z["X"], z["Xf"], z["y"], z["cell"]
    cyc, pos = z["audit_cycle"], z["audit_pos"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    cells = sorted(set(cell.tolist()))
    print(f"  창 {len(y):,}개, 계열 {X.shape[1]}x{X.shape[2]}, "
          f"고유 SOH {len(set(np.round(y,5)))}개, {dev}")
    print("  넘어야 할 기준선(창 단위): 손특징 0.0399, 저항 0.0442, 평균 0.0862\n")
    print(f"  {'홀드아웃':<20} {'n창':>7} {'n파일':>6} {'창 RMSE':>9} "
          f"{'파일 RMSE':>10} {'파일 MAE':>9} {'편향':>9}")

    ew, ef, preds = [], [], {}
    npar = 0
    for c in cells:
        tr, te = cell != c, cell == c
        mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), \
                 X[tr].reshape(-1, X.shape[2]).std(0) + 1e-8
        fmu, fsd = F[tr].mean(0), F[tr].std(0) + 1e-8
        Xtr = ((X[tr] - mu) / sd).astype(np.float32)
        Xte = ((X[te] - mu) / sd).astype(np.float32)
        Ftr = ((F[tr] - fmu) / fsd).astype(np.float32)
        Fte = ((F[te] - fmu) / fsd).astype(np.float32)
        ps = []
        for s in range(args.seeds):
            p, npar = train_fold(Xtr, Ftr, y[tr], Xte, Fte, args.epochs,
                                 args.lr, s, dev, args.with_feat, args.batch)
            ps.append(p)
        p = np.mean(ps, 0)
        preds[c] = (p, y[te], cyc[te], pos[te])
        # Written per fold, not once at the end. The first version of this file
        # saved nothing until all six folds finished, which meant an interrupted
        # run lost everything - the exact failure the project's standing rule
        # about checkpoints exists to prevent.
        np.savez(os.path.join(args.out, f"pred_{args.tag}_{c}.npz"),
                 pred=p, y=y[te], cycle=cyc[te], pos=pos[te])
        pf, yf = per_file(p, y[te], cell[te], cyc[te])
        ew.append(p - y[te]); ef.append(pf - yf)
        print(f"  {c:<20} {te.sum():>7,} {len(pf):>6} "
              f"{np.sqrt(np.mean((p-y[te])**2)):>9.4f} "
              f"{np.sqrt(np.mean((pf-yf)**2)):>10.4f} "
              f"{np.mean(np.abs(pf-yf)):>9.4f} {np.mean(pf-yf):>+9.4f}")
    EW = np.concatenate(ew); EF = np.concatenate(ef)
    print(f"  {'전체':<20} {len(EW):>7,} {len(EF):>6} "
          f"{np.sqrt(np.mean(EW**2)):>9.4f} {np.sqrt(np.mean(EF**2)):>10.4f} "
          f"{np.mean(np.abs(EF)):>9.4f} {np.mean(EF):>+9.4f}")
    print(f"\n  파라미터 {npar:,}개, 시드 {args.seeds}개 평균, "
          f"손특징 결합 {'예' if args.with_feat else '아니오'}")

    print("\n  === 시간축 지름길 감사 ===")
    print(f"  {'홀드아웃':<20} {'오차 vs 파일내위치':>18} {'오차 vs 사이클':>15}")
    for c in cells:
        p, yy, cc, pp = preds[c]
        e = p - yy
        print(f"  {c:<20} {np.corrcoef(e,pp)[0,1]:>+18.3f} "
              f"{np.corrcoef(e,cc.astype(float))[0,1]:>+15.3f}")
    print("  (파일내위치와 강한 상관이 있으면 열화가 아니라 기록 위치를 읽는 것)")

    np.savez(os.path.join(args.out, f"pred_{args.tag}.npz"),
             **{f"{c}_{k}": v for c in cells
                for k, v in zip(("pred", "y", "cycle", "pos"), preds[c])})


if __name__ == "__main__":
    main()
