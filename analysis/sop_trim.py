"""Train the two-multiplier trim, held out by cell, against measured pulse dV.

WHAT IS BEING LEARNED, AND WHY IT IS ONLY TWO NUMBERS
    k_f multiplies the fast branch (R0 and R1 together), k_s the slow branch.

    R0 and R1 are ONE regressor at every horizon this work uses. Measured tau1 is
    0.244 s median with 99.1 % below 1 s, so at tau = 2 s the fast branch is
    already 99.97 % developed and at 10 s it is complete. Splitting them is what
    ecm_refine.py did, and it drove R1's multiplier NEGATIVE while tripling
    held-out error. Here the degeneracy is removed structurally rather than
    clipped after the fact.

    tau, OCV and the hysteresis magnitude are NOT outputs. tau moves
    non-monotonically with temperature (ecm_temp_factor.py already declined to
    correct it); OCV and hysteresis do not appear in the loss at all, so there is
    no gradient that could estimate them - see below.

THE LOSS IS LINEAR IN THE OUTPUTS, SO THERE IS NO OPTIMISER INSIDE THE MODEL

        dV_hat(tau) = I * [ k_f * nf(tau) + k_s * ns(tau) ]

    with nf, ns the pooled nominal contributions, frozen per pulse at dataset
    build time. Autograd never touches LinearNDInterpolator, and the decode is a
    single multiply-add.

    OCV, hysteresis and the RC initial states are absent by construction: every
    kept pulse follows a long rest, so dV = V(t0+tau) - V(t0-) removes them
    exactly. That is the whole safety argument for the parameterisation - a
    45-72 mV OCV error at low SOH cannot enter a resistance multiplier through a
    term that is not in the equation.

THE BASELINE THAT MUST BE REPORTED BESIDE IT
    A linear 12->2 readout, 26 parameters. Given that k_f - 1 is close to
    dR_fast / R_fast_nom by construction, it may well be enough. If it is,
    ship it and say so: "the AI in the hybrid arm is 26 numbers" is a stronger
    deployment result than a 514-parameter network winning by a hair.

HELD OUT BY CELL, ALWAYS
    Same discipline as soh_cnn.py and train_soh.py. Neighbouring cycles of one
    cell are near-duplicates; the quantity being estimated is precisely what
    differs BETWEEN cells (5.58x in R1 at SOH 0.75), so a random split would
    report a number no new cell would ever see.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "cache", "trim")
OUT_DIR = os.path.join(HERE, "runs_trim")

# Bounds from the measured leave-one-cell-out requirement at rank 3
# (k_fast 0.70-1.28, k_slow 0.74-1.54), with margin. Sizing these from
# drive-cycle ridge fits instead produces a box that saturates on most points.
KF_SPAN, KS_SPAN = 0.470, 0.588


class TrimLinear(nn.Module):
    """26 parameters. The baseline the network has to beat."""

    def __init__(self, n_in=12):
        super().__init__()
        self.fc = nn.Linear(n_in, 2)

    def forward(self, x):
        return self.fc(x)


class TrimMLP(nn.Module):
    """514 parameters. tanh, not ReLU: the model is queried below SOH 0.72 where
    no OCV curve exists, and a saturating activation degrades to a constant out
    there rather than extrapolating linearly away."""

    def __init__(self, n_in=12, h=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 2),
        )

    def forward(self, x):
        return self.net(x)


def decode(u):
    """Bounded, symmetric in log space, centred at 1."""
    k_f = torch.exp(KF_SPAN * torch.tanh(u[:, 0]))
    k_s = torch.exp(KS_SPAN * torch.tanh(u[:, 1]))
    return k_f, k_s


def dv_hat(k_f, k_s, nom, I):
    """nom = [nf2, ns2, nf10, ns10] in ohms, I in amps, result in volts."""
    d2 = I * (k_f * nom[:, 0] + k_s * nom[:, 1])
    d10 = I * (k_f * nom[:, 2] + k_s * nom[:, 3])
    return torch.stack([d2, d10], dim=1)


def pinball(pred, y, I, q):
    """|dV_hat| 를 |dV| 의 q 분위수로 민다 - 즉 안전 여유를 q 가 직접 조종한다.

    WHY THIS IS THE RIGHT KNOB AND NOT A FUDGE FACTOR
        SOP 반전 I* = (V_min - V_pre) / R_eff 는 R 에 대해 단조 감소다. 단조
        변환은 분위수를 보존하므로

            P(|I*_hat| > |I*_true|)  =  P(R_hat < R_true)  =  1 - q

        즉 저항 공간에서 q 를 고르면 전류 공간의 초과율이 정확히 1-q 로 정해진다.
        상수 derate 는 같은 일을 상태와 무관하게 하지만, 여기서는 12개 특징이
        어디서 얼마나 조심할지 조건부로 정할 수 있다.

    부호 정규화
        방전은 I<0, dV<0 이고 충전은 둘 다 양수다. s = sign(I) 를 곱하면 두
        방향 모두 e = |dV| - |dV_hat| 가 되어 같은 코드가 쓰인다.
    """
    s = torch.sign(I).unsqueeze(1)
    e = s * (y - pred)
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


def load_cells(data_dir=DATA_DIR):
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "trim_*.npz"))):
        cell = os.path.basename(f)[5:-4]
        z = np.load(f, allow_pickle=True)
        out[cell] = {"X": z["X"], "Y": z["Y"], "NOM": z["NOM"], "I": z["I"],
                     "SOH": z["m_SOH"].astype(float),
                     "SOC": z["m_SOC"].astype(float),
                     "cycle": z["m_cycle"].astype(int),
                     "rank": z["m_rank"].astype(str),
                     "exc": z["m_exc"].astype(float)}
    return out


def train_fold(model_cls, tr, te, epochs, lr, seed, dev, lam=1e-2, delta=0.02,
               ablate=None, q=0.0):
    torch.manual_seed(seed)
    Xtr = tr["X"].copy(); Xte = te["X"].copy()
    if ablate:
        Xtr[:, ablate] = 0.0; Xte[:, ablate] = 0.0
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xtr = torch.from_numpy(((Xtr - mu) / sd).clip(-4, 4)).to(dev)
    xte = torch.from_numpy(((Xte - mu) / sd).clip(-4, 4)).to(dev)
    ytr = torch.from_numpy(tr["Y"]).to(dev)
    ntr = torch.from_numpy(tr["NOM"]).to(dev)
    itr = torch.from_numpy(tr["I"]).to(dev)

    m = model_cls(xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(epochs * f) for f in (0.5, 0.75)], 0.1)
    hub = nn.HuberLoss(delta=delta)      # 1.2 % of pulses have R(10s) < R(2s)
    n = len(xtr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, 256):
            b = perm[k:k + 256]
            opt.zero_grad(set_to_none=True)
            kf, ks = decode(m(xtr[b]))
            dh = dv_hat(kf, ks, ntr[b], itr[b])
            loss = (pinball(dh, ytr[b], itr[b], q) if q > 0
                    else hub(dh, ytr[b]))
            loss = loss + lam * ((torch.log(kf) ** 2).mean()
                                 + (torch.log(ks) ** 2).mean())
            loss.backward(); opt.step()
        sch.step()
    m.eval()
    state = {"model": m.state_dict(), "mu": mu, "sd": sd,
             "cls": model_cls.__name__, "n_in": xtr.shape[1],
             "ablate": ablate, "KF_SPAN": KF_SPAN, "KS_SPAN": KS_SPAN}
    with torch.no_grad():
        kf, ks = decode(m(xte))
        pred = dv_hat(kf, ks,
                      torch.from_numpy(te["NOM"]).to(dev),
                      torch.from_numpy(te["I"]).to(dev)).cpu().numpy()
        kf = kf.cpu().numpy(); ks = ks.cpu().numpy()
    return pred, kf, ks, sum(p.numel() for p in m.parameters()), state


def rmse_mv(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="독립 학습본을 만들어 오차 막대를 재기 위한 "
                         "시드 오프셋. 7.3 절이 잰 재현 산포를 "
                         "헤드라인 수치에 붙이려면 필요하다.")
    ap.add_argument("--lam", type=float, default=1e-2,
                    help="prior pulling log k towards 0")
    ap.add_argument("--save-pred", action="store_true",
                    help="dump per-row predictions so a later comparison does "
                         "not have to retrain to get them")
    ap.add_argument("--quantile", type=float, default=0.0,
                    help="0 = symmetric Huber (the shipped trim). >0 trains a\n                         pinball loss so |dV_hat| targets that quantile of\n                         |dV|, which sets the SOP exceedance rate to 1-q")
    ap.add_argument("--rung", default="A4",
                    help="A0 no model | A3 linear | A4 MLP | A5 strip residual "
                         "channels | A7 tie k_s=k_f | A8 dR_fast 하나만 "
                         "(29.3 이 잰 '잔차 채널이 일한다' 를 배치 형태로 시험)")
    args = ap.parse_args()

    cells = load_cells(args.data)
    if len(cells) < 3:
        sys.exit(f"셀이 {len(cells)}개뿐 — 데이터셋 구축이 끝났는지 확인")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    cls = TrimLinear if args.rung in ("A3", "A5", "A8") else TrimMLP
    ablate = (list(range(6)) if args.rung == "A5"
              else (list(range(1, 12)) if args.rung == "A8" else None))

    print(f"rung {args.rung}   셀 {list(cells)}   {dev}")
    print(f"  {'홀드아웃':<22} {'n':>6} {'A0(k=1)':>9} {'모델':>9} {'개선':>8} "
          f"{'k_f 중앙':>9} {'k_s 중앙':>9} {'상한포화':>8}")
    summ = []
    for c in cells:
        te = cells[c]
        tr = {k: np.concatenate([cells[o][k] for o in cells if o != c])
              for k in ("X", "Y", "NOM", "I")}
        base = dv_hat(torch.ones(len(te["I"])), torch.ones(len(te["I"])),
                      torch.from_numpy(te["NOM"]),
                      torch.from_numpy(te["I"])).numpy()
        b_rmse = rmse_mv(base, te["Y"])
        if args.rung == "A0":
            print(f"  {c:<22} {len(te['I']):>6,} {b_rmse:>8.2f}m "
                  f"{'-':>9} {'-':>8} {'-':>9} {'-':>9}")
            summ.append({"cell": c, "A0": b_rmse})
            if args.save_pred:
                np.savez(os.path.join(args.out, f"pred_A0_{c}.npz"),
                         pred=base, Y=te["Y"], I=te["I"], NOM=te["NOM"],
                         cycle=te["cycle"], SOC=te["SOC"], SOH=te["SOH"],
                         rank=te["rank"], exc=te["exc"])
            continue
        ps, kfs, kss, states = [], [], [], []
        for s in range(args.seed_base, args.seed_base + args.seeds):
            p, kf, ks, npar, st = train_fold(cls, tr, te, args.epochs, args.lr, s,
                                             dev, lam=args.lam, ablate=ablate,
                                             q=args.quantile)
            ps.append(p); kfs.append(kf); kss.append(ks); states.append(st)
        p = np.mean(ps, 0); kf = np.mean(kfs, 0); ks = np.mean(kss, 0)
        m_rmse = rmse_mv(p, te["Y"])
        sat = float(np.mean((kf > 0.98 * np.exp(KF_SPAN))
                            | (ks > 0.98 * np.exp(KS_SPAN))))
        print(f"  {c:<22} {len(te['I']):>6,} {b_rmse:>8.2f}m {m_rmse:>8.2f}m "
              f"{(1-m_rmse/b_rmse)*100:>+7.1f}% {np.median(kf):>9.3f} "
              f"{np.median(ks):>9.3f} {sat*100:>7.1f}%")
        if args.save_pred:
            # Weights, not just predictions. Without them the model cannot be
            # re-queried on perturbed inputs, which is exactly what the SOP error
            # propagation needed and could not do.
            torch.save({"seeds": states, "cell": c, "rung": args.rung},
                       os.path.join(args.out, f"model_{args.rung}_{c}.pt"))
            np.savez(os.path.join(args.out, f"pred_{args.rung}_{c}.npz"),
                     pred=p, base=base, Y=te["Y"], I=te["I"], NOM=te["NOM"],
                     k_f=kf, k_s=ks, cycle=te["cycle"], SOC=te["SOC"],
                     SOH=te["SOH"], rank=te["rank"], exc=te["exc"])
        summ.append({"cell": c, "A0": b_rmse, "model": m_rmse,
                     "k_f": float(np.median(kf)), "k_s": float(np.median(ks)),
                     "params": npar, "q": args.quantile, "sat": sat})
    a0 = np.mean([s["A0"] for s in summ])
    if any("model" in s for s in summ):
        mm = np.mean([s["model"] for s in summ])
        print(f"  {'전체 평균':<22} {'':>6} {a0:>8.2f}m {mm:>8.2f}m "
              f"{(1-mm/a0)*100:>+7.1f}%")
    with open(os.path.join(args.out, f"summary_{args.rung}.json"), "w") as f:
        json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
