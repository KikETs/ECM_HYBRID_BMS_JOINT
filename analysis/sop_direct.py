"""Regress the SOP current directly, instead of predicting voltage and inverting.

WHY THIS IS A DIFFERENT PROPOSITION FROM THE FULL-AI ARM
    The reference LSTM predicts voltage and the SOP answer comes from a binary
    search over current. That search does not converge on aged cells: it hits the
    bound on 57 % of rows for M1 and 96 % for M2, because SOP asks about 60-100 A
    while nothing in training exceeds 30 A, and a network does not extrapolate -
    M1 flattens (dV/ds falls 6x past s = 2) and M2 reverses sign, predicting
    4.589 V where the cell cannot exceed 4.2 (sop_hybrid_spec.md 11.4).

    Regressing I* removes the extrapolation entirely, because the target IS the
    answer. No search, no inversion, no query outside the training range.

WHAT IT IS ALLOWED TO SEE
    Exactly what the hybrid sees: SOC, SOH, the measured terminal voltage before
    the demand, temperature, and the twelve O(1) statistics the trim already
    computes from the preceding drive cycle. It must NOT see the pulse it is
    being asked about - a vehicle asking "how much can I draw" has not drawn it
    yet.

THREE FORMS, BECAUSE THE INTERESTING QUESTION IS WHERE THE PHYSICS GOES
    D0  I* from features alone. Pure data-driven.
    D1  I* / I*_ECM, i.e. a multiplicative correction on the physics answer -
        the same shape as the hybrid but learned in the ampere domain.
    D2  I* - I*_ECM, an additive correction.
    The baseline to beat is the hybrid's own inversion, 4.94 A, and the
    uncorrected ECM, 7.26 A, on the identical rows.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "sop_amps_eval.csv")
TRIM = os.path.join(HERE, "cache", "trim")
CELLS = ["BOOST", "BOOST_NEGPULSE", "BOOST_NEGPULSE_1S", "BOOST_REST",
         "CC", "CC_CELL2"]
FEAT = ("dR_fast", "dR_slow", "log_exc", "I_hi", "f_rest", "duty",
        "SOC", "SOH", "T", "I_rms", "R_fast_nom", "R_slow_nom")


E2E = os.path.join(HERE, "sop_end_to_end.csv")


def build(end_to_end=False):
    """end_to_end: SOC comes from the EKF, not the file.

    The ECM baseline is re-solved at the estimated SOC too, so the residual the
    model learns sits on top of the answer a BMS would actually have. The row set
    shrinks to the pulses the filter covers, so every method in the comparison is
    rescored on that subset rather than carried over.
    """
    rows = list(csv.DictReader(open(EVAL, encoding="utf-8")))
    if end_to_end:
        e2 = list(csv.DictReader(open(E2E, encoding="utf-8")))
        vp = {(r["cell"], int(r["cycle"]), round(float(r["SOC"]), 4)):
              r for r in rows}
        rows = []
        for r in e2:
            k = (r["cell"], int(r["cycle"]), round(float(r["SOC_true"]), 4))
            if k not in vp:
                continue
            q = dict(vp[k])
            q["SOC_est"] = r["SOC_ekf"]
            rows.append(q)
        # The ECM baseline must be re-solved at the ESTIMATED SOC as well.
        # Leaving it at the true-SOC answer would let the residual model learn on
        # top of a physics answer the BMS does not have, and would credit the
        # chain with an accuracy that only exists on paper.
        import sys as _sys
        _sys.path.insert(0, HERE)
        from ecm_pool import surfaces as _surf
        from eval_sop_amps import solve_I as _solve
        S = {}
        for q in rows:
            c = q["cell"]
            if c not in S:
                S[c] = _surf(c)[0]
            v = _solve(S[c], float(np.clip(float(q["SOC_est"]), 0.02, 1.0)),
                       float(q["SOH"]), float(q["V_pre_V"]), 2.5, 10.0, 1.0, 1.0,
                       I0=float(q["I_meas_A"]))
            q["I_A0_A"] = v if np.isfinite(v) else float(q["I_A0_A"])
        rows = [q for q in rows if np.isfinite(float(q["I_A0_A"]))]
    X, y, ecm, cell, extra = [], [], [], [], []
    for c in CELLS:
        z = np.load(os.path.join(TRIM, f"trim_{c}.npz"), allow_pickle=True)
        cy = z["m_cycle"].astype(int)
        # one feature vector per characterisation: the LAST block, which is the
        # most recent read and beat the twelve-block mean by 5.5 % (13.1)
        last = {}
        for i, q in enumerate(cy):
            last[int(q)] = i
        for r in rows:
            if r["cell"] != c:
                continue
            q = int(r["cycle"])
            if q not in last:
                continue
            f = z["X"][last[q]].copy()
            if "SOC_est" in r:
                f[FEAT.index("SOC")] = float(r["SOC_est"])
            X.append(f)
            # headroom is the single most physical quantity available and is not
            # in the twelve: how far the terminal voltage already is from the floor
            extra.append([float(r["V_pre_V"]) - 2.5])
            y.append(abs(float(r["I_meas_A"])))
            ecm.append(abs(float(r["I_A0_A"])))
            cell.append(c)
    return (np.array(X, np.float32), np.array(extra, np.float32),
            np.array(y, np.float32), np.array(ecm, np.float32), np.array(cell))


class Tiny(nn.Module):
    def __init__(self, n, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(Xtr, ytr, Xte, kind, epochs, seed):
    torch.manual_seed(seed)
    if kind == "linear":
        A = np.column_stack([Xtr, np.ones(len(Xtr))])
        w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ytr)
        return np.column_stack([Xte, np.ones(len(Xte))]) @ w
    m = Tiny(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(epochs * f) for f in (0.5, 0.75)], 0.1)
    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    lf = nn.SmoothL1Loss(beta=0.1)
    for ep in range(epochs):
        p = torch.randperm(len(xt))
        for k in range(0, len(xt), 256):
            b = p[k:k + 256]
            opt.zero_grad(set_to_none=True)
            lf(m(xt[b]), yt[b]).backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(Xte)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--end-to-end", action="store_true")
    ap.add_argument("--save-pred", default=None)
    args = ap.parse_args()

    X, ex, y, ecm, cell = build(args.end_to_end)
    meta = build.__dict__.get("meta")
    F = np.column_stack([X, ex]).astype(np.float32)
    print(f"  표본 {len(y):,}  특징 {F.shape[1]}  I* {y.min():.1f}~{y.max():.1f} A")
    print(f"  기준: ECM {np.sqrt(np.mean((ecm-y)**2)):.2f} A, "
          f"하이브리드 역산 4.94 A (동일 행 아님 — 아래 표에서 같은 행으로 재계산)")

    FORMS = [("D0  I* 직접", lambda p, e: p, lambda t, e: t),
             ("D1  I*/I_ecm", lambda p, e: p * e, lambda t, e: t / e),
             ("D2  I*-I_ecm", lambda p, e: p + e, lambda t, e: t - e)]
    print(f"\n  {'형태':<14} {'모델':<8} " + "".join(f"{c[:8]:>9}" for c in CELLS)
          + f"{'평균':>8} {'최악':>8}")
    for nm, dec, enc in FORMS:
        for kind, ne in (("linear", 0), ("tiny", args.epochs)):
            per = []
            for c in CELLS:
                tr, te = cell != c, cell == c
                mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
                a = ((F[tr] - mu) / sd).astype(np.float32)
                b = ((F[te] - mu) / sd).astype(np.float32)
                t = enc(y[tr], ecm[tr]).astype(np.float32)
                ps = [fit(a, t, b, kind, ne, s) for s in range(args.seeds if kind
                                                               == "tiny" else 1)]
                p = dec(np.mean(ps, 0), ecm[te])
                per.append(float(np.sqrt(np.mean((p - y[te]) ** 2))))
            print(f"  {nm:<14} {kind:<8} " + "".join(f"{v:>8.2f}A" for v in per)
                  + f"{np.mean(per):>7.2f}A {np.max(per):>7.2f}A")
    per = [float(np.sqrt(np.mean((ecm[cell == c] - y[cell == c]) ** 2)))
           for c in CELLS]
    print(f"  {'ECM (보정없음)':<14} {'-':<8} " + "".join(f"{v:>8.2f}A" for v in per)
          + f"{np.mean(per):>7.2f}A {np.max(per):>7.2f}A")
    rows = list(csv.DictReader(open(EVAL, encoding="utf-8")))
    hy = {}
    for c in CELLS:
        v = [(abs(float(r["I_A3_A"])) - abs(float(r["I_meas_A"])))
             for r in rows if r["cell"] == c]
        hy[c] = float(np.sqrt(np.mean(np.array(v) ** 2)))
    print(f"  {'하이브리드 역산':<14} {'-':<8} "
          + "".join(f"{hy[c]:>8.2f}A" for c in CELLS)
          + f"{np.mean(list(hy.values())):>7.2f}A {max(hy.values()):>7.2f}A")


if __name__ == "__main__":
    main()
