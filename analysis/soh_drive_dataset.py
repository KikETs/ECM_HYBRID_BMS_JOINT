"""Drive-cycle windows for SOH, with the channels that leak the label removed.

WHAT THIS ARM IS FOR
    The charge-curve CNN (soh_cnn.py) reaches 0.0128 but only fires when the car
    happens to charge at 1C through 3.55-4.05 V. This arm asks the same question
    of ordinary driving, where the answer is always available and the signal is
    much weaker. The comparison between the two IS the result: accuracy against
    availability, which is the trade a BMS actually has to make.

SOC IS EXCLUDED, AND NOT AS A PRECAUTION
    The cache's rated SOC is built as 1 - cap*(1 - SOC_aged/100)/3.0, so a file
    that discharges to empty reports 1 - cap/3.0 at its floor. Measured over 102
    files of CC_CELL2, the minimum SOC of a file correlates with SOH at
    r = -0.9996 and predicts it by simple regression with residual RMSE 0.00253 -
    five times better than the charge-curve CNN. Handing a model that channel
    would not be a subtle leak, it would be handing over the label.

TEMPERATURE IS EXCLUDED TOO, BUT FOR A WEAKER REASON
    T_mean correlates -0.445 with SOH. Some of that is real (an aged cell runs
    hotter at the same load) and some of it may be a months-long test drifting
    through ambient. It does not have to be settled, because dropping it costs
    nothing: at a 7200 s window the hand-feature baseline goes 0.0382 -> 0.0377
    WITHOUT temperature. A channel that is free to remove and might carry a
    calendar signal should be removed.

SUBSAMPLED, NOT AVERAGED
    The load-bearing statistic is the slope of V on I, which survives taking
    every Nth sample - both channels are read at the same instant, so the pairs
    stay valid. Block-averaging would destroy it: averaging current over 30 s
    while the voltage responds through two RC branches does not commute.

WHAT IS STORED FOR AUDIT AND MUST NEVER BE FED
    SOC, T, cycle and the window's offset within its file are kept in the
    archive under `audit_` names so the chronology question ("is the model
    reading the clock rather than the cell?") can be asked later. They are not
    inputs.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_t")
OUT = os.path.join(HERE, "cache", "soh_drive.npz")
CELLS = ("BOOST", "BOOST_NEGPULSE", "BOOST_NEGPULSE_1S", "BOOST_REST",
         "CC", "CC_CELL2")

WIN_S = 7200          # window length in seconds (data is 1 Hz)
STRIDE_S = 1800       # 75 % overlap; windows of one cell are correlated, which
                      # is why the split is by cell and never by window
SUB = 10              # keep every 10th sample -> 720 per window
MIN_I_STD = 1.0       # A, below which the window carries no excitation

FEAT_NAMES = ("R_mOhm", "OCV0", "V_mean", "V_std", "V_min", "V_max",
              "absI_mean", "I_std", "absI_max", "resid_mV")


def hand_features(v, i):
    """The ten O(1) statistics the network has to beat. No temperature."""
    R = np.cov(v, i)[0, 1] / np.var(i)
    ocv0 = v.mean() - R * i.mean()
    res = v - (ocv0 + R * i)
    return np.array([R * 1000, ocv0, v.mean(), v.std(), v.min(), v.max(),
                     np.abs(i).mean(), i.std(), np.abs(i).max(),
                     res.std() * 1000], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--win", type=int, default=WIN_S)
    ap.add_argument("--stride", type=int, default=STRIDE_S)
    ap.add_argument("--sub", type=int, default=SUB)
    args = ap.parse_args()

    W, S, U = args.win, args.stride, args.sub
    Xs, Xf, y, cell, a_soc, a_T, a_cyc, a_pos = [], [], [], [], [], [], [], []
    print(f"  창 {W}s, 보폭 {S}s, {U}배 솎음 -> {W // U} 샘플, 입력 (V, I)")
    print(f"  {'셀':<20} {'파일':>5} {'창':>7} {'제외: 무효':>10} {'여기부족':>9}")
    for cname in CELLS:
        p = os.path.join(args.cache, f"uypydj_{cname}_Fifteen_Drive_Cycles.npz")
        z = np.load(p, allow_pickle=True)
        lens, files = z["lens"], z["files"]
        off = np.concatenate([[0], np.cumsum(lens)])
        V, I, T, SOC, SOH, ok = (z[k] for k in
                                 ("V", "I", "T", "SOC", "SOH", "valid"))
        n_w = n_bad = n_dull = 0
        for k in range(len(lens)):
            m = re.match(r"(\d+)\|", str(files[k]))
            if not m:
                continue
            cyc = int(m.group(1))
            sl = slice(off[k], off[k] + lens[k])
            v, i, t, s, h, o = V[sl], I[sl], T[sl], SOC[sl], SOH[sl], ok[sl]
            good = (o.astype(bool) & np.isfinite(v) & np.isfinite(i)
                    & np.isfinite(t))
            for a in range(0, len(v) - W, S):
                b = a + W
                if not good[a:b].all():
                    n_bad += 1
                    continue
                vv, ii = v[a:b].astype(np.float32), i[a:b].astype(np.float32)
                if np.std(ii) < MIN_I_STD:
                    n_dull += 1
                    continue
                Xs.append(np.stack([vv[::U], ii[::U]], 1))
                Xf.append(hand_features(vv, ii))
                y.append(float(np.nanmedian(h[a:b])))
                cell.append(cname)
                a_soc.append([float(np.nanmin(s[a:b])),
                              float(np.nanmedian(s[a:b]))])
                a_T.append(float(np.nanmean(t[a:b])))
                a_cyc.append(cyc)
                a_pos.append(a / max(len(v) - W, 1))
                n_w += 1
        print(f"  {cname:<20} {len(lens):>5} {n_w:>7,} {n_bad:>10,} {n_dull:>9,}")

    Xs = np.stack(Xs).astype(np.float32)
    Xf = np.stack(Xf).astype(np.float32)
    y = np.array(y, np.float32)
    cell = np.array(cell)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, X=Xs, Xf=Xf, y=y, cell=cell,
        feat_names=np.array(FEAT_NAMES),
        audit_soc=np.array(a_soc, np.float32), audit_T=np.array(a_T, np.float32),
        audit_cycle=np.array(a_cyc, np.int32),
        audit_pos=np.array(a_pos, np.float32),
        win_s=W, stride_s=S, sub=U)
    mb = os.path.getsize(args.out) / 1e6
    print(f"\n  {args.out}  {mb:.1f} MB")
    print(f"  창 {len(y):,}개, 계열 {Xs.shape[1]}x{Xs.shape[2]}, "
          f"손특징 {Xf.shape[1]}개, SOH {y.min():.3f}~{y.max():.3f}")

    # the number a network has to beat, printed here so it cannot be chosen later
    print(f"\n  {'기준선 (leave-one-cell-out)':<32} {'RMSE':>8}")
    cs = sorted(set(cell.tolist()))
    e = np.concatenate([np.full((cell == c).sum(), y[cell != c].mean())
                        - y[cell == c] for c in cs])
    print(f"  {'평균 예측':<32} {np.sqrt(np.mean(e ** 2)):>8.4f}")
    for nm, F in (("저항 1개", Xf[:, [0]]), ("손특징 10개 릿지", Xf)):
        es = []
        for c in cs:
            tr, te = cell != c, cell == c
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
            A = np.column_stack([(F[tr] - mu) / sd, np.ones(tr.sum())])
            w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y[tr])
            es.append(np.column_stack([(F[te] - mu) / sd,
                                       np.ones(te.sum())]) @ w - y[te])
        e = np.concatenate(es)
        print(f"  {nm:<32} {np.sqrt(np.mean(e ** 2)):>8.4f}")
    print(f"  {'(참고) 충전곡선 CNN':<32} {0.0128:>8.4f}")


if __name__ == "__main__":
    main()
