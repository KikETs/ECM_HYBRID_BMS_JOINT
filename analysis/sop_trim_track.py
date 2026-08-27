"""Track the multiplier along the aging trajectory instead of re-guessing it.

WHY A TRACKER AND NOT BETTER FEATURES
    Measured variance decomposition (sop_hybrid_spec.md 13.6): log k_s varies
    79 % WITHIN a cell and only 21 % between cells, while log k_f is the other way
    round (63 % between). Redesigning the instantaneous features moved k_s not at
    all - five filter constants over a 16x range and a load gate all failed -
    because the thing being chased is not a cell-to-cell difference, it is one
    cell's path through time. The estimator currently treats every
    characterisation as independent, which throws that structure away.

THE MODEL, AND WHY ITS NOISES CAN BE FITTED WITHOUT LABELS
    x_n = x_{n-1} + w,   w ~ N(0, q * dcycle)      the multiplier drifts with age
    z_n = x_n + v,       v ~ N(0, R)               the trim's output is a read

    Differencing gives an MA(1): Var(dz) = q*dcycle + 2R and Cov(dz_n, dz_{n-1})
    = -R. So q and R come from the first two moments of the DIFFERENCED estimate
    sequence - no labels needed. A deployed BMS could do the same online, and
    here they are fitted on the training cells only.

CAUSAL AND NON-CAUSAL ARE BOTH REPORTED
    The filter is what a BMS can run. The RTS smoother sees the future and is
    reported only as the ceiling this model class could reach - it is not a
    deployable number and is labelled so.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["BOOST", "BOOST_NEGPULSE", "BOOST_NEGPULSE_1S", "BOOST_REST",
         "CC", "CC_CELL2"]


def per_characterisation(cell, runs=os.path.join(HERE, "runs_trim")):
    z = np.load(os.path.join(runs, f"pred_A3_{cell}.npz"), allow_pickle=True)
    key = np.array([f"{a}|{b:.4f}|{r}" for a, b, r in
                    zip(z["cycle"].astype(int), z["SOC"].astype(float),
                        z["rank"].astype(str))])
    uk, inv = np.unique(key, return_inverse=True)
    last = np.zeros(len(uk), int)
    for i, g in enumerate(inv):
        last[g] = i
    cy = z["cycle"].astype(int)[last]
    order = np.argsort(cy, kind="stable")
    cyc = np.unique(cy)
    seq = {"cycle": cyc,
           "kf": np.array([np.median(z["k_f"][last][cy == c]) for c in cyc]),
           "ks": np.array([np.median(z["k_s"][last][cy == c]) for c in cyc])}
    return seq, {"idx": last, "cycle": cy, "NOM": z["NOM"][last],
                 "I": z["I"][last], "Y": z["Y"][last], "SOH": z["SOH"][last]}


def fit_qr(seqs):
    """Moment identification of (q, R) from differenced sequences, per channel."""
    out = {}
    for ch in ("kf", "ks"):
        g0, g1, n = [], [], 0
        for s in seqs:
            y = np.log(s[ch]); d = np.diff(y)
            dc = np.maximum(np.diff(s["cycle"]).astype(float), 1.0)
            dn = d / np.sqrt(dc)              # normalise the walk part
            g0.append(dn ** 2)
            g1.append(d[1:] * d[:-1])
            n += len(d)
        g0 = float(np.mean(np.concatenate(g0)))
        g1 = float(np.mean(np.concatenate(g1)))
        R = max(-g1, 1e-8)
        q = max(g0 - 2 * R / np.mean([np.mean(np.diff(s["cycle"])) for s in seqs]),
                1e-10)
        out[ch] = (q, R)
    return out


def kalman(y, dc, q, R, smooth=False):
    n = len(y)
    xf = np.empty(n); Pf = np.empty(n); xp = np.empty(n); Pp = np.empty(n)
    x, P = y[0], R
    for i in range(n):
        if i:
            P = P + q * dc[i - 1]
        xp[i], Pp[i] = x, P
        K = P / (P + R)
        x = x + K * (y[i] - x); P = (1 - K) * P
        xf[i], Pf[i] = x, P
    if not smooth:
        return xf
    xs = xf.copy()
    for i in range(n - 2, -1, -1):
        A = Pf[i] / Pp[i + 1]
        xs[i] = xf[i] + A * (xs[i + 1] - xp[i + 1])
    return xs


def dv(kf, ks, NOM, I):
    return np.stack([I * (kf * NOM[:, 0] + ks * NOM[:, 1]),
                     I * (kf * NOM[:, 2] + ks * NOM[:, 3])], 1)


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs_trim"))
    args = ap.parse_args()

    S, P = {}, {}
    for c in CELLS:
        S[c], P[c] = per_characterisation(c, args.runs)

    print(f"  {'홀드아웃':<20} {'원본':>8} {'칼만(인과)':>11} {'평활(비인과)':>13} "
          f"{'오라클':>8} | {'개선':>7}")
    raw, kal, smo, orc = [], [], [], []
    for c in CELLS:
        qr = fit_qr([S[o] for o in CELLS if o != c])     # 학습 셀에서만
        d = P[c]; s = S[c]
        dc = np.maximum(np.diff(s["cycle"]).astype(float), 1.0)
        out = {}
        for mode in ("filt", "smooth"):
            kf = np.exp(kalman(np.log(s["kf"]), dc, *qr["kf"], mode == "smooth"))
            ks = np.exp(kalman(np.log(s["ks"]), dc, *qr["ks"], mode == "smooth"))
            m = {cc: i for i, cc in enumerate(s["cycle"])}
            gi = np.array([m[x] for x in d["cycle"]])
            out[mode] = dv(kf[gi], ks[gi], d["NOM"], d["I"])
        m = {cc: i for i, cc in enumerate(s["cycle"])}
        gi = np.array([m[x] for x in d["cycle"]])
        a = rmse(dv(s["kf"][gi], s["ks"][gi], d["NOM"], d["I"]), d["Y"])
        b = rmse(out["filt"], d["Y"]); e = rmse(out["smooth"], d["Y"])
        # 오라클: 특성화마다 최적 k
        op = np.empty_like(d["Y"])
        for cc in s["cycle"]:
            q = d["cycle"] == cc
            A = np.vstack([np.column_stack([d["I"][q] * d["NOM"][q, 0],
                                            d["I"][q] * d["NOM"][q, 1]]),
                           np.column_stack([d["I"][q] * d["NOM"][q, 2],
                                            d["I"][q] * d["NOM"][q, 3]])])
            k, *_ = np.linalg.lstsq(A, np.concatenate([d["Y"][q, 0], d["Y"][q, 1]]),
                                    rcond=None)
            op[q] = dv(k[0], k[1], d["NOM"][q], d["I"][q])
        o = rmse(op, d["Y"])
        raw.append(a); kal.append(b); smo.append(e); orc.append(o)
        print(f"  {c:<20} {a:>7.1f}m {b:>10.1f}m {e:>12.1f}m {o:>7.1f}m | "
              f"{(1-b/a)*100:>+6.1f}%")
    f = lambda v: (np.mean(v), np.max(v))
    print(f"  {'평균':<20} {np.mean(raw):>7.1f}m {np.mean(kal):>10.1f}m "
          f"{np.mean(smo):>12.1f}m {np.mean(orc):>7.1f}m | "
          f"{(1-np.mean(kal)/np.mean(raw))*100:>+6.1f}%")
    print(f"  {'최악 셀':<20} {np.max(raw):>7.1f}m {np.max(kal):>10.1f}m "
          f"{np.max(smo):>12.1f}m {np.max(orc):>7.1f}m | "
          f"{(1-np.max(kal)/np.max(raw))*100:>+6.1f}%")
    qr = fit_qr([S[c] for c in CELLS])
    for ch in ("kf", "ks"):
        q, R = qr[ch]
        print(f"  {ch}: q={q:.3e} /cycle, R={R:.3e}  -> 정상상태 게인 "
              f"{np.sqrt(q*37)/(np.sqrt(q*37)+np.sqrt(R)):.2f} (37 사이클 간격)")


if __name__ == "__main__":
    main()
