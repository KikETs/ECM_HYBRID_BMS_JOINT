"""Refine HPPC-derived ECM parameters on the drive cycles they will be used on.

WHY REFINE AT ALL
    The HPPC fit is excellent on its own terms - 0.455 mV median residual over
    27,891 pulses, reproducing an independently extracted Reff at r = 0.99999.
    But open loop on a drive cycle it degrades with age: 13.7 mV at SOH 1.00,
    108 mV at 0.70. Four alternative fits were tried (2 s window, 3RC, including
    60 s of relaxation) and all made the drive cycle WORSE, so this is not a
    fitting-window problem. The structural difference is the regime: every HPPC
    pulse starts from at least 30 s of rest, while a drive cycle never rests.

WHAT IS ALLOWED TO CHANGE, AND WHY SO LITTLE
    Only three scalars, one per branch:

        R0 -> m0 R0,   R1 -> m1 R1,   R2 -> m2 R2

    The SOC, SOH and current dependence measured from HPPC is kept exactly. The
    point is to correct a regime bias, not to refit the cell - with 20,000
    samples per run it would be easy to fit something that describes this run
    and nothing else.

THE FIT IS LINEAR, SO THERE IS NO OPTIMISER TO TUNE
    SOC comes from coulomb counting and does not depend on the multipliers, so
    theta along the trajectory can be precomputed once. Each RC branch voltage
    is then linear in its own resistance, giving

        V - OCV  =  m0 (I R0)  +  m1 v1_base  +  m2 v2_base

    with v1_base, v2_base simulated at unit multiplier. Ordinary least squares
    solves it exactly. No initial guess, no convergence to check.

HELD OUT BY CONSTRUCTION
    Multipliers are fitted on the first half of a run and reported on the
    second, and separately on a different run at similar SOH. A multiplier that
    only helps the samples it was fitted on is not a correction.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_surface import ECMSurface  # noqa: E402

Q_RATED_AH = 3.0


def trajectory(sd, sc, soc0, soh, I, T=None, dt=1.0):
    """Precompute theta and the unit-multiplier branch voltages along a run."""
    n = len(I)
    ocv = np.empty(n); r0 = np.empty(n)
    v1 = np.empty(n); v2 = np.empty(n)
    x1 = x2 = 0.0
    soc = soc0
    socs = np.empty(n)
    for k in range(n):
        s = sc if I[k] > 0 else sd
        th = s.theta(soc, soh, I[k], 25.0 if T is None else float(T[k]))
        o, _ = s.ocv(soc, soh)
        R0 = float(th["R0"][0]); R1 = float(th["R1"][0])
        t1 = float(th["tau1"][0]); R2 = float(th["R2"][0]); t2 = float(th["tau2"][0])
        ocv[k] = float(o[0]); r0[k] = R0
        v1[k] = x1; v2[k] = x2
        socs[k] = soc
        a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)
        x1 = x1 * a1 + R1 * (1 - a1) * I[k]
        x2 = x2 * a2 + R2 * (1 - a2) * I[k]
        soc = min(max(soc + I[k] * dt / 3600.0 / Q_RATED_AH, -0.05), 1.05)
    return {"ocv": ocv, "iR0": I * r0, "v1": v1, "v2": v2, "soc": socs}


def solve(tr, V, mask=None):
    A = np.column_stack([tr["iR0"], tr["v1"], tr["v2"]])
    y = V - tr["ocv"]
    if mask is not None:
        A, y = A[mask], y[mask]
    m, *_ = np.linalg.lstsq(A, y, rcond=None)
    return m


def apply(tr, m):
    return tr["ocv"] + m[0] * tr["iR0"] + m[1] * tr["v1"] + m[2] * tr["v2"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="CC")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=20000)
    args = ap.parse_args()

    sd = ECMSurface(args.cell, "discharge")
    sc = ECMSurface(args.cell, "charge")
    z = np.load(os.path.join(args.cache,
                             f"uypydj_{args.cell}_Fifteen_Drive_Cycles.npz"))
    lens = z["lens"]; off = np.concatenate([[0], np.cumsum(lens)])
    idx = np.linspace(0, len(lens) - 1, args.runs).astype(int)

    print(f"cell {args.cell}   보정 전/후, 전반부 적합 -> 후반부 검증\n")
    print(f"{'run':>4} {'SOH':>6} {'m0':>6} {'m1':>6} {'m2':>6} "
          f"{'전 RMSE':>9} {'후(적합부)':>11} {'후(검증부)':>11}")
    keep = []
    for k in idx:
        sl = slice(off[k], off[k] + lens[k])
        soc, V, I, SOH = z["SOC"][sl], z["V"][sl], z["I"][sl], z["SOH"][sl]
        T = z["T"][sl]
        ok = np.isfinite(soc) & np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        if ok.sum() < 2000:
            continue
        soc, V, I, T = (x[ok][:args.max_samples] for x in (soc, V, I, T))
        soh = float(np.nanmedian(SOH))
        tr = trajectory(sd, sc, float(soc[0]), soh, I, T)

        half = len(V) // 2
        mfit = np.zeros(len(V), bool); mfit[:half] = True
        m = solve(tr, V, mfit)
        pred = apply(tr, m)
        base = apply(tr, np.ones(3))
        keep.append((soh, m))
        print(f"{k:>4} {soh:>6.3f} {m[0]:>6.3f} {m[1]:>6.3f} {m[2]:>6.3f} "
              f"{rmse(base, V):>8.1f}m {rmse(pred[mfit], V[mfit]):>10.1f}m "
              f"{rmse(pred[~mfit], V[~mfit]):>10.1f}m")

    if len(keep) >= 3:
        a = np.array([[s] + list(m) for s, m in keep])
        print("\n  배율의 SOH 의존성 (기울기, SOH 1.00 -> 0.70)")
        for j, lab in ((1, "m0"), (2, "m1"), (3, "m2")):
            sl_ = np.polyfit(a[:, 0], a[:, j], 1)[0]
            print(f"    {lab}: {np.interp(1.0, a[::-1,0], a[::-1,j]):.3f} -> "
                  f"{np.interp(0.70, a[::-1,0], a[::-1,j]):.3f}  "
                  f"(기울기 {sl_:+.3f}/SOH)")


if __name__ == "__main__":
    main()
