"""How much of the 4.94 A survives when SOC and SOH come from estimators?

WHAT THE HEADLINE NUMBER ASSUMED
    eval_sop_amps.py reads SOC and SOH out of the data file. Those are the
    quantities a BMS has to ESTIMATE, and this project has measured how well:
    the EKF holds SOC to 0.8-3 % at SOH >= 0.77 and 4-6 % at 0.69, and the
    charge-curve CNN reaches 0.0128 RMSE in SOH. Reporting 4.94 A without
    feeding those errors back in is reporting an oracle.

TWO ERROR SHAPES, BECAUSE THEY DO NOT COST THE SAME
    `random`      zero-mean Gaussian, redrawn per row. Averages out across a
                  trajectory and mostly inflates variance.
    `systematic`  one constant offset per cell, held for that cell's whole life.
                  This is what both estimators actually produce - the SOH CNN's
                  per-cell bias runs -1.4 to +1.6 %p and the EKF's SOC error
                  drifts rather than dithering - and a bias does not average out
                  of a power limit.

WHAT THIS DOES NOT COVER
    SOC and SOH are also two of the trim's twelve input features, so an error in
    them perturbs k_f and k_s as well as R_eff. That second path is not
    evaluated here because sop_trim.py saves predictions rather than weights, so
    the model cannot be re-queried on perturbed features. The numbers below are
    therefore a LOWER bound on the degradation.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces  # noqa: E402
from eval_sop_amps import solve_I, trim_k  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "sop_amps_eval.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.eval, encoding="utf-8")))
    cells = sorted({r["cell"] for r in rows})
    surf = {c: surfaces(c)[0] for c in cells}
    K = {c: trim_k(c) for c in cells}
    cell = np.array([r["cell"] for r in rows])
    cyc = np.array([int(r["cycle"]) for r in rows])
    soc = np.array([float(r["SOC"]) for r in rows])
    soh = np.array([float(r["SOH"]) for r in rows])
    vpre = np.array([float(r["V_pre_V"]) for r in rows])
    meas = np.array([float(r["I_meas_A"]) for r in rows])
    base = np.array([float(r["I_A3_A"]) for r in rows])

    def run(dsoc, dsoh):
        out = np.empty(len(rows))
        for i in range(len(rows)):
            c = cell[i]
            kf, ks = K[c][cyc[i]]
            s = float(np.clip(soc[i] + dsoc[i], 0.02, 1.0))
            h = float(np.clip(soh[i] + dsoh[i], 0.60, 1.05))
            out[i] = solve_I(surf[c], s, h, vpre[i], 2.5, 10.0, kf, ks, I0=meas[i])
        return out

    rmse = lambda p: float(np.sqrt(np.nanmean((p - meas) ** 2)))
    opt = lambda p: float(np.nanmean(np.abs(p) - np.abs(meas) > 5.0) * 100)
    z = np.zeros(len(rows))
    print(f"  기준(오라클 상태): RMSE {rmse(base):.2f} A, "
          f"5A 이상 낙관 {opt(base):.1f} %")

    print(f"\n  === 단독 민감도 (계통 오차, 전 행에 상수) ===")
    print(f"  {'섭동':<22} {'RMSE':>8} {'변화':>8} {'5A+ 낙관':>10} "
          f"{'dI*/d 중앙':>12}")
    for name, ds, dh in (("SOC +0.02", 0.02, 0), ("SOC -0.02", -0.02, 0),
                         ("SOC +0.05", 0.05, 0), ("SOC -0.05", -0.05, 0),
                         ("SOH +0.013", 0, 0.013), ("SOH -0.013", 0, -0.013),
                         ("SOH +0.03", 0, 0.03), ("SOH -0.03", 0, -0.03)):
        p = run(z + ds, z + dh)
        d = np.nanmedian(np.abs(p) - np.abs(base))
        step = ds if ds else dh
        print(f"  {name:<22} {rmse(p):>7.2f}A {rmse(p)-rmse(base):>+7.2f}A "
              f"{opt(p):>9.1f}% {d/step:>11.1f} A/단위")

    print(f"\n  === 현실적 조합 ({args.seeds} 시드 평균) ===")
    print(f"  {'시나리오':<34} {'RMSE':>8} {'변화':>8} {'5A+ 낙관':>10}")
    scen = [
        ("무작위  SOC 2%, SOH 0.013", "rand", 0.02, 0.013),
        ("무작위  SOC 4%, SOH 0.013", "rand", 0.04, 0.013),
        ("계통    SOC 2%, SOH 0.013", "sys", 0.02, 0.013),
        ("계통    SOC 4%, SOH 0.013", "sys", 0.04, 0.013),
        ("계통    SOC 4%, SOH 0.025", "sys", 0.04, 0.025),
    ]
    for name, kind, ssoc, ssoh in scen:
        rs, os_ = [], []
        for sd in range(args.seeds):
            rng = np.random.default_rng(sd)
            if kind == "rand":
                ds = rng.normal(0, ssoc, len(rows))
                dh = rng.normal(0, ssoh, len(rows))
            else:
                bs = {c: rng.normal(0, ssoc) for c in cells}
                bh = {c: rng.normal(0, ssoh) for c in cells}
                ds = np.array([bs[c] for c in cell])
                dh = np.array([bh[c] for c in cell])
            p = run(ds, dh)
            rs.append(rmse(p)); os_.append(opt(p))
        print(f"  {name:<34} {np.mean(rs):>7.2f}A {np.mean(rs)-rmse(base):>+7.2f}A "
              f"{np.mean(os_):>9.1f}%")

    print(f"\n  === SOC 오차의 영향이 SOC 구간에 따라 다른가 (계통 +0.04) ===")
    p = run(z + 0.04, z)
    d = np.abs(p) - np.abs(base)
    print(f"  {'SOC 구간':<14} {'n':>6} {'|I*| 중앙':>9} {'I* 변화 중앙':>13}")
    for lo, hi in ((0.02, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
        m = (soc >= lo) & (soc < hi)
        if m.sum() < 30:
            continue
        print(f"  {f'{lo:.2f}~{hi:.2f}':<14} {m.sum():>6,} "
              f"{np.median(np.abs(meas[m])):>8.1f}A {np.nanmedian(d[m]):>+12.2f}A")


if __name__ == "__main__":
    main()
