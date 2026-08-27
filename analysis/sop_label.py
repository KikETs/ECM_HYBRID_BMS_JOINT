"""Measured SOP labels: the current that would drive V(tau) to the floor.

WHY THE EXISTING sop_reference.csv CANNOT BE THE LABEL
    It is derived from the same ECM the hybrid arm corrects, so scoring that arm
    against it grades the model on its own assumptions. Worse, its numbers are
    mostly not measurements: 84.8 % of its rows extrapolate past the largest
    current the cycler ever applied (median I* 66.7 A against a measured maximum
    of 29.0 A), and 82.7 % are decided by the current rating rather than by
    voltage - so those rows test nothing about a voltage model at all.

WHAT IS MEASURED HERE INSTEAD
    At every (cell, cycle, SOC group) the HPPC steps FOUR discharge rates - about
    1C, 4C, 8C and 11.5C when fresh - and records V(tau) for each. Four points of
    a V-versus-I characteristic at one operating point, measured, with no model
    in between. Solving that characteristic for V(tau) = V_min gives I*.

    Interpolation where the floor sits inside the measured fan, extrapolation
    where it does not. The distinction is not hidden: every row carries
    `extrap` = |I*| / max|I_measured|, and a row with extrap > 1 is a projection,
    not a measurement.

TWO FITS, BECAUSE THE CHARACTERISTIC IS NOT A LINE
    R1 and R2 fall to about 0.7x between 2.6 A and 29.6 A (findings.md 4.2), so a
    straight line through all four rates is pulled by the low-rate points into
    overstating the resistance at the high-rate end, which UNDERSTATES I*. The
    file therefore reports two solves per row:

      lin4    least squares on all four rates
      lin2hi  the two highest rates only - fewer points, but taken where SOP
              actually lives

    Their disagreement is the label's own uncertainty and is written out as
    `spread_A`. A row where the two disagree by more than a few amps should not
    be used to grade anything to better than that.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES_CSV = os.path.join(HERE, "uypydj_hppc_resistance.csv")
OUT = os.path.join(HERE, "sop_label_measured.csv")
OUT_CHG = os.path.join(HERE, "sop_label_charge.csv")
V_MIN = 2.5
V_MAX = 4.2
Q_RATED = 3.0


def solve(I, V, v_min):
    """Least-squares line V = a + b*I, solved for V = v_min. Returns (I*, r2)."""
    if len(I) < 2:
        return np.nan, np.nan
    b, a = np.polyfit(I, V, 1)
    if not np.isfinite(b) or abs(b) < 1e-9:
        return np.nan, np.nan
    pred = a + b * I
    ss = float(np.sum((V - pred) ** 2))
    tot = float(np.sum((V - V.mean()) ** 2))
    r2 = 1 - ss / tot if tot > 0 else np.nan
    return (v_min - a) / b, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES_CSV)
    # 기본 출력이 방향에 따라 달라야 한다. 예전에는 하나뿐이라
    # `--direction charge` 실행이 방전 라벨 파일을 통째로 덮어썼다
    # (2026-08-25 재생성에서 실제로 밟았다).
    ap.add_argument("--out", default=None)
    ap.add_argument("--v-min", type=float, default=V_MIN)
    ap.add_argument("--direction", default="discharge",
                    choices=["discharge", "charge"],
                    help="charge solves to V_MAX instead, which is the mirror "
                         "problem and better conditioned: 10.2 %% of measured "
                         "charge pulses already exceed 4.20 V, against the 8 %% "
                         "of discharge rows whose floor lies inside the fan")
    args = ap.parse_args()

    charge = args.direction == "charge"
    if args.out is None:
        args.out = OUT_CHG if charge else OUT
    v_target = V_MAX if charge else args.v_min
    rows = [r for r in csv.DictReader(open(args.res, encoding="utf-8"))
            if r["direction"] == args.direction]
    grp = collections.defaultdict(dict)
    for r in rows:
        k = (r["protocol"], int(r["cycle"]), int(r["soc_group"]), float(r["tau_s"]))
        grp[k][int(r["rate_rank"])] = r

    out, drop = [], collections.Counter()
    for (cell, cyc, g, tau), by in sorted(grp.items()):
        ranks = sorted(by)
        if len(ranks) < 3:
            drop["전류 단계 3개 미만"] += 1
            continue
        I = np.array([float(by[k]["I_A"]) for k in ranks])
        V = np.array([float(by[k]["V_tau_V"]) for k in ranks])
        # Discharge: most negative first. Charge: most positive first, so the
        # two-point fit uses the highest rates in both cases.
        o = np.argsort(-I) if charge else np.argsort(I)
        I, V = I[o], V[o]
        if V.max() - V.min() < 0.02:
            drop["전압 폭 20 mV 미만"] += 1
            continue
        i4, r2 = solve(I, V, v_target)
        i2, _ = solve(I[:2], V[:2], v_target)     # the two most extreme rates
        imax = float(np.abs(I).max())
        r0 = by[ranks[0]]
        if not np.isfinite(i4) or (i4 <= 0 if charge else i4 >= 0):
            drop["해 없음"] += 1
            continue
        out.append({
            "cell": cell, "cycle": cyc, "SOH": r0["SOH"], "CAP_Ah": r0["CAP_Ah"],
            "SOC": r0["SOC"], "soc_group": g, "tau_s": tau,
            "direction": args.direction,
            "V_pre_V": r0["V_pre_V"], "V_min_V": v_target,
            "n_rates": len(ranks),
            "I_max_meas_A": round(-imax, 3),
            "I_star_lin4_A": round(float(i4), 3),
            "I_star_lin2hi_A": round(float(i2), 3) if np.isfinite(i2) else "",
            "spread_A": round(abs(float(i4) - float(i2)), 3) if np.isfinite(i2) else "",
            "extrap": round(abs(i4) / imax, 3),
            "fit_r2": round(float(r2), 5) if np.isfinite(r2) else "",
        })
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    a = np.array([r["extrap"] for r in out])
    s = np.array([r["spread_A"] if r["spread_A"] != "" else np.nan for r in out],
                 dtype=float)
    r2 = np.array([r["fit_r2"] if r["fit_r2"] != "" else np.nan for r in out],
                  dtype=float)
    ist = np.array([r["I_star_lin4_A"] for r in out])
    print(f"  {args.out}\n  {len(out):,}행   제외 {dict(drop)}")
    print(f"\n  {'외삽 배수':<14} {'행':>7} {'비율':>7} {'I* 중앙':>9} "
          f"{'두 적합 차이 중앙':>16} {'적합 r2 중앙':>12}")
    for lo, hi, name in ((0, 1.0, "<= 1.0  내삽"), (1.0, 1.5, "1.0~1.5"),
                         (1.5, 2.5, "1.5~2.5"), (2.5, 1e9, "> 2.5")):
        m = (a > lo) & (a <= hi)
        if not m.any():
            continue
        print(f"  {name:<14} {m.sum():>7,} {m.mean()*100:>6.1f}% "
              f"{np.median(ist[m]):>8.1f}A {np.nanmedian(s[m]):>15.1f}A "
              f"{np.nanmedian(r2[m]):>12.4f}")
    m = a <= 1.0
    print(f"\n  내삽만: {m.sum():,}행, 셀당 "
          f"{m.sum()//len({r['cell'] for r in out})}행 정도")
    if m.any():
        soc = np.array([float(r["SOC"]) for r in out])[m]
        soh = np.array([float(r["SOH"]) for r in out])[m]
        print(f"    SOC {soc.min():.2f}~{soc.max():.2f}   SOH {soh.min():.3f}~{soh.max():.3f}")
        print(f"    I* {np.percentile(ist[m],5):.1f} ~ {np.percentile(ist[m],95):.1f} A")


if __name__ == "__main__":
    main()
