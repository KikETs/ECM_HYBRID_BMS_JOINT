"""Put the RPCWBY pulses into the schema ecm_pool already reads.

NO 2RC FIT IS NEEDED, AND THAT IS THE POINT
    ecm_pool never uses the fitted parameters as parameters - it uses them only to
    reconstruct the response at 2 s and 10 s, pools THAT, and re-derives R_fast
    and R_slow from the two horizons. RPCWBY measures those two responses
    directly, so the fitting step is skipped rather than reproduced: d2 and d10
    are the measured R(2 s) and R(10 s).

    tau2 is the one thing that cannot come from two horizons and a straight face,
    because the reduction needs it to separate the branches. It is taken as the
    UYPYDJ pooled median. That is defensible - measured tau2 varies far less
    across cells than the resistances do (findings.md 4.3.1) - and it is the same
    tau2 the UYPYDJ side of the pool is reduced at, so the two datasets are
    reduced identically rather than each on its own timescale.

CURRENT IS THE COMMON LANGUAGE, NOT RATE RANK
    UYPYDJ steps four discrete rates and labels them 0-3. RPCWBY's SOP search
    applies a continuum from 3 to 30 A. `rate_rank` is therefore rewritten as a
    CURRENT BIN index on both sides. ECMSurface already treats rank as a rung on
    a current ladder - it interpolates across ranks at |I| using each rank's
    median current - so nothing downstream has to change.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "rpcwby_resistance.csv")
UECM = os.path.join(HERE, "uypydj_ecm.csv")
OUT = os.path.join(HERE, "rpcwby_ecm.csv")
TAU_A, TAU_B = 2.0, 10.0
# Bin edges chosen to hold both ladders: UYPYDJ sits at about 3/12/24/34 A and
# RPCWBY spreads 3-30 A, so the edges must not put 30 and 34 in different bins
# when they are the same operating point to within Butler-Volmer curvature.
I_EDGES = (2.0, 7.0, 16.0, 26.0, 40.0)


def cur_bin(i):
    return int(np.clip(np.searchsorted(I_EDGES, abs(i)) - 1, 0, len(I_EDGES) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--uecm", default=UECM)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    u = list(csv.DictReader(open(args.uecm, encoding="utf-8")))
    t1 = float(np.median([float(r["tau1_s"]) for r in u]))
    t2 = float(np.median([float(r["tau2_s"]) for r in u]))
    a = 1 - np.exp(-TAU_A / t2)
    b = 1 - np.exp(-TAU_B / t2)
    print(f"  UYPYDJ 중앙 tau1 {t1:.4f}s, tau2 {t2:.3f}s  -> 이 값으로 환원")

    rows = list(csv.DictReader(open(args.res, encoding="utf-8")))
    grp = collections.defaultdict(dict)
    for r in rows:
        k = (r["cell"], int(r["cycle"]), r["direction"],
             round(float(r["SOC"]), 3), round(float(r["I_A"]), 1))
        grp[k][float(r["tau_s"])] = r

    out, drop = [], collections.Counter()
    for (cell, cyc, d, soc, ia), by in sorted(grp.items()):
        if TAU_A not in by or TAU_B not in by:
            drop["한 지평만"] += 1
            continue
        d2 = abs(float(by[TAU_A]["R_mOhm"]))
        d10 = abs(float(by[TAU_B]["R_mOhm"]))
        R_slow = (d10 - d2) / (b - a)
        R_fast = d2 - R_slow * a
        if not (np.isfinite(R_fast) and np.isfinite(R_slow)) or R_fast <= 0:
            drop["환원 실패"] += 1
            continue
        r0 = by[TAU_A]
        out.append({
            "cell": cell, "cycle": cyc, "SOH": r0["SOH"], "CAP_Ah": r0["CAP_Ah"],
            "SOC": round(soc, 4), "direction": d,
            "rate_rank": cur_bin(ia), "I_A": round(ia, 3),
            "V_pre_V": r0["V_pre_V"], "rest_before_s": r0["rest_before_s"],
            "n_points": 0,
            "R0_mOhm": round(max(R_fast, 1e-3), 4), "R1_mOhm": 1e-3,
            "tau1_s": round(t1, 4),
            "R2_mOhm": round(max(R_slow, 1e-3), 4), "tau2_s": round(t2, 4),
            "fit_rmse_mV": 0.0})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"  -> {args.out}  {len(out):,}행   제외 {dict(drop)}")

    c = collections.Counter((r["cell"], r["rate_rank"]) for r in out)
    print(f"\n  {'셀':<12} " + "".join(
        f"{f'bin{i} ({I_EDGES[i]:.0f}~{I_EDGES[i+1]:.0f}A)':>16}"
        for i in range(len(I_EDGES) - 1)))
    for cell in sorted({r["cell"] for r in out}):
        print(f"  {cell:<12} " + "".join(
            f"{c.get((cell, i), 0):>16,}" for i in range(len(I_EDGES) - 1)))
    uc = collections.Counter()
    for r in u:
        if r["direction"] == "discharge":
            uc[cur_bin(float(r["I_A"]))] += 1
    print(f"  {'UYPYDJ(방전)':<12} " + "".join(
        f"{uc.get(i, 0):>16,}" for i in range(len(I_EDGES) - 1)))
    s = np.array([float(r["SOH"]) for r in out])
    print(f"\n  SOH {s.min():.3f}~{s.max():.3f}")


if __name__ == "__main__":
    main()
