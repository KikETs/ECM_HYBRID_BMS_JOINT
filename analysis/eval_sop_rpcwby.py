"""Score the pooled ECM against the RPCWBY authors' own measured SOP.

WHY THIS IS THE ONLY REMAINING TEMPERATURE TEST
    Test#3 sweeps six temperatures but is an SOP SEARCH: it converges, so each
    (T, tau, SOC) keeps one current, not a fan, and no measured I* can be built
    from it (docs/sop_hybrid_spec.md section 18.3). Test#1/#2 instead carry the
    authors' hand-extracted SOP at 10 and 25 degC over 1994 cycles. Two
    temperatures is less than six, but it is measured rather than assumed.

WHICH ROWS ARE USABLE
    Their protocol clamps discharge at 30 A against a 2.55 V floor, so a row is
    either
        current limited   I = 30 A,      V_end > 2.55  ->  |SOP| = 30 * V_end
        voltage limited   V_end = 2.55,  I < 30 A      ->  |SOP| = 2.55 * I*
    and the two cases meet at exactly 2.55 * 30 = 76.5 W. Above that the cycler
    set the answer and the cell did not, so those rows cannot test a voltage
    model and are dropped. The split is self-consistent to the volt: the
    current-limited rows imply V_end in 2.551..3.759 (inside the 2.55..4.15
    window) and the voltage-limited rows imply I* in 1.8..29.9 A (under 30).

WHAT IS AND IS NOT ON TRIAL
    kf = ks = 1. The learned trim is keyed by UYPYDJ characterisation cycle and
    has no meaning on another lab's cell, so what is scored here is the bare
    pooled ECM plus the measured temperature factor - the physics baseline,
    nothing fitted to this data.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from ecm_pool import surfaces

HERE = os.path.dirname(os.path.abspath(__file__))
SOP = os.path.join(HERE, "rpcwby_sop_summary.csv")
RES = os.path.join(HERE, "rpcwby_resistance.csv")
OUT = os.path.join(HERE, "sop_rpcwby_eval.csv")

V_FLOOR = 2.55          # their floor, not this project's 2.5
I_CLAMP = 30.0
BOUND = V_FLOOR * I_CLAMP
TAU = 10.0              # readme: Test#1/#2 pulse length is 10 s
SHEET_CELL = {"Test#1": "RPC_CC", "Test#2": "RPC_US06"}


def soh_map():
    """cycle -> SOH per RPCWBY cell, from the capacity checks in the raw files."""
    rows = list(csv.DictReader(open(RES, encoding="utf-8")))
    out = {}
    for c in sorted({r["cell"] for r in rows}):
        pts = sorted({(float(r["cycle"]), float(r["SOH"]))
                      for r in rows if r["cell"] == c})
        a = np.array(pts)
        out[c] = (a[:, 0], a[:, 1])
    return out


def r_eff(surf, soc, soh, I, T):
    th = surf.theta(soc, soh, I, T)
    if not bool(np.atleast_1d(th["in_hull"])[0]):
        return np.nan, np.nan
    R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
    t1 = float(th["tau1"][0]); t2 = float(th["tau2"][0])
    return (R0 + R1 * (1 - np.exp(-TAU / t1))
            + R2 * (1 - np.exp(-TAU / t2))), float(th["g_temp"])


def solve_I(surf, soc, soh, v_pre, T, I0=-15.0, iters=24):
    I, g = I0, np.nan
    for _ in range(iters):
        R, g = r_eff(surf, soc, soh, I, T)
        if not np.isfinite(R) or R <= 0:
            return np.nan, np.nan
        nxt = float(np.clip((V_FLOOR - v_pre) / R, -400.0, -0.1))
        if abs(nxt - I) < 1e-3:
            return nxt, g
        I = 0.5 * I + 0.5 * nxt
    return I, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyst", choices=["none", "discharge"], default="discharge",
                    help="v_pre = OCV, or OCV - M for a cell resting on the "
                         "discharge branch (the sweep runs downward)")
    ap.add_argument("--soc-axis", choices=["as-is", "rated"], default="as-is",
                    help="their SOC may be normalised to the AGED capacity; "
                         "'rated' converts it to this project's axis by SOC*SOH")
    ap.add_argument("--holdouts", default="CC,CC_CELL2,BOOST,BOOST_REST,BOOST_NEGPULSE,BOOST_NEGPULSE_1S",
                    help="pool variants; the RPCWBY cell is external to all")
    a = ap.parse_args()

    smap = soh_map()
    rows = [r for r in csv.DictReader(open(SOP, encoding="utf-8"))
            if r["sheet"] in SHEET_CELL and r["SOP_disch"] not in ("", "nan")]
    holds = [h for h in a.holdouts.split(",") if h]

    recs, kept = [], 0
    for h in holds:
        sd = surfaces(h)[0]
        for r in rows:
            P = abs(float(r["SOP_disch"]))
            if P > BOUND:                     # current limited - cycler's answer
                continue
            cell = SHEET_CELL[r["sheet"]]
            soc = float(r["SOC"]); T = float(r["temp_C"]); cy = float(r["cycle"])
            cyc, soh_c = smap[cell]
            soh = float(np.interp(cy, cyc, soh_c))
            if a.soc_axis == "rated":
                soc *= soh
            v, ok = sd.ocv(soc, soh)
            v_pre = float(np.atleast_1d(v)[0])
            if a.hyst == "discharge":
                M, _ = sd.hyst_M(soc, soh)
                v_pre -= M
            I_pred, g = solve_I(sd, soc, soh, v_pre, T)
            I_meas = -P / V_FLOOR
            recs.append(dict(holdout=h, sheet=r["sheet"], cell=cell, cycle=cy,
                             SOH=round(soh, 4), SOC=soc, T_C=T,
                             I_meas_A=round(I_meas, 3),
                             I_pred_A=round(I_pred, 3) if np.isfinite(I_pred) else "",
                             P_meas_W=round(-P, 2),
                             P_pred_W=round(V_FLOOR * I_pred, 2) if np.isfinite(I_pred) else "",
                             g_temp=round(g, 4) if np.isfinite(g) else "",
                             v_pre=round(v_pre, 4)))
        kept = sum(1 for x in recs if x["holdout"] == h)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    im = np.array([x["I_meas_A"] for x in recs])
    ip = np.array([x["I_pred_A"] if x["I_pred_A"] != "" else np.nan for x in recs],
                  dtype=float)
    T = np.array([x["T_C"] for x in recs]); soh = np.array([x["SOH"] for x in recs])
    soc = np.array([x["SOC"] for x in recs]); hd = np.array([x["holdout"] for x in recs])
    ok = np.isfinite(ip)
    e = ip - im

    print(f"  전압 제한 행 {kept} x 풀 {len(holds)} = {len(recs)}, 유효 {int(ok.sum())}")
    print(f"  I* 측정 {im.min():.1f}~{im.max():.1f} A\n")
    print(f"  {'':<10}{'n':>5}{'RMSE':>8}{'MAE':>8}{'bias':>8}{'비 중앙':>9}")

    def line(tag, m):
        m = m & ok
        if m.sum() < 3:
            return
        print(f"  {tag:<10}{int(m.sum()):>5}{np.sqrt(np.mean(e[m]**2)):>7.2f}A"
              f"{np.mean(np.abs(e[m])):>7.2f}A{np.mean(e[m]):>7.2f}A"
              f"{np.median(ip[m]/im[m]):>9.3f}")

    line("전체", np.ones(len(recs), bool))
    for t in (10.0, 25.0):
        line(f"{t:.0f} C", T == t)
    for lo, hi in ((0.95, 1.01), (0.88, 0.95), (0.80, 0.88), (0.0, 0.80)):
        line(f"SOH {lo:.2f}+", (soh >= lo) & (soh < hi))
    for lo, hi in ((0.0, 0.05), (0.05, 0.12), (0.12, 1.01)):
        line(f"SOC {lo:.2f}+", (soc >= lo) & (soc < hi))
    print()
    for h in holds:
        line(f"풀-{h}", hd == h)


if __name__ == "__main__":
    main()
