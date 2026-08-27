"""SOP from the filter state - power limits that remember what the cell just did.

THE POINT, WHICH IS NOT ACCURACY BUT HISTORY
    A lookup table answers "what can this cell deliver at this SOC, SOH and
    temperature". It cannot answer "...given that it has been pulling 20 A for
    the last minute", because it has nowhere to put that. The ECM does: V1, V2
    and the hysteresis state h are already displaced when the request arrives,
    and they eat into the headroom before the new pulse even starts.

    So the same (SOC, SOH, T) gives a different SOP depending on the recent
    load. That difference is the reason for building the model rather than
    measuring a map.

CLOSED FORM, THEN ONE FIXED POINT FOR THE CURRENT DEPENDENCE
    Over a tau-second pulse at constant current the branch voltages are

        V1(tau) = V1(0) e1 + R1 (1 - e1) I,     e1 = exp(-tau/tau1)

    so the terminal voltage is LINEAR in I and V(tau) = V_lim inverts directly:

        I* = (V_lim - OCV - M h - V1(0) e1 - V2(0) e2)
             / (R0 + R1 (1 - e1) + R2 (1 - e2))

    The only nonlinearity is that R0, R1, R2 themselves depend on |I| (the
    Butler-Volmer effect measured in the HPPC: R1 and R2 fall to ~0.7x from
    2.6 A to 29.6 A). That is handled by iterating I* to a fixed point, which
    converges in a few steps because R varies far more slowly than 1/R.

WHAT LIMITS THE ANSWER
    Whichever binds first: the voltage floor, the 35 A continuous rating, or -
    for charge - the voltage ceiling and the charge current rating. The binding
    constraint is reported per row, because "current limited" and "voltage
    limited" behave completely differently as the cell ages (findings: the cold
    cell crosses from one to the other at cycle 1541 and its SOP ratio collapses
    from that point).

SOC DRIFT DURING THE PULSE IS INCLUDED
    A 35 A pulse for 10 s moves SOC by 0.032 on the rated axis, which is small
    but not nothing near the knee of the OCV curve, so OCV is evaluated at the
    mid-pulse SOC.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dekf_soh import DualEKF  # noqa: E402
from ecm_surface import ECMSurface  # noqa: E402

Q_RATED_AH = 3.0
V_MIN, V_MAX = 2.5, 4.2
I_MAX_DCH, I_MAX_CHG = 35.0, 15.0


def sop(surf_dis, soh, soc, v1, v2, h, tau, T=25.0, charge=False, iters=12,
        surf_chg=None):
    """Peak sustainable power for tau seconds from the CURRENT filter state.

    TWO DEFECTS FIXED HERE, BOTH FOUND BY REVIEW RATHER THAN BY A FAILING RUN.

    (a) DIRECTION-MATCHED SURFACE. The charge and discharge pulses were fitted
        separately because the cell is not symmetric, but this function used to
        take one surface and use it for both. Latent while main() never asked for
        charge, and about to become live when the charge limit is wired, so the
        charge surface is now a required argument whenever charge=True.

    (b) QUANTITIES RECOMPUTED AT THE FINAL OPERATING POINT. On the convergence
        break the loop published ocv/M/den from the PREVIOUS iterate's soc_mid.
        Measured over 400 random states: median 0.000 %, but wrong on 12.8 % of
        calls and up to +11.0 % - and one-sided OPTIMISTIC, i.e. it overstates
        the power the cell can deliver. That is the dangerous direction for a
        limit, so it is fixed even though it is rare.
    """
    surf = surf_chg if (charge and surf_chg is not None) else surf_dis
    if charge and surf_chg is None:
        raise ValueError("charge=True needs the charge surface (defect a)")
    v_lim = V_MAX if charge else V_MIN
    i_lim = I_MAX_CHG if charge else -I_MAX_DCH
    I = i_lim
    conv = False
    for _ in range(iters):
        soc_mid = np.clip(soc + I * tau / 2 / 3600.0 / Q_RATED_AH, 0.0, 1.0)
        th = surf.theta(soc_mid, soh, I, T)
        ocv, _ = surf.ocv(soc_mid, soh)
        M, _ = surf.hyst_M(soc_mid, soh)
        R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
        e1 = np.exp(-tau / float(th["tau1"][0]))
        e2 = np.exp(-tau / float(th["tau2"][0]))
        den = R0 + R1 * (1 - e1) + R2 * (1 - e2)
        num = v_lim - float(ocv[0]) - M * h - v1 * e1 - v2 * e2
        I_new = num / den
        if abs(I_new - I) < 1e-3:
            I = I_new
            conv = True
            break
        I = 0.5 * I + 0.5 * I_new
    # Defect (b), done SELF-CONSISTENTLY. The first attempt at this fix
    # re-evaluated theta at the final current but kept the I solved against the
    # PREVIOUS iterate's operating point, which left 11 of 200 voltage-limited
    # calls reporting a terminal voltage that was not the voltage limit. Solve
    # and report at the SAME point: then for a voltage-limited answer
    # V = OCV + Mh + I*den + v1 e1 + v2 e2 with I = (v_lim - OCV - Mh - v1 e1
    # - v2 e2)/den collapses to exactly v_lim, by construction rather than by
    # luck.
    soc_mid = np.clip(soc + I * tau / 2 / 3600.0 / Q_RATED_AH, 0.0, 1.0)
    th = surf.theta(soc_mid, soh, I, T)
    ocv, _ = surf.ocv(soc_mid, soh)
    M, _ = surf.hyst_M(soc_mid, soh)
    R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
    e1 = np.exp(-tau / float(th["tau1"][0]))
    e2 = np.exp(-tau / float(th["tau2"][0]))
    den = R0 + R1 * (1 - e1) + R2 * (1 - e2)
    I = (v_lim - float(ocv[0]) - M * h - v1 * e1 - v2 * e2) / den

    volt_limited = bool(np.isfinite(I) and abs(I) < abs(i_lim))
    I_out = I if volt_limited else i_lim
    v_end = float(ocv[0]) + M * h + I_out * den + v1 * e1 + v2 * e2
    return {"I": I_out, "V": v_end, "P": I_out * v_end,
            "limited_by": "voltage" if volt_limited else "current",
            "converged": bool(conv), "in_hull": bool(th["in_hull"][0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="CC")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--run", type=int, default=39)
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=10.0)
    args = ap.parse_args()

    sd = ECMSurface(args.cell, "discharge")
    sc = ECMSurface(args.cell, "charge")
    z = np.load(os.path.join(args.cache,
                             f"uypydj_{args.cell}_Fifteen_Drive_Cycles.npz"))
    lens = z["lens"]; off = np.concatenate([[0], np.cumsum(lens)])
    sl = slice(off[args.run], off[args.run] + lens[args.run])
    soc, V, I, SOH, T = (z[x][sl] for x in ("SOC", "V", "I", "SOH", "T"))
    ok = np.isfinite(soc) & np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
    soc, V, I, T = (x[ok][:args.max_samples] for x in (soc, V, I, T))
    soh = float(np.nanmedian(SOH))
    rv = float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))

    f = DualEKF(sd, sc, soh, rv); f.Pm = 0.0
    f.x = np.array([float(soc[0]), 0.0, 0.0])
    est = np.empty(len(I))
    state = []
    for k in range(len(I)):
        est[k], _, _ = f.step(I[k], V[k], T[k])
        state.append((f.x[0], f.x[1], f.x[2], f.h))

    print(f"cell {args.cell}  run {args.run}  SOH {soh:.3f}  tau {args.tau:.0f}s\n")
    print("같은 SOC에서 직전 이력에 따라 SOP가 얼마나 달라지는가")
    print(f"  {'t[s]':>7} {'SOC':>6} {'직전 I':>8} {'V1+V2':>9} {'h':>6} "
          f"{'SOP':>9} {'제한':>8} {'휴지 가정 SOP':>13} {'차이':>8}")
    # pick moments that share a SOC but differ in recent load
    tgt = np.nanmedian(est)
    cand = np.flatnonzero(np.abs(est - tgt) < 0.01)
    cand = cand[(cand > 200) & (cand < len(I) - 10)]
    if len(cand) > 8:
        cand = cand[np.linspace(0, len(cand) - 1, 8).astype(int)]
    for k in cand:
        s_, v1, v2, h = state[k]
        live = sop(sd, soh, s_, v1, v2, h, args.tau, float(T[k]), surf_chg=sc)
        rest = sop(sd, soh, s_, 0.0, 0.0, 0.0, args.tau, float(T[k]), surf_chg=sc)
        recent = float(np.mean(I[max(0, k - 60):k]))
        print(f"  {k:>7} {s_:>6.3f} {recent:>7.2f}A {(v1+v2)*1000:>8.1f}m "
              f"{h:>6.2f} {live['P']:>8.1f}W {live['limited_by']:>8} "
              f"{rest['P']:>12.1f}W {live['P']-rest['P']:>+7.1f}W")


if __name__ == "__main__":
    main()
