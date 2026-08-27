"""Open-loop ECM voltage simulation over a drive cycle - the gate for everything.

WHY THIS COMES BEFORE ANY FILTER
    A Kalman filter built on a model that cannot reproduce the measured voltage
    is an elaborate way to report the model's own bias. So the ECM is first run
    open loop - no feedback, no correction, SOC advanced only by coulomb counting
    from a known start - and its voltage error is compared against the same
    quantity the LSTM reproduction reports.

WHAT OPEN LOOP MEANS HERE
    SOC comes from integrating current against the cell's own measured capacity
    at that SOH. The RC states start at zero and evolve by the discrete update.
    Nothing reads the measured voltage. Any error is the model's.

DISCRETISATION
    V_k+1 = V_k * exp(-dt/tau) + R * (1 - exp(-dt/tau)) * I_k
    which is the exact zero-order-hold solution for a constant current over dt,
    not an Euler step - the drive cycle has 1 s steps against tau1 ~ 0.2-1 s, so
    an Euler step would be visibly wrong on the fast branch.

SIGN CONVENTION
    Negative current is discharge, as everywhere in this project. The terminal
    voltage is OCV + I*R0 + V1 + V2 with I signed, so discharge pulls it down.

THE SOC AXIS IS RATED, SO COULOMB COUNTING DIVIDES BY THE RATING
    dSOC/dt = I / 3.0 Ah, NOT I / (aged capacity). This module first divided by
    the cell's measured capacity and the SOC advanced 3.0/Q too fast - 1.44x at
    SOH 0.70, which walked the simulated SOC to -0.047 while the cell was really
    at 0.149 and produced a -411 mV voltage bias. The fresh cell looked fine
    (ratio 1.015) which is exactly how the bug hid.

    Aged capacity still matters, but as the SOC RANGE the cell can traverse, not
    as the divisor: a 2.09 Ah cell simply cannot go from SOC 1.0 to 0 on a 3.0 Ah
    axis. That is the same distinction findings.md section 2.1 makes about the
    published SOC column.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_surface import ECMSurface, available_cells  # noqa: E402


Q_RATED_AH = 3.0


def simulate(surf_dis, surf_chg, soc0, soh, I, dt, cap_Ah=None, T_cell=None,
             gamma=None, kf=1.0, ks=1.0):
    """Open-loop terminal voltage for a current series.

    Two surfaces because charge and discharge pulses were fitted separately -
    the cell is not symmetric, and using the discharge fit for regen would bias
    exactly the samples where the voltage rises.

    kf and ks are the hybrid arm's resistance multipliers - kf on the fast branch
    (R0 and R1 together, which are one regressor at every horizon this project
    uses) and ks on the slow one. At 1.0 this is the uncorrected pooled ECM, so
    both arms run through the same integrator and any difference between them is
    the correction and nothing else.
    """
    n = len(I)
    V = np.empty(n)
    v1 = v2 = 0.0
    # One-state hysteresis (Plett). h in [-1, 1] tracks which direction the cell
    # was last driven and relaxes towards sgn(I) at a rate set by throughput, so
    # it PERSISTS through zero current where an RC branch would have decayed.
    h = 0.0
    soc = soc0
    hull = np.ones(n, dtype=bool)
    for k in range(n):
        s = surf_chg if I[k] > 0 else surf_dis
        # MEASURED cell temperature, not the chamber setpoint. The campaign ran
        # at 25 degC ambient but the cell self-heats to 44 degC under fast
        # charge, where the resistance factor is 0.83 - a 17 % error if ignored.
        th = s.theta(soc, soh, I[k], 25.0 if T_cell is None else float(T_cell[k]))
        ocv, _ = s.ocv(soc, soh)
        r0, r1, t1, r2, t2 = (float(th["R0"][0]), float(th["R1"][0]),
                              float(th["tau1"][0]), float(th["R2"][0]),
                              float(th["tau2"][0]))
        hull[k] = bool(th["in_hull"][0])
        vh = 0.0
        if gamma:
            M, _ = s.hyst_M(soc, soh)
            vh = M * h
        V[k] = float(ocv[0]) + vh + I[k] * (kf * r0) + v1 + v2
        a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)
        v1 = v1 * a1 + (kf * r1) * (1 - a1) * I[k]
        v2 = v2 * a2 + (ks * r2) * (1 - a2) * I[k]
        if gamma:
            ah = np.exp(-abs(gamma * I[k] * dt / 3600.0 / Q_RATED_AH))
            h = ah * h + (1 - ah) * np.sign(I[k])
        soc = soc + I[k] * dt / 3600.0 / Q_RATED_AH
        soc = min(max(soc, -0.05), 1.05)
    return V, hull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="CC")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--runs", type=int, default=6, help="runs to simulate")
    ap.add_argument("--max-samples", type=int, default=40000)
    ap.add_argument("--ecm-csv", default=None,
                    help="alternative ECM parameter table to test")
    ap.add_argument("--gamma", type=float, default=None,
                    help="hysteresis rate constant (None disables the state)")
    ap.add_argument("--no-temp", action="store_true",
                    help="ignore measured temperature (25 C base only)")
    args = ap.parse_args()

    if args.cell not in available_cells():
        sys.exit(f"unknown cell; have {available_cells()}")
    kw = {"ecm_csv": args.ecm_csv} if args.ecm_csv else {}
    sd = ECMSurface(args.cell, "discharge", **kw)
    sc = ECMSurface(args.cell, "charge", **kw)
    print(f"cell {args.cell}   rank 전류 {sd.rank_I.round(1)} A")

    p = os.path.join(args.cache, f"uypydj_{args.cell}_Fifteen_Drive_Cycles.npz")
    z = np.load(p)
    lens, files = z["lens"], z["files"]
    SOC, V, I, SOH, valid = z["SOC"], z["V"], z["I"], z["SOH"], z["valid"]

    # Spread the sampled runs across the aging range rather than taking the first
    # few, which would all be nearly fresh.
    idx = np.linspace(0, len(lens) - 1, args.runs).astype(int)
    off = np.concatenate([[0], np.cumsum(lens)])
    print(f"\n{'run':>4} {'SOH':>6} {'n':>7} {'RMSE':>9} {'최대오차':>9} {'hull밖':>7}")
    tot = []
    for k in idx:
        sl = slice(off[k], off[k] + lens[k])
        soc, v, i, soh, ok = SOC[sl], V[sl], I[sl], SOH[sl], valid[sl]
        Tc_all = z["T"][sl]
        m = ok & np.isfinite(soc) & np.isfinite(v) & np.isfinite(i) & np.isfinite(Tc_all)
        if m.sum() < 1000:
            continue
        soc, v, i = soc[m][:args.max_samples], v[m][:args.max_samples], i[m][:args.max_samples]
        Tc = Tc_all[m][:args.max_samples]
        s = float(np.nanmedian(soh))
        vp, hull = simulate(sd, sc, float(soc[0]), s, i, 1.0,
                            T_cell=None if args.no_temp else Tc,
                            gamma=args.gamma)
        err = (vp - v) * 1000
        tot.append(np.sqrt(np.mean(err ** 2)))
        print(f"{k:>4} {s:>6.3f} {len(v):>7,} {tot[-1]:>8.2f}m "
              f"{np.abs(err).max():>8.1f}m {(~hull).mean()*100:>6.1f}%")
    if tot:
        print(f"\n  평균 RMSE {np.mean(tot):.2f} mV   "
              f"(LSTM 재현 기준선 29.92 mV, 논문 21.54 mV)")


if __name__ == "__main__":
    main()
