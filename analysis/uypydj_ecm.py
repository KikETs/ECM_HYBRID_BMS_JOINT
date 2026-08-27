"""Fit Thevenin ECM parameters per SOH from the UYPYDJ HPPC pulses.

WHAT THIS ADDS OVER uypydj_hppc_resistance.csv
    That file reports Reff at two fixed timescales, which is what a tau-second
    power limit needs but says nothing about WHERE the resistance growth lives.
    An ECM separates it:

        R0    ohmic - contacts, electrolyte conductivity, current collectors
        R1,t1 fast polarisation - charge transfer / double layer
        R2,t2 slow polarisation - diffusion

    Aging does not raise these equally, and a SOP model that must extrapolate in
    pulse duration needs the split, not one lumped number.

THE SAMPLING IS BETTER THAN THE FILE-LEVEL MEDIAN SUGGESTS
    The HPPC files look like 1 Hz data - that is the median interval over a 30 h
    record dominated by rests. Inside a pulse the cycler logs far denser: about
    101 samples in a 10 s pulse, with ~13 in the first second. The fast time
    constant is therefore resolved, not extrapolated.

MODEL ORDER, CHOSEN FROM RESIDUALS RATHER THAN HABIT
    Fitting a fresh cell's four discharge rates at SOC 1.0:

        rate        1RC      2RC      3RC
        -2.98 A    0.30     0.09     0.09  mV
        -34.17 A   2.33     0.47     0.17  mV

    2RC buys a 5x improvement over 1RC; 3RC only helps at the top rate, where the
    third time constant starts trading against the second. Default is 2RC and the
    per-fit residual is written out, so any row can be re-judged.

OCV REFERENCE
    V just before the current step, which is a rest voltage. Pulses whose
    preceding rest is shorter than MIN_REST_S are skipped rather than fitted
    against a still-relaxing voltage - the median rest here is 509 s, so this
    discards little.

SOH JOIN
    Interpolated from the SAME cell's drive-cycle runs by cycle number, never
    extrapolated and never across cells (findings.md section 4.1).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import Q_RATED_AH, load_uypydj_mat, uypydj_members  # noqa: E402
from uypydj_hppc_resistance import (PROTOCOLS, find_pulses,  # noqa: E402
                                    rank_pulses, soh_anchors)

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "UYPYDJ")
OUT = os.path.join(os.path.dirname(__file__), "uypydj_ecm.csv")

MIN_REST_S = 30.0        # rest before the pulse, so V_pre is a settled OCV
MIN_POINTS = 20          # samples inside the pulse
MAX_RMSE_MV = 5.0        # reject a fit that cannot describe its own pulse


def fit_rc(tp, dv, ip, n_rc, t_off=None, dv_off=None):
    """Least squares on dV(t) = ip * (R0 + sum_k Rk (1 - exp(-t/tau_k))).

    ip carries its sign, so discharge (ip < 0) gives dv < 0 and every R stays
    positive - a fitted negative resistance would be a bug, not a cell.
    """
    def resid(p):
        m = ip * p[0]
        for k in range(n_rc):
            m = m + ip * p[1 + 2 * k] * (1.0 - np.exp(-tp / p[2 + 2 * k]))
        r = m - dv
        if t_off is None:
            return r
        # Relaxation after the step: the ohmic drop vanishes instantly and each
        # RC branch decays from the value it had reached at the end of the pulse.
        tend = tp[-1]
        mo = np.zeros_like(t_off)
        for k in range(n_rc):
            Rk, tk = p[1 + 2 * k], p[2 + 2 * k]
            mo = mo + ip * Rk * (1.0 - np.exp(-tend / tk)) * np.exp(-t_off / tk)
        return np.concatenate([r, mo - dv_off])

    p0 = [0.008] + sum([[0.003, 10.0 ** k] for k in range(n_rc)], [])
    lo = [1e-4] + sum([[1e-5, 0.05] for _ in range(n_rc)], [])
    hi = [0.2] + sum([[0.2, 3000.0] for _ in range(n_rc)], [])
    s = least_squares(resid, p0, bounds=(lo, hi), max_nfev=8000)
    return s.x, float(np.sqrt(np.mean(s.fun ** 2)) * 1000.0)


from temp_defects import defective_hppc  # noqa: E402

_TEMP_BAD = defective_hppc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-rc", type=int, default=2, choices=[1, 2, 3])
    # A 10 s step response pins down R(1-exp(-t/tau)) over 0-10 s, but a drive
    # cycle reverses current every few seconds and is dominated by the FAST
    # branch. --fit-window restricts the fit to the first N seconds so the fast
    # dynamics are not averaged against the slow tail.
    ap.add_argument("--fit-window", type=float, default=None,
                    help="fit only the first N seconds of each pulse")
    # --with-relax adds the rest AFTER the pulse. The pulse alone constrains
    # R*(1-exp) as a product; the relaxation separates R from tau because the
    # decay has no driving current.
    ap.add_argument("--with-relax", type=float, default=0.0,
                    help="also fit N seconds of the following relaxation")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    n_temp = 0
    rows, skip = [], {"rest": 0, "points": 0, "rmse": 0, "range": 0, "fit": 0}
    keys = [args.only] if args.only else list(PROTOCOLS)
    for key in keys:
        zips = [os.path.join(args.raw, z) for z in PROTOCOLS[key]]
        acyc, asoh, acap = soh_anchors(zips)
        if len(acyc) < 2:
            continue
        n_ok = 0
        for zp in zips:
            if not os.path.exists(zp):
                continue
            for member, info in uypydj_members(zp, part="HPPC"):
                c = info["cycle"]
                if c is None or c < acyc.min() or c > acyc.max():
                    skip["range"] += 1
                    continue
                if (key, c) in _TEMP_BAD:
                    # 온도 채널 결함 특성화 — temp_defects.py 참조
                    n_temp += 1
                    continue
                try:
                    r = load_uypydj_mat(zp, member)
                except Exception:                            # noqa: BLE001
                    skip["fit"] += 1
                    continue
                t, V, I, Ah = r["t"], r["V"], r["I"], r["Ah"]
                soh = float(np.interp(c, acyc, asoh))
                cap = float(np.interp(c, acyc, acap))
                pulses = find_pulses(t, I)
                prev_end = None
                for (a, b, grp, rank) in rank_pulses(pulses, t, I, Ah):
                    rest = (t[a] - t[prev_end - 1]) if prev_end else 1e9
                    prev_end = b
                    if rest < MIN_REST_S:
                        skip["rest"] += 1
                        continue
                    if b - a < MIN_POINTS or a == 0:
                        skip["points"] += 1
                        continue
                    ip = float(np.median(I[a:b]))
                    tp = t[a:b] - t[a]
                    dv = V[a:b] - V[a - 1]
                    if args.fit_window:
                        keep = tp <= args.fit_window
                        if keep.sum() < MIN_POINTS:
                            skip["points"] += 1
                            continue
                        tp, dv = tp[keep], dv[keep]
                    t_off = dv_off = None
                    if args.with_relax > 0:
                        e = b
                        while e < len(t) and t[e] - t[b - 1] <= args.with_relax:
                            e += 1
                        if e - b >= 5:
                            t_off = t[b:e] - t[b - 1]
                            dv_off = V[b:e] - V[a - 1]
                    try:
                        p, rmse = fit_rc(tp, dv, ip, args.n_rc, t_off, dv_off)
                    except Exception:                        # noqa: BLE001
                        skip["fit"] += 1
                        continue
                    if not np.isfinite(rmse) or rmse > MAX_RMSE_MV:
                        skip["rmse"] += 1
                        continue
                    row = {
                        "cell": key, "cycle": c, "SOH": round(soh, 5),
                        "CAP_Ah": round(cap, 5),
                        "SOC": round(1.0 + float(Ah[a]) / Q_RATED_AH, 4),
                        "direction": "discharge" if ip < 0 else "charge",
                        "rate_rank": rank, "I_A": round(ip, 3),
                        "V_pre_V": round(float(V[a - 1]), 5),
                        "rest_before_s": round(float(min(rest, 99999)), 1),
                        "n_points": int(b - a),
                        "R0_mOhm": round(p[0] * 1000, 4),
                    }
                    for k in range(args.n_rc):
                        row[f"R{k+1}_mOhm"] = round(p[1 + 2 * k] * 1000, 4)
                        row[f"tau{k+1}_s"] = round(p[2 + 2 * k], 4)
                    row["fit_rmse_mV"] = round(rmse, 4)
                    rows.append(row)
                    n_ok += 1
        print(f"{key}: {n_ok}개 펄스 피팅 (SOH {asoh.min():.3f}~{asoh.max():.3f})")

    if not rows:
        sys.exit("피팅된 펄스가 없습니다")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    q = np.array([r["fit_rmse_mV"] for r in rows])
    print(f"\n{args.out}: {len(rows)}행 ({args.n_rc}RC)")
    print(f"  피팅 잔차: 중앙 {np.median(q):.3f} mV, 95% {np.percentile(q,95):.3f} mV")
    print(f"  제외: {skip}")


if __name__ == "__main__":
    main()
