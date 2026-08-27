"""Fit ECM parameters against TEMPERATURE from the Mendeley HPPC tests.

WHAT AXIS THIS FILLS
    uypydj_ecm.csv gives theta(SOC, SOH) at one temperature. This gives
    theta(SOC, T) at one age. Together with RPCWBY Test#1/#2, which carry SOP
    pulses at 10 AND 25 degC across the whole aging range, the three sources
    cover the axes no single one does - which is the point of using all of them.

SAME PULSE STRUCTURE AS UYPYDJ, SO THE SAME FITTER APPLIES
    Measured here: 10 s pulses, about 101 samples each, 0.1 s median interval.
    Model order was chosen from residuals on the UYPYDJ pulses (2RC, see
    uypydj_ecm.py) and the same choice is kept so the two tables are comparable.

THE PROTOCOL REDUCES CURRENT WHEN COLD, AND THAT IS NOT A DEFECT
    Discharge pulses reach -36 A at 25 degC but only -12 A at -20 degC. So the
    rate ladder is not the same across temperature and rate_rank means different
    currents at different T. The measured current is recorded per row; any
    comparison across temperature has to match on CURRENT, not on rank.

SOC AXIS
    1 + Ah/3.0, anchored at FULL, as everywhere in this project. Verified on
    these files: Ah starts at 0.000 with the cell at 4.19 V and falls to -2.94
    at 2.50 V. Anchoring at empty instead would need the capacity, which here is
    TEMPERATURE dependent (2.35 Ah at -20 degC against 2.945 at 25) - folding
    temperature into the axis is exactly what data30t.py's header forbids. A cold
    cell simply stops early; that shows up as a restricted SOC range.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import Q_RATED_AH, load_meas_mat, parse_name  # noqa: E402
from uypydj_ecm import MAX_RMSE_MV, MIN_POINTS, MIN_REST_S, fit_rc  # noqa: E402
from uypydj_hppc_resistance import find_pulses, rank_pulses  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw", "Mendeley")
OUT = os.path.join(HERE, "mendeley_ecm.csv")
TEMPS = (-20, -10, 0, 10, 25, 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-rc", type=int, default=2, choices=[1, 2, 3])
    args = ap.parse_args()

    rows, skip = [], {"rest": 0, "points": 0, "rmse": 0, "fit": 0, "nofile": 0}
    for temp in TEMPS:
        paths = sorted(glob.glob(os.path.join(args.raw, f"{temp}degC", "*HPPC*.mat")))
        if not paths:
            skip["nofile"] += 1
            continue
        n_ok = 0
        for p in paths:
            try:
                r = load_meas_mat(p)
            except Exception:                               # noqa: BLE001
                skip["fit"] += 1
                continue
            t, V, I, Ah = r["t"], r["V"], r["I"], r["Ah"]
            T = r["T"]
            # The final recharge sits after an Ah counter reset; keep only the
            # monotone HPPC sweep so the SOC axis stays a depth from full.
            jump = np.flatnonzero(np.abs(np.diff(Ah)) > 0.5)
            end = int(jump[0]) + 1 if len(jump) else len(Ah)
            t, V, I, Ah, T = t[:end], V[:end], I[:end], Ah[:end], T[:end]

            prev_end = None
            for (a, b, grp, rank) in rank_pulses(find_pulses(t, I), t, I, Ah):
                rest = (t[a] - t[prev_end - 1]) if prev_end else 1e9
                prev_end = b
                if rest < MIN_REST_S:
                    skip["rest"] += 1
                    continue
                if b - a < MIN_POINTS or a == 0:
                    skip["points"] += 1
                    continue
                ip = float(np.median(I[a:b]))
                try:
                    q, rmse = fit_rc(t[a:b] - t[a], V[a:b] - V[a - 1], ip, args.n_rc)
                except Exception:                           # noqa: BLE001
                    skip["fit"] += 1
                    continue
                if not np.isfinite(rmse) or rmse > MAX_RMSE_MV:
                    skip["rmse"] += 1
                    continue
                row = {
                    "temp_set_C": temp,
                    "T_cell_C": round(float(np.median(T[a:b])), 2),
                    "SOC": round(1.0 + float(Ah[a]) / Q_RATED_AH, 4),
                    "direction": "discharge" if ip < 0 else "charge",
                    "rate_rank": rank, "I_A": round(ip, 3),
                    "V_pre_V": round(float(V[a - 1]), 5),
                    "rest_before_s": round(float(min(rest, 99999)), 1),
                    "n_points": int(b - a),
                    "R0_mOhm": round(q[0] * 1000, 4),
                }
                for k in range(args.n_rc):
                    row[f"R{k+1}_mOhm"] = round(q[1 + 2 * k] * 1000, 4)
                    row[f"tau{k+1}_s"] = round(q[2 + 2 * k], 4)
                row["fit_rmse_mV"] = round(rmse, 4)
                row["file"] = os.path.basename(p)
                rows.append(row)
                n_ok += 1
        d = [r for r in rows if r["temp_set_C"] == temp and r["direction"] == "discharge"]
        cur = sorted({round(abs(r["I_A"])) for r in d})
        print(f"{temp:>4}degC: {n_ok:>4}개 펄스   방전 전류 {cur[:6]}"
              f"{' ...' if len(cur) > 6 else ''}")

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
