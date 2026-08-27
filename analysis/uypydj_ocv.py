"""Extract a per-SOH open-circuit-voltage curve from the UYPYDJ OCV tests.

WHY PER-SOH AND NOT ONCE
    The SOP reference in docs/soh_extension_design.md section 6 is

        I*(tau) = (OCV(SOC) - V_min) / R(tau, SOC, SOH)

    Resistance is already measured against SOH (uypydj_hppc_resistance.csv), but
    OCV moves with age too. Reusing a fresh OCV curve at SOH 0.70 would push that
    error straight into the reference the model is judged against, and it would
    do so in the direction that flatters the model.

WHAT THE TEST IS
    A 0.05C (+/-0.150 A) full discharge followed by a full charge, logged every
    60 s over about 41 h, run every ~75 cycles - 14 per cell.

PSEUDO-OCV, THE STANDARD CONSTRUCTION
    The two legs are averaged on a common SOC grid:

        OCV(SOC) = ( V_discharge(SOC) + V_charge(SOC) ) / 2

    At 0.05C the ohmic drop is small and, crucially, OPPOSITE in sign between
    the legs, so averaging cancels it to first order along with most of the
    hysteresis. This is an approximation and is labelled as one - it is a
    reference for comparison, not a measurement of thermodynamic OCV.

SOC AXIS - ANCHORED AT FULL
    SOC = 1 + Ah/3.0. See uypydj_hppc_resistance.py for why this anchoring is
    used project-wide and why mixing it with an empty-anchored axis is fatal.

SOH JOIN
    Same rule as the HPPC extraction: interpolate this cell's own drive-cycle
    cycle -> SOH curve, and DROP anything outside the anchor range rather than
    extrapolate. That costs the cycle-1 test on most cells.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import Q_RATED_AH, load_uypydj_mat, uypydj_members  # noqa: E402
from uypydj_hppc_resistance import PROTOCOLS, soh_anchors  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "UYPYDJ")
OUT = os.path.join(os.path.dirname(__file__), "uypydj_ocv.csv")

I_REST = 0.05            # A, below which the cell counts as resting
SOC_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)
MIN_LEG = 200            # samples; a leg shorter than this is not a full sweep


def legs(r, cap):
    """Split one OCV run into its discharge and charge sweeps."""
    I = np.asarray(r["I"], dtype=float)
    V = np.asarray(r["V"], dtype=float)
    ah = np.asarray(r["Ah"], dtype=float)
    if ah.ndim == 0 or len(ah) < 2 * MIN_LEG:
        return None, None
    soc = 1.0 + ah / Q_RATED_AH
    out = []
    for m in (I < -I_REST, I > I_REST):
        if m.sum() < MIN_LEG:
            out.append(None)
            continue
        s, v = soc[m], V[m]
        o = np.argsort(s)
        # Duplicate SOC values would make np.interp order-dependent.
        s, idx = np.unique(s[o], return_index=True)
        out.append((s, v[o][idx]))
    return out[0], out[1]


def curve(r, cap):
    """Pseudo-OCV on SOC_GRID, plus the two legs, or None if unusable."""
    dis, chg = legs(r, cap)
    if dis is None or chg is None:
        return None
    lo = max(dis[0].min(), chg[0].min())
    hi = min(dis[0].max(), chg[0].max())
    g = SOC_GRID[(SOC_GRID >= lo) & (SOC_GRID <= hi)]
    if len(g) < 10:
        return None
    vd = np.interp(g, dis[0], dis[1])
    vc = np.interp(g, chg[0], chg[1])
    return g, (vd + vc) / 2.0, vd, vc


from temp_defects import defective  # noqa: E402

_TEMP_BAD = defective("OCV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows, dropped = [], []
    for key, zs in PROTOCOLS.items():
        zips = [os.path.join(args.raw, z) for z in zs]
        acyc, asoh, acap = soh_anchors(zips)
        if len(acyc) < 2:
            dropped.append((key, "SOH 앵커 부족")); continue
        n_ok = 0
        for zp in zips:
            if not os.path.exists(zp):
                continue
            for member, info in uypydj_members(zp, part="OCV_0.05C"):
                c = info["cycle"]
                if c is None or c < acyc.min() or c > acyc.max():
                    dropped.append((info["file"], f"cycle {c} 앵커 범위 밖")); continue
                if (key, c) in _TEMP_BAD:
                    # 온도 채널 결함 — temp_defects.py 참조
                    dropped.append((info["file"], "온도 채널 결함")); continue
                try:
                    r = load_uypydj_mat(zp, member)
                except Exception as e:                    # noqa: BLE001
                    dropped.append((info["file"], type(e).__name__)); continue
                cap = float(np.interp(c, acyc, acap))
                cv = curve(r, cap)
                if cv is None:
                    dropped.append((info["file"], "스윕 불완전")); continue
                g, ocv, vd, vc = cv
                soh = float(np.interp(c, acyc, asoh))
                for j in range(len(g)):
                    rows.append({"cell": key, "cycle": c, "SOH": round(soh, 5),
                                 "SOC": g[j], "OCV_V": round(float(ocv[j]), 5),
                                 "V_dis_V": round(float(vd[j]), 5),
                                 "V_chg_V": round(float(vc[j]), 5),
                                 "hyst_mV": round(float(vc[j] - vd[j]) * 1000, 2)})
                n_ok += 1
        print(f"{key}: OCV {n_ok}개 (앵커 cycle {int(acyc.min())}-{int(acyc.max())})")

    if not rows:
        sys.exit("추출된 행이 없습니다")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{args.out}: {len(rows)}행")
    if dropped:
        print(f"제외 {len(dropped)}건: {dropped[:4]}")


if __name__ == "__main__":
    main()
