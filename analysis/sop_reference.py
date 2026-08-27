"""ECM-based SOP reference across (SOH, SOC, tau), for judging the model later.

WHAT THIS IS FOR
    docs/soh_extension_design.md section 6: there is NO measured SOP at aged
    states - the RPCWBY Test#3 SOP runs are all on fresh cells. So the aged
    reference is built from the two aged measurements that do exist, the HPPC
    pulse resistance and the per-SOH OCV curve.

    It is a REFERENCE, NOT GROUND TRUTH. It inherits every ECM assumption. The
    primary metric for the model stays the direct one - voltage error on the
    held-out cell's own HPPC pulses, which is measured, not modelled. This
    number is for interpretation.

THE CALCULATION IS SELF-CONSISTENT, AND HAS TO BE
    The textbook form

        I*(tau) = (OCV(SOC) - V_min) / R(tau, SOC, SOH)
        SOP(tau) = V_min * I*(tau)

    presumes one R. This cell does not have one: findings.md section 3.2 measured
    the 1C and 9C pulse resistances differing by 1.09x when fresh and 1.90x at
    SOH 0.74. Using the low-rate R would overstate the achievable current at end
    of life by nearly a factor of two.

    So R is taken at the rate rank whose measured current is closest to the
    current being solved for, and the solve is iterated to a fixed point. Where
    the answer lands outside the measured rate range it is flagged rather than
    extrapolated.

LIMITS CARRIED FROM THE INPUTS, ALL FLAGGED PER ROW
    - Below about SOC 0.29 there is no HPPC measurement at low SOH at all
      (findings.md section 4.3), so no row is produced there.
    - The pseudo-OCV averaging weakens below SOC 0.1, where hysteresis is 108.9 mV
      against 44-56 mV elsewhere (design section 6).
    - Cell-to-cell spread at equal SOH reaches 2.41x in R by SOH 0.75, so a
      reference computed per cell is the only honest form. Nothing is averaged
      across cells.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# Samsung INR21700-30T, from the manufacturer datasheet.
V_MIN, V_MAX = 2.5, 4.2
I_MAX_DCH = 35.0          # A, continuous discharge rating
SOC_FLOOR_OK = 0.29       # below this the aged HPPC coverage runs out
HYST_SUSPECT = 0.10       # below this SOC the pseudo-OCV average is weak


def load(path, cast=float):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ocv_lookup(ocv_rows):
    """cell -> sorted list of (SOH, curve), curve = (SOC, OCV) array."""
    by = {}
    for r in ocv_rows:
        by.setdefault((r["cell"], float(r["SOH"])), []).append(
            (float(r["SOC"]), float(r["OCV_V"])))
    out = {}
    for (cell, soh), v in by.items():
        out.setdefault(cell, []).append((soh, np.array(sorted(v))))
    for cell in out:
        out[cell].sort(key=lambda x: x[0])
    return out


def ocv_at(ocv_by, cell, soh, soc):
    """OCV interpolated in BOTH SOH and SOC, inside one cell.

    Taking the nearest curve instead threw away 562 of 7256 ladders and, worse,
    quantised the aging effect: the OCV tests run every ~75 cycles while the
    HPPC runs every ~37, so half the pulses would be paired with a curve up to
    37 cycles away. OCV moves about 150 mV over the life of the cell, so that
    mispairing is not negligible against the 40 mV the model is being judged at.

    Never across cells - findings.md section 4.1.
    """
    cur = ocv_by.get(cell)
    if not cur:
        return None, None
    usable = [(h, a) for h, a in cur if a[0, 0] <= soc <= a[-1, 0]]
    if not usable:
        return None, None
    if len(usable) == 1:
        h, a = usable[0]
        return float(np.interp(soc, a[:, 0], a[:, 1])), abs(h - soh)
    hs = np.array([h for h, _ in usable])
    vs = np.array([float(np.interp(soc, a[:, 0], a[:, 1])) for _, a in usable])
    o = np.argsort(hs)
    hs, vs = hs[o], vs[o]
    inside = hs.min() <= soh <= hs.max()
    return float(np.interp(soh, hs, vs)), (0.0 if inside
                                           else min(abs(hs - soh)))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--res", default=os.path.join(here, "uypydj_hppc_resistance.csv"))
    ap.add_argument("--ocv", default=os.path.join(here, "uypydj_ocv.csv"))
    ap.add_argument("--out", default=os.path.join(here, "sop_reference.csv"))
    ap.add_argument("--max-soh-gap", type=float, default=0.02,
                    help="reject if the nearest OCV curve is further than this in SOH")
    args = ap.parse_args()

    res = [r for r in load(args.res) if r["direction"] == "discharge"]
    ocv_by = ocv_lookup(load(args.ocv))

    # Group the resistance rows into rate ladders: one ladder per
    # (cell, cycle, soc_group, tau) holding its four rate ranks.
    ladders = {}
    for r in res:
        k = (r["protocol"], int(r["cycle"]), int(r["soc_group"]), float(r["tau_s"]))
        ladders.setdefault(k, []).append(
            (abs(float(r["I_A"])), float(r["R_mOhm"]) / 1000.0,
             float(r["SOC"]), float(r["SOH"])))

    rows, rejected = [], {"soc_floor": 0, "no_ocv": 0, "soh_gap": 0, "no_converge": 0}
    for (cell, cyc, grp, tau), lad in sorted(ladders.items()):
        lad.sort()
        I_meas = np.array([x[0] for x in lad])
        R_meas = np.array([x[1] for x in lad])
        soc = float(np.mean([x[2] for x in lad]))
        soh = float(lad[0][3])

        if soc < SOC_FLOOR_OK and soh < 0.80:
            rejected["soc_floor"] += 1
            continue
        v_ocv, gap = ocv_at(ocv_by, cell, soh, soc)
        if v_ocv is None:
            rejected["no_ocv"] += 1
            continue
        if gap > args.max_soh_gap:
            rejected["soh_gap"] += 1
            continue

        # Fixed point: pick R at the measured rate nearest the current we solve
        # for, recompute, repeat. Converges in a handful of steps because R
        # varies slowly with rate compared to the (OCV - Vmin)/R division.
        I = (v_ocv - V_MIN) / R_meas[0]
        conv = False
        for _ in range(25):
            j = int(np.argmin(np.abs(I_meas - min(I, I_MAX_DCH))))
            I_new = (v_ocv - V_MIN) / R_meas[j]
            if abs(I_new - I) < 1e-3:
                I, conv = I_new, True
                break
            I = 0.5 * I + 0.5 * I_new           # damped, avoids 2-cycle chatter
        if not conv:
            rejected["no_converge"] += 1
            continue

        # SOP AT THE LIMIT THAT ACTUALLY BINDS.
        # SOP = V_MIN * I is only right when VOLTAGE is the binding constraint.
        # For this cell it usually is not: at mid SOC the solved current runs
        # past the 35 A rating while the terminal voltage is still well above
        # 2.5 V, and pinning V to V_MIN there understates SOP. Computing the
        # terminal voltage at the limiting current covers both cases and reduces
        # to V_MIN * I* exactly when voltage is what binds.
        I_lim = min(I, I_MAX_DCH)
        v_at_lim = v_ocv - I_lim * R_meas[j]

        # A SECOND, ASSUMPTION-LIGHT NUMBER.
        # The line above usually needs R at a current the cycler never applied:
        # it clamps the top pulse rates at low SOC and low SOH, so the measured
        # ladder tops out near 24 A on an aged cell while the rating is 35 A.
        # 85 % of rows carry that extrapolation. So each row also reports what
        # the HIGHEST ACTUALLY MEASURED pulse supports, which needs no rate
        # extrapolation at all and is therefore a defensible lower bound on SOP.
        k_top = int(np.argmax(I_meas))
        v_meas = v_ocv - I_meas[k_top] * R_meas[k_top]
        sop_meas = float(v_meas * I_meas[k_top]) if v_meas > V_MIN else float("nan")
        rows.append({
            "cell": cell, "cycle": cyc, "SOH": round(soh, 5),
            "SOC": round(soc, 4), "tau_s": tau,
            "OCV_V": round(v_ocv, 5), "R_used_mOhm": round(R_meas[j] * 1000, 3),
            "I_at_rate_A": round(float(I_meas[j]), 3),
            "I_star_A": round(float(I), 3),
            "I_limited_A": round(float(I_lim), 3),
            "V_at_limit_V": round(float(v_at_lim), 4),
            "SOP_W": round(float(v_at_lim * I_lim), 2),
            "I_max_measured_A": round(float(I_meas[k_top]), 3),
            "SOP_measured_floor_W": (round(sop_meas, 2)
                                     if np.isfinite(sop_meas) else ""),
            "limited_by": "current_rating" if I > I_MAX_DCH else "voltage",
            # Flag against the LIMITING current: that is the one whose R is
            # actually used, and clipping to 35 A brings it back inside the
            # measured pulse ladder (which tops out near 34 A).
            "rate_extrapolated": int(I_lim > I_meas.max() * 1.05
                                     or I_lim < I_meas.min() * 0.95),
            "ocv_soh_gap": round(gap, 4),
            "low_soc_hyst": int(soc < HYST_SUSPECT),
        })

    if not rows:
        sys.exit("생성된 행이 없습니다")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{args.out}: {len(rows)}행")
    print(f"제외: {rejected}")
    ext = sum(r["rate_extrapolated"] for r in rows)
    cur = sum(r["limited_by"] == "current_rating" for r in rows)
    print(f"  전류정격 제한 {cur}행 ({cur/len(rows)*100:.1f}%), "
          f"측정 rate 범위 밖 {ext}행 ({ext/len(rows)*100:.1f}%), "
          f"저 SOC 히스테리시스 주의 {sum(r['low_soc_hyst'] for r in rows)}행")


if __name__ == "__main__":
    main()
