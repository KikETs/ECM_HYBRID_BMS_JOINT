"""Pull an (SOH x SOC x C-rate x direction) resistance grid out of the UYPYDJ HPPC tests.

WHY THIS EXISTS
    SOP is a voltage-limited quantity, so a SOH-dependent SOP model needs to know
    how the cell's pulse resistance grows as it ages. The UYPYDJ campaign runs an
    HPPC every 30 cycles all the way down to 70 % SOH, which is exactly that
    measurement - but it is buried one file per test inside nine multi-GB zips,
    and the HPPC files themselves DO NOT CARRY SOH.

SOH HAS TO BE JOINED IN, AND THE JOIN IS THE PART THAT CAN GO WRONG
    Only the Fifteen_Drive_Cycles files carry SOH/CAP. Both file kinds carry the
    aging-test cycle number in their name, so SOH is interpolated onto the HPPC's
    cycle number from the drive-cycle series of the SAME PROTOCOL. Interpolation
    is linear and NEVER extrapolated: an HPPC outside the protocol's drive-cycle
    cycle range is dropped rather than given an invented SOH. Protocols are kept
    strictly separate - they are different cells with different aging paths, and
    joining across them would be inventing data.

PULSE STRUCTURE (verified on cycle #5 of the CONSTANT CURRENT protocol, not assumed)
    104 pulses of 10 s each = 13 SOC levels x (4 discharge + 4 charge) rates.
    Discharge amplitudes -2.98 / -11.89 / -23.77 / -34.17 A, i.e. about 1C / 4C /
    8C / 11.5C on a 3 Ah cell. Logged at 1 Hz, so resistance is available at 2 s
    and 10 s into the pulse - the timescales the SOP work actually uses.

RESISTANCE DEFINITION
    R(tau) = (V(t0 + tau) - V(t0-)) / I_pulse
    with V(t0-) the last sample BEFORE the current step and I_pulse the median
    current during the pulse. This is the standard HPPC ratio; it includes the
    ohmic drop plus whatever polarisation has developed by tau, which is what a
    tau-second power limit needs.

SOC AXIS
    1.0 + Ah/3.0 - rated, this project's convention throughout. The HPPC file's
    own Ah counter starts at 0 with the cell full, so it is a depth-from-full and
    can be used directly, unlike the drive-cycle files' published SOC column,
    which is normalised by aged capacity. See data30t.load_uypydj_mat.

OUTPUT
    uypydj_hppc_resistance.csv - one row per (protocol, cycle, SOC, direction,
    C-rate, tau), carrying the joined SOH and the raw pieces of the ratio so any
    row can be re-derived without rerunning this.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import Q_RATED_AH, load_uypydj_mat, uypydj_members  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "UYPYDJ")
OUT = os.path.join(os.path.dirname(__file__), "uypydj_hppc_resistance.csv")

# Zips grouped by the cell/protocol they belong to. A protocol split across two
# zips is ONE aging series and must be joined as one, or the second half gets no
# SOH anchor points below its own first drive cycle.
PROTOCOLS = {
    "CC": ["03-CONSTANT CURRENT protocol_Cycles 0 to 1000.zip",
           "04-CONSTANT CURRENT protocol_Cycles 1000 to 1908.zip"],
    "BOOST": ["05-BOOST CHARGING protocol_Cycles 0 to 1000.zip",
              "06-BOOST CHARGING protocol_Cycles 1000 to 1908.zip"],
    "BOOST_REST": ["07-BOOST CHARGING WITH REST protocol.zip"],
    "BOOST_NEGPULSE": ["08-BOOST CHARGING WITH NEGATIVE PULSES protocol_Cycles 1 to 1000.zip",
                       "09-BOOST CHARGING WITH NEGATIVE PULSES protocol_Cycles 1000 to 1730.zip"],
    "BOOST_NEGPULSE_1S": ["10-BOOST CHARGING WITH NEGATIVE PULSES_1s_PERIOD_protocol.zip"],
    "CC_CELL2": ["11-CONSTANT CURRENT SECOND CELL protocol.zip"],
}

I_THRESHOLD = 1.0        # A, above which a sample counts as "in a pulse"
MIN_PULSE_S = 5.0        # reject anything too short to read a 2 s and 10 s point
TAUS = (2.0, 10.0)       # seconds into the pulse
# A nominal 10 s pulse measures 9.9 s between its first and last in-pulse sample,
# so demanding a sample at exactly t0+10 threw away EVERY 10 s row - the first
# run of this script produced 5167 rows that were all tau = 2 s and looked fine.
# The reported point is now the last sample at or before t0+tau, accepted only if
# it actually reached tau - TAU_TOL, with the achieved time recorded per row.
TAU_TOL_S = 1.5


CAP_MIN, CAP_MAX = 1.5, 3.2      # plausible capacity for a 3.0 Ah cell


def soh_anchors(zips: list) -> tuple:
    """(cycle, SOH, CAP) series for one protocol.

    ANCHORS COME FROM THE 0.5C CAPACITY TESTS, NOT THE DRIVE CYCLES. Both carry
    the measured capacity, but the capacity tests run at every characterisation
    - about twice as often - and, decisively, they START EARLIER AND END LATER:
    cycle 2-1909 against the drive cycles' 7-1893.

    That matters because everything else here refuses to extrapolate past the
    anchor range. With drive-cycle anchors the freshest HPPC (cycle 5) and the
    most aged one both fell outside and were dropped, which left the fitted
    surface spanning SOH 0.70-0.99 while the runs being simulated sat at 0.696
    and 1.000 - i.e. the two worst-performing runs were the two being
    extrapolated. Below SOH 0.72 there was exactly ONE OCV curve.

    One capacity value in 104 reads 7.117 Ah, which a 3.0 Ah cell cannot do; the
    range filter drops it rather than letting it distort the interpolation.
    """
    import zipfile as _zipfile

    import scipy.io as _sio

    pairs = {}
    for zp in zips:
        if not os.path.exists(zp):
            continue
        z = _zipfile.ZipFile(zp)
        for member, info in uypydj_members(zp, part="halfC_CAP"):
            try:
                m = _sio.loadmat(io.BytesIO(z.read(member)), squeeze_me=True,
                                 struct_as_record=False)
                st = m[next(k for k in m if not k.startswith("__"))]
                cs = np.atleast_1d(np.asarray(getattr(st, "cycle"), dtype=float))
                qs = np.atleast_1d(np.asarray(getattr(st, "halfC_cap"), dtype=float))
            except Exception:                            # noqa: BLE001
                continue
            for c, q in zip(cs, qs):
                if np.isfinite(c) and CAP_MIN <= q <= CAP_MAX:
                    pairs[int(c)] = float(q)
    if len(pairs) < 2:                                   # fall back on drive cycles
        cyc, soh, cap = [], [], []
        for zp in zips:
            if not os.path.exists(zp):
                continue
            for member, info in uypydj_members(zp, part="Fifteen_Drive_Cycles"):
                if info["cycle"] is None:
                    continue
                try:
                    r = load_uypydj_mat(zp, member)
                except Exception:                        # noqa: BLE001
                    continue
                if r["SOH"] is None or r["CAP_Ah"] is None:
                    continue
                cyc.append(info["cycle"]); soh.append(r["SOH"]); cap.append(r["CAP_Ah"])
        if not cyc:
            return np.array([]), np.array([]), np.array([])
        o = np.argsort(cyc)
        return np.array(cyc)[o], np.array(soh)[o], np.array(cap)[o]

    c = np.array(sorted(pairs))
    q = np.array([pairs[int(x)] for x in c])
    return c.astype(float), q / q.max(), q


def find_pulses(t, I):
    """Contiguous runs of |I| > threshold, as (start, stop) index pairs."""
    big = np.abs(I) > I_THRESHOLD
    if not big.any():
        return []
    edges = np.flatnonzero(np.diff(big.astype(int)))
    idx = list(edges + 1)
    if big[0]:
        idx.insert(0, 0)
    if big[-1]:
        idx.append(len(big))
    return [(idx[i], idx[i + 1]) for i in range(0, len(idx) - 1, 2)]


def rank_pulses(pulses, t, I, Ah):
    """Label each pulse with its SOC group and its rate rank inside that group.

    THE MEASURED CURRENT CANNOT BE USED AS THE RATE LABEL. The protocol steps
    four discharge rates (about 1C / 4C / 8C / 11.5C fresh) at each SOC, but at
    low SOC the top rates would drive the cell through the voltage floor, so the
    cycler clamps them - measured amplitudes smear into a continuum (92 distinct
    values when divided by rated capacity, 70 when divided by aged capacity, over
    what should be four levels). The protocol's own ordering survives that
    clamping, so rank within the SOC group is the label that stays meaningful.

    Groups are cut where the SOC drops between consecutive pulses, not by
    assuming a fixed pulse count per group.
    """
    out, grp, rank_d, rank_c, prev = [], 0, 0, 0, None
    for a, b in pulses:
        soc = 1.0 + float(Ah[a]) / Q_RATED_AH
        if prev is not None and prev - soc > 0.01:      # stepped down to next SOC
            grp += 1
            rank_d = rank_c = 0
        prev = soc
        if float(np.median(I[a:b])) < 0:
            r = rank_d; rank_d += 1
        else:
            r = rank_c; rank_c += 1
        out.append((a, b, grp, r))
    return out


def pulse_rows(t, V, I, Ah, a, b, grp, rank):
    """Resistance at each tau for one pulse, or [] if the pulse is unusable."""
    if t[b - 1] - t[a] < MIN_PULSE_S or a == 0:
        return []
    ip = float(np.median(I[a:b]))
    if abs(ip) < I_THRESHOLD:
        return []
    v0 = float(V[a - 1])                 # last sample before the step
    soc = 1.0 + float(Ah[a]) / Q_RATED_AH
    rel = t[a:b] - t[a]
    out = []
    for tau in TAUS:
        k = int(np.searchsorted(rel, tau, side="right")) - 1
        if k < 0 or rel[k] < tau - TAU_TOL_S:
            continue                     # pulse never got near this timescale
        out.append({
            "SOC": round(soc, 4),
            "soc_group": grp,
            "rate_rank": rank,
            "direction": "discharge" if ip < 0 else "charge",
            "I_A": round(ip, 3),
            "CRate_rated": round(abs(ip) / Q_RATED_AH, 3),
            "tau_s": tau,
            "tau_actual_s": round(float(rel[k]), 2),
            "V_pre_V": round(v0, 5),
            "V_tau_V": round(float(V[a + k]), 5),
            "R_mOhm": round((float(V[a + k]) - v0) / ip * 1000.0, 4),
        })
    return out


from temp_defects import defective_hppc  # noqa: E402

_TEMP_BAD = defective_hppc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default=None, help="run a single protocol key")
    args = ap.parse_args()

    rows, skipped, n_temp = [], [], 0
    keys = [args.only] if args.only else list(PROTOCOLS)
    for key in keys:
        zips = [os.path.join(args.raw, z) for z in PROTOCOLS[key]]
        cyc, soh, cap = soh_anchors(zips)
        if len(cyc) < 2:
            skipped.append((key, "SOH 앵커 부족")); continue
        print(f"{key}: SOH 앵커 {len(cyc)}개, cycle {cyc.min()}-{cyc.max()}, "
              f"SOH {soh.min():.3f}-{soh.max():.3f}")

        n_used = n_out = 0
        for zp in zips:
            if not os.path.exists(zp):
                continue
            for member, info in uypydj_members(zp, part="HPPC"):
                c = info["cycle"]
                # No extrapolation: an HPPC outside the anchor range gets no SOH.
                if c is None or c < cyc.min() or c > cyc.max():
                    n_out += 1
                    continue
                if (key, c) in _TEMP_BAD:
                    # 온도 채널 결함 특성화 — temp_defects.py 참조
                    n_temp += 1
                    continue
                try:
                    r = load_uypydj_mat(zp, member)
                except Exception as e:                   # noqa: BLE001
                    skipped.append((member, type(e).__name__)); continue
                t, V, I, Ah = r["t"], r["V"], r["I"], r["Ah"]
                s = float(np.interp(c, cyc, soh))
                q = float(np.interp(c, cyc, cap))
                got = 0
                for a, b, grp, rank in rank_pulses(find_pulses(t, I), t, I, Ah):
                    for row in pulse_rows(t, V, I, Ah, a, b, grp, rank):
                        row.update(protocol=key, cycle=c, SOH=round(s, 5),
                                   CAP_Ah=round(q, 5), file=info["file"])
                        rows.append(row); got += 1
                if got:
                    n_used += 1
        print(f"  HPPC {n_used}개 사용, {n_out}개 앵커 범위 밖으로 제외")

    if not rows:
        sys.exit("추출된 행이 없습니다")
    cols = ["protocol", "cycle", "SOH", "CAP_Ah", "SOC", "soc_group", "direction",
            "rate_rank", "I_A", "CRate_rated", "tau_s", "tau_actual_s",
            "V_pre_V", "V_tau_V", "R_mOhm", "file"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{args.out}: {len(rows)}행")
    if skipped:
        print(f"건너뜀 {len(skipped)}건, 예: {skipped[:3]}")


if __name__ == "__main__":
    main()
