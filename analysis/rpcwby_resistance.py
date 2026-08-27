"""Pulse resistances from the RPCWBY Samsung 30T aging cells.

WHY THIS FILE EXISTS
    The trim's remaining error is dominated by generalisation across cells:
    log k_f varies 63 % BETWEEN cells (sop_hybrid_spec.md 13.6), and three
    attempts to fix it inside the estimator - a guard, a feature redesign, a
    Kalman tracker - each bought about 2 % on the mean and made the worst cell
    worse. Six cells is the constraint, and RPCWBY carries two more Samsung 30T
    cells aged to 80 % SOH with the same 10 s pulse structure.

    Before any of it is pooled, the fresh-cell resistance has to land inside the
    UYPYDJ fresh spread. Two labs, two cyclers, two aging protocols; if the fresh
    cells already disagree then mixing them would put a lab difference into a
    surface that is supposed to describe a cell.

WHAT IS AND IS NOT THE SAME
    Same: cell model, 10 s pulses, 1 Hz logging, 25 C chamber.
    Different: voltage window 2.55-4.15 V against UYPYDJ's 2.5-4.2, aging by 1C
    CC discharge against fifteen-minute fast charge, and the SOC axis - RPCWBY
    logs Discharge_Capacity rather than a SOC column, so SOC is rebuilt here on
    the SAME rated axis this project uses (1 - Ah/3.0 from full) instead of
    taking whatever the file implies.
"""
from __future__ import annotations

import argparse
import csv as _csv
import io
import os
import re
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw", "RPCWBY")
TESTS = {"RPC_CC": "1_Test_1.zip", "RPC_US06": "2_Test_2.zip"}
OUT_ECM = os.path.join(HERE, "rpcwby_ecm.csv")
# A 10 s pulse cannot distinguish tau2 = 20 s from tau2 = 40 s - both look like
# a nearly straight ramp over the window - so the grid stops where the data can
# still tell them apart, and a fit that lands on either edge is discarded as
# unidentified rather than kept at the boundary. The first version ran the grid
# to 40 s, and the 5-95 % spread of the result piled up on that edge; the
# resulting R2 was inflated enough to fail the build gate on five of six cells
# with a systematic k_s of 0.95.
TAU2_LO, TAU2_HI = 1.5, 16.0
TAU2_GRID = np.exp(np.linspace(np.log(TAU2_LO), np.log(TAU2_HI), 80))
Q_RATED = 3.0
TAUS = (2.0, 10.0)
TAU_TOL = 1.5
I_THR = 1.0
MIN_PULSE_S = 5.0
# A measurement pulse is short and starts from rest. The first extraction took
# every |I| > 1 A stretch, which swept in the 1C discharge STEPS used to move SOC
# between measurements - 74 of them longer than 60 s in a single file. Those have
# a meaningless V_pre (the RC branches are already developed) and read 0.39-0.72x
# the UYPYDJ resistance at the same nominal 3 A, which is not a cell difference
# but a difference in what was called a pulse.
MAX_PULSE_S = 20.0
MIN_REST_S = 20.0
I_REST = 0.5
T_LO, T_HI = 20.0, 30.0        # keep the 25 C characterisations only


def read_csv(z, name):
    txt = z.read(name).decode("utf-8", errors="replace")
    rd = _csv.reader(io.StringIO(txt))
    hdr = [h.strip().lstrip("﻿") for h in next(rd)]
    ci = {h: i for i, h in enumerate(hdr)}
    cols = ["Test_Time(s)", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)",
            "Charge_Capacity(Ah)", "Aux_Temperature_3(C)"]
    out = {c: [] for c in cols}
    for r in rd:
        if len(r) <= ci["Voltage(V)"]:
            continue
        try:
            for c in cols:
                j = ci.get(c)
                out[c].append(float(r[j]) if j is not None and j < len(r)
                              and r[j] != "" else np.nan)
        except ValueError:
            for c in cols:
                if len(out[c]) and len(out[c]) > len(out["Test_Time(s)"]) - 1:
                    pass
            continue
    n = min(len(v) for v in out.values())
    return {c: np.array(v[:n], float) for c, v in out.items()}


def capacity(t, I, V, c_lo=-3.4, c_hi=-2.6, min_n=600):
    """Capacity from a contiguous ~1C discharge that reaches the floor.

    Step_Index is reused within a file - 164 distinct indices, one spanning
    919 minutes - so grouping by it splices unrelated segments and reported
    3.15 Ah for a 3.0 Ah cell. The test is identified by shape instead: about
    1C, starting near the top of the window, ending at the floor.
    """
    m = (I >= c_lo) & (I <= c_hi)
    d = np.diff(m.astype(int))
    s_ = np.flatnonzero(d == 1) + 1
    e_ = np.flatnonzero(d == -1) + 1
    if m[0]:
        s_ = np.r_[0, s_]
    if m[-1]:
        e_ = np.r_[e_, len(m)]
    best = None
    for a, b in zip(s_, e_):
        if b - a < min_n:
            continue
        if V[a] < 3.9 or V[b - 1] > 2.75:
            continue
        ah = float(-np.trapezoid(I[a:b], t[a:b]) / 3600.0)
        if ah > 1.5 and (best is None or ah > best):
            best = ah
    return best


def fit_pulse(rel, dv, ip, min_pts=6, min_dur=8.0):
    """Fit R_fast, R2 and tau2 to one pulse.

    Two horizons cannot separate the branches without a tau2, and borrowing a
    constant one is wrong where it matters most: measured tau2 falls to about
    half its fresh value in aged cells (sop_hybrid_spec.md 12.4), and these cells
    run to SOH 0.73. Reducing their aged pulses at a fresh tau2 mis-splits
    R_fast from R_slow exactly in the band the hybrid exists to correct.

        dv(t) = I * [ R_fast + R2 * (1 - exp(-t/tau2)) ]

    Only tau2 is nonlinear, so it is gridded and the other two solved in closed
    form - the same treatment used to recover tau2 from the stored horizons.

    tau1 is NOT fitted. RPCWBY logs at 1 Hz and measured tau1 is 0.244 s, so it
    is not resolvable here. That costs nothing: at horizons of 2 s and above the
    fast branch is fully developed and enters as the single number R0 + R1, which
    is why the trim carries one fast multiplier rather than two.
    """
    if len(rel) < min_pts or rel[-1] < min_dur:
        return None
    best = None
    for t2 in TAU2_GRID:
        A = np.column_stack([np.full(len(rel), ip), ip * (1 - np.exp(-rel / t2))])
        k, res, *_ = np.linalg.lstsq(A, dv, rcond=None)
        sse = float(np.sum((A @ k - dv) ** 2))
        if k[0] <= 0 or k[1] <= 0:
            continue
        if best is None or sse < best[0]:
            best = (sse, k[0], k[1], t2)
    if best is None:
        return None
    sse, rf, rs, t2 = best
    if t2 <= TAU2_GRID[1] or t2 >= TAU2_GRID[-2]:
        return None                       # unidentified at this pulse length
    return rf * 1000.0, rs * 1000.0, float(t2), float(np.sqrt(sse / len(rel)) * 1000)


def pulses(t, I):
    big = np.abs(I) > I_THR
    if not big.any():
        return []
    e = np.flatnonzero(np.diff(big.astype(int)))
    idx = list(e + 1)
    if big[0]:
        idx.insert(0, 0)
    if big[-1]:
        idx.append(len(big))
    return [(idx[i], idx[i + 1]) for i in range(0, len(idx) - 1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=os.path.join(HERE, "rpcwby_resistance.csv"))
    args = ap.parse_args()

    rows = []
    ecm = []
    anchors = {}
    for cell, zipname in TESTS.items():
        p = os.path.join(args.raw, zipname)
        if not os.path.exists(p):
            print(f"  {cell}: {zipname} 없음"); continue
        z = zipfile.ZipFile(p)
        ch = sorted([n for n in z.namelist()
                     if n.endswith(".CSV") and "__MACOSX" not in n
                     and "Charac" in n],
                    key=lambda n: int(re.search(r"Cycle(\d+)", n).group(1)))
        print(f"  {cell}: 특성화 {len(ch)}개")
        for name in ch:
            cyc = int(re.search(r"Cycle(\d+)", name).group(1))
            d = read_csv(z, name)
            t = d["Test_Time(s)"]; I = d["Current(A)"]; V = d["Voltage(V)"]
            T = d["Aux_Temperature_3(C)"]
            dch = d["Discharge_Capacity(Ah)"]; chg = d["Charge_Capacity(Ah)"]
            g = np.isfinite(t) & np.isfinite(I) & np.isfinite(V)
            if g.sum() < 1000:
                continue
            t, I, V, T, dch, chg = (x[g] for x in (t, I, V, T, dch, chg))
            # Net amp-hours from the start of the characterisation, on the rated
            # axis anchored at full - the same convention as data30t.py.
            ah = np.nan_to_num(chg) - np.nan_to_num(dch)
            soc = 1.0 + (ah - ah[0]) / Q_RATED
            n = 0
            for a, b in pulses(t, I):
                if a == 0 or not (MIN_PULSE_S <= t[b - 1] - t[a] <= MAX_PULSE_S):
                    continue
                # rest immediately before the step
                j = a - 1
                while j > 0 and abs(I[j]) < I_REST:
                    j -= 1
                rest = float(t[a - 1] - t[j])
                if rest < MIN_REST_S:
                    continue
                ip = float(np.median(I[a:b]))
                if abs(ip) < I_THR:
                    continue
                tc = float(np.nanmedian(T[a:b])) if np.isfinite(T[a:b]).any() else np.nan
                if np.isfinite(tc) and not (T_LO <= tc <= T_HI):
                    continue
                rel = t[a:b] - t[a]
                v0 = float(V[a - 1])
                fit = fit_pulse(rel, V[a:b] - v0, ip)
                if fit is not None:
                    rf, rs, t2f, rms = fit
                    ecm.append({
                        "cell": cell, "cycle": cyc, "SOH": None, "CAP_Ah": None,
                        "SOC": round(float(soc[a]), 4), "direction":
                            "discharge" if ip < 0 else "charge",
                        "rate_rank": None, "I_A": round(ip, 3),
                        "V_pre_V": round(v0, 5),
                        "rest_before_s": round(rest, 1),
                        "n_points": len(rel),
                        "R0_mOhm": round(rf, 4), "R1_mOhm": 1e-3,
                        "tau1_s": None,
                        "R2_mOhm": round(rs, 4), "tau2_s": round(t2f, 4),
                        "fit_rmse_mV": round(rms, 3)})
                for tau in TAUS:
                    k = int(np.searchsorted(rel, tau, side="right")) - 1
                    if k < 0 or rel[k] < tau - TAU_TOL:
                        continue
                    rows.append({
                        "cell": cell, "cycle": cyc,
                        "SOC": round(float(soc[a]), 4),
                        "direction": "discharge" if ip < 0 else "charge",
                        "I_A": round(ip, 3), "tau_s": tau,
                        "tau_actual_s": round(float(rel[k]), 2),
                        "T_C": round(tc, 2) if np.isfinite(tc) else "",
                        "rest_before_s": round(rest, 1),
                        "V_pre_V": round(v0, 5),
                        "V_tau_V": round(float(V[a + k]), 5),
                        "R_mOhm": round((float(V[a + k]) - v0) / ip * 1000, 4)})
                    n += 1
            cap = capacity(t, I, V)
            if cap is not None:
                anchors.setdefault(cell, []).append((cyc, cap))
            print(f"    cycle {cyc:>5}: 행 {n}  용량 "
                  f"{f'{cap:.4f} Ah' if cap else '없음'}")
    if not rows:
        print("  추출 없음"); return
    # SOH from the anchors, interpolated by cycle. Anchored at the FIRST
    # characterisation's capacity, the same convention soh_anchors uses.
    for cell, ac in anchors.items():
        ac.sort()
        cy = np.array([a for a, _ in ac], float)
        cp = np.array([c for _, c in ac], float)
        for r in rows:
            if r["cell"] != cell:
                continue
            c = float(np.interp(r["cycle"], cy, cp))
            r["CAP_Ah"] = round(c, 4)
            r["SOH"] = round(c / cp.max(), 5)
    print("\n  SOH 앵커:")
    for cell, ac in anchors.items():
        cp = [c for _, c in ac]
        print(f"    {cell}: {len(ac)}개  {max(cp):.3f} -> {min(cp):.3f} Ah  "
              f"SOH 1.000 -> {min(cp)/max(cp):.3f}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # SOH / CAP / tau1 / current bin onto the ECM-schema rows
    u = list(_csv.DictReader(open(os.path.join(HERE, "uypydj_ecm.csv"),
                                  encoding="utf-8")))
    t1 = float(np.median([float(r["tau1_s"]) for r in u]))
    edges = (2.0, 7.0, 16.0, 26.0, 40.0)
    for cell, ac in anchors.items():
        cy = np.array([a for a, _ in sorted(ac)], float)
        cp = np.array([c for _, c in sorted(ac)], float)
        for r in ecm:
            if r["cell"] != cell:
                continue
            c = float(np.interp(r["cycle"], cy, cp))
            r["CAP_Ah"] = round(c, 4); r["SOH"] = round(c / cp.max(), 5)
    for r in ecm:
        r["tau1_s"] = round(t1, 4)
        r["rate_rank"] = int(np.clip(np.searchsorted(edges, abs(r["I_A"])) - 1,
                                     0, len(edges) - 2))
    ecm = [r for r in ecm if r["SOH"] is not None]
    with open(OUT_ECM, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(ecm[0].keys()))
        w.writeheader(); w.writerows(ecm)
    t2 = np.array([r["tau2_s"] for r in ecm]); sh = np.array([r["SOH"] for r in ecm])
    rm = np.array([r["fit_rmse_mV"] for r in ecm])
    print(f"\n  -> {OUT_ECM}  {len(ecm):,}행  적합 RMSE 중앙 {np.median(rm):.2f} mV")
    print(f"  {'SOH 밴드':<12} {'n':>6} {'tau2 중앙':>10} {'5~95%':>14}")
    for lo, hi in ((0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85),
                   (0.72, 0.80)):
        m = (sh >= lo) & (sh < hi)
        if m.sum() < 20:
            continue
        print(f"  {f'{hi:.2f}-{lo:.2f}':<12} {m.sum():>6,} {np.median(t2[m]):>9.2f}s "
              f"{f'{np.percentile(t2[m],5):.1f}~{np.percentile(t2[m],95):.1f}':>14}")
    print(f"\n  -> {args.out}  {len(rows):,}행")


if __name__ == "__main__":
    main()
