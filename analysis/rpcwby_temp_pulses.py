"""Pulse resistances across six temperatures and three horizons (RPCWBY Test#3).

WHY THE TEMPERATURE AXIS IS MISSING AND WHY IT MATTERS
    Everything in this project so far is at 25 C. The reference paper's own
    figure sweeps -20 to 40 C and its equivalent-circuit model degrades from
    25 mV at 40 C to 77 mV at -20 C - the temperature axis is where an ECM is
    supposed to be weakest, so a claim that a corrected ECM beats a network is
    incomplete without it.

    Test#3 is one Samsung 30T cell measured at -20, -10, 0, 10, 25 and 40 C with
    2 s, 10 s and 30 s pulses - eighteen files, the exact axis needed. It is a
    DIFFERENT cell from the six aging cells, so any evaluation against it is a
    cross-dataset generalisation test and is labelled as such.

TAU = 2 s AND 30 s COME FREE
    The UYPYDJ HPPC only reaches 2 s and 10 s, and the discharge SOP label had
    zero interpolated rows at 2 s because 29 A cannot pull a fresh cell to 2.5 V
    in two seconds. Test#3 carries 2 s and 30 s directly.

SOC IS REBUILT ON THE RATED AXIS
    The capacity counters reset per step (verified on Test#1: 18 decreasing
    points in one file), so SOC is integrated from the current instead, anchored
    at full and divided by 3.0 Ah - the same convention as data30t.py.
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
ZIP = os.path.join(HERE, "..", "raw", "RPCWBY", "3_Test_3.zip")
OUT = os.path.join(HERE, "rpcwby_temp_pulses.csv")
Q_RATED = 3.0
I_THR = 1.0
MIN_REST_S = 20.0
I_REST = 0.5
TAU_TOL = 1.5


def read(z, name):
    rd = _csv.reader(io.StringIO(z.read(name).decode("utf-8", errors="replace")))
    hdr = [h.strip().lstrip("﻿") for h in next(rd)]
    ci = {h: i for i, h in enumerate(hdr)}
    cols = ["Test_Time(s)", "Current(A)", "Voltage(V)",
            "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)",
            "Aux_Temperature_5(C)"]
    out = {c: [] for c in cols}
    for r in rd:
        if len(r) <= ci["Voltage(V)"]:
            continue
        for c in cols:
            j = ci.get(c)
            try:
                out[c].append(float(r[j]) if j is not None and j < len(r)
                              and r[j] != "" else np.nan)
            except ValueError:
                out[c].append(np.nan)
    n = min(len(v) for v in out.values())
    return {c: np.array(v[:n], float) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    names = [n for n in z.namelist()
             if n.endswith(".CSV") and "__MACOSX" not in n]
    rows = []
    print(f"  {'파일':<34} {'설정 T':>7} {'실측 T':>7} {'tau':>5} "
          f"{'펄스':>6} {'SOC 범위':>14}")
    for name in sorted(names):
        m = re.search(r"_(-?\d+)degC_(\d+)s_", name)
        if not m:
            continue
        t_set = int(m.group(1)); tau = float(m.group(2))
        d = read(z, name)
        t = d["Test_Time(s)"]; I = d["Current(A)"]; V = d["Voltage(V)"]
        T = d["Aux_Temperature_5(C)"]
        chg = np.nan_to_num(d["Charge_Capacity(Ah)"])
        dch = np.nan_to_num(d["Discharge_Capacity(Ah)"])
        g = np.isfinite(t) & np.isfinite(I) & np.isfinite(V)
        t, I, V, T, chg, dch = (x[g] for x in (t, I, V, T, chg, dch))
        # integrate the current rather than trust counters that reset per step
        ah = np.concatenate([[0.0], np.cumsum(np.diff(t) * I[:-1] / 3600.0)])
        soc = np.clip(1.0 + (ah - ah[0]) / Q_RATED, -0.05, 1.05)
        big = np.abs(I) > I_THR
        e = np.flatnonzero(np.diff(big.astype(int)))
        idx = list(e + 1)
        if big[0]:
            idx.insert(0, 0)
        if big[-1]:
            idx.append(len(big))
        n = 0
        for a, b in [(idx[i], idx[i + 1]) for i in range(0, len(idx) - 1, 2)]:
            if a == 0:
                continue
            dur = t[b - 1] - t[a]
            if not (tau - TAU_TOL <= dur <= tau + 5.0):
                continue
            ip = float(np.median(I[a:b]))
            if abs(ip) < I_THR:
                continue
            j = a - 1
            while j > 0 and abs(I[j]) < I_REST:
                j -= 1
            rest = float(t[a - 1] - t[j])
            if rest < MIN_REST_S:
                continue
            rel = t[a:b] - t[a]
            k = int(np.searchsorted(rel, tau, side="right")) - 1
            if k < 0 or rel[k] < tau - TAU_TOL:
                continue
            v0 = float(V[a - 1])
            tc = float(np.nanmedian(T[a:b])) if np.isfinite(T[a:b]).any() else np.nan
            rows.append({
                "temp_set_C": t_set, "T_cell_C": round(tc, 2) if np.isfinite(tc) else "",
                "tau_s": tau, "SOC": round(float(soc[a]), 4),
                "direction": "discharge" if ip < 0 else "charge",
                "I_A": round(ip, 3), "rest_before_s": round(rest, 1),
                "V_pre_V": round(v0, 5), "V_tau_V": round(float(V[a + k]), 5),
                "R_mOhm": round((float(V[a + k]) - v0) / ip * 1000.0, 4)})
            n += 1
        ss = soc[np.abs(I) < I_REST]
        print(f"  {os.path.basename(name)[:34]:<34} {t_set:>6}C "
              f"{np.nanmedian(T):>6.1f}C {tau:>4.0f}s {n:>6,} "
              f"{f'{ss.min():.2f}~{ss.max():.2f}' if len(ss) else '-':>14}")
    if not rows:
        print("  추출 없음"); return
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  -> {args.out}  {len(rows):,}행")


if __name__ == "__main__":
    main()
