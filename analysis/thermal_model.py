"""Identify a lumped thermal model from the RPCWBY Test#1/#2 aging records.

WHY THESE FILES
    UYPYDJ has cell temperature but no ambient channel, so heat generation and
    cooling cannot be separated - a rise could be self-heating or the chamber
    drifting. RPCWBY Test#1 logs BOTH, at 1 Hz, for 192 h per file, with the
    chamber deliberately stepped between 10 and 25 degC. That is a forced
    excitation of exactly the state being identified.

MODEL
    C_th dT/dt = Q_gen - (T_cell - T_amb) / R_th

    Q_gen is taken as joule heating, I^2 * R, with R absorbed into the fitted
    coefficient. The textbook alternative Q = I*(V - OCV) is exact in principle
    and was tried first, but it needs an OCV trace: interpolating the rest
    voltage between rests does not track OCV through a long discharge, and the
    result came out NEGATIVE on 25 % of the high-current samples - a cell that
    absorbs heat while delivering 30 A. I^2 needs no OCV and cannot change sign.

    The reversible (entropic) term I*T*dOCV/dT is not included: dOCV/dT is not
    measured in any of the three datasets, and the joule term dominates at these
    currents.

IDENTIFICATION
    Dividing by C_th makes it linear in two coefficients

        dT/dt = a * I^2  -  b * (T_cell - T_amb),   a = R/C_th,  b = 1/(C_th R_th)

    Least squares gives both. The thermal time constant tau = C_th R_th = 1/b
    needs neither separately; C_th follows once R is taken from the ECM surface.

TIME HAS TO BE CLEANED FIRST
    The records contain 196 duplicate timestamps, one backwards step, and gaps
    up to 36,000 s where the test was paused. Differentiating through those
    produced divide-by-zero and a fit with r^2 = 0.004. The series is split at
    gaps and resampled onto a uniform grid per segment before anything else.

WHAT IS FITTED AND WHAT IS CHECKED
    Coefficients come from one file; the check simulates T_cell forward from its
    first sample using only current, voltage and ambient, and compares against
    the measured trace it never sees.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw", "RPCWBY")

COL_T = "Test_Time(s)"
COL_I = "Current(A)"
COL_V = "Voltage(V)"
COL_TC = "Aux_Temperature_3(C)"
COL_TA = "Aux_Temperature_38(C)"


def load_csv(zf, member):
    with zf.open(member) as f:
        rd = csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
        hdr = next(rd)
        rows = list(rd)
    idx = {c: hdr.index(c) for c in (COL_T, COL_I, COL_V, COL_TC, COL_TA)
           if c in hdr}
    if len(idx) < 5:
        return None
    out = {}
    for c, i in idx.items():
        a = np.empty(len(rows))
        for k, r in enumerate(rows):
            try:
                a[k] = float(r[i])
            except (ValueError, IndexError):
                a[k] = np.nan
        out[c] = a
    return out


def segments(t, max_gap=60.0, min_len=600):
    """Contiguous stretches of monotone time, split at pauses."""
    d = np.diff(t)
    cut = np.flatnonzero((d <= 0) | (d > max_gap)) + 1
    edges = np.concatenate([[0], cut, [len(t)]])
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= min_len]


def ocv_from_rest(t, I, V, win=60.0):
    """A crude OCV proxy: the voltage during rests, held between them.

    Good enough for the heat term, because Q = I*(V - OCV) is multiplied by I -
    wherever the OCV proxy is worst (long high-current stretches) the error in
    (V - OCV) is small compared with the overpotential itself.
    """
    rest = np.abs(I) < 0.2
    ocv = np.where(rest, V, np.nan)
    idx = np.arange(len(V))
    good = np.isfinite(ocv)
    if good.sum() < 10:
        return np.full_like(V, np.nanmedian(V))
    return np.interp(idx, idx[good], ocv[good])


def fit(d, dt_smooth=60):
    t, I, V = d[COL_T], d[COL_I], d[COL_V]
    Tc, Ta = d[COL_TC], d[COL_TA]
    ok = np.isfinite(t) & np.isfinite(I) & np.isfinite(Tc) & np.isfinite(Ta)
    t, I, Tc, Ta = t[ok], I[ok], Tc[ok], Ta[ok]

    A_all, y_all = [], []
    k = max(3, int(dt_smooth))
    box = np.ones(k) / k
    for a0, b0 in segments(t):
        ts, Is, Tcs, Tas = t[a0:b0], I[a0:b0], Tc[a0:b0], Ta[a0:b0]
        grid = np.arange(ts[0], ts[-1], 1.0)
        if len(grid) < 4 * k:
            continue
        Ig = np.interp(grid, ts, Is)
        Tg = np.interp(grid, ts, Tcs)
        Ag = np.interp(grid, ts, Tas)
        # Temperature is quantised, so a one-sample difference is mostly noise.
        Ts_ = np.convolve(Tg, box, mode="valid")
        Q_ = np.convolve(Ig ** 2, box, mode="valid")
        D_ = np.convolve(Tg - Ag, box, mode="valid")
        dT = np.gradient(Ts_, 1.0)
        A_all.append(np.column_stack([Q_, -D_]))
        y_all.append(dT)
    if not A_all:
        return None
    A = np.vstack(A_all); y = np.concatenate(y_all)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    pred = A @ coef
    ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"a": a, "b": b, "tau_s": 1.0 / b if b > 0 else np.nan,
            "r2": ss, "n": len(y)}


def simulate(d, a, b):
    t, I, V = d[COL_T], d[COL_I], d[COL_V]
    Tc, Ta = d[COL_TC], d[COL_TA]
    ok = np.isfinite(t) & np.isfinite(I) & np.isfinite(V) & np.isfinite(Tc) & np.isfinite(Ta)
    t, I, V, Tc, Ta = (x[ok] for x in (t, I, V, Tc, Ta))
    T = np.empty(len(t)); T[0] = Tc[0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        if not (0 < dt <= 60.0):
            T[k] = Tc[k]                 # re-seed across a pause
            continue
        T[k] = T[k - 1] + dt * (a * I[k - 1] ** 2 - b * (T[k - 1] - Ta[k - 1]))
    return T, Tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(RAW, "1_Test_1.zip"))
    ap.add_argument("--files", type=int, default=4)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = sorted(n for n in zf.namelist()
                   if n.lower().endswith(".csv") and "__MACOSX" not in n
                   and "aging" in n.lower())
    if not names:
        sys.exit("aging 파일이 없습니다")
    pick = names[:: max(1, len(names) // args.files)][:args.files]

    print(f"{'파일':<40} {'n':>9} {'a=R/Cth':>10} {'tau':>8} {'r2':>7}")
    fits = []
    for n in pick:
        d = load_csv(zf, n)
        if d is None:
            continue
        f = fit(d)
        if f is None:
            continue
        fits.append((n, d, f))
        print(f"{n.split('/')[-1][:40]:<40} {f['n']:>9,} {f['a']:>10.3e} "
              f"{f['tau_s']:>7.0f}s {f['r2']:>7.3f}")

    if not fits:
        sys.exit("적합 실패")
    A = np.array([[f["a"], f["b"]] for _, _, f in fits])
    a, b = A.mean(0)
    print(f"\n  평균 a = {a:.3e} K/(s·A²),  b = {b:.3e} 1/s")
    print(f"  열 시상수 tau = {1/b:.0f} s ({1/b/60:.1f} 분)")
    for R in (0.013, 0.020, 0.030):
        print(f"    R = {R*1000:.0f} mΩ 가정 시  C_th = {R/a:6.1f} J/K, "
              f"R_th = {1/(b*R/a):5.2f} K/W")

    print("\n개루프 온도 시뮬레이션 (평균 계수, 측정 T는 초기값만 사용)")
    print(f"  {'파일':<44} {'RMSE':>8} {'최대':>8}")
    for n, d, _ in fits:
        T, Tm = simulate(d, a, b)
        e = T - Tm
        print(f"  {n.split('/')[-1][:44]:<44} {np.sqrt(np.mean(e**2)):>7.2f}K "
              f"{np.abs(e).max():>7.2f}K")


if __name__ == "__main__":
    main()
