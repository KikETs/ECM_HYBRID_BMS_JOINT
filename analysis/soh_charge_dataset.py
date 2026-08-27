"""Build the partial-charge-window dataset for SOH estimation, and its baseline.

THE PROBLEM AS POSED
    Given only a SEGMENT of a 1C constant-current charge - the part a car
    actually gets when it plugs in partway - estimate SOH. The whole curve would
    be pointless to learn from: integrating it gives the capacity directly
    (measured 2.830 Ah at SOH 1.00 falling to 2.384 at 0.891), so a model would
    just be reimplementing an integral.

WHERE THE INFORMATION IS, MEASURED BEFORE DECIDING THE WINDOW
    Time spent crossing each 0.1 V band, against SOH:

        SOH     3.6-3.7  3.7-3.8  3.8-3.9  3.9-4.0  4.0-4.1
        1.000     533 s    414 s    360 s    457 s    300 s
        0.891     393 s    422 s    360 s    436 s    293 s

    The 3.6-3.7 V band moves 26 %; 3.8-3.9 V does not move at all. The signal
    sits at the low-voltage end, where the phase transition is. A window chosen
    for convenience rather than from this table would be measuring noise.

REPRESENTATION
    Each curve is resampled onto a fixed VOLTAGE grid and stored as dQ/dV, the
    incremental-capacity curve. Voltage rather than time because aged cells
    traverse the same voltage span in less time - putting time on the axis would
    make the sequences different lengths and hide the shape change in a stretch.

THE BASELINE THIS HAS TO BEAT
    A single number - seconds to cross 3.6-3.7 V - fitted by least squares
    against SOH. Any network that cannot beat that is not learning the curve
    shape, it is re-deriving one band's width the hard way.

SPLIT
    Leave-one-cell-out over the six cells. There are only ~50 curves per cell;
    a random split would put the same cell's neighbouring cycles on both sides
    and report a number no new cell would ever see.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import load_uypydj_mat, uypydj_members  # noqa: E402
from uypydj_hppc_resistance import PROTOCOLS  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "UYPYDJ")
OUT = os.path.join(os.path.dirname(__file__), "cache", "soh_charge.npz")

V_LO, V_HI, N_GRID = 3.55, 4.05, 64


def curve_features(r, v_lo=V_LO, v_hi=V_HI, n=N_GRID):
    """dQ/dV on a fixed voltage grid over the chosen window."""
    V = np.asarray(r["V"], dtype=float)
    Ah = np.asarray(r["Ah"], dtype=float)
    I = np.asarray(r["I"], dtype=float)
    t = np.asarray(r["t"], dtype=float)
    m = I > 0.5                                  # charging only
    if m.sum() < 40:
        return None
    V, Ah, t = V[m], Ah[m], t[m]
    o = np.argsort(V)
    V, Ah, t = V[o], Ah[o], t[o]
    V, idx = np.unique(V, return_index=True)
    Ah, t = Ah[idx], t[idx]
    if V.min() > v_lo or V.max() < v_hi:
        return None
    g = np.linspace(v_lo, v_hi, n + 1)
    q = np.interp(g, V, Ah)
    dqdv = np.diff(q) / np.diff(g)
    tt = np.interp(g, V, t)
    band = float(np.interp(3.7, V, t) - np.interp(3.6, V, t))
    return dqdv.astype(np.float32), np.diff(tt).astype(np.float32), band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # 온도 채널 결함 제외 — 특성화 층(uypydj_ecm / _ocv / _hppc_resistance)은
    # 이미 이것을 하는데 이 데이터셋만 빠져 있었다.  290 곡선 중 8 개가
    # 결함 사이클 위에 있었고, 그것들이 SOH 팔의 편향을 +0.0001 에서
    # +0.0010 으로 열 배 부풀리고 있었다.
    from temp_defects import defective
    BAD = set()
    for k in ('halfC', 'other', 'CAP', 'schedule', 'HPPC', 'OCV', 'drive'):
        BAD |= defective(k)
    n_drop = 0

    X, Xt, y, cell, cyc, band = [], [], [], [], [], []
    for key, zs in PROTOCOLS.items():
        n = 0
        for zp in [os.path.join(args.raw, x) for x in zs]:
            if not os.path.exists(zp):
                continue
            for member, info in uypydj_members(zp, part="ONE_C_charge"):
                try:
                    r = load_uypydj_mat(zp, member)
                except Exception:                        # noqa: BLE001
                    continue
                if r["SOH"] is None or not np.isfinite(r["SOH"]):
                    continue
                cy = info["cycle"] if info["cycle"] is not None else -1
                if (key, int(cy)) in BAD:
                    n_drop += 1
                    continue
                f = curve_features(r)
                if f is None:
                    continue
                X.append(f[0]); Xt.append(f[1]); band.append(f[2])
                y.append(float(r["SOH"])); cell.append(key)
                cyc.append(cy)
                n += 1
        print(f"  {key:<20} {n:>3}개")

    print(f"  온도 채널 결함으로 제외 {n_drop}개")
    X = np.stack(X); Xt = np.stack(Xt); y = np.array(y, dtype=np.float32)
    cell = np.array(cell); cyc = np.array(cyc); band = np.array(band, np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, Xt=Xt, y=y, cell=cell, cycle=cyc, band=band,
             v_lo=V_LO, v_hi=V_HI)
    print(f"\n{args.out}: {len(y)}개, 입력 {X.shape[1]}차원, "
          f"SOH {y.min():.3f}~{y.max():.3f}")

    print(f"\n기준선: 3.6~3.7 V 소요시간 하나로 선형회귀, leave-one-cell-out")
    print(f"  {'홀드아웃 셀':<20} {'n':>4} {'RMSE':>9} {'MAE':>9}")
    errs = []
    for c in sorted(set(cell)):
        tr, te = cell != c, cell == c
        if te.sum() < 3 or tr.sum() < 10:
            continue
        A = np.column_stack([band[tr], np.ones(tr.sum())])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p = band[te] * w[0] + w[1]
        e = p - y[te]
        errs.append(e)
        print(f"  {c:<20} {te.sum():>4} {np.sqrt(np.mean(e**2)):>9.4f} "
              f"{np.mean(np.abs(e)):>9.4f}")
    e = np.concatenate(errs)
    print(f"  {'전체':<20} {len(e):>4} {np.sqrt(np.mean(e**2)):>9.4f} "
          f"{np.mean(np.abs(e)):>9.4f}")
    print(f"\n  참고: SOH를 항상 평균으로 답하면 RMSE {np.std(y):.4f}")


if __name__ == "__main__":
    main()
