"""Test#8: 측정 직전 펄스만 바꿨을 때 트림이 그것을 읽는가.

구조 (여섯 파일 모두 동일하게 확인됨)
    [직전 펄스, 그 파일의 C-rate, 17~733 s]  ->  즉시  ->  [SOP 탐색 펄스, 9 s]
    9 s 펄스가 12~13 개, 그중 3~4 개가 V_end 2.55 V 에 수렴한다. 수렴한 것이
    측정 SOP 이고, 시트의 전압 제한 3 점(SOC 0.2 / 0.15 / 0.1)과 정확히 맞는다.

무엇을 재는가
    각 9 s 펄스가 시작되기 **직전** 샘플에서 트림 특징 12 개를 찍는다. 그 시점의
    통계에는 방금 끝난 직전 펄스가 들어 있다. 따라서 C-rate 가 다르면 특징이
    달라야 하고, 트림이 그 premise 위에 서 있다면 k 도 달라야 한다.

    600 s EW 창에서 60 s 펄스는 가중 1-exp(-60/600) = 9.5 % 를 받고, I^2 가중이
    있으므로 쉼보다 훨씬 크게 들어간다. 시간척도는 맞다.

정직하게 미리 적어 둘 것
    0 C 는 트림이 학습된 온도 범위(블록 T 특징 25.96~30.66 C) 밖이다. 입력 정규화
    가 +-4 sd 로 자르므로 발산하지는 않지만 외삽이고, 결과는 그 사실과 함께 읽어야
    한다.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces                    # noqa: E402
from sop_trim_features import TrimFeatures       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "raw", "RPCWBY", "8_Test_8.zip")
OUT = os.path.join(HERE, "cache", "t8")
Q_RATED = 3.0
V_FULL = 4.19
SOP_DUR = (8.0, 12.0)          # 9 s 펄스가 탐색 펄스다
RATE_PAT = [("_0C_SOP_0degC", 0.0), ("_C_3_SOP_0degC", 1 / 3),
            ("_1C_SOP_0degC", 1.0), ("_2C_SOP_0degC", 2.0),
            ("_3C_SOP_0degC", 3.0), ("_4C_SOP_0degC", 4.0)]


def soc_track(t, I, V):
    ah = np.concatenate([[0.0], np.cumsum(I[1:] * np.diff(t) / 3600.0)])
    idx = np.where(V >= V_FULL)[0]
    if len(idx):
        brk = np.where(np.diff(idx) > 1)[0]
        anc = [int(g[np.argmax(ah[g])]) for g in np.split(idx, brk + 1)]
    else:
        anc = [0]
    anc = np.array(sorted(set(anc)))
    k = np.clip(np.searchsorted(anc, np.arange(len(t)), side="right") - 1,
                0, len(anc) - 1)
    return 1.0 + (ah - ah[anc[k]]) / Q_RATED


def run_file(zf, name, rate, sd, sc, soh):
    with zf.open(name) as f:
        head = io.TextIOWrapper(f, encoding="utf-8-sig",
                                errors="replace").readline().strip().split(",")
    tc = [c for c in head if "Aux_Temperature" in c]
    with zf.open(name) as f:
        df = pd.read_csv(f, usecols=["Test_Time(s)", "Current(A)", "Voltage(V)"] + tc,
                         encoding="utf-8-sig", low_memory=False)
    t = df["Test_Time(s)"].to_numpy(float)
    I = df["Current(A)"].to_numpy(float)
    V = df["Voltage(V)"].to_numpy(float)
    T = df[tc[0]].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(I) & np.isfinite(V) & np.isfinite(T)
    t, I, V, T = t[ok], I[ok], V[ok], T[ok]
    soc = np.clip(soc_track(t, I, V), 0.01, 1.0)

    idx = np.where(I < -0.5)[0]
    segs = [s for s in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            if len(s) >= 2 and SOP_DUR[0] <= t[s[-1]] - t[s[0]] <= SOP_DUR[1]]
    starts = {int(s[0]): s for s in segs}

    tf = TrimFeatures(sd, sc)
    out = []
    for j in range(1, len(t)):
        if j in starts:
            s = starts[j]
            out.append(dict(C_rate=rate, t=float(t[j]), SOC=float(soc[j - 1]),
                            I_A=float(np.median(I[s])), V_pre=float(V[j - 1]),
                            V_end=float(V[s[-1]]), T_C=float(T[j - 1]),
                            x=tf.vector(float(soc[j - 1]), soh)))
        tf.update(float(t[j] - t[j - 1]), float(I[j]), float(V[j]),
                  float(T[j]), float(soc[j]), soh)
    return out, float(np.nanmin(soc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="CC")
    ap.add_argument("--soh", type=float, default=None,
                    help="미지정이면 파일의 만충->바닥 쿨롱량에서 추정")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sd, sc = surfaces(a.holdout)
    zf = zipfile.ZipFile(ZIP)
    names = [n for n in zf.namelist()
             if n.endswith(".csv") and not n.startswith("__MACOSX")]

    soh = a.soh if a.soh else 0.90       # 1 차 추정, 아래에서 갱신 보고
    rows = []
    print(f"  풀 {a.holdout}   SOH {soh:.3f}")
    print(f"  {'C-rate':>8}{'파일':<38}{'9s펄스':>7}{'수렴':>6}{'SOC 최저':>10}{'T 중앙':>8}")
    for pat, rate in RATE_PAT:
        cand = [n for n in names if pat in n]
        if not cand:
            print(f"  {rate:>8.2f}  파일 없음 ({pat})")
            continue
        got, socmin = run_file(zf, cand[0], rate, sd, sc, soh)
        conv = [g for g in got if abs(g["V_end"] - 2.55) <= 0.04]
        rows += got
        print(f"  {rate:>8.2f}{cand[0].split('/')[-1][:37]:<38}{len(got):>7}"
              f"{len(conv):>6}{socmin:>10.3f}"
              f"{np.median([g['T_C'] for g in got]):>8.1f}")
    if not rows:
        raise SystemExit("  펄스를 찾지 못했다")
    np.savez(os.path.join(a.out, f"t8_feats_{a.holdout}.npz"),
             X=np.stack([r["x"] for r in rows]).astype(np.float32),
             C_rate=np.array([r["C_rate"] for r in rows]),
             SOC=np.array([r["SOC"] for r in rows]),
             I_A=np.array([r["I_A"] for r in rows]),
             V_pre=np.array([r["V_pre"] for r in rows]),
             V_end=np.array([r["V_end"] for r in rows]),
             T_C=np.array([r["T_C"] for r in rows]),
             soh=soh)
    print(f"\n  저장 {len(rows)} 펄스 x 12 특징")


if __name__ == "__main__":
    main()
