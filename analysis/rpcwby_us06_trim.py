"""US06 주행에서 트림 특징을 뽑는다 — 외부 셀에서 하이브리드를 채점하기 위해.

WHY THIS FILE IS THE STRONGEST TEST THE PROJECT CAN RUN
    지금까지 하이브리드는 언제나 UYPYDJ 안에서만 평가됐다. 셀 홀드아웃이었지만
    같은 실험실, 같은 사이클러, 같은 프로토콜이다. RPCWBY Test#2 는 **주행(US06)과
    측정 SOP 를 한 셀에 함께** 담은 유일한 파일이므로, 여기서만 트림을 완전히
    바깥에서 시험할 수 있다.

    18.6 절은 보정 없는 pooled ECM 만 채점했다 — 트림이 UYPYDJ 특성화 사이클로
    색인돼 있어서 옮길 수 없었기 때문이다. 이 파일이 그 색인을 없앤다: 특징을
    RPCWBY 자신의 주행에서 다시 계산한다.

짝짓기 규칙은 UYPYDJ 와 같다 (sop_trim_dataset.pair)
    각 특성화는 **직전 주행**의 마지막 12 블록(600 s)을 가져간다. 특성화 자체에서는
    절대 가져오지 않는다 — 차량은 HPPC 를 하지 않는다.

    다만 여기서는 게이트가 하나 더 필요하다. Test#2 의 주행 파일은 특성화 직전에
    만충전으로 끝나므로 마지막 2 h 가 통째로 CC-CV 충전이다. 그대로 12 블록을
    가져오면 학습 분포 밖의 충전 블록만 담긴다. 그래서 duty(|I|>5 A 의 EW 비율)가
    DUTY_MIN 을 넘는 블록만 센다. UYPYDJ 블록의 duty 5 %tile 이 0.028 이다.

SOC 는 적분해서 세운다
    용량 카운터가 스텝마다 리셋되므로(rpcwby_temp_pulses.py 에서 확인) 쓸 수 없다.
    주행 구간은 만충전으로 끝나므로 그 지점을 SOC=1 로 못박고 뒤로 적분한다.
    축은 이 프로젝트의 정격 축(Q_RATED=3.0)이고, RPCWBY readme 117 행의 정의와
    같은 규약이다.

온도는 이 시험의 진짜 위험이다
    UYPYDJ 블록의 T 특징은 25.96~30.66 C 범위다. Test#2 는 챔버가 10 과 25 C 를
    오간다. 즉 트림은 **한 번도 본 적 없는 온도**를 받는다. 입력 정규화가 ±4 sd 로
    자르므로 발산하지는 않지만 외삽이고, 그 사실을 결과와 함께 보고해야 한다.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces                      # noqa: E402
from sop_trim_features import TrimFeatures         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZIPS = {2: os.path.join(HERE, "..", "raw", "RPCWBY", "2_Test_2.zip"),
        1: os.path.join(HERE, "..", "raw", "RPCWBY", "1_Test_1.zip")}
CELLS = {2: "RPC_US06", 1: "RPC_CC"}
RES = os.path.join(HERE, "rpcwby_resistance.csv")
OUT = os.path.join(HERE, "cache", "us06")

Q_RATED = 3.0
BLOCK_S = 600.0
N_BLOCKS = 12
DUTY_MIN = 0.05
WINDOW_H = 14.0
V_FULL, I_FULL = 4.19, 0.30            # end of CV
BASE_COLS = ["Test_Time(s)", "Current(A)", "Voltage(V)"]


def cell_temp_col(header):
    """어느 Aux_Temperature 가 셀이고 어느 것이 챔버인가.

    WHY NOT JUST TAKE THE LARGER STANDARD DEVIATION
        Test#2 는 챔버를 10 과 25 C 사이로 순환시키므로 챔버 채널의 표준편차가
        셀보다 크다(5.71 대 5.54) - 크기 규칙은 거기서 틀린 채널을 고른다.
        셀은 부하에 초 단위로 반응하고 챔버는 시간 단위로 움직이므로, 가르는
        것은 크기가 아니라 **단기 변동**이다. 차분의 표준편차를 쓴다.
    """
    return [c for c in header if "Aux_Temperature" in c]


def soh_of(cycle, cell):
    import csv as _csv
    rows = [r for r in _csv.DictReader(open(RES, encoding="utf-8"))
            if r["cell"] == cell]
    pts = sorted({(float(r["cycle"]), float(r["SOH"])) for r in rows})
    a = np.array(pts)
    return float(np.interp(cycle, a[:, 0], a[:, 1]))


def pairs(zf):
    """(Charac cycle) -> 직전 aging 주행의 마지막 파트."""
    names = [n for n in zf.namelist()
             if n.endswith(".CSV") and not n.startswith("__MACOSX")]
    ag, ch = {}, set()
    for n in names:
        b = n.split("/")[-1]
        c = re.search(r"Cycle(\d+)", b)
        w = re.search(r"Wb_(\d+)", b)
        if not c:
            continue
        cy, wb = int(c.group(1)), int(w.group(1)) if w else 1
        if "aging" in b.lower():
            ag.setdefault(cy, []).append((wb, n))
        else:
            ch.add(cy)
    A = sorted(ag)
    out = {}
    for c in sorted(ch):
        prev = [a for a in A if a < c]
        if prev:
            out[c] = sorted(ag[prev[-1]])[-1][1]
    return out


def soc_track(t, I, V):
    """만충전마다 SOC=1 로 재고정하며 적분.

    WHY THE ANCHOR CANNOT DEPEND ON THE TAPER CURRENT
        Test#2 는 도중에 CC-CV 종료 전류를 바꾼다 — 초반 파일은 0.15 A 에서
        끊고 후반 파일은 약 1 A 에서 끊는다. |I| 문턱으로 만충을 찾으면 후반
        파일에서 한 점도 못 찾고, 그러면 SOC 가 1 을 넘는 값으로 새어 나온다
        (측정된 실패: 사이클 1333 이후 SOC 1.00~1.62).

        그리고 후반 주행은 만충이 아니라 방전으로 끝난다. 따라서 '창의 마지막
        만충' 하나로는 부족하고, 만충 구간마다 다시 못박아야 한다. 그렇게 하면
        쿨롱 효율에 의한 표류도 매 사이클 지워진다.

        앵커는 V >= V_FULL 인 연속 구간마다 누적 전하가 최대인 점이다 - 전류
        문턱이 필요 없다.
    """
    ah = np.concatenate([[0.0], np.cumsum(I[1:] * np.diff(t) / 3600.0)])
    hi = V >= V_FULL
    idx = np.where(hi)[0]
    anchors = []
    if len(idx):
        brk = np.where(np.diff(idx) > 1)[0]
        for grp in np.split(idx, brk + 1):
            anchors.append(int(grp[np.argmax(ah[grp])]))
    if not anchors:
        anchors = [len(t) - 1]
    anchors = np.array(sorted(set(anchors)))
    # 각 샘플에 가장 가까운 '이전' 앵커(없으면 첫 앵커)를 붙인다
    k = np.searchsorted(anchors, np.arange(len(t)), side="right") - 1
    k = np.clip(k, 0, len(anchors) - 1)
    return 1.0 + (ah - ah[anchors[k]]) / Q_RATED, len(anchors)


def blocks_for(zf, path, cycle, tf, cell, window_h=WINDOW_H):
    import io as _io
    with zf.open(path) as f:
        head = _io.TextIOWrapper(f, encoding="utf-8-sig",
                                 errors="replace").readline().strip().split(",")
    tcols = cell_temp_col(head)
    with zf.open(path) as f:
        df = pd.read_csv(f, usecols=BASE_COLS + tcols, encoding="utf-8-sig",
                         low_memory=False)
    t = df["Test_Time(s)"].to_numpy(float)
    I = df["Current(A)"].to_numpy(float)
    V = df["Voltage(V)"].to_numpy(float)
    cand = {c: df[c].to_numpy(float) for c in tcols}
    tname = max(cand, key=lambda c: np.nanstd(np.diff(cand[c])))
    T = cand[tname]
    keep = np.isfinite(t) & np.isfinite(I) & np.isfinite(V) & np.isfinite(T)
    t, I, V, T = t[keep], I[keep], V[keep], T[keep]
    m = t >= t.max() - window_h * 3600.0
    t, I, V, T = t[m], I[m], V[m], T[m]

    soc, n_anchor = soc_track(t, I, V)
    soh = soh_of(cycle, cell)
    tf.reset()
    out, mark = [], t[0] + BLOCK_S
    for j in range(1, len(t)):
        if tf.update(float(t[j] - t[j - 1]), float(I[j]), float(V[j]),
                     float(T[j]), float(np.clip(soc[j], 0.02, 1.0)), soh) is None:
            continue
        if t[j] >= mark:
            x = tf.vector(float(np.clip(soc[j], 0.02, 1.0)), soh)
            out.append(dict(x=x, duty=float(x[5]), T=float(x[8]),
                            soc=float(soc[j]), t=float(t[j])))
            mark = t[j] + BLOCK_S
    drive = [b for b in out if b["duty"] >= DUTY_MIN]
    bad = float(np.mean((soc > 1.02) | (soc < 1.0 - soh - 0.05)))
    return out, drive[-N_BLOCKS:], soh, (n_anchor, bad, tname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=2, choices=[1, 2],
                    help="2 = US06 drive, 1 = constant-current aging")
    ap.add_argument("--duty-min", type=float, default=DUTY_MIN)
    ap.add_argument("--holdout", default="CC",
                    help="어느 pooled 표면을 쓸지. RPCWBY 는 여섯 폴드 모두에게 "
                         "외부이므로 어느 것을 써도 홀드아웃이다.")
    ap.add_argument("--window-h", type=float, default=WINDOW_H)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    sd, sc = surfaces(a.holdout)
    tf = TrimFeatures(sd, sc)
    cell = CELLS[a.test]
    zf = zipfile.ZipFile(ZIPS[a.test])
    globals()["DUTY_MIN"] = a.duty_min
    P = pairs(zf)
    print(f"  Test#{a.test} ({cell})  풀 홀드아웃 {a.holdout}   "
          f"특성화 {len(P)} 개  duty>={a.duty_min}")
    print(f"  {'Charac':>7}{'SOH':>7}{'전체블록':>9}{'주행블록':>9}"
          f"{'duty 중앙':>10}{'T 중앙':>8}{'SOC 범위':>16}{'앵커':>5}{'SOC이탈':>8}")
    keep = {}
    for cyc, path in P.items():
        allb, drive, soh, (n_anchor, bad, tname) = blocks_for(
            zf, path, cyc, tf, cell, a.window_h)
        if len(drive) < N_BLOCKS:
            print(f"  {cyc:>7}{soh:>7.3f}{len(allb):>9}{len(drive):>9}"
                  f"      블록 부족 — 제외")
            continue
        X = np.stack([b["x"] for b in drive])
        keep[cyc] = X
        socs = [b["soc"] for b in drive]
        print(f"  {cyc:>7}{soh:>7.3f}{len(allb):>9}{len(drive):>9}"
              f"{np.median([b['duty'] for b in drive]):>10.3f}"
              f"{np.median([b['T'] for b in drive]):>8.1f}"
              f"   {min(socs):>5.2f}~{max(socs):<5.2f}"
              f"{n_anchor:>5}{bad*100:>7.1f}%")
    np.savez(os.path.join(a.out, f"t{a.test}_feats_{a.holdout}.npz"),
             cycles=np.array(sorted(keep)),
             X=np.stack([keep[c] for c in sorted(keep)]))
    print(f"\n  저장 {len(keep)} 특성화 x {N_BLOCKS} 블록 x 12 특징")


if __name__ == "__main__":
    main()
