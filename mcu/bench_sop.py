"""보드에서 하이브리드 SOP 의 연산 비용을 잰다.

단계를 따로 재는 이유
    26 절이 보였듯 지배하는 것은 트림 26 파라미터가 아니라 **표 조회 x 반전 반복**
    이다. 합계만 재면 그 사실이 숫자로 남지 않는다.

    REFF  표 조회 1 회
    TRIM  12->2 선형 1 회 (26 파라미터)
    SOLVE 고정점 반전 (표 조회 N 회)
    FULL  TRIM + SOLVE = SOP 1 회

DWT CYCCNT 를 인터럽트 끄고 재며, 측정 구간은 호출 자체만이다. 기존 CEMA 벤치와
같은 규약이라 숫자를 나란히 놓을 수 있다.
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time

import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = 0x43454D41
CMD = dict(QUERY=0x60, REFF=0x61, TRIM=0x62, SOLVE=0x63, FULL=0x64,
           FEAT=0x65, EKF=0x66, EKF_P=0x67,
           # 채택 구성(A8)의 특징 갱신 — dR_fast 하나, EW 상태 2 개.
           # FEAT(12 개) 와 나란히 재야 절감이 얼마인지 안다.
           FEAT_A8=0x6D)
NFEAT = 12
REQ = "<B6f" + f"{NFEAT}f"   # dir + soc,soh,v_pre,v_limit,tau,current
RES = "<IIII4f"


def query(p):
    p.write(bytes([CMD["QUERY"]])); p.flush()
    d = p.read(32)
    if len(d) != 32:
        raise SystemExit(f"  QUERY 응답 {len(d)}바이트 — 펌웨어 확인")
    v = struct.unpack("<8I", d)
    if v[0] != MAGIC:
        raise SystemExit(f"  magic 불일치 {v[0]:#x}")
    return dict(version=v[1], clock_hz=v[2], req_bytes=v[3], res_bytes=v[4],
                ecm_bytes=v[5], ocv_bytes=v[6], nfeat=v[7])


def call(p, cmd, dirn, soc, soh, v_pre, v_lim, tau, cur, x12):
    # 명령 바이트를 먼저 보내고 잠깐 쉰다.  펌웨어는 명령 1 바이트를 받은 뒤에야
    # 본문 수신에 들어가므로, 921600 baud 로 붙여 보내면 앞부분을 놓친다.
    p.write(bytes([CMD[cmd]])); p.flush()
    time.sleep(0.002)
    p.write(struct.pack(REQ, dirn, soc, soh, v_pre, v_lim, tau, cur, *x12))
    p.flush()
    d = p.read(struct.calcsize(RES))
    if len(d) != struct.calcsize(RES):
        raise SystemExit(f"  {cmd} 응답 {len(d)}바이트")
    m, cyc, it, hw, kf, ks, reff, ist = struct.unpack(RES, d)
    if m != MAGIC:
        raise SystemExit(f"  {cmd} magic 불일치")
    return dict(cycles=cyc, iters=it, stack=hw, kf=kf, ks=ks,
                r_eff=reff, i_star=ist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--feats", default=os.path.join(HERE, "..", "analysis",
                                                    "cache", "trim", "trim_CC.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "sop_mcu_bench.csv"))
    a = ap.parse_args()

    X = np.load(a.feats, allow_pickle=True)["X"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), a.n, replace=False)

    p = serial.Serial(a.port, a.baud, timeout=3)
    time.sleep(0.3); p.reset_input_buffer()
    q = query(p)
    print(f"  보드 {q['clock_hz']/1e6:.0f} MHz   요청 {q['req_bytes']}B / 응답 "
          f"{q['res_bytes']}B   특징 {q['nfeat']}")
    print(f"  ECM 격자 {q['ecm_bytes']/1024:.1f} KB + OCV {q['ocv_bytes']/1024:.1f} KB")
    if q["nfeat"] != NFEAT:
        raise SystemExit("  특징 수 불일치")

    rows = []
    for t in idx:
        soc = float(rng.uniform(0.2, 0.9)); soh = float(rng.uniform(0.75, 0.98))
        vpre = float(rng.uniform(3.3, 3.9)); tau = float(rng.choice([2.0, 10.0]))
        x = [float(v) for v in X[t]]
        for cmd in ("REFF", "TRIM", "SOLVE", "FULL", "FEAT", "FEAT_A8", "EKF", "EKF_P"):
            r = call(p, cmd, 0, soc, soh, vpre, 2.5, tau, 10.0, x)
            rows.append(dict(cmd=cmd, soc=round(soc, 4), soh=round(soh, 4),
                             tau_s=tau, **r,
                             us=round(r["cycles"] / q["clock_hz"] * 1e6, 3)))
    p.close()

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n  {'단계':<8}{'n':>5}{'중앙 us':>10}{'p95 us':>10}{'최대 us':>10}"
          f"{'중앙 cyc':>10}{'반복':>7}")
    for cmd in ("FEAT", "FEAT_A8", "EKF_P", "EKF", "REFF", "TRIM", "SOLVE", "FULL"):
        s = [r for r in rows if r["cmd"] == cmd]
        us = np.array([r["us"] for r in s]); cy = np.array([r["cycles"] for r in s])
        it = np.array([r["iters"] for r in s])
        print(f"  {cmd:<8}{len(s):>5}{np.median(us):>10.2f}{np.percentile(us,95):>10.2f}"
              f"{us.max():>10.2f}{np.median(cy):>10.0f}"
              f"{(f'{np.median(it):.0f}' if it.max() else '-'):>7}")
    hw = max(r["stack"] for r in rows)
    print(f"\n  스택 최고수위 {hw} B")
    print(f"  {a.out}")


if __name__ == "__main__":
    main()
