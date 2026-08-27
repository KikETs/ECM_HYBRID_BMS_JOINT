"""Test#3 의 측정 SOP 로 pooled ECM 을 채점한다 — 처음으로 온도 축 위에서.

무엇이 새로 열리는가
    -20 과 -10 C 에서는 14 행이 **전부** 전압 제한이고 SOC 0.02~1.00 전 구간을
    덮는다. 전압 제한 SOP 가 SOC 축 전체를 덮은 것은 이 프로젝트에서 처음이다.
    따뜻한 쪽은 30 A 상한이 2.55 V 에 못 닿아 저 SOC 몇 행만 남는다 — 18.3 절의
    물리가 여기서도 그대로다.

SOH 는 블록별 용량이 아니라 25 C 용량이다
    시트는 온도 블록마다 용량을 적어 두었다(2.6333 / 2.6 / 2.4193 / 2.1352 Ah).
    그것을 SOH 로 쓰면 -20 C 에서 SOH 0.712 가 되는데, 이는 노화가 아니라 저온
    용량 감소다. 파일 날짜가 증거다 — 시험 순서는 0, 10, 25, 40, -20, -10, ...
    으로 **온도 순이 아니다**. 온도에 따라 단조인 양이 시간에 따라 단조가 아닌
    순서로 측정됐다면 그것은 노화일 수 없다. 따라서 SOH = 2.6 / 3.0 = 0.867 을
    전 블록에 쓴다.

온도는 실측 셀 온도다
    18.1 절에서 실측이 설정보다 일관되게 높다(-20 설정 -> -17.9 실측). 온도 인자가
    T_cell 을 받으므로 rpcwby_temp_pulses.csv 의 설정->실측 중앙값으로 옮긴다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_pool import surfaces  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "rpcwby_sop_test3.csv")
TPULSE = os.path.join(HERE, "rpcwby_temp_pulses.csv")
OUT = os.path.join(HERE, "sop_test3_eval.csv")

V_FLOOR, I_CLAMP = 2.55, 30.0
BOUND = V_FLOOR * I_CLAMP
SOH_25C = 2.6 / 3.0


def tcell_map():
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(TPULSE, encoding="utf-8")):
        d[int(float(r["temp_set_C"]))].append(float(r["T_cell_C"]))
    return {k: float(np.median(v)) for k, v in d.items()}


def r_eff(surf, soc, soh, I, T, tau, kf=1.0, ks=1.0):
    th = surf.theta(soc, soh, I, T)
    if not bool(np.atleast_1d(th["in_hull"])[0]):
        return np.nan, np.nan
    R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
    t1 = float(th["tau1"][0]); t2 = float(th["tau2"][0])
    return (kf * (R0 + R1 * (1 - np.exp(-tau / t1)))
            + ks * R2 * (1 - np.exp(-tau / t2))), float(th["g_temp"])


def solve_I(surf, soc, soh, v_pre, T, tau, iters=30):
    I, g = -12.0, np.nan
    for _ in range(iters):
        R, g = r_eff(surf, soc, soh, I, T, tau)
        if not np.isfinite(R) or R <= 0:
            return np.nan, np.nan
        nxt = float(np.clip((V_FLOOR - v_pre) / R, -400.0, -0.1))
        if abs(nxt - I) < 1e-3:
            return nxt, g
        I = 0.5 * I + 0.5 * nxt
    return I, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts",
                    default="CC,CC_CELL2,BOOST,BOOST_REST,BOOST_NEGPULSE,"
                            "BOOST_NEGPULSE_1S")
    ap.add_argument("--soh", type=float, default=SOH_25C)
    ap.add_argument("--t-cell", choices=["measured", "setpoint"],
                    default="measured")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    TC = tcell_map()
    rows = [r for r in csv.DictReader(open(LAB, encoding="utf-8"))
            if r["SOP_disch"] not in ("", "nan")
            and abs(float(r["SOP_disch"])) <= BOUND]
    print(f"  전압 제한 {len(rows)}행   SOH {a.soh:.3f}   온도 {a.t_cell}")

    out, drop = [], collections.Counter()
    for h in a.holdouts.split(","):
        sd = surfaces(h)[0]
        for r in rows:
            soc = float(r["SOC"]); tau = float(r["tau_s"])
            Tset = int(float(r["temp_C"]))
            T = TC.get(Tset, float(Tset)) if a.t_cell == "measured" else float(Tset)
            v, _ = sd.ocv(soc, a.soh)
            M, _ = sd.hyst_M(soc, a.soh)
            v_pre = float(np.atleast_1d(v)[0]) - M
            ip, g = solve_I(sd, soc, a.soh, v_pre, T, tau)
            if not np.isfinite(ip):
                drop["hull 밖"] += 1
                continue
            out.append(dict(holdout=h, tau_s=tau, T_set=Tset, T_cell=round(T, 2),
                            SOC=soc, SOH=round(a.soh, 4),
                            I_meas_A=round(-abs(float(r["SOP_disch"])) / V_FLOOR, 3),
                            I_pred_A=round(ip, 3), g_temp=round(g, 4),
                            v_pre=round(v_pre, 4)))
    if not out:
        sys.exit("  채점 가능한 행이 없다")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    im = np.abs([r["I_meas_A"] for r in out]); ip = np.abs([r["I_pred_A"] for r in out])
    tau = np.array([r["tau_s"] for r in out]); Ts = np.array([r["T_set"] for r in out])
    soc = np.array([r["SOC"] for r in out])
    print(f"  {a.out}  {len(out)}행  제외 {dict(drop)}\n")
    print(f"  {'tau':>5}{'T설정':>7}{'n':>5}{'초과율':>8}{'필요 lambda':>12}"
          f"{'배율 중앙':>10}{'최악':>8}{'g_temp':>8}")

    def line(t0, T0, m):
        if m.sum() < 2:
            return
        rat = ip[m] / im[m]
        lam = float(1.0 / np.max(rat))          # 이 칸에서 초과를 없애는 lambda
        gt = np.median([r["g_temp"] for r, q in zip(out, m) if q])
        print(f"  {t0:>4.0f}s{T0:>7}{int(m.sum()):>5}"
              f"{np.mean(ip[m] > im[m])*100:>7.1f}%{lam:>12.3f}"
              f"{np.median(rat):>10.3f}{np.max(ip[m]-im[m]):>7.2f}A{gt:>8.3f}")

    for t0 in (2.0, 10.0, 30.0):
        for T0 in sorted(set(Ts[tau == t0])):
            line(t0, str(T0), (tau == t0) & (Ts == T0))
    print()
    for t0 in (2.0, 10.0, 30.0):
        line(t0, "전체", tau == t0)


if __name__ == "__main__":
    main()
