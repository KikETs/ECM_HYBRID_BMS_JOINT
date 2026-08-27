"""Test#8: 다른 것을 모두 고정하고 '측정 직전 펄스'만 바꾼 SOP.

WHY THIS IS THE TRIM'S OWN PREMISE ON TRIAL
    트림은 "최근 이력이 유효 저항을 바꾸고, 그것을 12 개 O(1) 통계로 읽을 수
    있다" 는 가정 위에 서 있다. UYPYDJ 에서는 그 가정을 시험할 수 없었다 — 이력이
    바뀌면 셀도 사이클도 함께 바뀌기 때문이다.

    Test#8 은 **같은 셀, 같은 0 C, 같은 SOC** 에서 측정 직전 펄스의 C-rate 만
    0 / C3 / 1C / 2C / 3C / 4C 로 바꾼다. 저자들이 붙인 그림 이름이
    fig_historyEffect.png 다.

시트 레이아웃
    r1  B "0°C"
    r2  B "0C" | C "C/3" | D "1C" | E "2C" | F "3C" | G "4C"
    r3~ A = SOC, B~G = SOP_disch

온도가 교란이지만 방향이 반대다
    저자 그림의 막대는 측정 직전 셀 온도이고, C-rate 가 오르면 자기발열로
    0.8 -> 5 C 로 오른다. 따뜻하면 저항이 낮아 SOP 는 **올라가야** 한다. 그런데
    측정된 SOP 는 20 % SOC 에서 75.5 -> 58.1 W 로 **떨어진다.** 이력 효과가 온도
    이득을 이긴다 - 온도를 보정하면 효과는 더 커진다.
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "..", "raw", "RPCWBY", "0_SOP_summary.xlsx")
OUT = os.path.join(HERE, "rpcwby_sop_test8.csv")

RATE = {"0C": 0.0, "C/3": 1 / 3, "1C": 1.0, "2C": 2.0, "3C": 3.0, "4C": 4.0}


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=XLSX)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    wb = openpyxl.load_workbook(a.xlsx, data_only=True, read_only=True)
    rows = [list(r) for r in wb["Test#8"].iter_rows(values_only=True)]

    hdr = None
    for j, r in enumerate(rows[:6]):
        cols = {c: v.strip() for c, v in enumerate(r)
                if isinstance(v, str) and v.strip() in RATE}
        if len(cols) >= 4:
            hdr = (j, cols)
            break
    if hdr is None:
        raise SystemExit("  C-rate 헤더를 찾지 못했다")
    j0, cols = hdr
    temp = 0.0
    for r in rows[:j0]:
        for v in r:
            if isinstance(v, str):
                m = re.match(r"\s*(-?\d+)\s*°?C", v)
                if m:
                    temp = float(m.group(1))
    out = []
    for j in range(j0 + 1, len(rows)):
        soc = _num(rows[j][0]) if rows[j] else None
        if soc is None or not (0.0 <= soc <= 1.0):
            continue
        for c, name in cols.items():
            v = _num(rows[j][c]) if c < len(rows[j]) else None
            if v is None:
                continue
            out.append({"sheet": "Test#8", "temp_C": temp, "rate_label": name,
                        "C_rate": RATE[name], "SOC": round(soc, 4),
                        "SOP_disch": round(v, 4)})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    import numpy as np
    socs = sorted({r["SOC"] for r in out}, reverse=True)
    print(f"  {a.out}  {len(out)}행   온도 {temp:.0f} C   "
          f"C-rate {sorted({r['C_rate'] for r in out})}")
    BOUND = 2.55 * 30.0
    print(f"\n  {'SOC':>6}" + "".join(f"{n:>9}" for n in cols.values())
          + f"{'진폭':>8}{'전압제한':>9}")
    for s0 in socs:
        g = {r["rate_label"]: r["SOP_disch"] for r in out if r["SOC"] == s0}
        vals = [g.get(n) for n in cols.values()]
        ok = [v for v in vals if v is not None]
        if len(ok) < 2:
            continue
        amp = (max(ok) - min(ok)) / abs(np.mean(ok)) * 100
        vl = sum(1 for v in ok if abs(v) <= BOUND)
        print(f"  {s0:>6.3f}" + "".join(f"{v:>9.1f}" if v is not None else f"{'-':>9}"
                                        for v in vals)
              + f"{amp:>7.1f}%{vl:>6}/{len(ok)}")


if __name__ == "__main__":
    main()
