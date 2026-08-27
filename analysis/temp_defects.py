"""온도 채널이 결함인 HPPC 특성화 목록 — 전수조사(temp_audit_all.csv)에서.

WHY A SHARED MODULE
    findings.md 7.1 의 마스크는 주행 캐시에만 걸렸다. HPPC 저항 표에는 온도
    컬럼이 없어 같은 필터를 걸 수 없었고, 그래서 셀 온도 3.8 C 에서 측정된
    BOOST_NEGPULSE#487 이 방전 SOP 안전계수를 정하고 있었다
    (sop_hybrid_spec.md 26.5, findings.md 7.2).

    저항 표도 SOP 반전도 25 C 를 전제한다. 다른 온도에서 나온 라벨은 비교 대상이
    아니다. 추출 단계에서 한 번 걸러야 하류 전체가 깨끗해진다.

왜 사이클 단위로 통째로 버리는가
    부분 결함(BOOST_NEGPULSE#487, 무효 19.6 %)은 저 SOC 구간에만 있고 고 SOC 는
    멀쩡하다. 그 절반만 살릴 수도 있지만 그러지 않는다 — 7.1 이 "죽은 열전대는
    0 을 읽지 않고 배회한다" 고 확인했으므로 결함의 경계를 신뢰할 수 없다.
    한 특성화의 2.1 % 를 버리는 비용이 경계를 추측하는 위험보다 싸다.
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "temp_audit_all.csv")
BAD_FRAC = 0.02


def defective_hppc(audit=AUDIT, bad_frac=BAD_FRAC):
    """{(protocol, cycle)} — 온도 채널이 결함인 HPPC 특성화."""
    if not os.path.exists(audit):
        raise FileNotFoundError(
            f"{audit} 없음 — analysis/temp_audit_all.py 를 먼저 돌린다")
    return defective("HPPC", audit, bad_frac)


def defective(kind, audit=AUDIT, bad_frac=BAD_FRAC):
    """{(protocol, cycle)} — 주어진 파일 종류에서 온도 채널이 결함인 것."""
    out = set()
    for r in csv.DictReader(open(audit, encoding="utf-8")):
        if r["kind"] != kind or r["note"] or int(r["n"]) == 0:
            continue
        if float(r["bad_frac"]) > bad_frac:
            out.add((r["cell"], int(r["cycle"])))
    return out


if __name__ == "__main__":
    d = sorted(defective_hppc())
    print(f"  온도 결함 HPPC 특성화 {len(d)}개")
    for c, y in d:
        print(f"   {c:<20}{y:>7}")
