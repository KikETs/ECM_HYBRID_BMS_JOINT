"""UYPYDJ 원자료 전체의 온도 채널을 전수 조사한다 — 셀 x 파일종류 x 사이클.

WHY
    findings.md 7.1 은 온도 결함을 찾아 마스크를 만들었지만 **주행 캐시에만**
    적용됐다. HPPC 저항 표에는 온도 컬럼이 없어 같은 필터를 걸 수 없었고, 그래서
    BOOST_NEGPULSE 사이클 487 이 저 SOC 구간을 3.8 C 에서 측정한 채로 라벨이 되어
    방전 SOP 안전계수 12 조합 중 6 개를 혼자 정하고 있었다(26 절).

    저항 표도 SOP 반전도 25 C 를 전제한다. 다른 온도에서 나온 라벨은 비교 대상이
    아니다. 그러므로 어느 파일에 그런 구간이 있는지 전부 알아야 한다.

무엇을 이상으로 보는가
    T_LO..T_HI 밖이거나 유한하지 않은 샘플. 하한 15 C 는 findings 7.1 이 여섯 셀의
    진짜 최저를 16.4 C 로 확인한 데서 온다. 죽은 열전대는 0 을 읽지 않고 배회하므로
    (7.1) 비율과 함께 최소·중앙·저전압구간 중앙을 같이 적는다.
"""
from __future__ import annotations
import argparse, csv, io, os, re, sys, zipfile
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw", "UYPYDJ")
PROTOCOLS = {
 "CC": ["03-CONSTANT CURRENT protocol_Cycles 0 to 1000.zip",
        "04-CONSTANT CURRENT protocol_Cycles 1000 to 1908.zip"],
 "BOOST": ["05-BOOST CHARGING protocol_Cycles 0 to 1000.zip",
           "06-BOOST CHARGING protocol_Cycles 1000 to 1908.zip"],
 "BOOST_REST": ["07-BOOST CHARGING WITH REST protocol.zip"],
 "BOOST_NEGPULSE": ["08-BOOST CHARGING WITH NEGATIVE PULSES protocol_Cycles 1 to 1000.zip",
                    "09-BOOST CHARGING WITH NEGATIVE PULSES protocol_Cycles 1000 to 1730.zip"],
 "BOOST_NEGPULSE_1S": ["10-BOOST CHARGING WITH NEGATIVE PULSES_1s_PERIOD_protocol.zip"],
 "CC_CELL2": ["11-CONSTANT CURRENT SECOND CELL protocol.zip"]}
T_LO, T_HI = 15.0, 45.0


def kind_of(base):
    b = base.lower()
    for k in ("hppc", "drive", "ocv", "cap", "halfc", "schedule"):
        if k in b:
            return {"halfc": "halfC", "cap": "CAP"}.get(k, k.upper() if k in ("hppc", "ocv") else k)
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "temp_audit_all.csv"))
    a = ap.parse_args()
    rows = []
    for cell, zs in PROTOCOLS.items():
        for zn in zs:
            zp = os.path.join(RAW, zn)
            if not os.path.exists(zp):
                print(f"  없음: {zn}", flush=True); continue
            z = zipfile.ZipFile(zp)
            mats = [n for n in z.namelist() if n.endswith(".mat")]
            for j, n in enumerate(mats):
                base = n.split("/")[-1]
                mc = re.search(r"cycle_#(\d+)", base)
                rec = dict(cell=cell, zipname=zn, file=base, kind=kind_of(base),
                           cycle=int(mc.group(1)) if mc else -1)
                try:
                    m = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True,
                                    struct_as_record=False)["meas"]
                    T = np.asarray(m.Battery_temp_DegC, float).ravel()
                    V = np.asarray(m.Voltage, float).ravel()
                except Exception as e:
                    rec.update(n=0, bad_frac=1.0, T_min="", T_med="", T_lowV_med="",
                               note=f"read:{type(e).__name__}")
                    rows.append(rec); continue
                if T.size == 0:
                    rec.update(n=0, bad_frac=1.0, T_min="", T_med="", T_lowV_med="",
                               note="빈 온도")
                    rows.append(rec); continue
                bad = ~np.isfinite(T) | (T < T_LO) | (T > T_HI)
                lo = np.isfinite(V) & (V < 3.55) if V.size == T.size else np.zeros(T.size, bool)
                fin = np.isfinite(T)
                rec.update(n=int(T.size), bad_frac=round(float(np.mean(bad)), 4),
                           T_min=round(float(T[fin].min()), 2) if fin.any() else "",
                           T_med=round(float(np.median(T[fin])), 2) if fin.any() else "",
                           T_lowV_med=(round(float(np.median(T[lo & fin])), 2)
                                       if (lo & fin).any() else ""),
                           note="")
                rows.append(rec)
            print(f"  {cell:<20} {zn[:38]:<40} {len(mats):>5} 파일 완료", flush=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  {a.out}  {len(rows)}행")


if __name__ == "__main__":
    main()
