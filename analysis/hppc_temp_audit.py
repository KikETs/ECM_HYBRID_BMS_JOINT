"""HPPC 특성화의 온도를 전수 조사한다.

WHY THIS WAS MISSING
    findings.md 7.1 이 온도 채널 결함을 찾아 마스크를 만들었지만, 그것은 **주행
    캐시**(cache_t)에만 적용됐다. HPPC 저항 표(uypydj_hppc_resistance.csv)에는
    온도 컬럼이 아예 없어서 같은 필터를 걸 수 없었다.

    결과: BOOST_NEGPULSE 사이클 487 의 저 SOC 구간이 셀 온도 3.8 C 에서 측정됐고
    (V<3.55 구간 중앙 3.8 C, 0.4~27.6), 저항이 25 C 대비 2~3 배로 나왔다. 그 라벨
    하나가 충전 SOP 안전계수 전체를 정하고 있었다(26.5).

    저항 표도 SOP 반전도 25 C 를 전제한다. 4 C 셀에서 나온 라벨은 비교 대상이
    아니다.
"""
from __future__ import annotations
import argparse, csv, io, os, re, zipfile
import numpy as np
import scipy.io as sio

HERE=os.path.dirname(os.path.abspath(__file__))
RAW=os.path.join(HERE,"..","raw","UYPYDJ")
PROTOCOLS={
 "CC":["03-CONSTANT CURRENT protocol_Cycles 0 to 1000.zip",
       "04-CONSTANT CURRENT protocol_Cycles 1000 to 1908.zip"],
 "BOOST":["05-BOOST CHARGING protocol_Cycles 0 to 1000.zip",
          "06-BOOST CHARGING protocol_Cycles 1000 to 1908.zip"],
 "BOOST_REST":["07-BOOST CHARGING WITH REST protocol.zip"],
 "BOOST_NEGPULSE":["08-BOOST CHARGING WITH NEGATIVE PULSES protocol_Cycles 1 to 1000.zip",
                   "09-BOOST CHARGING WITH NEGATIVE PULSES protocol_Cycles 1000 to 1730.zip"],
 "BOOST_NEGPULSE_1S":["10-BOOST CHARGING WITH NEGATIVE PULSES_1s_PERIOD_protocol.zip"],
 "CC_CELL2":["11-CONSTANT CURRENT SECOND CELL protocol.zip"]}
T_LO, T_HI = 15.0, 45.0        # 여섯 셀 진짜 최저가 16.4 C (findings 7.1)
FRAC = 0.02                    # 이 비율 넘게 벗어나면 그 특성화를 표시

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=os.path.join(HERE,"hppc_temp_audit.csv"))
    a=ap.parse_args(); rows=[]
    for cell,zs in PROTOCOLS.items():
        for zn in zs:
            zp=os.path.join(RAW,zn)
            if not os.path.exists(zp): continue
            z=zipfile.ZipFile(zp)
            for n in z.namelist():
                if 'HPPC' not in n or not n.endswith('.mat'): continue
                mc=re.search(r'cycle_#(\d+)',n)
                if not mc: continue
                try:
                    m=sio.loadmat(io.BytesIO(z.read(n)),squeeze_me=True,struct_as_record=False)['meas']
                    T=np.asarray(m.Battery_temp_DegC,float); V=np.asarray(m.Voltage,float)
                except Exception as e:
                    rows.append(dict(cell=cell,cycle=int(mc.group(1)),n=0,bad_frac=1.0,
                                     T_min="",T_med="",T_lowV_med="",note=f"읽기 실패 {e}")); continue
                bad=~np.isfinite(T)|(T<T_LO)|(T>T_HI)
                lo=np.isfinite(V)&(V<3.55)
                rows.append(dict(cell=cell,cycle=int(mc.group(1)),n=len(T),
                                 bad_frac=round(float(np.mean(bad)),4),
                                 T_min=round(float(np.nanmin(T)),2),
                                 T_med=round(float(np.nanmedian(T)),2),
                                 T_lowV_med=round(float(np.nanmedian(T[lo])),2) if lo.sum() else "",
                                 note=""))
    rows.sort(key=lambda r:(r["cell"],r["cycle"]))
    with open(a.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    bad=[r for r in rows if r["bad_frac"]>FRAC]
    print(f"  특성화 {len(rows)}개 조사, 이상 {len(bad)}개 (무효 비율 > {FRAC*100:.0f}%)\n")
    print(f"  {'셀':<20}{'사이클':>7}{'무효비율':>9}{'T 최소':>8}{'T 중앙':>8}{'저전압구간 T':>13}  비고")
    for r in bad:
        print(f"  {r['cell']:<20}{r['cycle']:>7}{r['bad_frac']*100:>8.1f}%{str(r['T_min']):>8}"
              f"{str(r['T_med']):>8}{str(r['T_lowV_med']):>13}  {r['note']}")
    print(f"\n  {a.out}")

if __name__=="__main__": main()
