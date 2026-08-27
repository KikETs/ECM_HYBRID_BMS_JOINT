"""전류 옵셋 상태 2차 진단 — 면 선택 불연속이 원인인지.

_theta 는 s = sc if I > 0 else sd 로 충전면/방전면을 고른다.  전류 옵셋
상태를 넣으면서 보정 전류 Ic = I - b 를 넘겼는데, 휴지 구간에서는 참
전류가 0 이므로 Ic 의 부호 = -b 의 부호다.  즉 옵셋 추정치가 0 을
지날 때마다 OCV 표가 통째로 바뀐다.  게이트는 갱신을 휴지 구간에만
허용하므로 그 불연속 위에서만 갱신이 일어난다.

(1) 두 면의 OCV 차이가 실제로 얼마인지 잰다
(2) 면은 측정 전류 I 로 고르고 크기만 Ic 로 쓰는 판을 만들어 비교한다
"""
import numpy as np
import pickle

RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))


def gap():
    print("  (1) 충전면 - 방전면 OCV 차이 (mV)\n", flush=True)
    print(f"  {'셀':<20}{'SOH':>6}" + ''.join(f"{f'SOC {x:.2f}':>10}"
                                             for x in (0.2, 0.4, 0.6, 0.8)),
          flush=True)
    seen = set()
    for r in RUNS:
        key = (r['cell'], round(r['soh'], 2))
        if key in seen:
            continue
        seen.add(key)
        row = ''
        for x in (0.2, 0.4, 0.6, 0.8):
            oc, _ = r['sc'].ocv(x, r['soh'])
            od, _ = r['sd'].ocv(x, r['soh'])
            row += f"{(float(np.atleast_1d(oc)[0]) - float(np.atleast_1d(od)[0]))*1000:>10.1f}"
        print(f"  {r['cell']:<20}{r['soh']:>6.2f}{row}", flush=True)


if __name__ == '__main__':
    gap()
