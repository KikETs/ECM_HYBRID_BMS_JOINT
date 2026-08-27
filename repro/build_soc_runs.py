"""SOC 벤치가 쓰는 주행 런 묶음을 만든다.

이 단계가 따로 있는 이유: 30 절의 SOC 실험들이 원래 /tmp/soc_runs.pkl 을
읽었다.  탐색 중에는 편했지만 재현 패키지가 /tmp 에 기대면 안 된다 —
재부팅 한 번에 사라지고, 무엇으로 만들어졌는지 기록이 없다.

여섯 셀에서 드라이브 사이클 파일을 균등 간격으로 6 개씩 골라 36 런을 만든다.
고르는 규칙은 np.linspace 라 결정적이다.  각 런은 앞 20,000 표본만 쓴다
(1 Hz 이므로 약 5.5 시간).

주의: BOOST_NEGPULSE_1S 는 파일 수가 적어 linspace 가 같은 파일을 두 번
집는다 (30.11 에 기록).  그 셀의 유효 런은 5 개다.  고치지 않고 남긴 것은
지금까지의 모든 수치가 이 구성 위에서 나왔기 때문이다.
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(os.path.dirname(HERE), 'analysis')
sys.path.insert(0, ANALYSIS)

CELLS = ['CC', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_REST', 'CC_CELL2',
         'BOOST_NEGPULSE_1S']
NRUN_PER_CELL = 6
NMAX = 20000
MIN_VALID = 2000


def build(cache_dir):
    from ecm_surface import ECMSurface
    runs = []
    for cell in CELLS:
        sd = ECMSurface(cell, 'discharge')
        sc = ECMSurface(cell, 'charge')
        z = np.load(os.path.join(cache_dir,
                                 f'uypydj_{cell}_Fifteen_Drive_Cycles.npz'))
        lens = z['lens']
        off = np.concatenate([[0], np.cumsum(lens)])
        for k in np.linspace(0, len(lens) - 1, NRUN_PER_CELL).astype(int):
            sl = slice(off[k], off[k] + lens[k])
            soc, V, I, SOH, T = (z[x][sl] for x in
                                 ('SOC', 'V', 'I', 'SOH', 'T'))
            ok = (np.isfinite(soc) & np.isfinite(V) & np.isfinite(I)
                  & np.isfinite(T))
            if ok.sum() < MIN_VALID:
                continue
            runs.append(dict(cell=cell, cyc=int(k), sd=sd, sc=sc,
                             soc=soc[ok][:NMAX], V=V[ok][:NMAX],
                             I=I[ok][:NMAX], T=T[ok][:NMAX],
                             soh=float(np.nanmedian(SOH))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(ANALYSIS, 'cache_t'))
    ap.add_argument('--out',
                    default=os.path.join(ANALYSIS, 'results', 'soc_runs.pkl'))
    ap.add_argument('--check-against', default=None,
                    help='기존 pkl 과 같은지 확인만 한다')
    a = ap.parse_args()

    runs = build(a.cache)
    print(f'  {len(runs)} 런  ({len(set(r["cell"] for r in runs))} 셀)',
          flush=True)
    for c in CELLS:
        rs = [r for r in runs if r['cell'] == c]
        sohs = sorted({round(r['soh'], 4) for r in rs})
        note = '  <- 중복' if len(sohs) < len(rs) else ''
        print(f'    {c:<20} {len(rs)} 런, 서로 다른 SOH {len(sohs)} 개{note}',
              flush=True)

    if a.check_against:
        old = pickle.load(open(a.check_against, 'rb'))
        same = len(old) == len(runs) and all(
            o['cell'] == n['cell'] and abs(o['soh'] - n['soh']) < 1e-12
            and len(o['soc']) == len(n['soc'])
            and np.array_equal(o['soc'], n['soc'])
            and np.array_equal(o['I'], n['I'])
            for o, n in zip(old, runs))
        print(f"\n  {a.check_against} 와 {'동일' if same else '다름'}", flush=True)
        return 0 if same else 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(runs, f)
    print(f'\n  -> {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
