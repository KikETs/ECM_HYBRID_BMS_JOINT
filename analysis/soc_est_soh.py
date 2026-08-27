"""29.4 재측정 — 추정 SOH 의 대가를 섭동 벤치에서 다시 잰다.

29.4 는 순환 벤치에서 3.11 -> 3.35 %p (+0.24) 를 냈다.  그 벤치는 전압을
더 쓰는 모든 변화를 벌하므로, SOH 를 높게 보아 R_volt 가 작아진 것과
SOH 자체가 틀린 것을 가를 수 없었다.

SOH 는 ekf_soc 에 두 경로로 들어간다:
  (a) 저항·OCV 표의 축      run(..., soh=...)
  (b) 측정 잡음 R_volt 스케줄  0.70 -> 110 mV, 1.00 -> 15 mV
네 판을 돌려 두 경로를 분리한다.
"""
import multiprocessing as mp
import pickle
import re

import os
import numpy as np

RUNS = None
SOHEST = None

PERTURB = [('없음', {}),
           ('초기SOC+10', dict(dsoc=+0.10)),
           ('초기SOC-10', dict(dsoc=-0.10)),
           ('옵셋+0.1A', dict(ibias=+0.10)),
           ('옵셋-0.1A', dict(ibias=-0.10)),
           ('이득+1%', dict(igain=+0.01)),
           ('이득-1%', dict(igain=-0.01))]

ARMS = [('둘 다 정답', False, False),
        ('표 축만 추정', True, False),
        ('R_volt 만 추정', False, True),
        ('둘 다 추정', True, True)]


def rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def build_map():
    """드라이브 런 -> 직전 충전에서 추정한 SOH."""
    z = np.load(os.environ.get('SOH_PRED', 'results/soh_pred.npz'))
    out = []
    for r in RUNS:
        cache = np.load(f"cache_t/uypydj_{r['cell']}_Fifteen_Drive_Cycles.npz")
        fn = str(cache['files'][r['cyc']])
        cyc = int(re.match(r'(\d+)\|', fn).group(1))
        cc, pp = z[f"{r['cell']}_cycle"], z[f"{r['cell']}_pred"]
        prior = cc <= cyc
        if prior.any():                       # 직전 충전
            est = float(pp[prior][np.argmax(cc[prior])])
        else:                                 # 그 앞이 없으면 가장 이른 것
            est = float(pp[np.argmin(cc)])
        out.append((cyc, est))
    return out


def _one(job):
    from ekf_soc import run as ekf_run
    ai, pi = job
    _, use_axis, use_rv = ARMS[ai]
    pkw = PERTURB[pi][1]
    err = []
    for r, (_, est) in zip(RUNS, SOHEST):
        soh_axis = est if use_axis else r['soh']
        rv = rvolt(est if use_rv else r['soh'])
        If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
        s0 = min(max(float(r['soc'][0]) + pkw.get('dsoc', 0.0), 0.02), 0.98)
        e, _ = ekf_run(r['sd'], r['sc'], soh_axis, If, r['V'], r['T'],
                       s0, rv, gamma=0.0, i_gate=1.0, rest_hold_s=30.0)
        err.append(float(np.sqrt(np.mean((e - r['soc']) ** 2))))
    return job, np.array(err)


def main():
    global RUNS, SOHEST
    RUNS = pickle.load(open(os.environ.get('SOC_RUNS',
                            'results/soc_runs.pkl'), 'rb'))
    SOHEST = build_map()

    print(f"  {'셀':<20}{'사이클':>7}{'참 SOH':>9}{'추정':>9}{'오차':>9}",
          flush=True)
    for r, (cyc, est) in zip(RUNS, SOHEST):
        print(f"  {r['cell']:<20}{cyc:>7}{r['soh']:>9.4f}{est:>9.4f}"
              f"{est - r['soh']:>+9.4f}", flush=True)
    d = np.array([e - r['soh'] for r, (_, e) in zip(RUNS, SOHEST)])
    print(f"  {'전체':<20}{'':>7}{'':>9}{'':>9}"
          f"  RMSE {np.sqrt(np.mean(d**2)):.4f}  편향 {d.mean():+.4f}\n",
          flush=True)

    jobs = [(a, p) for a in range(len(ARMS)) for p in range(len(PERTURB))]
    with mp.Pool(14) as pool:
        res = dict(pool.map(_one, jobs))
    np.savez('results/soc_est_soh.npz',
             soh_true=np.array([r['soh'] for r in RUNS]),
             soh_est=np.array([e for _, e in SOHEST]),
             cell=np.array([r['cell'] for r in RUNS]),
             vals=np.array([[res[(a, p)] for p in range(len(PERTURB))]
                            for a in range(len(ARMS))]))

    print(f"  {'SOH 입력':<16}{'평균':>8}"
          + ''.join(f"{n:>12}" for n, _ in PERTURB), flush=True)
    print('  ' + '-' * (24 + 12 * len(PERTURB)), flush=True)
    base = None
    for a, (nm, _, _) in enumerate(ARMS):
        m = np.mean([res[(a, p)].mean() for p in range(len(PERTURB))])
        if base is None:
            base = m
        print(f"  {nm:<16}{m*100:>8.2f}"
              + ''.join(f"{res[(a, p)].mean()*100:>12.2f}"
                        for p in range(len(PERTURB)))
              + (f"   {(m-base)*100:+.2f}" if a else ''), flush=True)

    print("\n  == 셀별 (둘 다 정답 -> 둘 다 추정, 섭동 7 개 평균 %p)", flush=True)
    cell = np.array([r['cell'] for r in RUNS])
    for c in sorted(set(cell)):
        m = cell == c
        b = np.mean([res[(0, p)][m].mean() for p in range(len(PERTURB))]) * 100
        v = np.mean([res[(3, p)][m].mean() for p in range(len(PERTURB))]) * 100
        de = np.mean([e - r['soh'] for r, (_, e) in zip(RUNS, SOHEST)
                      if r['cell'] == c])
        print(f"    {c:<20} SOH 오차 {de:+.4f}   {b:>6.2f} -> {v:>6.2f}"
              f"  ({v-b:+.2f})", flush=True)


if __name__ == '__main__':
    main()
