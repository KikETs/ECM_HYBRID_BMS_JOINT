"""최종 구성을 셀 하나씩 빼고 확정한다.

고를 것: R_volt 배율, 게이트, 전류 옵셋 상태의 p0_ib (끔 포함).
고르는 것도 홀드아웃 셀을 빼고 하고, 성능은 고를 때 안 본 셀에서 잰다.

인자로 재조정에서 살아남은 후보를 좁혀 넘긴다.
"""
import argparse
import itertools
import multiprocessing as mp
import pickle

import numpy as np

RUNS = None

PERTURB = [('없음', {}),
           ('초기SOC+10', dict(dsoc=+0.10)),
           ('초기SOC-10', dict(dsoc=-0.10)),
           ('옵셋+0.1A', dict(ibias=+0.10)),
           ('옵셋-0.1A', dict(ibias=-0.10)),
           ('이득+1%', dict(igain=+0.01)),
           ('이득-1%', dict(igain=-0.01))]

SCALES = None
GATES = None
P0S = None


def _one(job):
    from ekf_soc import run as ekf_run
    si, gi, pj, pi = job
    kw = dict(GATES[gi][1])
    if P0S[pj] is not None:
        kw.update(q_ib=1e-10, p0_ib=P0S[pj], ib_clip=0.3)
    pkw = PERTURB[pi][1]
    err = []
    for r in RUNS:
        rv = float(np.interp(r['soh'], [0.70, 0.90, 1.00],
                             [0.110, 0.035, 0.015])) * SCALES[si]
        If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
        s0 = min(max(float(r['soc'][0]) + pkw.get('dsoc', 0.0), 0.02), 0.98)
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, **kw)
        err.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
    return job, np.array(err)


def main():
    global RUNS, SCALES, GATES, P0S
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='1.0')
    ap.add_argument('--gates', default='1A 30s')
    ap.add_argument('--p0', default='off,1e-6,1e-5,1e-4,1e-3')
    a = ap.parse_args()

    SCALES = [float(x) for x in a.scales.split(',')]
    allg = {'게이트 없음': dict(),
            '1A 즉시': dict(i_gate=1.0, rest_hold_s=0.0),
            '1A 30s': dict(i_gate=1.0, rest_hold_s=30.0),
            '1A 120s': dict(i_gate=1.0, rest_hold_s=120.0),
            '3A 30s': dict(i_gate=3.0, rest_hold_s=30.0)}
    GATES = [(g, allg[g]) for g in a.gates.split(',')]
    P0S = [None if x == 'off' else float(x) for x in a.p0.split(',')]

    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    cell = np.array([r['cell'] for r in RUNS])
    jobs = list(itertools.product(range(len(SCALES)), range(len(GATES)),
                                  range(len(P0S)), range(len(PERTURB))))
    with mp.Pool(14) as pool:
        res = dict(pool.map(_one, jobs))
    E = {(s, g, p): np.stack([res[(s, g, p, i)] for i in range(len(PERTURB))])
         for s, g, p in itertools.product(range(len(SCALES)),
                                          range(len(GATES)), range(len(P0S)))}

    print("  == 섭동 7 개 평균 RMSE (%p), 36 run", flush=True)
    print(f"  {'배율':>6}{'게이트':>12}{'p0_ib':>10}{'평균':>9}"
          + ''.join(f"{n:>12}" for n, _ in PERTURB), flush=True)
    print('  ' + '-' * (37 + 12 * len(PERTURB)), flush=True)
    for k in sorted(E, key=lambda k: E[k].mean()):
        s, g, p = k
        nm = 'off' if P0S[p] is None else f'{P0S[p]:.0e}'
        print(f"  {SCALES[s]:>6.2f}{GATES[g][0]:>12}{nm:>10}"
              f"{E[k].mean()*100:>9.2f}"
              + ''.join(f"{E[k][i].mean()*100:>12.2f}"
                        for i in range(len(PERTURB))), flush=True)

    print("\n  == 셀 하나씩 빼고 고른 뒤, 안 본 셀에서 측정", flush=True)
    cur = None
    for k in E:
        if (abs(SCALES[k[0]] - 1.0) < 1e-9 and GATES[k[1]][0] == '1A 30s'
                and P0S[k[2]] is None):
            cur = k
    print(f"    {'홀드아웃':<20}{'고른 것':>34}{'현재채택':>10}{'새것':>9}{'변화':>9}",
          flush=True)
    tb, tv = [], []
    for c in sorted(set(cell)):
        te, tr = cell == c, cell != c
        best = min(E, key=lambda k: E[k][:, tr].mean())
        nm = ('off' if P0S[best[2]] is None else f'{P0S[best[2]]:.0e}')
        lab = f"x{SCALES[best[0]]:.2f} / {GATES[best[1]][0]} / p0={nm}"
        b = E[cur][:, te].mean() if cur else np.nan
        v = E[best][:, te].mean()
        tb.append(E[cur][:, te].ravel() if cur else np.array([]))
        tv.append(E[best][:, te].ravel())
        print(f"    {c:<20}{lab:>34}{b*100:>10.2f}{v*100:>9.2f}"
              f"{(v-b)/b*100:>+8.1f}%", flush=True)
    TB, TV = np.concatenate(tb), np.concatenate(tv)
    print(f"    {'합계':<20}{'':>34}{TB.mean()*100:>10.2f}{TV.mean()*100:>9.2f}"
          f"{(TV.mean()-TB.mean())/TB.mean()*100:>+8.1f}%", flush=True)


if __name__ == '__main__':
    main()
