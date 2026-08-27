"""R_volt 스케줄과 게이트를 섭동 벤치에서 다시 고른다.

현재 값 — R_volt = SOH 스케줄 (0.70 -> 110 mV, 1.00 -> 15 mV), 게이트는
|I| <= 1 A 를 30 s 유지 — 은 전부 순환 벤치에서 고른 것이다.  그 벤치의
최적해가 "전압을 믿지 마라" 였으므로, 두 값 모두 전압을 덜 쓰는 쪽으로
과하게 밀려 있을 수 있다.

여기서는 라벨을 만든 전류와 필터가 보는 전류를 어긋나게 한 뒤 다시 고른다.
고르는 것도 셀 하나씩 빼고 한다.
"""
import numpy as np
import pickle
import itertools
import multiprocessing as mp

RUNS = None

SCALES = [0.25, 0.5, 1.0, 2.0]          # SOH 스케줄에 곱한다
GATES = [('게이트 없음', dict()),
         ('1A 즉시', dict(i_gate=1.0, rest_hold_s=0.0)),
         ('1A 30s', dict(i_gate=1.0, rest_hold_s=30.0)),
         ('1A 120s', dict(i_gate=1.0, rest_hold_s=120.0)),
         ('3A 30s', dict(i_gate=3.0, rest_hold_s=30.0))]
PERTURB = [('없음', {}),
           ('초기SOC+10', dict(dsoc=+0.10)),
           ('초기SOC-10', dict(dsoc=-0.10)),
           ('옵셋+0.1A', dict(ibias=+0.10)),
           ('옵셋-0.1A', dict(ibias=-0.10)),
           ('이득+1%', dict(igain=+0.01)),
           ('이득-1%', dict(igain=-0.01))]


def _one(job):
    from ekf_soc import run as ekf_run
    si, gi, pi = job
    _, gkw = GATES[gi]
    _, pkw = PERTURB[pi]
    kw = dict(gkw)
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
    global RUNS
    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    cell = np.array([r['cell'] for r in RUNS])
    jobs = list(itertools.product(range(len(SCALES)), range(len(GATES)),
                                  range(len(PERTURB))))
    with mp.Pool(14) as pool:
        res = dict(pool.map(_one, jobs))
    # 계산이 비싸므로 먼저 저장하고 표를 만든다
    np.savez('/tmp/soc_retune.npz', cell=cell,
             keys=np.array([f'{a}|{b}|{c}' for a, b, c in jobs]),
             vals=np.array([res[j] for j in jobs]))
    # 구성별로 섭동 7 개를 이어 붙인다 (run x 섭동)
    E = {}
    for s in range(len(SCALES)):
        for g in range(len(GATES)):
            E[(s, g)] = np.stack([res[(s, g, p)]
                                  for p in range(len(PERTURB))])   # 7 x 36

    print("  == 섭동 7 개 평균 RMSE (%p).  세로 = R_volt 배율, 가로 = 게이트",
          flush=True)
    print(f"  {'배율':<8}" + ''.join(f"{n:>14}" for n, _ in GATES), flush=True)
    print('  ' + '-' * (8 + 14 * len(GATES)), flush=True)
    for s, sc in enumerate(SCALES):
        print(f"  x{sc:<7.2f}" + ''.join(f"{E[(s, g)].mean()*100:>14.2f}"
                                         for g in range(len(GATES))),
              flush=True)

    print("\n  == 틀어짐이 없을 때만 (%p) — 순환 벤치가 보던 것", flush=True)
    print(f"  {'배율':<8}" + ''.join(f"{n:>14}" for n, _ in GATES), flush=True)
    for s, sc in enumerate(SCALES):
        print(f"  x{sc:<7.2f}" + ''.join(f"{E[(s, g)][0].mean()*100:>14.2f}"
                                         for g in range(len(GATES))),
              flush=True)

    print("\n  == 전류 옵셋일 때만 (%p) — 실제 약점", flush=True)
    print(f"  {'배율':<8}" + ''.join(f"{n:>14}" for n, _ in GATES), flush=True)
    for s, sc in enumerate(SCALES):
        print(f"  x{sc:<7.2f}" + ''.join(
            f"{E[(s, g)][3:5].mean()*100:>14.2f}" for g in range(len(GATES))),
            flush=True)

    print("\n  == 셀 하나씩 빼고 고른 뒤, 안 본 셀에서 측정", flush=True)
    cur = (2, 2)   # 배율 x1.0, 게이트 1A 30s = 현재 채택
    print(f"    {'홀드아웃':<20}{'고른 배율':>10}{'고른 게이트':>14}"
          f"{'현재채택':>10}{'재조정':>10}{'변화':>9}", flush=True)
    tb, tv = [], []
    for c in sorted(set(cell)):
        te, tr = cell == c, cell != c
        best = min(E, key=lambda k: E[k][:, tr].mean())
        b = E[cur][:, te].mean()
        v = E[best][:, te].mean()
        tb.append(E[cur][:, te]); tv.append(E[best][:, te])
        print(f"    {c:<20}{SCALES[best[0]]:>10.2f}{GATES[best[1]][0]:>14}"
              f"{b*100:>10.2f}{v*100:>10.2f}{(v-b)/b*100:>+8.1f}%", flush=True)
    TB = np.concatenate([x.ravel() for x in tb])
    TV = np.concatenate([x.ravel() for x in tv])
    print(f"    {'합계':<20}{'':>10}{'':>14}{TB.mean()*100:>10.2f}"
          f"{TV.mean()*100:>10.2f}{(TV.mean()-TB.mean())/TB.mean()*100:>+8.1f}%",
          flush=True)


if __name__ == '__main__':
    main()
