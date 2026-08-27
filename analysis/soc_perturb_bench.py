"""순환을 끊은 SOC 벤치.

기존 벤치는 필터에게 라벨을 만든 것과 똑같은 전류를 주고 정확한 초기
SOC 에서 출발시켰다.  라벨이 SOC = 1 + Ah/3.0 이고 필터 예측이
soc + I dt/3600/3.0 이므로 두 식이 같다.  그래서 전압을 안 쓰는 쪽이
무조건 이긴다 (측정: 순수 전류 적분 0.12 %p vs 최선의 필터 0.84 %p).

칼만 필터를 쓰는 이유는 실차에서 그 세 가지가 안 맞기 때문이다:
초기 SOC 를 모르고, 전류 센서에 옵셋이 있고, 이득 오차가 있다.
여기서는 그 셋을 하나씩 넣고 다시 잰다.  라벨은 참 전류로 만들고,
필터에게는 틀어진 전류만 준다.

전체 RMSE 와 뒤쪽 4 분의 1 RMSE 를 같이 낸다.  앞은 수렴 속도까지
섞이고, 뒤는 수렴한 뒤에 남는 오차만 본다.
"""
import os
import numpy as np
import pickle
import multiprocessing as mp

RUNS = None

CONFIGS = [
    ('순수 전류 적분', dict(_open=True)),
    ('EKF 게이트 없음', dict()),
    ('EKF 현재 채택 (게이트)', dict(i_gate=1.0, rest_hold_s=30.0)),
    ('EKF 게이트 + 퍼짐 k=20', dict(i_gate=1.0, rest_hold_s=30.0,
                                 r_var_k=20., ew_rate=0.003)),
    ('EKF 게이트 + 퍼짐 k=200', dict(i_gate=1.0, rest_hold_s=30.0,
                                  r_var_k=200., ew_rate=0.003)),
]

PERTURB = [
    ('틀어짐 없음', dict()),
    ('초기 SOC +10 %p', dict(dsoc=+0.10)),
    ('초기 SOC -10 %p', dict(dsoc=-0.10)),
    ('전류 옵셋 +0.10 A', dict(ibias=+0.10)),
    ('전류 옵셋 -0.10 A', dict(ibias=-0.10)),
    ('전류 이득 +1 %', dict(igain=+0.01)),
    ('전류 이득 -1 %', dict(igain=-0.01)),
]


def _one(job):
    from ekf_soc import run as ekf_run
    ci, pi = job
    cname, ckw = CONFIGS[ci]
    pname, pkw = PERTURB[pi]
    ckw = dict(ckw)
    open_loop = ckw.pop('_open', False)
    full, tail = [], []
    for r in RUNS:
        rv = 1e4 if open_loop else float(
            np.interp(r['soh'], [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))
        # 필터가 보는 전류만 틀고, 라벨은 참 전류로 만들어진 것을 그대로 쓴다
        If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
        s0 = float(r['soc'][0]) + pkw.get('dsoc', 0.0)
        s0 = min(max(s0, 0.02), 0.98)
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], If, r['V'], r['T'],
                         s0, rv, gamma=0.0, **ckw)
        e = est - r['soc']
        full.append(float(np.sqrt(np.mean(e ** 2))))
        tail.append(float(np.sqrt(np.mean(e[-len(e) // 4:] ** 2))))
    return ci, pi, np.array(full), np.array(tail)


def main():
    global RUNS
    RUNS = pickle.load(open(os.environ.get('SOC_RUNS',
                            'results/soc_runs.pkl'), 'rb'))
    soh = np.array([r['soh'] for r in RUNS])
    jobs = [(c, p) for c in range(len(CONFIGS)) for p in range(len(PERTURB))]
    with mp.Pool(14) as pool:
        res = pool.map(_one, jobs)
    F = {(c, p): f for c, p, f, _ in res}
    T = {(c, p): t for c, p, _, t in res}
    import os as _os, csv as _csv
    _os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/soc_perturb.csv', 'w', newline='',
              encoding='utf-8') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['config', 'perturbation', 'rmse_full_pct',
                     'rmse_tail_pct', 'worst_pct'])
        for _c, (_cn, _) in enumerate(CONFIGS):
            for _p, (_pn, _) in enumerate(PERTURB):
                _w.writerow([_cn, _pn, f'{F[(_c, _p)].mean()*100:.3f}',
                             f'{T[(_c, _p)].mean()*100:.3f}',
                             f'{F[(_c, _p)].max()*100:.3f}'])
    print('  -> results/tables/soc_perturb.csv', flush=True)
    np.savez('results/soc_perturb.npz', soh=soh,
             full=np.array([F[(c, p)] for c in range(len(CONFIGS))
                            for p in range(len(PERTURB))]),
             tail=np.array([T[(c, p)] for c in range(len(CONFIGS))
                            for p in range(len(PERTURB))]))

    for tag, D in (('전체 RMSE (%p, 36 run 평균)', F),
                   ('뒤쪽 1/4 RMSE (%p) — 수렴 뒤에 남는 오차', T)):
        print(f"\n  == {tag}", flush=True)
        print(f"  {'틀어짐':<20}" + ''.join(f"{n[:16]:>18}"
                                          for n, _ in CONFIGS), flush=True)
        print('  ' + '-' * (20 + 18 * len(CONFIGS)), flush=True)
        for p, (pname, _) in enumerate(PERTURB):
            row = ''.join(f"{D[(c, p)].mean()*100:>18.2f}"
                          for c in range(len(CONFIGS)))
            print(f"  {pname:<20}{row}", flush=True)
        row = ''.join(f"{np.mean([D[(c, p)].mean() for p in range(1, len(PERTURB))])*100:>18.2f}"
                      for c in range(len(CONFIGS)))
        print(f"  {'틀어짐 6개 평균':<20}{row}", flush=True)


if __name__ == '__main__':
    main()
