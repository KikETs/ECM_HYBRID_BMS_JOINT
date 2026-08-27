"""전류 옵셋 상태의 고정 비용을 줄일 수 있는지 판다.

옵셋이 0 일 때 상태를 켜면 1.51 -> 2.41 %p 로 0.90 %p 를 문다.  이 값은
넣은 옵셋 크기와 무관하게 평평하므로 옵셋 때문이 아니라 상태를 하나 더
둔 값이다.  상태가 참 SOC 오차의 일부를 대신 흡수하기 때문으로 보인다.

세 손잡이를 돌린다:
  q_ib      상태가 얼마나 빨리 움직일 수 있는가
  p0_ib     처음에 얼마나 모른다고 두는가
  ib_clip   제한.  실제 센서 옵셋은 만스케일(30 A)의 1 % 를 넘지 않으므로
            0.3 A 면 충분하다.  좁히면 흡수를 막는다.
"""
import numpy as np
import pickle
import itertools
import multiprocessing as mp

RUNS = None
GATE = dict(i_gate=1.0, rest_hold_s=30.0)
BIASES = [0.0, 0.10, -0.10, 0.20]


def _rv(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def _drive(r, If, soc0, kw):
    from ekf_soc import EKF
    f = EKF(r['sd'], r['sc'], r['soh'], R_volt=_rv(r['soh']), gamma=0.0, **kw)
    f.x = np.zeros(f.n)
    f.x[0] = soc0
    est = np.empty(len(If))
    for i in range(len(If)):
        est[i], _ = f.step(If[i], r['V'][i], r['T'][i])
    return est, f


def _one(job):
    q_ib, p0, clip = job
    out = {}
    for b in BIASES:
        err, found = [], []
        for r in RUNS:
            kw = dict(GATE)
            if q_ib is not None:
                kw.update(q_ib=q_ib, p0_ib=p0, ib_clip=clip)
            est, f = _drive(r, r['I'] + b, float(r['soc'][0]), kw)
            err.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
            found.append(float(f.x[f.iib]) if f.estimate_ib else np.nan)
        out[b] = (np.array(err), np.array(found))
    return job, out


def main():
    global RUNS
    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    jobs = [(None, 0, 0)]
    jobs += list(itertools.product([1e-12, 1e-10, 1e-8],
                                   [1e-4, 1e-2, 0.25],
                                   [0.3, 2.0]))
    with mp.Pool(14) as pool:
        res = dict(pool.map(_one, jobs))

    print(f"  {'q_ib':>8}{'p0_ib':>8}{'제한':>7}"
          + ''.join(f"{f'{b:+.2f}A':>10}" for b in BIASES)
          + f"{'4개 평균':>10}{'옵셋오차':>10}", flush=True)
    print('  ' + '-' * (23 + 10 * len(BIASES) + 20), flush=True)
    rows = []
    for job in jobs:
        o = res[job]
        means = [o[b][0].mean() * 100 for b in BIASES]
        if job[0] is None:
            nm = f"{'상태 끔':>23}"
            ferr = '-'
        else:
            nm = f"{job[0]:>8.0e}{job[1]:>8.0e}{job[2]:>7.1f}"
            ferr = f"{np.mean([abs(np.median(o[b][1]) - b) for b in BIASES])*1000:.0f} mA"
        rows.append((np.mean(means), job))
        print(f"  {nm}" + ''.join(f"{m:>10.2f}" for m in means)
              + f"{np.mean(means):>10.2f}{ferr:>10}", flush=True)
    rows.sort()
    print(f"\n  4개 평균 최고: {rows[0][1]}  ->  {rows[0][0]:.2f} %p", flush=True)
    print(f"  상태 끔 평균: {np.mean([res[(None,0,0)][b][0].mean()*100 for b in BIASES]):.2f} %p",
          flush=True)


if __name__ == '__main__':
    main()
