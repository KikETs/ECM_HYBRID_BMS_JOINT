"""전류 옵셋 상태가 왜 일부 run 에서 폭주하는지 가른다.

가설 A  기계 자체가 고장 — 옵셋을 안 넣어도 터진다
가설 B  게이트 탓 — 휴지 때만 갱신하니 한 번의 잔차로 SOC 오차와
        옵셋 오차를 갈라낼 수 없다.  게이트를 풀면 나아진다
가설 C  초기 불확실성 P0 가 커서 초반에 크게 튄다
"""
import numpy as np
import pickle
import multiprocessing as mp

RUNS = None


def _rv(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def _one(job):
    from ekf_soc import EKF
    nm, bias, kw = job
    err, found = [], []
    for r in RUNS:
        f = EKF(r['sd'], r['sc'], r['soh'], R_volt=_rv(r['soh']), gamma=0.0,
                **kw)
        If = r['I'] + bias
        est = np.empty(len(If))
        for i in range(len(If)):
            est[i], _ = f.step(If[i], r['V'][i], r['T'][i])
        err.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
        found.append(float(f.x[f.iib]) if f.estimate_ib else np.nan)
    return nm, np.array(err), np.array(found)


G = dict(i_gate=1.0, rest_hold_s=30.0)


def main():
    global RUNS
    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    jobs = [
        # A: 옵셋을 안 넣고 상태만 켠다.  터지면 기계 문제.
        ('A  옵셋 0, 상태 켬 (게이트)', 0.0, dict(q_ib=1e-9, **G)),
        ('A  옵셋 0, 상태 끔 (게이트)', 0.0, dict(**G)),
        # B: 게이트를 풀어 매 시각 갱신.
        ('B  옵셋 +0.1, 상태 켬, 게이트 없음', 0.10, dict(q_ib=1e-9)),
        ('B  옵셋 +0.1, 상태 끔, 게이트 없음', 0.10, {}),
        # C: 초기 불확실성을 낮춘다 (P0 0.25 -> 0.0025, 즉 표준편차 0.05 A)
        ('C  옵셋 +0.1, 상태 켬, P0 작게', 0.10, dict(q_ib=1e-9, p0_ib=2.5e-3, **G)),
        ('C  옵셋 +0.1, 상태 켬, P0 작게+게이트없음', 0.10,
         dict(q_ib=1e-9, p0_ib=2.5e-3)),
        # 기준
        ('참고  옵셋 +0.1, 상태 끔 (게이트)', 0.10, dict(**G)),
    ]
    with mp.Pool(7) as pool:
        res = pool.map(_one, jobs)
    print(f"  {'구성':<38}{'RMSE':>8}{'중앙':>8}{'최악':>8}"
          f"{'찾은 옵셋':>12}{'폭주 run':>9}", flush=True)
    for nm, e, fd in res:
        run = int(np.sum(np.abs(fd) > 0.5)) if np.isfinite(fd).any() else 0
        s = f"{np.nanmedian(fd):>11.3f}A" if np.isfinite(fd).any() else f"{'-':>12}"
        print(f"  {nm:<38}{e.mean()*100:>8.2f}{np.median(e)*100:>8.2f}"
              f"{e.max()*100:>8.2f}{s}{run:>9}", flush=True)


if __name__ == '__main__':
    main()
