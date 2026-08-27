"""전류 옵셋 상태 시범 확인.

(1) 끄면 (q_ib=None) 예전 숫자가 그대로 나오는가 — 회귀 확인
(2) 켜면 넣어 준 옵셋을 실제로 찾아내는가 — 관측 가능성 확인
"""
import numpy as np
import pickle
import multiprocessing as mp

RUNS = None
BIAS = 0.10


def _rv(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def _regress(_):
    from ekf_soc import run as ekf_run
    e = []
    for r in RUNS:
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], r['I'], r['V'], r['T'],
                         float(r['soc'][0]), _rv(r['soh']), gamma=0.0,
                         i_gate=1.0, rest_hold_s=30.0)
        e.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
    return ('회귀: q_ib 끔, 틀어짐 없음', np.array(e), None)


def _observe(q_ib):
    """옵셋을 넣고, 필터가 추정한 옵셋이 넣은 값에 가는지 본다."""
    from ekf_soc import EKF
    err, found = [], []
    for r in RUNS:
        f = EKF(r['sd'], r['sc'], r['soh'], R_volt=_rv(r['soh']), gamma=0.0,
                i_gate=1.0, rest_hold_s=30.0, q_ib=q_ib)
        If = r['I'] + BIAS
        est = np.empty(len(If))
        for i in range(len(If)):
            est[i], _ = f.step(If[i], r['V'][i], r['T'][i])
        err.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
        found.append(float(f.x[f.iib]))
    return (f'q_ib={q_ib:.0e}', np.array(err), np.array(found))


def main():
    global RUNS
    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    with mp.Pool(6) as pool:
        res = ([_regress(None)]
               + pool.map(_observe, [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]))
    print(f"  넣은 전류 옵셋 {BIAS:+.2f} A  (30 A 만스케일의 {BIAS/30*100:.1f}%)\n",
          flush=True)
    print(f"  {'구성':<26}{'RMSE':>8}{'최악':>8}{'찾은 옵셋 중앙':>16}"
          f"{'찾은 범위':>18}", flush=True)
    for nm, e, fd in res:
        s = (f"{np.median(fd):>15.3f}A"
             f"{f'{fd.min():+.2f} ~ {fd.max():+.2f}':>18}"
             if fd is not None else f"{'-':>15} {'-':>17}")
        print(f"  {nm:<26}{e.mean()*100:>8.2f}{e.max()*100:>8.2f}{s}", flush=True)
    print("\n  참고: 이전 측정  q_ib 끔 / 틀어짐 없음 = 1.51,  "
          "옵셋 +0.1A 인데 q_ib 끔 = 2.86", flush=True)


if __name__ == '__main__':
    main()
