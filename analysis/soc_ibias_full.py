"""전류 옵셋 상태 전면 평가 — 초기화를 바로잡고 다시.

앞선 진단은 EKF 를 직접 만들면서 f.x[0] = soc0 를 빠뜨려 SOC 가 0 에서
출발했다.  run() 은 그 줄을 갖고 있다.  여기서는 같은 초기화를 쓰는
헬퍼로 통일한다.

1부  q_ib x 넣은 옵셋 격자.  옵셋을 찾아내는가, RMSE 가 나아지는가.
2부  섭동 벤치 전체에 전류 옵셋 상태를 6번째 구성으로 넣어 비교.
"""
import numpy as np
import pickle
import multiprocessing as mp

RUNS = None


def _rv(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def _drive(r, If, soc0, kw):
    """run() 과 같은 초기화로 돌리고, 마지막 내부 상태까지 돌려준다."""
    from ekf_soc import EKF
    f = EKF(r['sd'], r['sc'], r['soh'], R_volt=_rv(r['soh']), gamma=0.0, **kw)
    f.x = np.zeros(f.n)
    f.x[0] = soc0
    est = np.empty(len(If))
    for i in range(len(If)):
        est[i], _ = f.step(If[i], r['V'][i], r['T'][i])
    return est, f


GATE = dict(i_gate=1.0, rest_hold_s=30.0)


def _grid(job):
    q_ib, bias = job
    err, tail, found = [], [], []
    for r in RUNS:
        kw = dict(GATE)
        if q_ib is not None:
            kw['q_ib'] = q_ib
        est, f = _drive(r, r['I'] + bias, float(r['soc'][0]), kw)
        e = est - r['soc']
        err.append(float(np.sqrt(np.mean(e ** 2))))
        tail.append(float(np.sqrt(np.mean(e[-len(e) // 4:] ** 2))))
        found.append(float(f.x[f.iib]) if f.estimate_ib else np.nan)
    return q_ib, bias, np.array(err), np.array(tail), np.array(found)


def main():
    global RUNS
    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    QS = [None, 1e-10, 1e-9, 1e-8, 1e-7]
    BS = [0.0, 0.05, 0.10, 0.20, -0.10]
    jobs = [(q, b) for q in QS for b in BS]
    with mp.Pool(14) as pool:
        res = {(q, b): (e, t, f) for q, b, e, t, f in pool.map(_grid, jobs)}

    print("  == 전체 RMSE (%p).  세로 = 옵셋 상태의 q_ib, 가로 = 넣은 전류 옵셋",
          flush=True)
    print(f"  {'q_ib':<12}" + ''.join(f"{f'{b:+.2f} A':>11}" for b in BS),
          flush=True)
    print('  ' + '-' * (12 + 11 * len(BS)), flush=True)
    for q in QS:
        nm = '상태 끔' if q is None else f'{q:.0e}'
        print(f"  {nm:<12}" + ''.join(f"{res[(q, b)][0].mean()*100:>11.2f}"
                                      for b in BS), flush=True)

    print("\n  == 뒤쪽 1/4 RMSE (%p) — 수렴 뒤 남는 오차", flush=True)
    print(f"  {'q_ib':<12}" + ''.join(f"{f'{b:+.2f} A':>11}" for b in BS),
          flush=True)
    print('  ' + '-' * (12 + 11 * len(BS)), flush=True)
    for q in QS:
        nm = '상태 끔' if q is None else f'{q:.0e}'
        print(f"  {nm:<12}" + ''.join(f"{res[(q, b)][1].mean()*100:>11.2f}"
                                      for b in BS), flush=True)

    print("\n  == 찾아낸 옵셋 (A, 36 run 중앙값 / 최소~최대)", flush=True)
    for q in QS[1:]:
        row = ''
        for b in BS:
            fd = res[(q, b)][2]
            row += f"{np.median(fd):>+7.3f}"
        print(f"  q_ib={q:.0e}" + '   ' + row
              + '     넣은 값 ' + ' '.join(f'{b:+.2f}' for b in BS), flush=True)

    print("\n  == 폭주한 run 수 (|찾은 옵셋| > 0.5 A)", flush=True)
    for q in QS[1:]:
        print(f"  q_ib={q:.0e}   " + ''.join(
            f"{int(np.sum(np.abs(res[(q, b)][2]) > 0.5)):>11}" for b in BS),
            flush=True)

    np.savez('/tmp/soc_ibias_full.npz',
             qs=np.array([-1 if q is None else q for q in QS]),
             bs=np.array(BS),
             err=np.array([[res[(q, b)][0] for b in BS] for q in QS]),
             tail=np.array([[res[(q, b)][1] for b in BS] for q in QS]),
             found=np.array([[res[(q, b)][2] for b in BS] for q in QS]))


if __name__ == '__main__':
    main()
