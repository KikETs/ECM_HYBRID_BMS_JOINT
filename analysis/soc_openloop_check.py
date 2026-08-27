"""퍼짐 규칙의 이득이 순환 논리인지 확인한다.

SOC 정답 라벨은 SOC = 1 + Ah/3.0, 즉 전류 적분이다.  R 을 키우는 것은
전압 보정을 줄여 전류 적분에 가까워지는 방향이므로, 이득이 "라벨이
전류 적분이라서" 생긴 것일 수 있다.  전압 보정을 아예 끈 경우와
비교하면 판별된다.  전류 적분만으로도 같은 성능이 나오면 순환이고,
전류 적분이 훨씬 나쁘면 퍼짐 규칙은 실제로 전압 정보를 쓰고 있다.
"""
import numpy as np
import pickle
import multiprocessing as mp

RUNS = None


def _rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def _one(job):
    from ekf_soc import run as ekf_run, EKF
    nm, rv_over, kw, want_frac = job
    err, frac, mult = [], [], []
    for r in RUNS:
        rv = _rvolt(r['soh']) if rv_over is None else rv_over
        if want_frac:
            f = EKF(r['sd'], r['sc'], r['soh'], R_volt=rv, gamma=0.0,
                    i_gate=1.0, rest_hold_s=30.0, **kw)
            hit = tot = 0
            mm = []
            for i in range(len(r['I'])):
                f.step(r['I'][i], r['V'][i], r['T'][i])
                if f._v_ew:
                    q = kw['r_var_k'] ** 2 * f._v_ew
                    tot += 1
                    if q > f.R:
                        hit += 1
                        mm.append(q / f.R)
            frac.append(hit / max(tot, 1))
            mult.append(float(np.median(mm)) if mm else 1.0)
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], r['I'], r['V'], r['T'],
                         float(r['soc'][0]), rv, gamma=0.0, i_gate=1.0,
                         rest_hold_s=30.0, **kw)
        err.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
    extra = (np.array(frac), np.array(mult)) if want_frac else None
    return nm, np.array(err), extra


def main():
    global RUNS
    RUNS = pickle.load(open('/tmp/soc_runs.pkl', 'rb'))
    soh = np.array([r['soh'] for r in RUNS])
    jobs = [
        ('기준 (현재 채택)', None, {}, False),
        ('퍼짐 k=20, w=0.003', None, dict(r_var_k=20., ew_rate=0.003), True),
        ('퍼짐 k=50, w=0.003', None, dict(r_var_k=50., ew_rate=0.003), False),
        ('퍼짐 k=200, w=0.003', None, dict(r_var_k=200., ew_rate=0.003), False),
        ('전압 보정 끔 (순수 전류 적분)', 1e4, {}, False),
        ('전압 보정 강함 (R_volt=0.005)', 0.005, {}, False),
    ]
    with mp.Pool(6) as pool:
        res = pool.map(_one, jobs)
    print(f"  {'구성':<34}{'전체':>7}{'최악':>7}{'노화':>7}", flush=True)
    for nm, e, extra in res:
        print(f"  {nm:<34}{e.mean()*100:>7.2f}{e.max()*100:>7.2f}"
              f"{e[soh < 0.80].mean()*100:>7.2f}", flush=True)
        if extra is not None:
            fr, mu = extra
            print(f"    -> 하한이 실제로 R 을 올린 시간 비율 {fr.mean()*100:.1f}%"
                  f"  (run 별 {fr.min()*100:.0f}~{fr.max()*100:.0f}%)", flush=True)
            print(f"    -> 올렸을 때 R 배수 중앙값 {np.median(mu):.1f}x", flush=True)


if __name__ == '__main__':
    main()
