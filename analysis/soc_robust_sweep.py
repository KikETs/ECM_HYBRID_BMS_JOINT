"""SOC EKF 로버스트 이득 규칙 스윕.

현재 채택된 게이트(|I|<=1A 를 30 s 유지)는 그대로 두고, 그 위에
잔차 크기로 신뢰를 연속 조절하는 규칙을 얹어 본다.

  huber_c     잔차가 c 를 넘으면 칼만 이득을 c/|r| 로 줄인다
  r_floor_k   R 하한을 잔차 이동평균의 k 배 (제곱해서 분산으로)
  r_max_mult  부풀린 R 의 상한 (보정이 완전히 멎는 것을 막음)

사용:  python3 soc_robust_sweep.py [--jobs 14]
"""
import argparse, os, pickle
import numpy as np

CELLS = ['CC', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_REST', 'CC_CELL2',
         'BOOST_NEGPULSE_1S']
CACHE = '/tmp/soc_runs.pkl'
NRUN_PER_CELL = 6
NMAX = 20000


def build():
    from ecm_surface import ECMSurface
    runs = []
    for cell in CELLS:
        try:
            sd = ECMSurface(cell, 'discharge')
            sc = ECMSurface(cell, 'charge')
            z = np.load(f'cache_t/uypydj_{cell}_Fifteen_Drive_Cycles.npz')
        except Exception:
            continue
        lens = z['lens']
        off = np.concatenate([[0], np.cumsum(lens)])
        for k in np.linspace(0, len(lens) - 1, NRUN_PER_CELL).astype(int):
            sl = slice(off[k], off[k] + lens[k])
            soc, V, I, SOH, T = (z[x][sl] for x in ('SOC', 'V', 'I', 'SOH', 'T'))
            ok = (np.isfinite(soc) & np.isfinite(V) & np.isfinite(I)
                  & np.isfinite(T))
            if ok.sum() < 2000:
                continue
            runs.append(dict(cell=cell, cyc=int(lens[:k].sum() and k or k),
                             sd=sd, sc=sc,
                             soc=soc[ok][:NMAX], V=V[ok][:NMAX],
                             I=I[ok][:NMAX], T=T[ok][:NMAX],
                             soh=float(np.nanmedian(SOH))))
    return runs


def load_runs():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    runs = build()
    with open(CACHE, 'wb') as f:
        pickle.dump(runs, f)
    return runs


RUNS = None


def _one(job):
    """한 구성 x 전체 run.  (이름, run 별 RMSE 배열, SOH 배열)"""
    from ekf_soc import run as ekf_run
    name, kw = job
    err, soh = [], []
    for r in RUNS:
        rv = float(np.interp(r['soh'], [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], r['I'], r['V'], r['T'],
                         float(r['soc'][0]), rv, gamma=0.0,
                         i_gate=1.0, rest_hold_s=30.0, **kw)
        err.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
        soh.append(r['soh'])
    return name, np.array(err), np.array(soh)


def configs():
    '''2 차 스윕: 편향을 뺀 퍼짐만으로 R 을 키우는 규칙 (r_var_k) 을
    1 차의 최고 구성들과 나란히 놓는다.'''
    c = [('기준 (현재 채택)', {}),
         ('[1차 최고] R 하한 k=5, 상한 9x',
          dict(r_floor_k=5., r_max_mult=9.)),
         ('[1차 최고] R 하한 k=8, 상한 25x',
          dict(r_floor_k=8., r_max_mult=25.))]
    for k in (2, 3, 4, 5, 6, 8, 12, 16):
        c.append((f'퍼짐 k={k}', dict(r_var_k=float(k))))
    for k in (5, 8, 12):
        for m in (9, 25, 100):
            c.append((f'퍼짐 k={k}, 상한 {m}x',
                      dict(r_var_k=float(k), r_max_mult=float(m))))
    for w in (0.003, 0.03, 0.1):
        for k in (5, 8):
            c.append((f'퍼짐 k={k}, 창 w={w}',
                      dict(r_var_k=float(k), ew_rate=w)))
    for cc in (0.02, 0.05):
        for k in (5, 8):
            c.append((f'Huber {cc*1000:.0f} mV + 퍼짐 k={k}',
                      dict(huber_c=cc, r_var_k=float(k))))
    return c


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--out', default='/tmp/soc_robust.npz')
    a = ap.parse_args()

    RUNS = load_runs()
    print(f'  {len(RUNS)} run  ({len(set(r["cell"] for r in RUNS))} 셀)', flush=True)

    jobs = configs()
    import multiprocessing as mp
    with mp.Pool(a.jobs) as pool:
        res = pool.map(_one, jobs)

    soh = res[0][2]
    base = res[0][1]
    hdr = (f'  {"구성":<28}{"전체":>7}{"최악":>7}{"신품":>7}'
           f'{"중간":>7}{"노화":>7}{"기준대비":>9}')
    print('\n' + hdr, flush=True)
    print('  ' + '-' * 70, flush=True)
    out = {}
    for name, e, _ in res:
        out[name] = e
        g = [e[soh >= 0.90].mean(), e[(soh >= 0.80) & (soh < 0.90)].mean(),
             e[soh < 0.80].mean()]
        d = (e.mean() - base.mean()) / base.mean() * 100
        print(f'  {name:<28}{e.mean()*100:>7.2f}{e.max()*100:>7.2f}'
              f'{g[0]*100:>7.2f}{g[1]*100:>7.2f}{g[2]*100:>7.2f}'
              f'{d:>8.1f}%', flush=True)
    np.savez(a.out, soh=soh,
             names=np.array(list(out)), **{f'e{i}': v for i, v in
                                           enumerate(out.values())})
    print(f'\n  {a.out}', flush=True)


if __name__ == '__main__':
    main()
