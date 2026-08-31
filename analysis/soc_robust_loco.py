"""퍼짐 규칙 (k, w) 을 세밀 격자로 훑고, 셀 하나씩 빼서 정직하게 검증한다.

검증 방식은 λ 보정과 같다: 홀드아웃 셀을 빼고 나머지 5 셀에서 (k, w) 를
고른 뒤, 고를 때 안 본 셀에서 성능을 잰다. 36 run 전부로 고른 값을
같은 36 run 에서 재는 것과 얼마나 차이 나는지가 핵심이다.
"""
import numpy as np, pickle, multiprocessing as mp, itertools

CACHE = '/tmp/soc_runs.pkl'
KS = [3, 4, 5, 6, 8, 10, 12, 16, 20]
WS = [0.001, 0.002, 0.003, 0.005, 0.01]
RUNS = None


def _one(job):
    from ekf_soc import run as ekf_run
    k, w = job
    out = []
    for r in RUNS:
        rv = float(np.interp(r['soh'], [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))
        kw = {} if k is None else dict(r_var_k=float(k), ew_rate=w)
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], r['I'], r['V'], r['T'],
                         float(r['soc'][0]), rv, gamma=0.0,
                         i_gate=1.0, rest_hold_s=30.0, **kw)
        out.append(float(np.sqrt(np.mean((est - r['soc']) ** 2))))
    return (k, w), np.array(out)


def main():
    global RUNS
    RUNS = pickle.load(open(CACHE, 'rb'))
    cell = np.array([r['cell'] for r in RUNS])
    soh = np.array([r['soh'] for r in RUNS])
    jobs = [(None, None)] + list(itertools.product(KS, WS))
    with mp.Pool(14) as pool:
        res = dict(pool.map(_one, jobs))
    base = res[(None, None)]
    np.savez('/tmp/soc_loco.npz', cell=cell, soh=soh,
             keys=np.array([f'{k}|{w}' for k, w in res]),
             vals=np.array(list(res.values())))

    print(f"  기준: 전체 {base.mean()*100:.2f}  최악 {base.max()*100:.2f}  "
          f"노화 {base[soh<0.80].mean()*100:.2f}\n", flush=True)
    print("  (a) 36 run 전부로 고른 경우 — 평균 기준 상위 6", flush=True)
    print(f"    {'k':>4}{'w':>8}{'전체':>8}{'최악':>8}{'노화':>8}", flush=True)
    order = sorted([x for x in res if x[0] is not None],
                   key=lambda x: res[x].mean())
    for k, w in order[:6]:
        e = res[(k, w)]
        print(f"    {k:>4}{w:>8.3f}{e.mean()*100:>8.2f}{e.max()*100:>8.2f}"
              f"{e[soh<0.80].mean()*100:>8.2f}", flush=True)
    print("\n  (b) 최악을 기준보다 나쁘게 만들지 않는 것 중 평균 최고", flush=True)
    ok = [x for x in order if res[x].max() <= base.max()]
    for k, w in ok[:6]:
        e = res[(k, w)]
        print(f"    {k:>4}{w:>8.3f}{e.mean()*100:>8.2f}{e.max()*100:>8.2f}"
              f"{e[soh<0.80].mean()*100:>8.2f}", flush=True)
    if not ok:
        print("    없음", flush=True)

    print("\n  (c) 셀 하나씩 빼고 고른 뒤, 안 본 셀에서 측정", flush=True)
    print(f"    {'홀드아웃':<20}{'고른 k':>7}{'고른 w':>8}"
          f"{'기준':>8}{'적용':>8}{'변화':>9}", flush=True)
    tot_b, tot_v = [], []
    for c in sorted(set(cell)):
        te, tr = cell == c, cell != c
        k, w = min([x for x in res if x[0] is not None],
                   key=lambda x: res[x][tr].mean())
        b, v = base[te].mean(), res[(k, w)][te].mean()
        tot_b.append(base[te]); tot_v.append(res[(k, w)][te])
        print(f"    {c:<20}{k:>7}{w:>8.3f}{b*100:>8.2f}{v*100:>8.2f}"
              f"{(v-b)/b*100:>+8.1f}%", flush=True)
    tb, tv = np.concatenate(tot_b), np.concatenate(tot_v)
    print(f"    {'합계':<20}{'':>7}{'':>8}{tb.mean()*100:>8.2f}"
          f"{tv.mean()*100:>8.2f}{(tv.mean()-tb.mean())/tb.mean()*100:>+8.1f}%",
          flush=True)
    print(f"    최악  기준 {tb.max()*100:.2f} -> 적용 {tv.max()*100:.2f}", flush=True)
    m = np.concatenate([soh[cell == c] for c in sorted(set(cell))]) < 0.80
    print(f"    노화  기준 {tb[m].mean()*100:.2f} -> 적용 {tv[m].mean()*100:.2f}",
          flush=True)


if __name__ == '__main__':
    main()
