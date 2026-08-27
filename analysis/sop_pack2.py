"""팩 수준 재측정 — 28 절을 추정 SOH 위에서 다시.

28 절의 결론(28.4)은 "채택 lambda 에서는 N=1 부터 192 까지 초과 0 이고,
위험은 여유를 줄이려 할 때만 드러난다" 였다.  그런데 그 lambda 는 정답
SOH 로 잡은 값이다 (29 절 지적).  추정 SOH 를 쓰면 lambda 가 조여지므로
(방전 0.679 -> 0.594) 팩도 그 위에서 다시 봐야 한다.

채택 평가 구성.  31.1 에서 A3 판이 16 절의 lambda 를 재현하는 것으로
설정을 특정했고(방전 허용 0.0 A, 충전 허용 0.5 A), 32.7 이후 채택 트림이
A8 로 바뀌었다.  A8 의 lambda 는 방전 0.683 / 0.470, 충전 0.586 / 0.560.

여기서는 lambda 를 홀드아웃 셀을 빼고 다시 잡은 뒤(정답판·추정판 각각),
그 lambda 로 N 셀 팩을 모사해 초과와 쓸 수 있는 전류를 낸다.
"""
import argparse
import csv

import numpy as np

from sop_safety import EXTRAP_MAX

NSIM = 4000
NS = [1, 12, 48, 96, 192]
SOH_EDGES = [0.0, 0.75, 0.85, 0.95, 2.0]
SOC_EDGES = [0.0, 0.35, 0.55, 0.75, 2.0]


def load(path):
    r = list(csv.DictReader(open(path, encoding='utf-8')))

    def g(k):
        return np.array([float(x[k]) if x[k] not in ('', 'nan') else np.nan
                         for x in r])
    return dict(meas=np.abs(g('I_meas_A')), hyb=np.abs(g('I_A3_A')),
                extrap=g('extrap'), soh=g('SOH'), soc=g('SOC'),
                tau=g('tau_s'), cell=np.array([x['cell'] for x in r]))


def keep(d, tau):
    return (np.isfinite(d['meas']) & np.isfinite(d['hyb'])
            & (d['meas'] > 0.5) & (d['extrap'] <= EXTRAP_MAX)
            & (np.round(d['tau'], 1) == tau))


def lam_loco(d, tau, tol):
    """홀드아웃 셀을 빼고 잡은 lambda 의 중앙값 (16 절과 같은 방식)."""
    m = keep(d, tau)
    out = []
    for c in sorted(set(d['cell'][m])):
        tr = m & (d['cell'] != c)
        if tr.sum() < 25:
            continue
        lo, hi = 0.02, 1.6
        for _ in range(70):
            mid = (lo + hi) / 2
            if np.max(mid * d['hyb'][tr] - d['meas'][tr]) > tol:
                hi = mid
            else:
                lo = mid
        out.append(lo)
    return float(np.median(out)) if out else np.nan


def pack(d, tau, lam, rng, min_group=8):
    m = keep(d, tau)
    sh = np.digitize(d['soh'][m], SOH_EDGES)
    sc = np.digitize(d['soc'][m], SOC_EDGES)
    g = np.array([f'{a}|{b}' for a, b in zip(sh, sc)])
    meas, pred = d['meas'][m], d['hyb'][m]
    ks, cnt = np.unique(g, return_counts=True)
    ks = ks[cnt >= min_group]
    if len(ks) == 0:
        return None
    idx = {k: np.where(g == k)[0] for k in ks}
    w = np.array([len(idx[k]) for k in ks], float)
    w /= w.sum()
    out = {}
    for N in NS:
        gsel = rng.choice(len(ks), size=NSIM, p=w)
        pm = np.empty(NSIM)
        pp = np.empty(NSIM)
        for i, gi in enumerate(gsel):
            ii = idx[ks[gi]]
            pick = ii[rng.integers(0, len(ii), N)]
            pm[i] = meas[pick].min()
            pp[i] = pred[pick].min()
        o = lam * pp - pm
        out[N] = dict(exc=float(np.mean(o > 0) * 100),
                      worst=float(max(o.max(), 0.0)),
                      util=float(np.median(lam * pp / pm) * 100))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-dir', default='results/eval')
    ap.add_argument('--out', default='results/tables/pack.csv')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    E = a.eval_dir
    j = lambda n: __import__('os').path.join(E, n + '.csv')
    SETS = [('방전', 10.0, 0.0, j('a8_disc_oracle'), j('a8_disc_est')),
            ('방전', 2.0, 0.0, j('a8_disc_oracle'), j('a8_disc_est')),
            ('충전', 10.0, 0.5, j('a8_char_oracle'), j('a8_char_est')),
            ('충전', 2.0, 0.5, j('a8_char_oracle'), j('a8_char_est'))]

    print(f"  {'방향':<6}{'tau':>5}{'SOH 입력':>10}{'lambda':>9}{'셀 n':>7}"
          + ''.join(f"{f'N={n} 초과':>11}" for n in NS)
          + f"{'N=192 최악':>12}{'N=192 전류':>12}", flush=True)
    print('  ' + '-' * (37 + 11 * len(NS) + 24), flush=True)
    store = {}
    for nm, tau, tol, po, pe in SETS:
        for arm, path in (('정답', po), ('추정', pe)):
            d = load(path)
            lam = lam_loco(d, tau, tol)
            r = pack(d, tau, lam, rng)
            store[(nm, tau, arm)] = (lam, r)
            print(f"  {nm:<6}{tau:>5.0f}{arm:>10}{lam:>9.3f}{keep(d, tau).sum():>7}"
                  + ''.join(f"{r[n]['exc']:>10.1f}%" for n in NS)
                  + f"{r[NS[-1]]['worst']:>11.2f}A{r[NS[-1]]['util']:>11.1f}%",
                  flush=True)

    import csv as _csv, os as _os
    _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['direction', 'tau_s', 'soh', 'lambda', 'N',
                    'exceed_pct', 'worst_A', 'usable_pct'])
        for (nm, tau, arm), (lam, r) in store.items():
            for N in NS:
                w.writerow([nm, tau, arm, f'{lam:.3f}', N,
                            f"{r[N]['exc']:.1f}", f"{r[N]['worst']:.2f}",
                            f"{r[N]['util']:.1f}"])
    print(f"\n  -> {a.out}", flush=True)

    print("\n  == 정답 SOH 의 lambda 를 추정 SOH 판에 그대로 쓰면"
          " (여유를 안 다시 잡았을 때)", flush=True)
    print(f"  {'방향':<6}{'tau':>5}{'쓴 lambda':>11}"
          + ''.join(f"{f'N={n} 초과':>11}" for n in NS)
          + f"{'N=192 최악':>12}", flush=True)
    for nm, tau, tol, po, pe in SETS:
        lam_o = store[(nm, tau, '정답')][0]
        d = load(pe)
        r = pack(d, tau, lam_o, rng)
        print(f"  {nm:<6}{tau:>5.0f}{lam_o:>11.3f}"
              + ''.join(f"{r[n]['exc']:>10.1f}%" for n in NS)
              + f"{r[NS[-1]]['worst']:>11.2f}A", flush=True)


if __name__ == '__main__':
    main()
