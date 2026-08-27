"""팩 수준 SOP — min 연산이 오차를 어떻게 바꾸는가.

직렬 팩은 전류가 공통이므로  팩 SOP = min over cells (셀별 I*).
셀 추정기를 팩에 올리면 그 min 을 거친다.

방법 (28.2 와 같음): 셀 홀드아웃 오차를 조건 그룹(SOH 대역 x SOC 대역 x
tau) 안에서 재표집해 N 셀 팩을 모사한다.  팩 셀은 SOH 가 비슷하므로 같은
그룹에서만 뽑는다.

28 절과 다른 점: --soh-est 로 만든 평가를 쓰면 추정 SOH 판이 된다.
28 절의 수치는 전부 정답 SOH 위에 있었다 (29 절 지적).

사용:
  python3 sop_pack.py --disc /tmp/pk_discharge_oracle.csv \\
                      --chg  /tmp/pk_charge_oracle.csv
"""
import argparse
import csv

import numpy as np

from sop_safety import EXTRAP_MAX


def load(path):
    """sop_safety.load 와 같되 SOC 와 cycle 까지 읽는다."""
    r = list(csv.DictReader(open(path, encoding='utf-8')))

    def g(k):
        return np.array([float(x[k]) if x[k] not in ('', 'nan') else np.nan
                         for x in r])
    return dict(meas=np.abs(g('I_meas_A')), hyb=np.abs(g('I_A3_A')),
                ecm=np.abs(g('I_A0_A')), extrap=g('extrap'),
                soh=g('SOH'), soc=g('SOC'), tau=g('tau_s'), cycle=g('cycle'),
                cell=np.array([x['cell'] for x in r]))

NSIM = 4000
NS = [1, 12, 96]
LAMS = [0.90, 0.80, 0.70]
SOH_EDGES = [0.0, 0.75, 0.85, 0.95, 2.0]
SOC_EDGES = [0.0, 0.35, 0.55, 0.75, 2.0]


def groups(d, mask):
    """조건 그룹 라벨 — SOH 대역 x SOC 대역 x tau."""
    sh = np.digitize(d['soh'][mask], SOH_EDGES)
    sc = np.digitize(d['soc'][mask], SOC_EDGES)
    ta = np.round(d['tau'][mask], 3)
    return np.array([f"{a}|{b}|{c}" for a, b, c in zip(sh, sc, ta)])


def simulate(path, rng, min_group=8):
    d = load(path)
    m = (np.isfinite(d['meas']) & np.isfinite(d['hyb'])
         & (np.abs(d['meas']) > 0.5) & (d['extrap'] <= EXTRAP_MAX))
    g = groups(d, m)
    meas = np.abs(d['meas'][m])
    pred = np.abs(d['hyb'][m])
    out = {}
    keys, counts = np.unique(g, return_counts=True)
    keys = keys[counts >= min_group]
    if len(keys) == 0:
        return None, 0
    idx = {k: np.where(g == k)[0] for k in keys}
    w = np.array([len(idx[k]) for k in keys], float)
    w /= w.sum()
    for N in NS:
        # 그룹을 크기 비례로 뽑고, 그 안에서 N 개 셀을 복원추출
        gsel = rng.choice(len(keys), size=NSIM, p=w)
        pm = np.empty(NSIM)
        pp = np.empty(NSIM)
        for i, gi in enumerate(gsel):
            ii = idx[keys[gi]]
            pick = ii[rng.integers(0, len(ii), N)]
            pm[i] = meas[pick].min()
            pp[i] = pred[pick].min()
        for lam in LAMS:
            out[(N, lam)] = float(np.mean(lam * pp > pm))
    return out, m.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--disc', required=True)
    ap.add_argument('--chg', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    res = {}
    for nm, path in (('방전', a.disc), ('충전', a.chg)):
        r, n = simulate(path, rng)
        res[nm] = r
        print(f"  {nm}  {path}  신뢰 라벨 {n}", flush=True)

    print(f"\n  == 팩 초과율 {a.tag}  ({NSIM:,} 회 모사)", flush=True)
    print(f"  {'lambda':<9}" + ''.join(f"{f'방전 N={n}':>12}" for n in NS)
          + '   ' + ''.join(f"{f'충전 N={n}':>12}" for n in NS), flush=True)
    print('  ' + '-' * (9 + 12 * 2 * len(NS) + 3), flush=True)
    for lam in LAMS:
        row = ''.join(f"{res['방전'][(n, lam)]*100:>11.1f}%" for n in NS)
        row += '   ' + ''.join(f"{res['충전'][(n, lam)]*100:>11.1f}%" for n in NS)
        print(f"  {lam:<9.2f}{row}", flush=True)


if __name__ == '__main__':
    main()
