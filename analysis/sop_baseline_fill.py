"""32.6 의 두 빈칸을 채운다.

(1) 충전 방향의 축소 계수 / 직접 대입 판을 만든다.
(2) HPPC-RLS 의 k 를 사이클 단위로 모아 전류로 옮긴다.

모두 트림 디렉터리 형식(`pred_A3_<cell>.npz` 에 k_f, k_s, cycle)으로 내보내
`eval_sop_amps.py --trim` 이 그대로 읽게 한다.
"""
import argparse
import os

import numpy as np

import sop_trim
from sop_trim import KF_SPAN, KS_SPAN
from sop_rls import rls_cell

KF_LO, KF_HI = np.exp(-KF_SPAN), np.exp(KF_SPAN)
KS_LO, KS_HI = np.exp(-KS_SPAN), np.exp(KS_SPAN)


def ratios(d):
    X = d['X'].astype(float)
    rf = X[:, 0] / np.where(np.abs(X[:, 10]) < 1e-6, np.nan, X[:, 10])
    rs = X[:, 1] / np.where(np.abs(X[:, 11]) < 1e-6, np.nan, X[:, 11])
    return np.nan_to_num(rf), np.nan_to_num(rs)


def pred_dv(kf, ks, d):
    NOM, I = d['NOM'].astype(float), d['I'].astype(float)
    return np.stack([I * (kf * NOM[:, 0] + ks * NOM[:, 1]),
                     I * (kf * NOM[:, 2] + ks * NOM[:, 3])], 1)


def fit_alpha(ds, grid=np.linspace(0.0, 1.0, 51)):
    best, ba, bb = np.inf, 0.0, 0.0
    for af in grid:
        for asl in grid:
            s = 0.0
            for d in ds:
                rf, rs = ratios(d)
                p = pred_dv(np.clip(np.exp(af * rf), KF_LO, KF_HI),
                            np.clip(np.exp(asl * rs), KS_LO, KS_HI), d)
                s += float(np.sum((p - d['Y'].astype(float)) ** 2))
            if s < best:
                best, ba, bb = s, af, asl
    return ba, bb


def dump(tag, cell, kf, ks, d):
    os.makedirs(tag, exist_ok=True)
    np.savez(os.path.join(tag, f'pred_A3_{cell}.npz'),
             k_f=np.clip(kf, KF_LO, KF_HI).astype(np.float32),
             k_s=np.clip(ks, KS_LO, KS_HI).astype(np.float32),
             cycle=d['cycle'].astype(np.int64), SOC=d['SOC'], SOH=d['SOH'],
             rank=d['rank'], exc=d['exc'], I=d['I'], NOM=d['NOM'], Y=d['Y'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cache/trim')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--rls-ff', type=float, default=1.0,
                    help='HPPC-RLS 의 망각계수.  1.0 = 전 이력 (32.4 의 '
                         '49.92 mV).  0.95 는 사실상 국소 보간이라 배포 '
                         '해석이 어렵다.')
    a = ap.parse_args()

    cells = sop_trim.load_cells(a.data)
    names = sorted(cells)
    sx = a.suffix
    print(f"  데이터 {a.data}   셀 {len(names)}개\n", flush=True)
    print(f"  {'홀드아웃':<20}{'n':>8}{'a_f':>7}{'a_s':>7}"
          f"{'A0':>10}{'직접':>10}{'축소':>10}{'RLS':>10}", flush=True)
    print('  ' + '-' * 82, flush=True)
    acc = {k: [] for k in ('A0', '직접', '축소', 'RLS')}
    for c in names:
        d = cells[c]
        Y = d['Y'].astype(float)
        rf, rs = ratios(d)
        af, asl = fit_alpha([cells[o] for o in names if o != c])
        one = np.ones(len(rf))
        p_rls, K = rls_cell(d, ff=a.rls_ff)

        dump(f'runs_trim_direct{sx}', c, 1.0 + rf, 1.0 + rs, d)
        dump(f'runs_trim_shrink{sx}', c, np.exp(af * rf), np.exp(asl * rs), d)
        dump(f'runs_trim_rls{sx}', c, K[:, 0], K[:, 1], d)

        ps = [pred_dv(one, one, d),
              pred_dv(np.clip(1 + rf, KF_LO, KF_HI),
                      np.clip(1 + rs, KS_LO, KS_HI), d),
              pred_dv(np.clip(np.exp(af * rf), KF_LO, KF_HI),
                      np.clip(np.exp(asl * rs), KS_LO, KS_HI), d),
              p_rls]
        for k, p in zip(acc, ps):
            acc[k].append((p - Y).ravel())
        r = lambda p: float(np.sqrt(np.mean((p - Y) ** 2)) * 1000)
        print(f"  {c:<20}{len(rf):>8,}{af:>7.3f}{asl:>7.3f}"
              + ''.join(f"{r(p):>9.2f}m" for p in ps), flush=True)
    g = lambda k: float(np.mean([np.sqrt(np.mean(x ** 2)) for x in acc[k]]) * 1000)
    print(f"  {'전체':<20}{'':>8}{'':>7}{'':>7}"
          + ''.join(f"{g(k):>9.2f}m" for k in acc), flush=True)
    print(f"\n  -> runs_trim_direct{sx} / runs_trim_shrink{sx} / runs_trim_rls{sx}",
          flush=True)


if __name__ == '__main__':
    main()
