"""파라미터 사다리 — 0 개, 1 개, 4 개, 26 개.

직접 대입(sop_baselines)이 실패한 이유는 부호가 아니라 크기다.  dR_fast 는
주행 전류(대개 5 A 미만)에서 재는데 SOP 는 30 A 부근의 저항을 필요로 하고,
R 은 전류에 크게 의존한다.  그래서 저전류 잔차 기울기가 1:1 로 전이되지
않는다.  학습된 사상이 하는 일이 그 **전달비** 다.

그렇다면 전달비 하나면 되는가?  축소 계수 alpha 를 하나 두고

    k_f = exp(alpha_f * dR_fast / R_fast_nom)
    k_s = exp(alpha_s * dR_slow / R_slow_nom)

를 셀 하나씩 빼고 맞춘다.  파라미터 두 개짜리 방법이다.

사다리:  A0 0 개  ->  축소 2 개  ->  A8 4 개  ->  A3 26 개
"""
import numpy as np

from sop_trim import load_cells, KF_SPAN, KS_SPAN

KF_LO, KF_HI = np.exp(-KF_SPAN), np.exp(KF_SPAN)
KS_LO, KS_HI = np.exp(-KS_SPAN), np.exp(KS_SPAN)


def ratios(d):
    X = d['X'].astype(float)
    rf = X[:, 0] / np.where(np.abs(X[:, 10]) < 1e-6, np.nan, X[:, 10])
    rs = X[:, 1] / np.where(np.abs(X[:, 11]) < 1e-6, np.nan, X[:, 11])
    return np.nan_to_num(rf), np.nan_to_num(rs)


def pred(d, af, asl):
    rf, rs = ratios(d)
    kf = np.clip(np.exp(af * rf), KF_LO, KF_HI)
    ks = np.clip(np.exp(asl * rs), KS_LO, KS_HI)
    NOM, I = d['NOM'].astype(float), d['I'].astype(float)
    return np.stack([I * (kf * NOM[:, 0] + ks * NOM[:, 1]),
                     I * (kf * NOM[:, 2] + ks * NOM[:, 3])], 1)


def sse(ds, af, asl):
    return sum(float(np.sum((pred(d, af, asl) - d['Y'].astype(float)) ** 2))
               for d in ds)


def fit(ds, grid=np.linspace(0.0, 1.0, 51)):
    best, ba, bb = np.inf, 0.0, 0.0
    for af in grid:
        for asl in grid:
            s = sse(ds, af, asl)
            if s < best:
                best, ba, bb = s, af, asl
    return ba, bb


def main():
    cells = load_cells()
    names = sorted(cells)
    print("  축소 계수를 셀 하나씩 빼고 맞춘다 "
          "(k = exp(alpha * dR/R_nom), 파라미터 2 개)\n", flush=True)
    print(f"  {'홀드아웃':<20}{'맞춘 a_f':>10}{'맞춘 a_s':>10}"
          f"{'A0':>10}{'직접(a=1 상당)':>16}{'축소':>10}", flush=True)
    print('  ' + '-' * 76, flush=True)
    acc0, accd, accs, accp = [], [], [], []
    for c in names:
        tr = [cells[o] for o in names if o != c]
        af, asl = fit(tr)
        d = cells[c]
        Y = d['Y'].astype(float)
        NOM, I = d['NOM'].astype(float), d['I'].astype(float)
        p0 = np.stack([I * (NOM[:, 0] + NOM[:, 1]),
                       I * (NOM[:, 2] + NOM[:, 3])], 1)
        pd_ = pred(d, 1.0, 1.0)            # exp 형태의 alpha=1
        rf, rs = ratios(d)
        kfp = np.clip(1.0 + rf, KF_LO, KF_HI)
        ksp = np.clip(1.0 + rs, KS_LO, KS_HI)
        pp = np.stack([I * (kfp * NOM[:, 0] + ksp * NOM[:, 1]),
                       I * (kfp * NOM[:, 2] + ksp * NOM[:, 3])], 1)
        ps = pred(d, af, asl)
        acc0.append((p0 - Y).ravel())
        accd.append((pd_ - Y).ravel())
        accs.append((ps - Y).ravel())
        accp.append((pp - Y).ravel())
        r = lambda p: float(np.sqrt(np.mean((p - Y) ** 2)) * 1000)
        print(f"  {c:<20}{af:>10.3f}{asl:>10.3f}{r(p0):>9.2f}m"
              f"{r(pd_):>15.2f}m{r(ps):>9.2f}m", flush=True)
    # 트림 표와 같은 규약 — 셀별 RMSE 를 낸 뒤 평균한다 (통합 RMSE 가 아님)
    g = lambda a: float(np.mean([np.sqrt(np.mean(x ** 2)) for x in a]) * 1000)
    print(f"  {'전체':<20}{'':>10}{'':>10}{g(acc0):>9.2f}m"
          f"{g(accd):>15.2f}m{g(accs):>9.2f}m", flush=True)
    print("\n  == 파라미터 사다리 (같은 셀 홀드아웃, 전압 RMSE)", flush=True)
    rows = [('A0  보정 없음', 0, g(acc0)),
            ('직접 대입  k = 1 + dR/R', 0, g(accp)),
            ('직접 대입  k = exp(dR/R)', 0, g(accd)),
            ('축소 계수  k = exp(a dR/R)', 2, g(accs)),
            ('A8  dR_fast 하나', 4, 62.81),
            ('A3  12 특징', 26, 58.76)]
    print(f"  {'방법':<26}{'파라미터':>9}{'RMSE':>10}{'A0 대비':>10}", flush=True)
    for nm, p, v in rows:
        print(f"  {nm:<26}{p:>9}{v:>9.2f}m{(v/g(acc0)-1)*100:>+9.1f}%", flush=True)


if __name__ == '__main__':
    main()
