"""문헌 비교군 정리 — 트림은 무엇을 이기는가.

네 판을 같은 데이터·같은 셀 홀드아웃 위에 세운다.

  A0        보정 없음.  명목 표를 그대로 (k = 1).
            = HPPC 저항표 기반 고전 SOP (FreedomCAR / Plett 의 ECM 역산).

  직접 대입   dR_fast 를 배수로 그대로 쓴다:  k_f = 1 + dR_fast / R_fast_nom.
            **학습이 없다.**  dR_fast = EW{I*r} / EW{I*I} 는 잔차를 전류에
            회귀시킨 지수가중 최소자승이므로, 이것이 곧 주행 전압으로 도는
            온라인 RLS 다 — 즉 적응형 SOP 의 표준 기법 그 자체다.

  트림       같은 dR_fast 를 포함한 특징에서 배수를 만드는 26 파라미터 회귀.
            A3 = 12 특징, A8 = dR_fast 하나.

  HPPC-RLS  같은 셀의 **과거 HPPC 측정 dV** 로 RLS 를 돌린다.  주기적
            특성화를 실제로 수행하는 시스템의 상한이다.  실차 BMS 는 HPPC 를
            돌리지 않으므로 배포 가능한 비교군이 아니다 — 상한으로만 읽는다.

읽는 법: 직접 대입을 트림이 이겨야 "배운 사상"이 값을 한다.  못 이기면
26 파라미터는 불필요하고 dR_fast 를 그대로 쓰면 된다.
"""
import argparse

import numpy as np

from sop_trim import load_cells, KF_SPAN, KS_SPAN
from sop_rls import rls_cell

KF_LO, KF_HI = np.exp(-KF_SPAN), np.exp(KF_SPAN)
KS_LO, KS_HI = np.exp(-KS_SPAN), np.exp(KS_SPAN)
I_DR_FAST, I_DR_SLOW, I_RF_NOM, I_RS_NOM = 0, 1, 10, 11


def dv(K, NOM, I):
    """k -> 예측 dV (2 s, 10 s)."""
    return np.stack([I * (K[:, 0] * NOM[:, 0] + K[:, 1] * NOM[:, 1]),
                     I * (K[:, 0] * NOM[:, 2] + K[:, 1] * NOM[:, 3])], 1)


def direct(d, clip=True):
    """dR_fast / R_fast_nom 을 배수로 그대로."""
    X = d['X'].astype(float)
    kf = 1.0 + X[:, I_DR_FAST] / np.where(np.abs(X[:, I_RF_NOM]) < 1e-6,
                                          np.nan, X[:, I_RF_NOM])
    ks = 1.0 + X[:, I_DR_SLOW] / np.where(np.abs(X[:, I_RS_NOM]) < 1e-6,
                                          np.nan, X[:, I_RS_NOM])
    kf = np.nan_to_num(kf, nan=1.0)
    ks = np.nan_to_num(ks, nan=1.0)
    if clip:
        kf = np.clip(kf, KF_LO, KF_HI)
        ks = np.clip(ks, KS_LO, KS_HI)
    return np.stack([kf, ks], 1)


def rmse_mv(p, Y):
    return float(np.sqrt(np.mean((p - Y) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rls-ff', type=float, default=0.95)
    a = ap.parse_args()
    cells = load_cells()

    cols = ['A0 (k=1)', '직접 대입', '직접 (자르기 없음)', f'HPPC-RLS ff={a.rls_ff}']
    print(f"  {'홀드아웃':<20}{'n':>8}" + ''.join(f"{c:>18}" for c in cols),
          flush=True)
    print('  ' + '-' * (28 + 18 * len(cols)), flush=True)
    acc = {c: [] for c in cols}
    KDIR = {}
    for c in sorted(cells):
        d = cells[c]
        Y, NOM, I = d['Y'].astype(float), d['NOM'].astype(float), d['I'].astype(float)
        one = np.ones((len(I), 2))
        p_rls, _ = rls_cell(d, ff=a.rls_ff)
        KDIR[c] = direct(d)
        preds = [dv(one, NOM, I), dv(KDIR[c], NOM, I),
                 dv(direct(d, clip=False), NOM, I), p_rls]
        for col, p in zip(cols, preds):
            acc[col].append((p - Y).ravel())
        print(f"  {c:<20}{len(I):>8,}"
              + ''.join(f"{rmse_mv(p, Y):>17.2f}m" for p in preds), flush=True)
    print(f"  {'전체':<20}{'':>8}"
          + ''.join(f"{float(np.sqrt(np.mean(np.concatenate(acc[c])**2))*1000):>17.2f}m"
                    for c in cols), flush=True)
    print(f"\n  트림 (같은 홀드아웃):  A3 12 특징 58.76m   A8 dR_fast 하나 62.81m",
          flush=True)

    print("\n  == 직접 대입의 k 분포 (트림 한계 "
          f"k_f {KF_LO:.3f}~{KF_HI:.3f}, k_s {KS_LO:.3f}~{KS_HI:.3f})", flush=True)
    print(f"  {'셀':<20}{'k_f 중앙':>10}{'k_f 5~95%':>18}{'k_f 포화':>10}"
          f"{'k_s 중앙':>10}{'k_s 포화':>10}", flush=True)
    for c in sorted(cells):
        K = KDIR[c]
        raw = direct(cells[c], clip=False)
        sf = np.mean((raw[:, 0] <= KF_LO) | (raw[:, 0] >= KF_HI)) * 100
        ss = np.mean((raw[:, 1] <= KS_LO) | (raw[:, 1] >= KS_HI)) * 100
        q = np.percentile(K[:, 0], [5, 95])
        print(f"  {c:<20}{np.median(K[:, 0]):>10.3f}"
              f"{f'{q[0]:.2f}~{q[1]:.2f}':>18}{sf:>9.1f}%"
              f"{np.median(K[:, 1]):>10.3f}{ss:>9.1f}%", flush=True)

    np.savez('/tmp/k_direct.npz',
             **{c: KDIR[c] for c in sorted(cells)},
             **{f'{c}_cycle': cells[c]['cycle'] for c in sorted(cells)})
    print("\n  -> /tmp/k_direct.npz", flush=True)


if __name__ == '__main__':
    main()
