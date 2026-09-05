"""문헌의 표준 비교군 — 온라인 RLS 로 저항 배수를 적응시킨다.

WHY THIS IS THE RIGHT BASELINE
    트림은 12 개 특징에서 두 배수 (k_f, k_s) 를 만드는 26 파라미터 회귀다.
    리뷰어가 물을 당연한 질문은 "오프라인으로 배울 것 없이 온라인 RLS 로
    저항을 갱신하면 되지 않나" 이고, 그것이 적응형 SOP 의 표준 기법이다.

    다행히 정면 비교가 된다.  트림의 예측은 k 에 대해 선형이기 때문이다:

        dV(2 s)  = I * (k_f * nf2  + k_s * ns2 )
        dV(10 s) = I * (k_f * nf10 + k_s * ns10)

    그래서 RLS 는 같은 출력을 같은 관측(측정 dV) 에서 학습 없이 추정한다.

FAIRNESS
    RLS 쪽이 유리하게 놓았다 — 트림은 홀드아웃 셀을 한 번도 못 보고
    나머지 다섯 셀에서만 배우는데, RLS 는 **그 셀 자신의 과거** 를 쓴다.
    인과성은 pulse 단위로 지킨다: 한 pulse 의 예측은 모두 그 pulse 의
    관측을 쓰기 전에 만든다 (rls_cell 참조).
    허수아비가 되지 않도록 망각계수도 훑는다.

    k 의 범위도 트림과 같게 자른다 (exp(+-0.470), exp(+-0.588)).
"""
import argparse

import numpy as np

from sop_trim import load_cells, KF_SPAN, KS_SPAN

KF_LO, KF_HI = np.exp(-KF_SPAN), np.exp(KF_SPAN)
KS_LO, KS_HI = np.exp(-KS_SPAN), np.exp(KS_SPAN)


def rls_cell(d, ff=0.999, p0=1e2, warm=0):
    """PULSE-causal RLS.  Predict every row of a pulse before updating on it.

    THE UNIT IS THE PHYSICAL PULSE, NOT THE ROW
        sop_trim_dataset.py pairs each measured pulse with twelve preceding
        drive blocks, so one physical pulse is twelve rows carrying the SAME
        target (verified: CC has 896 pulses in 10752 rows, and no key holds
        two different Y).  The first version of this function walked rows:
        predict, update, next row.  Rows 2..12 of a pulse were therefore
        predicted by a theta already updated up to 22 times on that pulse's
        own answer.

        Measured with a probe that perturbs one pulse's Y by 50 mV and asks
        whether that pulse's own predictions move: they moved on 6 of 6
        probed pulses, by up to 4.81 mV.  Under this version they do not move
        at all.

    ORDER, AND WHAT IS NOT ASSUMED
        Pulses are ordered by (cycle, -SOC): within a cycle the HPPC steps SOC
        down monotonically, which the data shows.  The order of the two
        current ranks at one SOC is NOT established anywhere, so pulses
        sharing (cycle, SOC) are treated as SIMULTANEOUS - every prediction at
        that step is made before any update from that step.  Guessing an order
        would be inventing causality the files do not record.

    WHAT THE FIX COSTS
        Little here, and that is a property of the shipped setting rather than
        of the defect.  sop_baseline_fill.py runs at ff = 1.0, where P shrinks
        monotonically and a twelfth update on the same target adds almost
        nothing: pooled voltage RMSE moves 49.89 -> 50.02 mV on discharge and
        33.59 -> 33.69 mV on charge.  At ff = 0.999 the same defect is worth
        32.06 -> 44.14 mV.  Fixed regardless: a baseline that reads its own
        answer is not a baseline, whatever the forgetting factor happens to be.

    반환: pred (n,2) 예측 dV,  K (n,2) 그 시점의 (k_f, k_s)
    """
    NOM, I, Y = d['NOM'].astype(float), d['I'].astype(float), d['Y'].astype(float)
    n = len(I)
    cyc = np.asarray(d['cycle']).astype(int)
    soc = np.round(np.asarray(d['SOC'], float), 6)
    rank = np.asarray(d['rank'])

    steps = {}
    for i in range(n):
        steps.setdefault((cyc[i], -soc[i]), {}) \
             .setdefault((cyc[i], soc[i], str(rank[i])), []).append(i)

    th = np.array([1.0, 1.0])
    P = np.eye(2) * p0
    pred = np.empty((n, 2))
    K = np.empty((n, 2))
    seen = 0

    for step in sorted(steps):
        pulses = steps[step]
        kf = min(max(th[0], KF_LO), KF_HI)
        ks = min(max(th[1], KS_LO), KS_HI)
        for rows in pulses.values():                   # 예측이 먼저 — 전부
            for t in rows:
                H = np.array([[I[t] * NOM[t, 0], I[t] * NOM[t, 1]],
                              [I[t] * NOM[t, 2], I[t] * NOM[t, 3]]])
                K[t] = (kf, ks)
                pred[t] = H @ np.array([kf, ks])
        for rows in pulses.values():                   # 그 다음 갱신 — pulse 당 한 번
            seen += 1
            if seen <= warm:
                continue
            t = rows[0]            # 한 pulse 의 12 행은 Y, NOM, I 가 모두 같다
            H = np.array([[I[t] * NOM[t, 0], I[t] * NOM[t, 1]],
                          [I[t] * NOM[t, 2], I[t] * NOM[t, 3]]])
            for r in range(2):
                h = H[r]
                den = ff + h @ P @ h
                if den <= 0 or not np.isfinite(den):
                    continue
                g = P @ h / den
                th = th + g * (Y[t, r] - h @ th)
                P = (P - np.outer(g, h @ P)) / ff
                if not np.isfinite(P).all():
                    P = np.eye(2) * p0
            th = np.clip(th, [KF_LO, KS_LO], [KF_HI, KS_HI])
    return pred, K


def rmse_mv(pred, Y):
    return float(np.sqrt(np.mean((pred - Y) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trim', default='runs_trim_v2',
                    help='비교할 트림 run 디렉터리')
    a = ap.parse_args()
    cells = load_cells()

    FFS = [1.0, 0.9999, 0.999, 0.99, 0.95]
    print(f"  {'홀드아웃':<20}{'n':>8}{'A0 (k=1)':>11}"
          + ''.join(f"{f'RLS ff={f}':>13}" for f in FFS), flush=True)
    print('  ' + '-' * (39 + 13 * len(FFS)), flush=True)
    tot = {f: [] for f in FFS}
    tot0 = []
    KEEP = {}
    for c in sorted(cells):
        d = cells[c]
        Y, NOM, I = d['Y'], d['NOM'].astype(float), d['I'].astype(float)
        a0 = np.stack([I * (NOM[:, 0] + NOM[:, 1]),
                       I * (NOM[:, 2] + NOM[:, 3])], 1)
        tot0.append((a0 - Y).ravel())
        row = ''
        for f in FFS:
            p, K = rls_cell(d, ff=f)
            tot[f].append((p - Y).ravel())
            row += f"{rmse_mv(p, Y):>12.2f}m"
            KEEP[(c, f)] = K
        print(f"  {c:<20}{len(I):>8,}{rmse_mv(a0, Y):>10.2f}m{row}", flush=True)
    e0 = np.concatenate(tot0)
    row = ''.join(f"{float(np.sqrt(np.mean(np.concatenate(tot[f])**2))*1000):>12.2f}m"
                  for f in FFS)
    print(f"  {'전체':<20}{'':>8}{float(np.sqrt(np.mean(e0**2))*1000):>10.2f}m{row}",
          flush=True)

    best = min(FFS, key=lambda f: np.mean(np.concatenate(tot[f]) ** 2))
    print(f"\n  최적 망각계수 {best}", flush=True)
    print("  참고 — 트림(같은 데이터, 셀 홀드아웃): A0 85.36m -> A3 58.76m, "
          "A8 62.81m", flush=True)

    np.savez('/tmp/rls_k.npz',
             **{f'{c}': KEEP[(c, best)] for c in sorted(cells)},
             **{f'{c}_cycle': cells[c]['cycle'] for c in sorted(cells)},
             ff=best)
    print("  -> /tmp/rls_k.npz  (최적 망각계수의 k 를 저장)", flush=True)


if __name__ == '__main__':
    main()
