"""전압 RMSE 표를 만든다 — 지금까지 그림과 문서에 손으로 박혀 있던 값.

왜 따로 두는가: `fig_ladder.py` 가 전압 RMSE 여섯 개 x 두 방향을 상수로
갖고 있었다.  표가 바뀌어도 그림은 안 바뀌고, 그림의 숫자가 맞는지
확인할 방법도 없었다.  verify.py 가 볼 수 있게 표로 뺀다.

규약은 트림 표와 같다 — **셀별 RMSE 를 낸 뒤 평균** 한다 (통합 RMSE 가
아니다).  이것을 안 맞추면 A0 가 85.36 이 아니라 87.24 로 나온다.

    python3 repro/run_voltage.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')
sys.path.insert(0, ANALYSIS)

# (표에 쓸 이름, 방전 run 디렉터리, 충전 run 디렉터리)
# A0 는 디렉터리의 pred_A0_*.npz 에 들어 있다 (k=1 판).
RUNS = [
    ('A0', 'runs_trim_a8', 'runs_trim_a8_chg'),
    ('direct', 'runs_trim_direct', 'runs_trim_direct_chg'),
    ('shrink', 'runs_trim_shrink', 'runs_trim_shrink_chg'),
    ('A8', 'runs_trim_a8', 'runs_trim_a8_chg'),
    ('A3', 'runs_trim_v2', 'runs_trim_chg_v2'),
    ('RLS', 'runs_trim_rls', 'runs_trim_rls_chg'),
]


def per_cell_rmse(run_dir, rung):
    """셀별 전압 RMSE (mV).  rung='A0' 이면 보정 없는 판."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ANALYSIS, run_dir,
                                            'pred_A*_*.npz'))):
        base = os.path.basename(f)
        if base.startswith('pred_A0_'):
            continue
        cell = base.split('_', 2)[2][:-4]
        z = np.load(f, allow_pickle=True)
        if rung == 'A0':
            # 보정 없는 판은 학습 때 이미 저장돼 있다 (`base`).  직접 다시
            # 계산하면 sop_trim.py 와 미세하게 어긋날 수 있으므로 그것을 쓴다.
            p = z['base']
        elif 'pred' in z:
            p = z['pred']
        else:
            I, NOM = z['I'].astype(float), z['NOM'].astype(float)
            kf, ks = z['k_f'].astype(float), z['k_s'].astype(float)
            p = np.stack([I * (kf * NOM[:, 0] + ks * NOM[:, 1]),
                          I * (kf * NOM[:, 2] + ks * NOM[:, 3])], 1)
        out[cell] = float(np.sqrt(np.mean((p - z['Y']) ** 2)) * 1000)
    return out


def main():
    rows = []
    for name, dis, chg in RUNS:
        for direction, run in (('discharge', dis), ('charge', chg)):
            if not os.path.isdir(os.path.join(ANALYSIS, run)):
                continue
            try:
                per = per_cell_rmse(run, 'A0' if name == 'A0' else name)
            except Exception as e:                       # noqa: BLE001
                print(f'  건너뜀 {name}/{direction}: {e}', flush=True)
                continue
            if not per:
                continue
            mean = float(np.mean(list(per.values())))
            rows.append([direction, name, len(per), f'{mean:.2f}']
                        + [f'{per[c]:.2f}' for c in sorted(per)])
            print(f'  {direction:<10}{name:<8}{mean:>9.2f} mV   '
                  f'({len(per)} 셀)', flush=True)

    cells = sorted(per_cell_rmse('runs_trim_a8', 'A8'))
    os.makedirs(TABLES, exist_ok=True)
    out = os.path.join(TABLES, 'voltage.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['direction', 'method', 'n_cells', 'rmse_mV'] + cells)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(out, ROOT)}  ({len(rows)} 행)', flush=True)
    print('  규약: 셀별 RMSE 의 평균 (통합 RMSE 가 아님 — sop_trim.py 와 같게)',
          flush=True)


if __name__ == '__main__':
    main()
