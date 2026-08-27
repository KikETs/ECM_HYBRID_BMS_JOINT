"""SOH 궤적 — 부분 충전 구간 CNN 이 셀 하나씩 빼고 무엇을 맞히나.

각 셀은 그 셀을 한 번도 못 본 모델이 맞힌다 (나머지 다섯 셀로 학습).
파라미터 10,945 개, 시드 3 개 평균.

    python3 repro/fig_soh_traj.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, 'analysis', 'results', 'soh_pred.npz')

CELLS = ['CC', 'CC_CELL2', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S',
         'BOOST_REST']
C_TRUE = '#222222'
C_PRED = '#1f6fb4'
C_BAD = '#c0392b'


def main():
    z = np.load(PRED)
    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.92],
                          hspace=0.52, wspace=0.28)

    all_err = []
    for i, c in enumerate(CELLS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        cy, y, p = z[f'{c}_cycle'], z[f'{c}_y'], z[f'{c}_pred']
        o = np.argsort(cy)
        cy, y, p = cy[o], y[o], p[o]
        e = (p - y) * 100
        all_err.append(e)
        ax.plot(cy, y * 100, color=C_TRUE, lw=2.0, label='참 SOH', zorder=3)
        bad = np.abs(e) > 2.0
        ax.scatter(cy[~bad], p[~bad] * 100, s=17, color=C_PRED, zorder=4,
                   label='추정 (홀드아웃)')
        if bad.any():
            ax.scatter(cy[bad], p[bad] * 100, s=26, color=C_BAD, zorder=5,
                       label='오차 > 2 %p')
        ax.set_title(f'{c}   RMSE {np.sqrt(np.mean(e**2)):.2f} %p   '
                     f'편향 {e.mean():+.2f}',
                     fontsize=11, weight='bold', loc='left', pad=6)
        ax.set_ylim(64, 104)
        ax.grid(color='#ececec', lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('SOH  [%]', fontsize=10.5)
        if i >= 3:
            ax.set_xlabel('사이클', fontsize=10.5)
        if i == 0:
            ax.legend(fontsize=9, loc='lower left', framealpha=0.94)

    # 아래 폭: 오차 분포와 1:1
    axl = fig.add_subplot(gs[2, :2])
    for c in CELLS:
        cy = z[f'{c}_cycle']
        e = (z[f'{c}_pred'] - z[f'{c}_y']) * 100
        o = np.argsort(cy)
        axl.plot(cy[o], e[o], lw=1.5, alpha=0.85, label=c)
    axl.axhline(0, color=C_TRUE, lw=1.4)
    axl.axhspan(-2, 2, color='#ececec', zorder=0)
    axl.set_xlabel('사이클', fontsize=10.5)
    axl.set_ylabel('오차  [%p]', fontsize=10.5)
    axl.set_title('셀별 오차 — 회색 띠는 ±2 %p', fontsize=11,
                  weight='bold', loc='left', pad=6)
    axl.legend(fontsize=8.5, ncol=3, framealpha=0.94, loc='lower left')
    axl.grid(color='#ececec', lw=0.8)
    for sp in ('top', 'right'):
        axl.spines[sp].set_visible(False)

    axr = fig.add_subplot(gs[2, 2])
    yy = np.concatenate([z[f'{c}_y'] for c in CELLS]) * 100
    pp = np.concatenate([z[f'{c}_pred'] for c in CELLS]) * 100
    axr.scatter(yy, pp, s=14, color=C_PRED, alpha=0.7)
    lo, hi = 64, 104
    axr.plot([lo, hi], [lo, hi], color=C_TRUE, lw=1.4)
    axr.set_xlim(lo, hi)
    axr.set_ylim(lo, hi)
    axr.set_xlabel('참 SOH  [%]', fontsize=10.5)
    axr.set_ylabel('추정  [%]', fontsize=10.5)
    e = pp - yy
    axr.set_title(f'전체  RMSE {np.sqrt(np.mean(e**2)):.2f} %p  '
                  f'편향 {e.mean():+.2f}',
                  fontsize=11, weight='bold', loc='left', pad=6)
    axr.grid(color='#ececec', lw=0.8)
    for sp in ('top', 'right'):
        axr.spines[sp].set_visible(False)

    fig.suptitle('SOH — 부분 충전 구간 CNN, 셀 하나씩 빼고',
                 fontsize=15, weight='bold', y=0.988)
    fig.text(0.5, 0.938,
             '파라미터 10,945 개, 시드 3 개 평균.  각 셀은 그 셀을 한 번도 '
             '못 본 모델이 맞힌다.  '
             '편향이 +0.10 %p 로 위험한 쪽이다 — SOH 를 높게 보면 SOP 가 '
             '낙관적이 된다 (29.1).',
             ha='center', fontsize=10.4, color='#555555')
    fig.subplots_adjust(left=0.058, right=0.982, top=0.872, bottom=0.075)

    out = os.path.join(ROOT, 'results_fig_soh_traj.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')
    print(f'    전체 RMSE {np.sqrt(np.mean(e**2))/100:.4f}  '
          f'편향 {e.mean()/100:+.4f}')


if __name__ == '__main__':
    main()
