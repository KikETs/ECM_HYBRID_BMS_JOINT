"""SOC 궤적 — 순환 벤치가 무엇을 보고 있었나.

한 장에 두 가지를 보인다.

  위  틀어짐이 없을 때.  순수 전류 적분이 정답 라벨 위에 정확히 얹힌다.
      라벨이 SOC = 1 + Ah/3.0 이고 필터 예측이 soc + I dt/3600/3.0 이라
      같은 식이기 때문이다.  이 벤치에서는 전압을 쓰는 쪽이 질 수밖에 없다.

  아래 전류 센서에 0.1 A 옵셋을 넣었을 때.  적분은 갈 곳 없이 표류하고,
      전압 보정이 그것을 잡는다.  칼만 필터를 쓰는 이유가 여기 있고,
      순환 벤치는 이 장면을 볼 수 없었다.

    python3 repro/fig_soc_traj.py [--cell CC --pick 4]
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

IBIAS = 0.10
C_TRUE = '#222222'
C_OPEN = '#c0392b'
C_EKF = '#1f6fb4'


def rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='CC')
    ap.add_argument('--pick', type=int, default=4,
                    help='그 셀의 몇 번째 런 (0 = 신품, 5 = 가장 노화)')
    a = ap.parse_args()

    from ekf_soc import run as ekf_run
    runs = pickle.load(open(os.path.join(ANALYSIS, 'results', 'soc_runs.pkl'),
                            'rb'))
    rs = [r for r in runs if r['cell'] == a.cell]
    r = rs[a.pick]
    t = np.arange(len(r['soc'])) / 3600.0
    rv = rvolt(r['soh'])
    kw = dict(gamma=0.0, i_gate=1.0, rest_hold_s=30.0)

    def go(bias, open_loop):
        est, _ = ekf_run(r['sd'], r['sc'], r['soh'], r['I'] + bias, r['V'],
                         r['T'], float(r['soc'][0]),
                         1e4 if open_loop else rv, **kw)
        return est

    tr = r['soc']
    series = {
        'clean_open': go(0.0, True),
        'clean_ekf': go(0.0, False),
        'bias_open': go(IBIAS, True),
        'bias_ekf': go(IBIAS, False),
    }
    rmse = {k: np.sqrt(np.mean((v - tr) ** 2)) * 100
            for k, v in series.items()}

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6),
                             gridspec_kw={'width_ratios': [2.6, 1],
                                          'hspace': 0.34, 'wspace': 0.20})

    for row, (tag, sub) in enumerate((
            ('clean', '틀어짐 없음 — 순환 벤치가 보던 것'),
            ('bias', f'전류 센서 옵셋 +{IBIAS:.2f} A — 실차의 조건'))):
        ax, axe = axes[row]
        ax.plot(t, tr * 100, color=C_TRUE, lw=2.4, label='참 SOC (라벨)',
                zorder=4)
        ax.plot(t, series[f'{tag}_open'] * 100, color=C_OPEN, lw=1.7,
                ls=(0, (5, 2)),
                label=f'순수 전류 적분   RMSE {rmse[f"{tag}_open"]:.2f} %p',
                zorder=3)
        ax.plot(t, series[f'{tag}_ekf'] * 100, color=C_EKF, lw=1.7,
                label=f'EKF 채택 구성   RMSE {rmse[f"{tag}_ekf"]:.2f} %p',
                zorder=3)
        ax.set_ylabel('SOC  [%]', fontsize=11)
        ax.set_title(sub, fontsize=12.5, weight='bold', loc='left', pad=8)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.94)
        ax.grid(color='#ececec', lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if row == 1:
            ax.set_xlabel('시간  [h]', fontsize=11)

        # 오른쪽: 오차만
        for k, c, ls in ((f'{tag}_open', C_OPEN, (0, (5, 2))),
                         (f'{tag}_ekf', C_EKF, '-')):
            axe.plot(t, (series[k] - tr) * 100, color=c, lw=1.6, ls=ls)
        axe.axhline(0, color=C_TRUE, lw=1.6, zorder=4)
        axe.set_ylabel('오차  [%p]', fontsize=10.5)
        axe.set_title('오차', fontsize=11, loc='left', pad=8, color='#555555')
        axe.grid(color='#ececec', lw=0.8)
        for sp in ('top', 'right'):
            axe.spines[sp].set_visible(False)
        if row == 1:
            axe.set_xlabel('시간  [h]', fontsize=10.5)

    # 오차 패널의 범위는 실제 값에서 잡는다 (잘라 내면 오해를 준다)
    for row, tag in ((0, 'clean'), (1, 'bias')):
        e = np.concatenate([(series[f'{tag}_{k}'] - tr) * 100
                            for k in ('open', 'ekf')])
        lo, hi = float(e.min()), float(e.max())
        pad = max(0.35, (hi - lo) * 0.10)
        axes[row][1].set_ylim(lo - pad, hi + pad)

    def note(ax, txt, xf, yf, color):
        ax.text(xf, yf, txt, transform=ax.transAxes, fontsize=10.5,
                color=color, weight='bold', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.42', fc='white',
                          ec=color, lw=1.1, alpha=0.93), zorder=6)

    note(axes[0][0], '적분이 라벨 위에 정확히 얹힌다 — 두 식이 같다',
         0.05, 0.13, C_OPEN)
    note(axes[1][0], '옵셋을 적분하니 표류한다', 0.05, 0.13, C_OPEN)
    note(axes[1][0], '전압 보정이 되돌린다', 0.05, 0.27, C_EKF)
    note(axes[0][1], '휴지마다 되돌림', 0.05, 0.12, C_EKF)

    fig.suptitle('SOC 벤치가 순환이었다 — 라벨과 필터 예측이 같은 식이다',
                 fontsize=15, weight='bold', y=0.982)
    fig.text(0.5, 0.945,
             f"{a.cell},  SOH {r['soh']:.2f},  드라이브 사이클 "
             f"{len(t) / 3600:.1f} 시간.   "
             '라벨 SOC = 1 + Ah/3.0,  예측 soc + I·dt/3600/3.0',
             ha='center', fontsize=10.6, color='#555555')
    fig.subplots_adjust(left=0.062, right=0.982, top=0.885, bottom=0.085)

    out = os.path.join(ROOT, 'results_fig_soc_traj.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')
    for k, v in rmse.items():
        print(f'    {k:<12} RMSE {v:6.2f} %p')


if __name__ == '__main__':
    main()
