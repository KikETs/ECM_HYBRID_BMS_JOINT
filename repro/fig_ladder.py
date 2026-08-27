"""논문의 중심 그림 — 전압으로 고르면 순위가 뒤집힌다.

같은 여섯 판을 왼쪽은 전압 RMSE 로, 오른쪽은 배치를 결정하는 값(쓸 수 있는
전류)으로 세운다.  선이 교차하면 그 두 지표가 순위를 다르게 매긴다는 뜻이다.

    python3 repro/fig_ladder.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

def voltage():
    """전압 RMSE 를 표에서 읽는다.

    전에는 이 파일에 상수 열두 개로 박혀 있었다.  표가 바뀌어도 그림은
    안 바뀌고, 그림의 숫자가 맞는지 확인할 방법도 없었다.
    """
    # voltage.csv 는 영문 이름을, ladder.csv 매핑은 ORDER 의 키를 쓴다.
    ALIAS = {'A0': 'A0', 'direct': '직접', 'shrink': '축소',
             'A8': 'A8', 'A3': 'A3', 'RLS': 'RLS'}
    rows = list(csv.DictReader(
        open(os.path.join(TABLES, 'voltage.csv'), encoding='utf-8')))
    out = {}
    for r in rows:
        out.setdefault(r['direction'], {})[ALIAS[r['method']]] = \
            float(r['rmse_mV'])
    return out


KEY = {'A0  보정 없음': 'A0', '직접 대입': '직접', '축소 계수': '축소',
       'A8  dR_fast 하나': 'A8', 'A3  12 특징': 'A3',
       '[상한] HPPC-RLS': 'RLS'}
LABEL = {'A0': 'A0  보정 없음\n(고전 HPPC 표)', '직접': '직접 대입\n(RLS 그대로)',
         '축소': '축소 계수\n(2개)', 'A8': 'A8\n(4개)', 'A3': 'A3\n(26개)',
         'RLS': 'HPPC-RLS\n(상한)'}
COLOR = {'A0': '#8c8c8c', '직접': '#c0392b', '축소': '#e08b3c',
         'A8': '#1f6fb4', 'A3': '#7cb0d8', 'RLS': '#4d9e6a'}
ORDER = ['A0', '직접', '축소', 'A8', 'A3', 'RLS']


def usable():
    rows = list(csv.DictReader(
        open(os.path.join(TABLES, 'ladder.csv'), encoding='utf-8')))
    out = {}
    for r in rows:
        out.setdefault((r['direction'], r['tau_s']), {})[
            KEY[r['method']]] = float(r['usable_pct'])
    return out


def rank(d, better_low):
    """1 이 가장 좋다."""
    s = sorted(d, key=lambda k: d[k], reverse=not better_low)
    return {k: i + 1 for i, k in enumerate(s)}


def panel(ax, direction, tau, title, use, VOLT):
    u = use[(direction, tau)]
    rv = rank(VOLT[direction], better_low=True)
    ru = rank(u, better_low=False)
    for k in ORDER:
        crossed = rv[k] != ru[k]
        ax.plot([0, 1], [rv[k], ru[k]], color=COLOR[k],
                lw=3.0 if crossed else 1.6,
                alpha=0.95 if crossed else 0.45,
                solid_capstyle='round', zorder=3 if crossed else 2)
        ax.scatter([0, 1], [rv[k], ru[k]], s=54, color=COLOR[k],
                   zorder=4, edgecolor='white', linewidth=1.4)
        ax.annotate(f"{VOLT[direction][k]:.1f} mV", (0, rv[k]),
                    xytext=(-10, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=9.5,
                    color=COLOR[k], weight='bold')
        ax.annotate(f"{u[k]:.1f} %", (1, ru[k]),
                    xytext=(10, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=9.5,
                    color=COLOR[k], weight='bold')
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(6.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['전압 RMSE 로 세운 순위\n(작을수록 좋다)',
                        '쓸 수 있는 전류로 세운 순위\n(클수록 좋다)'],
                       fontsize=10.5)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f'{i}위' for i in range(1, 7)], fontsize=9.5)
    from scipy.stats import spearmanr
    rho = spearmanr([VOLT[direction][k] for k in ORDER],
                    [-u[k] for k in ORDER]).statistic
    n_cross = sum(rv[k] != ru[k] for k in ORDER)
    note = ('순위 그대로' if n_cross == 0
            else f'{n_cross}/6 판이 자리를 바꾼다')
    ax.set_title(f"{title}\n스피어만 {rho:+.2f}   —   {note}",
                 fontsize=12, weight='bold', pad=10,
                 color='#333333' if n_cross else '#8c8c8c')
    ax.grid(axis='y', color='#e8e8e8', lw=0.9, zorder=0)
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.tick_params(length=0)


def main():
    use = usable()
    VOLT = voltage()
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 12.4))
    panel(axes[0][0], 'discharge', '10.0', '방전  τ = 10 s', use, VOLT)
    panel(axes[0][1], 'charge', '10.0', '충전  τ = 10 s', use, VOLT)
    panel(axes[1][0], 'discharge', '2.0', '방전  τ = 2 s', use, VOLT)
    panel(axes[1][1], 'charge', '2.0', '충전  τ = 2 s', use, VOLT)

    fig.suptitle('전압 오차로 SOP 기법을 고르면, 네 설정 중 셋에서 잘못 고른다',
                 fontsize=15.5, weight='bold', y=0.983)
    fig.text(0.5, 0.955,
             '같은 여섯 판, 같은 셀 홀드아웃.  굵은 선 = 두 지표가 순위를 '
             '다르게 매긴 판.',
             ha='center', fontsize=10.8, color='#555555')

    handles = [Line2D([], [], color=COLOR[k], lw=3,
                      label=LABEL[k].replace('\n', ' ')) for k in ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.003))

    fig.text(0.5, 0.062,
             '전압 오차를 전류로 바꾸는 것은 저항으로 나누는 연산이라 순위를 '
             '보존할 이유가 없다.\n'
             '예외가 하필 방전 10 s 다 — 문헌이 표준으로 보고하는 설정이라, '
             '그것만 보면 이 문제를 못 본다.\n'
             'HPPC-RLS 는 전압에서 네 설정 모두 1 위지만 충전 전류에서는 '
             '4 위다.  직접 대입만 어디서나 꼴찌다.',
             ha='center', fontsize=10.2, color='#333333', linespacing=1.7)

    fig.subplots_adjust(left=0.075, right=0.955, top=0.905, bottom=0.135,
                        wspace=0.42, hspace=0.40)
    out = os.path.join(ROOT, 'results_fig_ladder.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')

    from scipy.stats import spearmanr
    for d, nm in (('discharge', '방전'), ('charge', '충전')):
        for tau in ('10.0', '2.0'):
            u = use[(d, tau)]
            rv, ru = rank(VOLT[d], True), rank(u, False)
            cross = [k for k in ORDER if rv[k] != ru[k]]
            rho = spearmanr([VOLT[d][k] for k in ORDER],
                            [-u[k] for k in ORDER]).statistic
            print(f'  {nm} tau={tau:>4}: {len(cross)}/6 바뀜  '
                  f'스피어만 {rho:+.3f}  {", ".join(cross) or "-"}')


if __name__ == '__main__':
    main()
