"""SOP 궤적 — 노화에 따라 쓸 수 있는 전류가 어떻게 줄어드나.

셀마다 사이클을 가로축에 두고 세 가지를 겹친다:

  검은 점   HPPC 에서 **측정한** |I*|  (신뢰 라벨만, extrap <= 1.5)
  회색 선   A0 — 명목 저항표만 (보정 없음)
  파란 선   A8 — 채택 트림의 원 예측
  파란 면   lambda x A8 — **실제로 BMS 가 허용하는 전류**

파란 면이 검은 점 아래에 있으면 초과가 없다는 뜻이고, 그 사이의 간격이
안전을 위해 버리는 몫이다.  lambda 는 셀 하나씩 빼고 잡았으므로 그
셀은 자기 lambda 를 정할 때 쓰이지 않았다.

    python3 repro/fig_sop_traj.py [--direction discharge --tau 10]
"""
import argparse
import csv
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
sys.path.insert(0, HERE)
from run_safety import load, keep, lam_loco, TOL      # noqa: E402

EVAL = os.path.join(ROOT, 'analysis', 'results', 'eval')
CELLS = ['CC', 'CC_CELL2', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S',
         'BOOST_REST']
C_MEAS = '#222222'
C_A0 = '#9a9a9a'
C_A8 = '#1f6fb4'


def binned(x, y, nb=9):
    """사이클 구간마다 중앙값 — 같은 사이클에 SOC 가 여럿이라 흩어진다."""
    if len(x) < 4:
        return x, y
    e = np.linspace(x.min(), x.max() + 1e-6, nb + 1)
    xs, ys = [], []
    for i in range(nb):
        m = (x >= e[i]) & (x < e[i + 1])
        if m.sum():
            xs.append(float(np.median(x[m])))
            ys.append(float(np.median(y[m])))
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', default='discharge',
                    choices=['discharge', 'charge'])
    ap.add_argument('--tau', type=float, default=10.0)
    a = ap.parse_args()
    tag = 'disc' if a.direction == 'discharge' else 'char'
    d = load(os.path.join(EVAL, f'a8_{tag}_oracle.csv'))
    cyc = np.array([float(x) for x in
                    (r['cycle'] for r in csv.DictReader(
                        open(os.path.join(EVAL, f'a8_{tag}_oracle.csv'),
                             encoding='utf-8')))])
    tol = TOL[a.direction]
    lam = lam_loco(d, a.tau, tol)
    m0 = keep(d, a.tau)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    exc_tot = n_tot = 0
    for i, c in enumerate(CELLS):
        ax = axes[i // 3][i % 3]
        m = m0 & (d['cell'] == c)
        if m.sum() == 0:
            ax.set_visible(False)
            continue
        x = cyc[m]
        # 같은 사이클에 SOC 가 여러 개라 |I*| 가 흩어진다.  선으로 그리면
        # 짝이 안 맞는 점과 비교돼 없는 초과가 보이므로, 짝지어 점으로 그린다.
        dep = lam * d['hyb'][m]
        ax.vlines(x, dep, d['meas'][m], color='#cfe0ee', lw=1.0, zorder=2)
        ax.scatter(x, d['meas'][m], s=15, color=C_MEAS, zorder=5,
                   label='측정 |I*|')
        ax.scatter(x, dep, s=13, color=C_A8, zorder=4, alpha=0.85,
                   label=f'λ x A8  허용 전류  (λ={lam:.3f})')
        bx, by = binned(x, d['ecm'][m])
        ax.plot(bx, by, color=C_A0, lw=1.6, zorder=3, label='A0  보정 없음')
        bx, by = binned(x, d['hyb'][m])
        ax.plot(bx, by, color=C_A8, lw=1.6, ls=(0, (4, 2)), zorder=3,
                label='A8  원 예측 (λ 곱하기 전)')

        o = lam * d['hyb'][m] - d['meas'][m]
        nx = int((o > tol).sum())
        exc_tot += nx
        n_tot += int(m.sum())
        u = np.median(lam * d['hyb'][m] / d['meas'][m]) * 100
        ax.set_title(f'{c}   n={m.sum()}   쓸 수 있는 전류 {u:.1f} %'
                     + (f'   초과 {nx}' if nx else '   초과 없음'),
                     fontsize=10.5, weight='bold', loc='left', pad=6)
        ax.grid(color='#ececec', lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('|I*|  [A]', fontsize=10.5)
        if i >= 3:
            ax.set_xlabel('사이클', fontsize=10.5)
        if i == 0:
            ax.legend(fontsize=8.2, loc='lower left', framealpha=0.94)

    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_ylim(0, None)

    dn = '방전' if a.direction == 'discharge' else '충전'
    ub = np.median(lam * d['hyb'][m0] / d['meas'][m0]) * 100
    fig.suptitle(f'SOP {dn}  τ = {a.tau:.0f} s — 노화에 따라 쓸 수 있는 '
                 f'전류가 어떻게 줄어드나', fontsize=15, weight='bold', y=0.988)
    fig.text(0.5, 0.933,
             f'채택 구성 A8 (파라미터 4 개).  λ = {lam:.3f} 는 셀 하나씩 '
             f'빼고 잡았다 — 각 셀은 자기 λ 를 정할 때 쓰이지 않았다.\n'
             f'신뢰 라벨 {n_tot} 개 중 초과 {exc_tot} 개 (허용 {tol:.1f} A).  '
             f'전체 쓸 수 있는 전류 {ub:.1f} %.  '
             f'세로선 하나가 그 펄스의 여유 — 파란 점(허용)과 검은 점(측정)의 간격.',
             ha='center', fontsize=10.2, color='#555555', linespacing=1.7)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.826, bottom=0.078,
                        hspace=0.36, wspace=0.22)

    out = os.path.join(ROOT, f'results_fig_sop_traj_{a.direction}.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')
    print(f'    λ={lam:.3f}  n={n_tot}  초과 {exc_tot}  '
          f'쓸 수 있는 전류 {ub:.1f} %')


if __name__ == '__main__':
    main()
