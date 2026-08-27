"""SOP trajectories — how the usable current falls with ageing.

For each cell, cycles run along the x axis and four things are overlaid:

  black dots  |I*| **measured** at HPPC  (trustworthy labels only,
              extrap <= 1.5)
  grey line   A0 — the nominal resistance table alone (no correction)
  blue line   A8 — the adopted trim's raw prediction
  blue dots   lambda x A8 — **the current the BMS actually allows**

A blue dot below its black dot means no exceedance, and the gap between them
is what is discarded for safety.  lambda was set leaving one cell out, so a
cell was never used to set its own lambda.

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

plt.rcParams['font.family'] = 'DejaVu Sans'
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
    """Median per cycle bin — several SOCs share a cycle, so it scatters."""
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
        # Several SOCs share a cycle, so |I*| scatters.  Drawing lines would
        # compare unpaired points and show exceedances that do not exist, so
        # the pairs are drawn as dots joined by a vertical line.
        dep = lam * d['hyb'][m]
        ax.vlines(x, dep, d['meas'][m], color='#cfe0ee', lw=1.0, zorder=2)
        ax.scatter(x, d['meas'][m], s=15, color=C_MEAS, zorder=5,
                   label='measured |I*|')
        ax.scatter(x, dep, s=13, color=C_A8, zorder=4, alpha=0.85,
                   label=f'λ x A8  allowed current  (λ={lam:.3f})')
        bx, by = binned(x, d['ecm'][m])
        ax.plot(bx, by, color=C_A0, lw=1.6, zorder=3,
                label='A0  no correction')
        bx, by = binned(x, d['hyb'][m])
        ax.plot(bx, by, color=C_A8, lw=1.6, ls=(0, (4, 2)), zorder=3,
                label='A8  raw prediction (before λ)')

        o = lam * d['hyb'][m] - d['meas'][m]
        nx = int((o > tol).sum())
        exc_tot += nx
        n_tot += int(m.sum())
        u = np.median(lam * d['hyb'][m] / d['meas'][m]) * 100
        ax.set_title(f'{c}   n={m.sum()}   usable {u:.1f} %'
                     + (f'   {nx} exc' if nx else '   0 exc'),
                     fontsize=10, weight='bold', loc='left', pad=6)
        ax.grid(color='#ececec', lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('|I*|  [A]', fontsize=10.5)
        if i >= 3:
            ax.set_xlabel('cycle', fontsize=10.5)
        if i == 0:
            ax.legend(fontsize=8.2, loc='lower left', framealpha=0.94)

    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_ylim(0, None)

    dn = a.direction
    ub = np.median(lam * d['hyb'][m0] / d['meas'][m0]) * 100
    fig.suptitle(f'SOP {dn}  τ = {a.tau:.0f} s — how the usable current falls '
                 f'with ageing', fontsize=15, weight='bold', y=0.985)
    fig.text(0.5, 0.947,
             f'Adopted configuration A8 (4 parameters).  λ = {lam:.3f} was set '
             f'leaving one cell out — no cell set its own λ.\n'
             f'{exc_tot} exceedances of {n_tot} trustworthy labels '
             f'(tolerance {tol:.1f} A).  Overall usable current {ub:.1f} %.\n'
             f'Each vertical line is that pulse\'s margin — blue dot '
             f'(allowed) to black dot (measured).',
             ha='center', va='top', fontsize=10.2, color='#555555',
             linespacing=1.6)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.800, bottom=0.078,
                        hspace=0.36, wspace=0.24)

    out = os.path.join(ROOT, f'results_fig_sop_traj_{a.direction}.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')
    print(f'    λ={lam:.3f}  n={n_tot}  {exc_tot} exceed  '
          f'usable current {ub:.1f} %')


if __name__ == '__main__':
    main()
