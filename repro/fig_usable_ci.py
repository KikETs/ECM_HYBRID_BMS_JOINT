"""Usable current with its uncertainty — the figure the ladder cannot be.

    python3 repro/fig_usable_ci.py

Why this exists
    Every table in this project carries a cell-cluster bootstrap interval and
    no figure showed one.  The ladder chart plots ranks, so a point estimate
    is all it can carry, and a rank chart makes six methods look decisively
    ordered whether or not they are.  They are not: A3, A8, LSTM and GRU
    overlap in both directions.  A reader shown only the ladder would
    conclude otherwise.

    Cells are the independent unit, not rows, so the interval resamples the
    six cells.  With six clusters the intervals are wide, and that width is
    the point.

    Sorted by point estimate, with the worst cell marked separately: a method
    can lead on the mean and still be the one that fails a cell.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

LABEL = {'a8': 'A8  (4 effective coeff.)', 'a3': 'A3  (26 coeff.)',
         'lstm': 'LSTM  (5,954 par.)', 'gru': 'GRU  (4,482 par.)',
         'shrink': 'shrinkage coeff.  (2)', 'ffrls': 'FFRLS adaptive ECM'}
COLOR = {'a8': '#1f6fb4', 'a3': '#7cb0d8', 'lstm': '#4d9e6a',
         'gru': '#7fc49a', 'shrink': '#e08b3c', 'ffrls': '#c0392b'}


def rows():
    out = []
    for f in glob.glob(os.path.join(TABLES, 'safety_strict_*oracle.csv')):
        b = os.path.basename(f)
        if 'percell' in b or 'tolsens' in b:
            continue
        for r in csv.DictReader(open(f, encoding='utf-8')):
            if r.get('method') in LABEL:
                out.append(r)
    return out


def main():
    R = rows()
    if not R:
        print('  no safety_strict_*_oracle.csv found')
        return 1
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex='col')
    for ax, (direction, tau) in zip(
            axes.ravel(),
            [('discharge', '10.0'), ('charge', '10.0'),
             ('discharge', '2.0'), ('charge', '2.0')]):
        sel = [r for r in R if r['direction'] == direction
               and r['tau_s'] == tau]
        sel.sort(key=lambda r: float(r['usable_mean_pct']))
        y = np.arange(len(sel))
        for i, r in enumerate(sel):
            m = float(r['usable_mean_pct'])
            lo = float(r['usable_boot_lo_pct'])
            hi = float(r['usable_boot_hi_pct'])
            w = float(r['usable_worst_pct'])
            c = COLOR[r['method']]
            ax.plot([lo, hi], [i, i], color=c, lw=3.2, solid_capstyle='round',
                    alpha=0.55, zorder=2)
            ax.scatter([m], [i], s=70, color=c, zorder=4, edgecolor='white',
                       linewidth=1.4)
            ax.scatter([w], [i], s=52, color=c, zorder=3, marker='|',
                       linewidth=2.2)
            ax.annotate(f'{m:.1f}', (m, i), xytext=(0, 9),
                        textcoords='offset points', ha='center',
                        fontsize=9, color=c, weight='bold')
        # A8's interval as a reference band: anything inside it is not
        # distinguishable from the adopted model on this evidence.
        a8 = [r for r in sel if r['method'] == 'a8']
        if a8:
            ax.axvspan(float(a8[0]['usable_boot_lo_pct']),
                       float(a8[0]['usable_boot_hi_pct']),
                       color='#1f6fb4', alpha=0.07, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[r['method']] for r in sel], fontsize=9.5)
        ax.set_title(f'{direction}   τ = {float(tau):.0f} s',
                     fontsize=12, weight='bold', loc='left', pad=8)
        ax.grid(axis='x', color='#ececec', lw=0.9, zorder=0)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel('usable current  [%]', fontsize=10.5)

    # Derived, not asserted.  A hard-coded "four of six" was wrong in two
    # of the four panels - the same mistake as the SOH caption and the
    # ladder footnote earlier in this audit.
    inside_counts = []
    for direction, tau in (('discharge', '10.0'), ('charge', '10.0'),
                           ('discharge', '2.0'), ('charge', '2.0')):
        s = [r for r in R if r['direction'] == direction
             and r['tau_s'] == tau]
        a8r = [r for r in s if r['method'] == 'a8']
        if not a8r:
            continue
        lo = float(a8r[0]['usable_boot_lo_pct'])
        hi = float(a8r[0]['usable_boot_hi_pct'])
        inside_counts.append(sum(lo <= float(r['usable_mean_pct']) <= hi
                                 for r in s))
    lo_n, hi_n = min(inside_counts), max(inside_counts)
    span = f'{lo_n}' if lo_n == hi_n else f'{lo_n}–{hi_n}'
    fig.suptitle(f'Usable current with cell-cluster uncertainty — {span} of '
                 f'six methods sit inside A8\'s interval',
                 fontsize=14.5, weight='bold', y=0.985)
    fig.text(0.5, 0.945,
             'Dot = mean over cells, bar = 95 % bootstrap over the six cells, '
             'tick = worst cell.  Shaded band is A8\'s interval.',
             ha='center', fontsize=10.2, color='#555555')
    handles = [Line2D([], [], color='#444', marker='o', ls='none',
                      label='mean over cells'),
               Line2D([], [], color='#444', lw=3.2, alpha=0.55,
                      label='95 % cell-cluster bootstrap'),
               Line2D([], [], color='#444', marker='|', ls='none',
                      markersize=10, label='worst cell')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.885, bottom=0.115,
                        wspace=0.42, hspace=0.42)
    out = os.path.join(ROOT, 'results_fig_usable_ci.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')
    for direction in ('discharge', 'charge'):
        sel = [r for r in R if r['direction'] == direction
               and r['tau_s'] == '10.0']
        a8 = [r for r in sel if r['method'] == 'a8'][0]
        lo, hi = (float(a8['usable_boot_lo_pct']),
                  float(a8['usable_boot_hi_pct']))
        inside = [r['method'] for r in sel
                  if lo <= float(r['usable_mean_pct']) <= hi]
        print(f'  {direction:<10}tau 10 s: inside A8\'s interval '
              f'[{lo:.1f}, {hi:.1f}] -> {", ".join(sorted(inside))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
