"""Journal-style Figures 5 and 8 (matching the manuscript's Figure 4/9 style: Times New Roman, top legend without frame,
bold outside panel labels, left/bottom spines, light grid). Data: review2 outputs (E5 curves, Figure 8 data) and the
Test #3 counts from review1 follow-up summary (cold3, tau = 10 s, original factors).
Palette validated with dataviz/validate_palette.js (light): 8-slot set passes with CVD floor 7.2 -> line styles added;
2-slot set (Figure 8) passes."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2
FIG = _os.path.join(R2, 'out', 'figs')

import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 10.5, 'axes.linewidth': 0.8,
                     'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'legend.frameon': False,
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'savefig.dpi': 600, 'axes.edgecolor': '#333333', 'text.color': '#111111',
                     'axes.labelcolor': '#111111', 'xtick.color': '#333333', 'ytick.color': '#333333'})
INK2 = '#52514e'

# ------------------------------------------------------------------ Figure 5
C = pd.read_csv(_os.path.join(R2, 'out', 'E5', 'E5_curves.csv'))
S = [  # key, label, color, linestyle, linewidth (fixed order; RC-SOP first)
    ('a8', 'RC-SOP', '#2a78d6', '-', 2.2),
    ('hppc', 'HPPC lookup', '#eb6834', '-', 1.5),
    ('a3', 'Linear (12 features)', '#1baf7a', (0, (6, 2)), 1.2),
    ('gru', 'GRU', '#6250d6', (0, (4, 1.5, 1, 1.5)), 1.2),
    ('lstm', 'LSTM', '#e87ba4', (0, (2, 1.2)), 1.3),
    ('shrink', 'Shrinkage', '#eda100', (0, (8, 2, 1, 2, 1, 2)), 1.2),
    ('ffrls', 'FFRLS', '#008300', (0, (1, 1.6)), 1.4),
    ('a0', 'Nominal 2RC', '#e34948', (0, (5, 3)), 1.3),
]
fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.9))
cases = [('(a)', 'discharge', 10.0, '10 s discharge'), ('(b)', 'discharge', 2.0, '2 s discharge'),
         ('(c)', 'charge', 10.0, '10 s charge'), ('(d)', 'charge', 2.0, '2 s charge')]
for ax, (lab, d, tau, name) in zip(axs.flat, cases):
    for k, label, col, ls, lw in S:
        g = C[(C.direction == d) & (C.tau_s == tau) & (C.tol == 0.0) & (C.method == k)].sort_values('lam')
        g = g[(g.exceed_rate_pct <= 10) & (g.usable_pct >= 30)]
        g = g.groupby('exceed_rate_pct', as_index=False).usable_pct.max().sort_values('exceed_rate_pct')
        g['usable_pct'] = g.usable_pct.cummax()
        ax.step(g.exceed_rate_pct, g.usable_pct, where='post', color=col, ls=ls, lw=lw, zorder=3 if k == 'a8' else 2,
                solid_capstyle='round', dash_capstyle='round')
    ax.set_xlim(0, 10); ax.set_ylim(30, 90); ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(30, 91, 10))
    ax.grid(True, axis='both'); ax.set_axisbelow(True)
    ax.set_xlabel('Reference exceedance rate (%)'); ax.set_ylabel('Mean usable current (%)')
    ax.text(0.03, 0.95, name, transform=ax.transAxes, ha='left', va='top', color=INK2, fontsize=10.5)
    ax.text(-0.17, 1.04, lab, transform=ax.transAxes, ha='left', va='bottom', fontweight='bold', fontsize=12)
handles = [Line2D([0], [0], color=col, ls=ls, lw=lw if lw > 1.3 else 1.4) for _, _, col, ls, lw in S]
fig.legend(handles, [s[1] for s in S], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.6, w_pad=1.8)
fig.savefig(_os.path.join(FIG, 'Figure5_revised.png'), facecolor='white')
fig.savefig(_os.path.join(FIG, 'Figure5_revised.tif'), facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# ------------------------------------------------------------------ Figure 8
M = pd.read_csv(_os.path.join(FIG, 'Figure8a_data.csv'))
surf = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
short = {'BOOST': 'BOOST', 'BOOST_NEGPULSE': 'NEG', 'BOOST_NEGPULSE_1S': 'NEG 1s', 'BOOST_REST': 'REST', 'CC': 'CC', 'CC_CELL2': 'CC2'}
COL = {'Test #1': '#2a78d6', 'Test #2': '#eb6834'}
MK = {'discharge': 'o', 'charge': 's'}
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={'width_ratios': [1.55, 1]})
ax = axs[0]; x = np.arange(len(surf))
off = {('Test #1', 'discharge'): -0.18, ('Test #2', 'discharge'): -0.06, ('Test #1', 'charge'): 0.06, ('Test #2', 'charge'): 0.18}
for (t, dr), dx in off.items():
    v = [M[(M.test == t) & (M.direction == dr) & (M.surface == s)].min_margin.min() for s in surf]
    filled = dr == 'discharge'
    ax.scatter(x + dx, v, s=34, marker=MK[dr], facecolor=COL[t] if filled else 'white', edgecolor=COL[t], linewidth=1.3, zorder=3)
ax.axhline(1.0, color='#666666', lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(-0.45, 1.012, 'margin = 1', ha='left', va='bottom', fontsize=9.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([short[s] for s in surf]); ax.set_xlim(-0.5, len(surf) - 0.5)
ax.set_ylim(0.4, 1.4); ax.set_ylabel('Worst margin'); ax.set_xlabel('Internal surface')
ax.grid(True, axis='y'); ax.set_axisbelow(True)
ax.text(-0.13, 1.04, '(a)', transform=ax.transAxes, fontweight='bold', fontsize=12)
cold = [(-20, 66, 288), (-10, 7, 288), (0, 0, 288), (10, 0, 288), (25, 0, 274), (40, 0, 276)]
ax = axs[1]
pct = [100 * e / n for _, e, n in cold]
ax.bar(range(len(cold)), pct, width=0.6, color='#2a78d6', zorder=3)
for i, (_, e, n) in enumerate(cold):
    ax.text(i, pct[i] + 0.7, f'{e}/{n}', ha='center', va='bottom', fontsize=9, color=INK2)
ax.set_xticks(range(len(cold))); ax.set_xticklabels([f'{t}'.replace('-', '−') for t, _, _ in cold])
ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Exceedances (%)'); ax.set_ylim(0, 30)
ax.grid(True, axis='y'); ax.set_axisbelow(True)
ax.text(-0.22, 1.04, '(b)', transform=ax.transAxes, fontweight='bold', fontsize=12)
handles = [Line2D([0], [0], marker=MK[dr], ls='', markersize=6.5, markerfacecolor=COL[t] if dr == 'discharge' else 'white',
                  markeredgecolor=COL[t], markeredgewidth=1.3) for t, dr in off]
fig.legend(handles, [f'{t} {dr}' for t, dr in off], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
fig.savefig(_os.path.join(FIG, 'Figure8_revised.png'), facecolor='white')
fig.savefig(_os.path.join(FIG, 'Figure8_revised.tif'), facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
print('figures written')
