"""Revision figures: Figure 5 (usable current vs exceedance, eight methods) and Figure 8 (external transfer)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
B1 = REV1
OUT = os.path.join(R2, 'out', 'figs'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'legend.frameon': False, 'savefig.dpi': 600})
NAMES = {'a8': 'RC-SOP', 'a0': 'Nominal 2RC', 'a3': 'Linear (12 features)', 'shrink': 'Shrinkage', 'lstm': 'LSTM', 'gru': 'GRU',
         'ffrls': 'FFRLS', 'hppc': 'HPPC lookup'}
COL = {'a8': '#d62728', 'a0': '#7f7f7f', 'a3': '#1f77b4', 'shrink': '#17becf', 'lstm': '#9467bd', 'gru': '#8c564b', 'ffrls': '#bcbd22',
       'hppc': '#2ca02c'}
C = pd.read_csv(os.path.join(R2, 'out', 'E5', 'E5_curves.csv'))
fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.2))
for ax, (lab, d, tau) in zip(axs.flat, [('(a)', 'discharge', 10.0), ('(b)', 'discharge', 2.0), ('(c)', 'charge', 10.0), ('(d)', 'charge', 2.0)]):
    for m in NAMES:
        g = C[(C.direction == d) & (C.tau_s == tau) & (C.tol == 0.0) & (C.method == m)].sort_values('lam')
        g = g[g.exceed_rate_pct <= 10]
        ax.step(g.exceed_rate_pct, g.usable_pct, where='post', color=COL[m], lw=1.8 if m == 'a8' else 0.9, label=NAMES[m])
    ax.set_xlim(-0.2, 10); ax.set_ylim(30, 90)
    ax.set_title(f'{lab} {d.capitalize()}, {tau:.0f} s', loc='left', fontsize=8)
    ax.set_xlabel('Reference exceedance rate (%)'); ax.set_ylabel('Mean usable current (%)')
    ax.grid(alpha=0.25, lw=0.4)
axs[0, 0].legend(fontsize=6.5, loc='lower right', ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure5_revised.png'))

# Figure 8
rows = []
for tag, pat in (('Test #1', B1 + '/followup/external/Test1_*_safety.csv'), ('Test #2', B1 + '/external/*_safety.csv')):
    for f in glob.glob(pat):
        s = os.path.basename(f).replace('_safety.csv', '').replace('Test1_', '')
        if s.startswith('Test') or s.startswith('test'):
            continue
        t = pd.read_csv(f)
        for dr, g in t.groupby('direction'):
            rows.append(dict(test=tag, surface=s, direction=dr, min_margin=g.margin.min(), exceed=g.exceed.sum()))
M = pd.DataFrame(rows)
surf = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
short = {'BOOST': 'BOOST', 'BOOST_NEGPULSE': 'NEG', 'BOOST_NEGPULSE_1S': 'NEG 1s', 'BOOST_REST': 'REST', 'CC': 'CC', 'CC_CELL2': 'CC2'}
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={'width_ratios': [1.6, 1]})
ax = axs[0]; x = np.arange(len(surf)); w = 0.2
for k, (tag, dr, col, hatch) in enumerate([('Test #1', 'discharge', '#1f77b4', ''), ('Test #2', 'discharge', '#aec7e8', ''),
                                            ('Test #1', 'charge', '#d62728', ''), ('Test #2', 'charge', '#ff9896', '')]):
    v = [M[(M.test == tag) & (M.direction == dr) & (M.surface == s)].min_margin.min() for s in surf]
    ax.bar(x + (k - 1.5) * w, v, w, color=col, label=f'{tag} {dr}')
ax.axhline(1.0, color='k', lw=0.7, ls='--')
ax.set_xticks(x); ax.set_xticklabels([short[s] for s in surf]); ax.set_ylabel('Worst margin')
ax.set_title('(a) Frozen RC-SOP, 10 and 25 °C', loc='left', fontsize=8); ax.legend(fontsize=6.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
summ = json.load(open(B1 + '/followup/summary.json'))
cold = [r for r in summ['cold3'] if str(r['tau_s']) in ('10', '10.0')]
cold = sorted(cold, key=lambda r: int(r['temp_C']))
ax = axs[1]
T = [int(r['temp_C']) for r in cold]; E = [100 * r['original_exceed'] / r['n_model_pulse'] for r in cold]
ax.bar(range(len(T)), E, color='#1f77b4')
for i, r in enumerate(cold):
    ax.text(i, E[i] + 0.6, f"{r['original_exceed']}/{r['n_model_pulse']}", ha='center', fontsize=6)
ax.set_xticks(range(len(T))); ax.set_xticklabels([f'{t}' for t in T]); ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Exceedances (% of model–pulse cases)'); ax.set_ylim(0, max(E) * 1.25 + 1)
ax.set_title('(b) Frozen RC-SOP, Test #3 discharge', loc='left', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure8_revised.png'))
M.to_csv(os.path.join(OUT, 'Figure8a_data.csv'), index=False)
print(M.groupby(['test', 'direction']).min_margin.agg(['min', 'max']))
