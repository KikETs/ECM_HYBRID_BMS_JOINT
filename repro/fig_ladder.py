"""The paper's central figure — picking by voltage flips the ranking.

The same six versions are ranked on the left by voltage RMSE and on the right
by the value that decides deployment (usable current).  A crossing line means
the two metrics rank that version differently.

    python3 repro/fig_ladder.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

def voltage():
    """Read the voltage RMSEs from the table.

    These used to be twelve constants in this file.  The figure would not
    change when the table did, and there was no way to check that the
    figure's numbers were right.
    """
    # voltage.csv uses the short method names; ladder.csv is mapped onto the
    # keys in ORDER.
    ALIAS = {'A0': 'A0', 'direct': 'direct', 'shrink': 'shrink',
             'A8': 'A8', 'A3': 'A3', 'RLS': 'RLS'}
    rows = list(csv.DictReader(
        open(os.path.join(TABLES, 'voltage.csv'), encoding='utf-8')))
    out = {}
    for r in rows:
        out.setdefault(r['direction'], {})[ALIAS[r['method']]] = \
            float(r['rmse_mV'])
    return out


KEY = {'A0  no correction': 'A0', 'direct plug-in': 'direct',
       'shrinkage coefficient': 'shrink', 'A8  dR_fast alone': 'A8',
       'A3  12 features': 'A3', '[upper bound] HPPC-RLS': 'RLS'}
LABEL = {'A0': 'A0  no correction (classical HPPC)',
         'direct': 'direct plug-in (RLS as-is)',
         'shrink': 'shrinkage coeff. (2 params)', 'A8': 'A8 (4 params)',
         'A3': 'A3 (26 params)', 'RLS': 'HPPC-RLS (upper bound)'}
COLOR = {'A0': '#8c8c8c', 'direct': '#c0392b', 'shrink': '#e08b3c',
         'A8': '#1f6fb4', 'A3': '#7cb0d8', 'RLS': '#4d9e6a'}
ORDER = ['A0', 'direct', 'shrink', 'A8', 'A3', 'RLS']


def usable():
    rows = list(csv.DictReader(
        open(os.path.join(TABLES, 'ladder.csv'), encoding='utf-8')))
    out = {}
    for r in rows:
        out.setdefault((r['direction'], r['tau_s']), {})[
            KEY[r['method']]] = float(r['usable_pct'])
    return out


def rank(d, better_low):
    """Rank 1 is best."""
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
    ax.set_xticklabels(['ranked by voltage RMSE\n(lower is better)',
                        'ranked by usable current\n(higher is better)'],
                       fontsize=10.5)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([f'{i}' for i in range(1, 7)], fontsize=9.5)
    from scipy.stats import spearmanr
    rho = spearmanr([VOLT[direction][k] for k in ORDER],
                    [-u[k] for k in ORDER]).statistic
    n_cross = sum(rv[k] != ru[k] for k in ORDER)
    note = ('ranking preserved' if n_cross == 0
            else f'{n_cross}/6 change place')
    ax.set_title(f"{title}\nSpearman {rho:+.2f}   —   {note}",
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
    panel(axes[0][0], 'discharge', '10.0', 'discharge  τ = 10 s', use, VOLT)
    panel(axes[0][1], 'charge', '10.0', 'charge  τ = 10 s', use, VOLT)
    panel(axes[1][0], 'discharge', '2.0', 'discharge  τ = 2 s', use, VOLT)
    panel(axes[1][1], 'charge', '2.0', 'charge  τ = 2 s', use, VOLT)

    fig.suptitle('Picking an SOP method by voltage error picks wrong in '
                 'three of the four settings',
                 fontsize=15.5, weight='bold', y=0.983)
    fig.text(0.5, 0.955,
             'Same six versions, same cell holdout.  A thick line = the two '
             'metrics rank that version differently.',
             ha='center', fontsize=10.8, color='#555555')

    handles = [Line2D([], [], color=COLOR[k], lw=3, label=LABEL[k])
               for k in ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.003),
               columnspacing=1.4, handlelength=1.8)

    fig.text(0.5, 0.062,
             'Turning voltage error into current is a division by resistance, '
             'so it has no reason to preserve rank.\n'
             'The exception is, of all settings, discharge 10 s — the one the '
             'literature reports as standard, so looking only there hides the '
             'problem.\n'
             'HPPC-RLS ranks first on voltage in all four settings but fourth '
             'on charge current.  Only the direct plug-in is last everywhere.',
             ha='center', fontsize=10.2, color='#333333', linespacing=1.7)

    fig.subplots_adjust(left=0.075, right=0.955, top=0.905, bottom=0.135,
                        wspace=0.42, hspace=0.40)
    out = os.path.join(ROOT, 'results_fig_ladder.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')

    from scipy.stats import spearmanr
    for d, nm in (('discharge', 'discharge'), ('charge', 'charge')):
        for tau in ('10.0', '2.0'):
            u = use[(d, tau)]
            rv, ru = rank(VOLT[d], True), rank(u, False)
            cross = [k for k in ORDER if rv[k] != ru[k]]
            rho = spearmanr([VOLT[d][k] for k in ORDER],
                            [-u[k] for k in ORDER]).statistic
            print(f'  {nm:<10} tau={tau:>4}: {len(cross)}/6 changed  '
                  f'Spearman {rho:+.3f}  {", ".join(cross) or "-"}')


if __name__ == '__main__':
    main()
