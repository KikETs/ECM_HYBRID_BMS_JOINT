"""SOH trajectories — what the SOH model predicts leaving one cell out.

Each cell is predicted by a model that never saw it (trained on the other
five).

The model's name and size are read from results/soh_pred.npz, never written
here.  This figure spent one commit captioned "partial-charge CNN, 10,945
parameters, mean over 3 seeds" while plotting ridge, because the caption was
a string in this file.  If the metadata is missing the figure refuses to
render rather than guessing.

    python3 repro/fig_soh_traj.py
"""
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
PRED = os.path.join(ROOT, 'analysis', 'results', 'soh_pred.npz')

CELLS = ['CC', 'CC_CELL2', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S',
         'BOOST_REST']
C_TRUE = '#222222'
C_PRED = '#1f6fb4'
C_BAD = '#c0392b'


def main():
    z = np.load(PRED)
    miss = [k for k in ('model', 'detail') if k not in z.files]
    if miss:
        print(f'  {os.path.relpath(PRED, ROOT)} carries no {miss} — it was '
              f'written by a producer that does not record which model made '
              f'it, so this figure cannot label itself.  Re-run the soh '
              f'stage:\n    python3 analysis/soh_ridge.py --save-model '
              f'runs_soh_ridge --save-pred results/soh_pred.npz --deployment',
              file=sys.stderr)
        return 1
    model_name = str(z['model'])
    model_detail = str(z['detail'])
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
        ax.plot(cy, y * 100, color=C_TRUE, lw=2.0, label='true SOH',
                zorder=3)
        bad = np.abs(e) > 2.0
        ax.scatter(cy[~bad], p[~bad] * 100, s=17, color=C_PRED, zorder=4,
                   label='estimated (holdout)')
        if bad.any():
            ax.scatter(cy[bad], p[bad] * 100, s=26, color=C_BAD, zorder=5,
                       label='error > 2 %p')
        ax.set_title(f'{c}   RMSE {np.sqrt(np.mean(e**2)):.2f} %p   '
                     f'bias {e.mean():+.2f}',
                     fontsize=11, weight='bold', loc='left', pad=6)
        ax.set_ylim(64, 104)
        ax.grid(color='#ececec', lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('SOH  [%]', fontsize=10.5)
        if i >= 3:
            ax.set_xlabel('cycle', fontsize=10.5)
        if i == 0:
            ax.legend(fontsize=9, loc='lower left', framealpha=0.94)

    # bottom band: error over cycles, and the 1:1 plot
    axl = fig.add_subplot(gs[2, :2])
    for c in CELLS:
        cy = z[f'{c}_cycle']
        e = (z[f'{c}_pred'] - z[f'{c}_y']) * 100
        o = np.argsort(cy)
        axl.plot(cy[o], e[o], lw=1.5, alpha=0.85, label=c)
    axl.axhline(0, color=C_TRUE, lw=1.4)
    axl.axhspan(-2, 2, color='#ececec', zorder=0)
    axl.set_xlabel('cycle', fontsize=10.5)
    axl.set_ylabel('error  [%p]', fontsize=10.5)
    axl.set_title('error per cell — the grey band is ±2 %p', fontsize=11,
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
    axr.set_xlabel('true SOH  [%]', fontsize=10.5)
    axr.set_ylabel('estimated  [%]', fontsize=10.5)
    e = pp - yy
    axr.set_title(f'overall  RMSE {np.sqrt(np.mean(e**2)):.2f} %p  '
                  f'bias {e.mean():+.2f}',
                  fontsize=11, weight='bold', loc='left', pad=6)
    axr.grid(color='#ececec', lw=0.8)
    for sp in ('top', 'right'):
        axr.spines[sp].set_visible(False)

    fig.suptitle(f'SOH — {model_name}, leave one cell out',
                 fontsize=15, weight='bold', y=0.988)
    # The bias is read off the predictions, not hard-coded: an earlier caption
    # said "+0.10 %p, the dangerous side", which came from the run that still
    # contained the 8 temperature-fault curves.  Sec 30.12 retracted that.
    fig.text(0.5, 0.947,
             f'{model_detail}.  Each cell is predicted '
             'by a model that never saw it.\n'
             f'Bias {e.mean():+.2f} %p — reading SOH high makes SOP '
             'optimistic (29.1), so the sign matters; here it is effectively '
             'zero (30.12).',
             ha='center', va='top', fontsize=10.4, color='#555555',
             linespacing=1.6)
    fig.subplots_adjust(left=0.058, right=0.982, top=0.862, bottom=0.075)

    out = os.path.join(ROOT, 'results_fig_soh_traj.png')
    fig.savefig(out, dpi=190, facecolor='white')
    print(f'  -> {out}')
    print(f'    overall RMSE {np.sqrt(np.mean(e**2))/100:.4f}  '
          f'bias {e.mean()/100:+.4f}')


if __name__ == '__main__':
    main()
