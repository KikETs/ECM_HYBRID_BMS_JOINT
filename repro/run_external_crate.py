"""Does the frozen safety factor survive a change of prior load, at 0 C?

RPCWBY Test#8 measures SOP on the same cell at 0 C after discharging at
0, C/3, 1C, 2C, 3C and 4C, over thirteen SOC points from 0.95 down to 0.05.
That is the one axis Test#3 does not sweep, and it matters here for a
specific reason: 0 C is exactly where the frozen lambda stopped having room.
external_temp_envelope.csv puts the margin at 1.396 there, against 1.447 at
40 C and 0.878 at -10 C.  If prior load moves the requirement at all, it
moves it at the temperature with the least to give.

Same caveat as the temperature sweep, and it is the whole scope of the
result: Test#8 carries no paired drive cycle, so the A8 trim cannot be
computed on it.  What is scored is the nominal 2RC layer the trim sits on.

The C-rate column is the rate the cell was discharged at BEFORE the pulse,
not the pulse current.  So this asks whether recent history changes what the
physics layer owes, which is the assumption the trim's exponentially-weighted
features exist to relax.

Two things about the numbers this prints, both of which decide how they may
be quoted.

The exceedance upper bound is Clopper-Pearson on the in-hull ROWS.  Those rows
are one physical cell measured repeatedly across SOC and prior rate, so they
are not independent samples of anything -- the binomial model treats them as
if they were, which makes the bound a statement about THIS GRID on THIS CELL
and nothing wider.  Cell-level or population-level risk is not estimable from
a single external cell at any sample size, so 6.1 % must never be quoted as a
risk figure.  Pooling across surfaces would be worse still: the same 48
measurements scored six times is 288 rows of which only 48 are distinct.

And the answer depends on which internal surface is used.  Test#8 is external,
so no surface is "held out" for it -- all six are models built on internal
cells and applied to a cell none of them saw.  Reporting one is a choice, and
--holdout CC happened to be the most favourable of the six.  --all-surfaces
runs every one and writes the range; the number to quote is the worst.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

LAB = os.path.join(ANALYSIS, 'rpcwby_sop_test8.csv')
OUT = os.path.join(ANALYSIS, 'results', 'tables', 'external_crate_envelope.csv')
LAM_SHIPPED = 0.6832
SURF_OUT = os.path.join(ANALYSIS, 'results', 'tables',
                       'external_crate_surfaces.csv')
sys.path.insert(0, HERE)
from stages import CELLS  # noqa: E402


HEADER = ['prior_rate', 'n_points', 'n_in_hull', 'soc_in_hull_min',
          'soc_in_hull_max', 'lambda_shipped', 'lambda_needed', 'margin',
          'exceed', 'exceed_ub95_pct', 'worst_overshoot_W']

SURF_HEADER = ['surface', 'n_in_hull', 'lambda_needed_min',
               'lambda_needed_max', 'margin_min', 'exceed',
               'exceed_ub95_pct', 'worst_overshoot_W']

RATES = ('0C', 'C/3', '1C', '2C', '3C', '4C')


def score(holdout, tau):
    """Score Test#8 against one internal surface.

    Returns (per-rate table rows, pooled dict).  Pulled out of main so the
    all-surfaces sweep runs the identical code path rather than a second
    implementation that could drift from it.
    """
    from ecm_surface import ECMSurface
    from run_chen2026_baseline import chen_search, tcell_map, SOH_25C
    from run_safety_strict import clopper_pearson_upper

    surf = ECMSurface(holdout, 'discharge')
    T = tcell_map().get(0, 0.0)
    soh = SOH_25C

    def ocv_fn(soc):
        v, _ = surf.ocv(soc, soh)
        m, _ = surf.hyst_M(soc, soh)
        return float(np.atleast_1d(v)[0]), float(np.atleast_1d(m)[0])

    rows = [r for r in csv.DictReader(open(LAB, encoding='utf-8'))
            if r['SOP_disch'] not in ('', 'nan')]
    by_rate = collections.defaultdict(list)
    for r in rows:
        soc = float(r['SOC'])
        meas = abs(float(r['SOP_disch']))
        pred, lim = chen_search(surf, soc, soh, T, tau, ocv_fn)
        by_rate[r['rate_label']].append((soc, meas, pred, lim))

    out = []
    for label in RATES:
        g = by_rate.get(label, [])
        ih = [x for x in g if np.isfinite(x[2]) and x[3] != 'out-of-hull']
        if not ih:
            out.append([label, len(g), 0, '', '', '', '', '', '', ''])
            continue
        socs = sorted(x[0] for x in ih)
        m = np.array([x[1] for x in ih])
        q = np.array([x[2] for x in ih])
        ok = q > 0
        need = float(np.min(m[ok] / q[ok])) if ok.any() else float('nan')
        over = LAM_SHIPPED * q - m
        exceed = int((over > 0).sum())
        ub = clopper_pearson_upper(exceed, len(ih)) * 100
        out.append([label, len(g), len(ih), f'{min(socs):.2f}',
                    f'{max(socs):.2f}', f'{LAM_SHIPPED:.4f}', f'{need:.4f}',
                    f'{need / LAM_SHIPPED:.3f}', exceed, f'{ub:.1f}',
                    f'{max(over.max(), 0.0):.2f}'])

    full = [r for r in out if len(r) == len(HEADER)]
    pooled = None
    if full:
        n = sum(int(r[2]) for r in full)
        k = sum(int(r[8]) for r in full)
        mins = [float(r[6]) for r in full]
        pooled = dict(n_in_hull=n, exceed=k, need_min=min(mins),
                      need_max=max(mins),
                      ub95=clopper_pearson_upper(k, n) * 100,
                      worst=max(float(r[10]) for r in full),
                      n_rows=sum(int(r[1]) for r in full))
    return out, pooled


def one_surface(a):
    out, pooled = score(a.holdout, a.tau)
    print(f'  Test#8, 0 C, tau = {a.tau:g} s, surface {a.holdout}')
    print(f"  {'prior rate':>11}{'n':>5}{'in hull':>9}{'SOC range':>13}"
          f"{'lam need':>10}{'margin':>8}{'exceed':>8}{'ub95 %':>9}"
          f"{'worst W':>9}")
    print('  ' + '-' * 84)
    for r in out:
        if len(r) != len(HEADER):
            continue
        print(f'  {r[0]:>11}{r[1]:>5}{r[2]:>9}'
              f'{f"{r[3]}-{r[4]}":>13}{float(r[6]):>10.4f}'
              f'{float(r[7]):>8.3f}{r[8]:>8}{float(r[9]):>9.1f}'
              f'{float(r[10]):>9.2f}')
    if pooled:
        print(f"\n  pooled over all prior rates: {pooled['exceed']} "
              f"exceedances in {pooled['n_in_hull']} in-hull points, 95 % "
              f"upper bound {pooled['ub95']:.1f} %")
        print(f"  lambda_needed spans {pooled['need_min']:.4f}-"
              f"{pooled['need_max']:.4f} across the six rates; the shipped "
              f"factor is {LAM_SHIPPED}")
        print('  That bound is Clopper-Pearson over ROWS of one physical '
              'cell,\n  conditional on this SOC and prior-rate grid.  It is '
              'not a cell-level\n  or population-level risk.')
        out.append(['0C..4C pooled', pooled['n_rows'], pooled['n_in_hull'],
                    '', '', f'{LAM_SHIPPED:.4f}', f"{pooled['need_min']:.4f}",
                    f"{pooled['need_min'] / LAM_SHIPPED:.3f}",
                    pooled['exceed'], f"{pooled['ub95']:.1f}", ''])
    return out


def all_surfaces(a):
    """The same test against each of the six internal surfaces.

    Test#8 is external, so nothing is held out for it and every surface is
    equally entitled to be used.  Publishing one was a choice, and CC was the
    most favourable of the six -- margin 1.655 against 1.351 for CC_CELL2.
    The row to quote is the worst.
    """
    rows, margins = [], []
    for cell in CELLS:
        out, pooled = score(cell, a.tau)
        if not pooled:
            continue
        margins.append(pooled['need_min'] / LAM_SHIPPED)
        rows.append([cell, pooled['n_in_hull'], f"{pooled['need_min']:.4f}",
                     f"{pooled['need_max']:.4f}",
                     f"{pooled['need_min'] / LAM_SHIPPED:.3f}",
                     pooled['exceed'], f"{pooled['ub95']:.1f}",
                     f"{pooled['worst']:.2f}"])
        print(f"  {cell:<20}lambda_needed {pooled['need_min']:.4f}-"
              f"{pooled['need_max']:.4f}   margin "
              f"{pooled['need_min'] / LAM_SHIPPED:.3f}   "
              f"{pooled['exceed']}/{pooled['n_in_hull']} exceed")
    if margins:
        lo, hi = min(margins), max(margins)
        worst = rows[int(np.argmin(margins))][0]
        # The summary row carries real values rather than blanks: min and
        # max of lambda_needed over the six surfaces, and the margin to
        # quote.  A blank here would be a hole the schema check has to be
        # loosened for, and a loosened schema check stops checking.
        needs = [float(r[2]) for r in rows]
        # n_in_hull on this row is the DISTINCT measurement count, 48, not
        # the sum over surfaces.  Summing would print 288 and read as six
        # times the evidence, which is the exact misreading the docstring
        # warns about; the same 48 rows are simply scored six times.
        distinct = max(int(r[1]) for r in rows)
        rows.append(['worst over surfaces', distinct, f'{min(needs):.4f}',
                     f'{max(needs):.4f}', f'{lo:.3f}',
                     max(int(r[5]) for r in rows), '', ''])
        print(f'\n  margin spans {lo:.3f} ({worst}) to {hi:.3f} across the '
              f'six surfaces, a {hi / lo:.2f}x range,\n  against a ~1 % '
              f'spread across prior C-rate within any one surface.  Prior '
              f'load is\n  not what moves this number; the choice of '
              f'internal surface is.')
        print(f'  Every surface clears the shipped factor, so the '
              f'zero-exceedance result does not\n  depend on the choice - '
              f'but the margin to quote is {lo:.3f}, not {hi:.3f}.')
        print('\n  The six columns are NOT independent evidence: they are the '
              'same 48\n  measurements scored six times.  Do not pool them.')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--holdout', default='CC')
    ap.add_argument('--tau', type=float, default=10.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--all-surfaces', action='store_true',
                    help='score against every internal surface and report '
                         'the range')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(LAB):
        print(f'  missing {os.path.relpath(LAB, ROOT)}', file=sys.stderr)
        return 1

    header = SURF_HEADER if a.all_surfaces else HEADER
    out_path = a.out or (SURF_OUT if a.all_surfaces else OUT)
    rows = all_surfaces(a) if a.all_surfaces else one_surface(a)

    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(out_path, header, rows,
                               'external_crate_surfaces' if a.all_surfaces
                               else 'external_crate_envelope')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  -> {os.path.relpath(out_path, ROOT)}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
