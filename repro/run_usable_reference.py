"""What is "usable current" a percentage OF?

The safety tables report `usable = median(lambda * pred / meas)`, so 100 %
means the system permits exactly what the cell can actually deliver.  On the
adopted trim that lands near 65 % on discharge and 54 % on charge, which reads
as a weak result and is not one.  The denominator is the wrong thing to judge
the method by.

Lambda is the largest factor with no exceedance on the training cells, so it
is set by the single worst row of the most demanding cell -- BOOST_NEGPULSE_1S
in most conditions.  Every OTHER cell is then scored against its own true
capability, which no single-lambda policy can reach by construction.  The
metric is measuring the price of a fleet-wide safety margin and reporting it
as if it were model error.

Three reference points, all computed here, because each answers a different
question and only the first is currently published.

  usable_pct            against the cell's own measured SOP.  Perfect
                        prediction scores 100.  This is the published number.

  vs_fleet_pct          against the best single lambda fitted with ALL six
                        cells visible, including the evaluated one.  This is
                        the ceiling for any one-lambda policy, and it answers
                        "what does holding a cell out cost?"

                        READ IT WITH THE EXCEEDANCE COLUMNS.  A ratio above
                        100 does not mean free utility: it means the
                        leave-one-cell-out lambda came out LARGER than the
                        all-cells one, which is a more permissive policy, and
                        more permissive is less safe.  Comparing usable
                        current between two policies that carry different
                        exceedance is comparing two points on a safety/utility
                        trade-off and calling one of them better.  The first
                        version of this table did exactly that and concluded
                        "holding a cell out costs nothing"; it is not true,
                        and exceed_deployed / exceed_fleet are here so the
                        claim cannot be made again without the risk beside
                        it.

  vs_cell_oracle_pct    against a lambda fitted on the evaluated cell itself.
                        This is the ceiling for a per-cell field calibration,
                        and it answers "what would calibrating in the field
                        buy?"

The last two use the evaluated cell's own labels and are ORACLE BOUNDS.  They
may be reported and must never be used to choose lambda, the tolerance, or a
model -- that is selection on the test set, which is the defect this audit
exists to remove.

load(), keep() and fit_lambda() are imported from the modules that produce the
published tables rather than reimplemented.  A first draft of this analysis
did reimplement them, dropped the `meas > 0.5` and extrapolation filters that
keep() applies, and produced per-cell numbers that disagreed with
safety_strict_percell_est.csv by up to 4 %p.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
EVAL = os.path.join(ANALYSIS, 'results', 'eval')
OUT = os.path.join(ANALYSIS, 'results', 'tables', 'usable_reference.csv')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

HEADER = ['direction', 'tau_s', 'method', 'cell', 'n', 'lambda_deployed',
          'lambda_fleet', 'lambda_cell_oracle', 'usable_pct', 'vs_fleet_pct',
          'vs_cell_oracle_pct', 'exceed_deployed', 'exceed_fleet',
          'risk_matched', 'binding_cell']

METHODS = ['a8', 'a3', 'lstm', 'gru', 'ffrls', 'shrink']
DETAIL = 'a8'          # the adopted method gets a row per cell
TAUS = [10.0, 2.0]
MIN_TRAIN = 25


def condition_rows(method, direction, tau, arm='est'):
    from run_safety import load, keep, TOL
    from run_safety_strict import fit_lambda

    tag = 'disc' if direction == 'discharge' else 'char'
    path = os.path.join(EVAL, f'{method}_{tag}_{arm}.csv')
    if not os.path.exists(path):
        return []
    d = load(path)
    tol = TOL[direction]
    m = keep(d, tau)
    cells = sorted(set(d['cell'][m]))
    if len(cells) < 3:
        return []

    # The ceiling for a single lambda that has seen everything.
    lam_fleet = fit_lambda(d['hyb'][m], d['meas'][m], tol)

    per = []
    for c in cells:
        tr, te = m & (d['cell'] != c), m & (d['cell'] == c)
        if tr.sum() < MIN_TRAIN or te.sum() == 0:
            continue
        lam_dep = fit_lambda(d['hyb'][tr], d['meas'][tr], tol)
        lam_or = fit_lambda(d['hyb'][te], d['meas'][te], tol)
        p, y = d['hyb'][te], d['meas'][te]
        # Exceedance under BOTH policies, because vs_fleet_pct compares two
        # factors that do not carry the same risk.  Where lam_dep > lam_fleet
        # the deployed policy permits more current, and any figure above 100 %
        # is bought with that, not free.
        per.append(dict(
            cell=c, n=int(te.sum()), lam_dep=lam_dep, lam_or=lam_or,
            usable=float(np.median(lam_dep * p / y) * 100),
            u_fleet=float(np.median(lam_fleet * p / y) * 100),
            u_oracle=float(np.median(lam_or * p / y) * 100),
            k_dep=int((lam_dep * p - y > tol).sum()),
            k_fleet=int((lam_fleet * p - y > tol).sum())))
    if not per:
        return []
    binding = min(per, key=lambda r: r['lam_or'])['cell']

    rows = []
    if method == DETAIL:
        for r in per:
            rows.append([direction, f'{tau:.1f}', method, r['cell'], r['n'],
                         f"{r['lam_dep']:.4f}", f'{lam_fleet:.4f}',
                         f"{r['lam_or']:.4f}", f"{r['usable']:.2f}",
                         f"{100 * r['usable'] / r['u_fleet']:.1f}",
                         f"{100 * r['usable'] / r['u_oracle']:.1f}",
                         r['k_dep'], r['k_fleet'],
                         'yes' if r['k_dep'] == r['k_fleet'] else 'NO',
                         binding])
    # Row-weighted, the same convention safety_strict.py uses for
    # usable_mean_pct.  A plain per-cell mean gives 67.01 where the published
    # table says 68.89 for the same quantity, and two numbers for one thing
    # is how a reader stops trusting either.
    w = np.array([r['n'] for r in per], float)
    # vs_fleet is averaged over the RISK-MATCHED folds only.  A fold where the
    # two factors differ in exceedance is not comparable on usable current,
    # and averaging it in is what produced the 100.4 that read as "free".
    mt = [r for r in per if r['k_dep'] == r['k_fleet']]
    wm = np.array([r['n'] for r in mt], float)
    vs_fleet = (f"{np.average([100 * r['usable'] / r['u_fleet'] for r in mt], weights=wm):.1f}"
                if mt else '')
    rows.append([direction, f'{tau:.1f}', method, '(mean)',
                 int(w.sum()), '', f'{lam_fleet:.4f}', '',
                 f"{np.average([r['usable'] for r in per], weights=w):.2f}",
                 vs_fleet,
                 f"{np.average([100 * r['usable'] / r['u_oracle'] for r in per], weights=w):.1f}",
                 sum(r['k_dep'] for r in per), sum(r['k_fleet'] for r in per),
                 f'{len(mt)}/{len(per)}', binding])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', default='est', choices=['oracle', 'est'],
                    help='which SOH the inversion was given; est is the '
                         'deployment condition and the default')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    rows = []
    for method in METHODS:
        for direction in ('discharge', 'charge'):
            for tau in TAUS:
                rows += condition_rows(method, direction, tau, a.arm)
    if not rows:
        print(f'  no {a.arm} evaluation files found in '
              f'{os.path.relpath(EVAL, ROOT)}', file=sys.stderr)
        return 1

    det = [r for r in rows if r[3] != '(mean)']
    print(f'  adopted trim, per cell, SOH arm = {a.arm}\n')
    print(f"  {'condition':<17}{'cell':<20}{'lam_dep':>9}{'usable%':>9}"
          f"{'vs fleet':>10}{'exc dep':>9}{'exc fleet':>10}")
    print('  ' + '-' * 86)
    for r in det:
        flag = '  <- looser than the fleet factor' if r[11] > r[12] else ''
        print(f"  {r[0] + ' t=' + r[1][:-2]:<17}{r[3]:<20}{r[5]:>9}"
              f"{r[8]:>9}{r[9]:>10}{r[11]:>9}{r[12]:>10}{flag}")

    print("\n  every method, condition means\n")
    print(f"  {'method':<9}{'condition':<17}{'usable%':>9}{'vs fleet':>10}"
          f"{'vs cell oracle':>16}   binding cell")
    print('  ' + '-' * 88)
    for r in [x for x in rows if x[3] == '(mean)']:
        print(f"  {r[2]:<9}{r[0] + ' t=' + r[1][:-2]:<17}{r[8]:>9}{r[9]:>10}"
              f"{r[10]:>16}   {r[11]}")

    same = [r for r in det if r[5] == r[6]]
    loose = [r for r in det if r[5] != r[6]]
    kd = sum(r[11] for r in loose)
    kf = sum(r[12] for r in loose)
    print(f'\n  For {len(same)} of {len(det)} held-out cells the deployed '
          f'lambda EQUALS the all-cells lambda\n  exactly, so vs_fleet is '
          f'100.0 by identity, not by measurement: the binding cell is in\n'
          f'  the training set and fixes the factor.')
    print(f'  The other {len(loose)} are the binding cell itself held out.  '
          f'There the deployed lambda is\n  LOOSER than the fleet factor, '
          f'which is why vs_fleet exceeds 100 -- and it is not free:\n'
          f'  those rows carry {kd} exceedances against {kf} under the fleet '
          f'factor.  Every exceedance\n  in this arm is there.')
    print('  So "holding a cell out costs nothing" is wrong as a summary.  It '
          'costs nothing in\n  usable current precisely because it costs '
          'something in exceedance, and the two\n  policies being compared '
          'do not carry the same risk.')
    oc = [float(r[10]) for r in rows if r[3] == '(mean)']
    print(f'  Against a per-cell oracle lambda it scores {min(oc):.1f}-'
          f'{max(oc):.1f} %, so a field calibration\n  per cell would be '
          f'worth roughly {100 - max(oc):.0f}-{100 - min(oc):.0f} %p of '
          f'usable current.')
    print('\n  Both comparisons use the evaluated cell\'s own labels.  They '
          'are oracle bounds:\n  report them, never select on them.')

    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(OUT, HEADER, rows, 'usable_reference')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(OUT, ROOT)}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
