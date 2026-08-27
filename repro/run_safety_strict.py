"""Strict held-out-cell safety calibration.

    python3 repro/run_safety_strict.py

**What this fixes.**  `run_safety.py` calibrates lambda leave-one-cell-out
six times and then takes the *median* of the six as a single lambda applied
to every cell.  Cell i therefore contributes to the lambda under which cell
i is scored: five of the six folds that enter the median were fitted on data
containing cell i.  That is not a held-out calibration, and the exceedance
count it produces is optimistic by construction.

Here each outer held-out cell i gets its own lambda_i, fitted with cell i
removed entirely, and lambda_i scores only cell i.  Nothing else about the
model changes — same trim, same aggregation, same tolerance.

**Reported, because a single pooled number hides the thing that matters.**
Per-cell lambda, per-cell exceedance and usable current, the worst cell, a
one-sided Clopper-Pearson upper bound on the exceedance rate, and a
cell-cluster bootstrap interval on usable current.  Cells are the
independent unit here, not rows: rows within a cell share a cell, so a
row-level bootstrap would understate the interval.

**Language.**  Zero observed exceedances is a measurement.  It is not zero
risk and not a guarantee — with n rows and no events the 95 % upper bound on
the true rate is roughly 3/n, which this table prints.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_safety import load, keep, TOL, EXTRAP_MAX   # noqa: E402

EVAL = os.path.join(ROOT, 'analysis', 'results', 'eval')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')
TAUS = (10.0, 2.0)
MIN_TRAIN = 25
N_BOOT = 4000
BOOT_SEED = 12345


def fit_lambda(pred, meas, tol, lo=0.02, hi=1.6, iters=70):
    """Largest lambda with no row exceeding `tol`, by bisection."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if np.max(mid * pred - meas) > tol:
            hi = mid
        else:
            lo = mid
    return lo


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided upper bound on a binomial rate.  k=0 gives 1-alpha^(1/n)."""
    if n == 0:
        return float('nan')
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def strict(d, tau, tol, key='hyb'):
    """Per-cell held-out calibration.  Returns per-cell records."""
    m = keep(d, tau, key)
    cells = sorted(set(d['cell'][m]))
    recs = []
    for c in cells:
        tr = m & (d['cell'] != c)
        te = m & (d['cell'] == c)
        if tr.sum() < MIN_TRAIN or te.sum() == 0:
            continue
        lam = fit_lambda(d[key][tr], d['meas'][tr], tol)
        p, y = d[key][te], d['meas'][te]
        over = lam * p - y
        recs.append(dict(
            cell=c, n=int(te.sum()), lam=lam,
            exceed=int((over > tol).sum()),
            worst=float(max(over.max(), 0.0)),
            usable=float(np.median(lam * p / y) * 100),
            optimism=float(np.mean(p > y) * 100),
            rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        ))
    return recs


def pooled_lambda(d, tau, tol, key='hyb'):
    """The current shipped estimator: median of the six LOCO lambdas."""
    m = keep(d, tau, key)
    out = []
    for c in sorted(set(d['cell'][m])):
        tr = m & (d['cell'] != c)
        if tr.sum() < MIN_TRAIN:
            continue
        out.append(fit_lambda(d[key][tr], d['meas'][tr], tol))
    return float(np.median(out)) if out else float('nan')


def bootstrap_usable(recs, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cell-cluster bootstrap CI on the row-weighted usable current."""
    if len(recs) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    u = np.array([r['usable'] for r in recs])
    w = np.array([r['n'] for r in recs], float)
    stat = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(recs), len(recs))
        stat[b] = np.average(u[i], weights=w[i])
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default=EVAL)
    ap.add_argument('--arm', default='oracle', choices=['oracle', 'est'],
                    help='oracle = true SOH fed to the inversion; '
                         'est = the SOH arm\'s own prediction')
    ap.add_argument('--tol-sweep', default='0.0,0.25,0.5,1.0',
                    help='tolerances for the sensitivity table, in A')
    ap.add_argument('--method', default='a8',
                    help='which evaluation tag to score: a8, a3, lstm, gru, '
                         'ffrls, shrink, direct, rls')
    a = ap.parse_args()

    os.makedirs(TABLES, exist_ok=True)
    percell, summary, sens = [], [], []

    for direction in ('discharge', 'charge'):
        tag = 'disc' if direction == 'discharge' else 'char'
        path = os.path.join(a.eval_dir, f'{a.method}_{tag}_{a.arm}.csv')
        if not os.path.exists(path):
            print(f'  MISSING {path} — run repro/run_evals.py', file=sys.stderr)
            return 1
        d = load(path)
        tol = TOL[direction]
        for tau in TAUS:
            recs = strict(d, tau, tol)
            if not recs:
                print(f'  no usable rows for {direction} tau={tau}',
                      file=sys.stderr)
                continue
            pool = pooled_lambda(d, tau, tol)
            n = sum(r['n'] for r in recs)
            k = sum(r['exceed'] for r in recs)
            lo, hi = bootstrap_usable(recs)
            ub = clopper_pearson_upper(k, n)
            lams = np.array([r['lam'] for r in recs])
            us = np.array([r['usable'] for r in recs])
            ws = np.array([r['n'] for r in recs], float)
            worst_i = int(np.argmin(us))

            for r in recs:
                percell.append([direction, f'{tau:.1f}', a.method, a.arm, r['cell'],
                                r['n'], f'{r["lam"]:.4f}', r['exceed'],
                                f'{r["worst"]:.3f}', f'{r["usable"]:.2f}',
                                f'{r["optimism"]:.1f}', f'{r["rmse"]:.3f}'])
            summary.append([
                direction, f'{tau:.1f}', a.method, a.arm, len(recs), n, k,
                f'{np.average(us, weights=ws):.2f}',
                f'{np.median(us):.2f}', f'{us[worst_i]:.2f}',
                recs[worst_i]['cell'],
                f'{lams.min():.4f}', f'{np.median(lams):.4f}',
                f'{lams.max():.4f}', f'{pool:.4f}',
                f'{ub * 100:.3f}', f'{lo:.2f}', f'{hi:.2f}'])

            for ts in [float(x) for x in a.tol_sweep.split(',')]:
                rs = strict(d, tau, ts)
                if not rs:
                    continue
                nn = sum(r['n'] for r in rs)
                kk = sum(r['exceed'] for r in rs)
                uu = np.array([r['usable'] for r in rs])
                ww = np.array([r['n'] for r in rs], float)
                sens.append([direction, f'{tau:.1f}', a.method, a.arm,
                             f'{ts:.2f}', nn,
                             kk, f'{np.average(uu, weights=ww):.2f}',
                             f'{uu.min():.2f}',
                             f'{clopper_pearson_upper(kk, nn) * 100:.3f}'])

    def write(name, header, rows):
        p = os.path.join(TABLES, name)
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f'  -> {os.path.relpath(p, ROOT)}  ({len(rows)} rows)')

    suf = a.arm if a.method == 'a8' else f'{a.method}_{a.arm}'
    write(f'safety_strict_percell_{suf}.csv',
          ['direction', 'tau_s', 'method', 'soh_arm', 'cell', 'n', 'lambda_i',
           'exceed', 'worst_A', 'usable_pct', 'optimism_pct', 'rmse_A'],
          percell)
    write(f'safety_strict_{suf}.csv',
          ['direction', 'tau_s', 'method', 'soh_arm', 'n_cells', 'n_rows',
           'exceed',
           'usable_mean_pct', 'usable_median_pct', 'usable_worst_pct',
           'worst_cell', 'lambda_min', 'lambda_median', 'lambda_max',
           'lambda_pooled_shipped', 'exceed_ub95_pct',
           'usable_boot_lo_pct', 'usable_boot_hi_pct'],
          summary)
    write(f'safety_strict_tolsens_{suf}.csv',
          ['direction', 'tau_s', 'method', 'soh_arm', 'tolerance_A', 'n_rows',
           'exceed',
           'usable_mean_pct', 'usable_worst_pct', 'exceed_ub95_pct'],
          sens)

    print(f"\n  {'direction':<10}{'tau':>5}{'lam min':>9}{'lam med':>9}"
          f"{'lam max':>9}{'shipped':>9}{'exc':>5}{'/n':>7}"
          f"{'ub95%':>8}{'usable':>8}{'worst':>8}  worst cell")
    print('  ' + '-' * 104)
    for s in summary:
        print(f'  {s[0]:<10}{s[1]:>5}{s[11]:>9}{s[12]:>9}{s[13]:>9}{s[14]:>9}'
              f'{s[6]:>5}{s[5]:>7}{s[15]:>8}{s[7]:>8}{s[9]:>8}  {s[10]}')
    print('\n  Zero observed exceedance is a measurement, not a guarantee: '
          'read exceed_ub95_pct.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
