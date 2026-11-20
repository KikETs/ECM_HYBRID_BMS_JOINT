"""Shared helpers for the review-2 analyses (read-only use of the reproduced baseline)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import csv, os, pickle, sys
from collections import defaultdict
import numpy as np

REPO = REPO_ROOT
ARMS = os.path.join(R2, 'baseline', 'arms', 'arm_rows_replace_level2_hull2.pkl')
CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
DIRECTIONS = ('discharge', 'charge')
TAUS = (2.0, 10.0)
TOL = {'discharge': 0.0, 'charge': 0.5}   # submitted policy; charge 0.0 is the retrospective zero-tolerance policy


def out_dir(name):
    d = os.path.join(R2, 'out', name)
    os.makedirs(d, exist_ok=True)
    return d


def load_arms(path=ARMS):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty ' + path)
    keys = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fit_lambda(pred, meas, tol, lo=0.02, hi=1.6, iters=70):
    """identical to repro/run_safety_strict.fit_lambda"""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if np.max(mid * pred - meas) > tol:
            hi = mid
        else:
            lo = mid
    return lo


def keep_factory(tau, extrap_max=1.5):
    def keep(r):
        return r['tau'] == tau and r['status'] == 'ok' and r['meas'] > .5 and r['extrap'] <= extrap_max
    return keep


def split_rows(arms, direction, tau, c, arm='eo', extrap_max=1.5, inner_source=None):
    keep = keep_factory(tau, extrap_max)
    src = inner_source if inner_source is not None else arms
    outer = [r for r in arms['outer', 'NONE', c, direction, arm, 'replace'] if keep(r)]
    inner = [r for j in CELLS if j != c for r in src['inner', c, j, direction, arm, 'replace'] if keep(r)]
    return inner, outer


def paper_usable(per_cell):
    """per_cell: list of (median_usable_pct, n_outer) -> row-weighted mean (manuscript definition)"""
    a = np.array([x[0] for x in per_cell]); n = np.array([x[1] for x in per_cell])
    return float(np.average(a, weights=n))


def cell_boot(values, weights, B=4000, seed=12345):
    v = np.asarray(values, float); w = np.asarray(weights, float)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(v), (B, len(v)))
    return np.quantile(np.sum(v[idx] * w[idx], axis=1) / np.sum(w[idx], axis=1), [.025, .975])


def score_condition(arms, direction, tau, model='hyb', tol=None, yfun=None, extrap_max=1.5, rowfilter=None,
                    lam_override=None, arm='eo', min_inner=25):
    """Leave-one-cell-out: lambda_c on inner rows of the other cells, scored on the outer cell.
    yfun(r) -> reference current (abs) or None to drop the row; default r['meas'] (published reference).
    Returns per-cell records."""
    tol = TOL[direction] if tol is None else tol
    yfun = yfun or (lambda r: r['meas'])
    recs = []
    for c in CELLS:
        inner, outer = split_rows(arms, direction, tau, c, arm, extrap_max)
        if rowfilter is not None:
            inner = [r for r in inner if rowfilter(r)]; outer = [r for r in outer if rowfilter(r)]
        yi = [(r[model], yfun(r)) for r in inner]; yi = [(p, y) for p, y in yi if y is not None and np.isfinite(y) and y > .5]
        yo = [(r[model], yfun(r), r) for r in outer]; yo = [(p, y, r) for p, y, r in yo if y is not None and np.isfinite(y) and y > .5]
        if not yo or len(yi) < min_inner:
            continue
        lam = fit_lambda(np.array([p for p, _ in yi]), np.array([y for _, y in yi]), tol) if lam_override is None else lam_override[c]
        p = lam * np.array([q for q, _, _ in yo]); y = np.array([q for _, q, _ in yo])
        d = p - y
        recs.append(dict(direction=direction, tau_s=tau, cell=c, model=model, tol=tol, lam=lam, n_inner=len(yi), n_outer=len(yo),
                         exceed=int((d > tol).sum()), exceed_strict0=int((d > 0).sum()), max_pos_A=max(0., float(d.max())),
                         usable_med_pct=float(np.median(p / y) * 100), usable_mean_pct=float(np.mean(p / y) * 100),
                         ratios=p / y, rows=[r for _, _, r in yo]))
    return recs


def summarize(recs):
    if not recs:
        return {}
    pc = [(r['usable_med_pct'], r['n_outer']) for r in recs]
    lo, hi = cell_boot([x[0] for x in pc], [x[1] for x in pc])
    return dict(n_cells=len(recs), n_rows=int(sum(r['n_outer'] for r in recs)), exceed=int(sum(r['exceed'] for r in recs)),
                exceed_gt0=int(sum(r['exceed_strict0'] for r in recs)), max_pos_A=max(r['max_pos_A'] for r in recs),
                usable_pct=paper_usable(pc), boot_lo=float(lo), boot_hi=float(hi),
                usable_cell_equal_pct=float(np.mean([x[0] for x in pc])),
                lam_min=min(r['lam'] for r in recs), lam_max=max(r['lam'] for r in recs))
