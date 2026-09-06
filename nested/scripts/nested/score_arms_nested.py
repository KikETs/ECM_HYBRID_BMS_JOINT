"""Score the four state-input arms from arm_rows_<rule>[_<solver>].pkl.

Protocols (all with per-outer-cell lambda_c, MIN_TRAIN 25, shipped keep():
|meas| > 0.5 A, extrap <= 1.5, both models finite):

  nested   lambda_c fitted on the INNER rows of outer c (assets built without
           c), applied to c's outer rows                    <- the R01 protocol
  shipped  lambda_c fitted on the OUTER rows of the other five cells
           (assets that include c), as run_safety_strict does  <- reference only

Tables written (per rule):
  full        every eligible row of each arm, its own calibration
  paired      rows common to the four arms (outer keys), calibration on the
              inner rows common to the four arms (per outer c)
  fixed       paired rows, lambda taken from the oo arm's paired calibration
  drift       outer rows outside the common set; calibration on the inner rows
              outside the inner common set (unscored if < MIN_TRAIN)
  counts      raw / no_k / no_prior_drive / hull / eligible / common / outside
              / calibratable / scored / unscored per arm and condition
"""
import csv
import os
import pickle
import sys

import numpy as np

REPO = os.environ['RCSOP_REPO']
HERE = os.environ.get('OUTDIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'repro'))
from run_safety import TOL, EXTRAP_MAX                     # noqa: E402
from run_safety_strict import (fit_lambda, clopper_pearson_upper,  # noqa: E402
                               bootstrap_usable, MIN_TRAIN)

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST',
         'CC', 'CC_CELL2']
ARMS = ['oo', 'eo', 'oe', 'ee']
LABEL = {'oo': 'oracle SOH + oracle SOC', 'eo': 'estimated SOH + oracle SOC',
         'oe': 'oracle SOH + estimated SOC',
         'ee': 'estimated SOH + estimated SOC'}
TAUS = (10.0, 2.0)


def keep(r, tau, key='hyb'):
    return (r['status'] == 'ok' and r['tau'] == tau and r['meas'] > 0.5
            and r['extrap'] <= EXTRAP_MAX and np.isfinite(r[key]))


def score(rows, lam, tol, key):
    p = np.array([r[key] for r in rows]); y = np.array([r['meas'] for r in rows])
    over = lam * p - y
    return dict(n=len(rows), exceed=int((over > tol).sum()),
                worst=float(max(over.max(), 0.0)),
                usable=float(np.median(lam * p / y) * 100))


def agg(recs):
    n = sum(r['n'] for r in recs); k = sum(r['exceed'] for r in recs)
    us = np.array([r['usable'] for r in recs]); w = np.array([r['n'] for r in recs], float)
    lo, hi = bootstrap_usable(recs)
    return dict(n_cells=len(recs), n=n, exceed=k, cp95=clopper_pearson_upper(k, n) * 100,
                worst=max(r['worst'] for r in recs), usable=float(np.average(us, weights=w)),
                usable_worst=float(us.min()), worst_cell=recs[int(np.argmin(us))]['cell'],
                lam_min=min(r['lam'] for r in recs), lam_med=float(np.median([r['lam'] for r in recs])),
                lam_max=max(r['lam'] for r in recs), lo=lo, hi=hi,
                cells_exceed=sum(1 for r in recs if r['exceed'] > 0),
                per_cell=' '.join(f"{r['cell']}:{r['exceed']}/{r['n']}" for r in recs))


def main(rule='replace', tag='replace', key='hyb'):
    D = pickle.load(open(os.path.join(HERE, f'arm_rows_{tag}.pkl'), 'rb'))
    outer = {(c, d, a): D[('outer', 'NONE', c, d, a, rule)] for c in CELLS
             for d in ('discharge', 'charge') for a in ARMS}
    inner = {(c, d, a): [r for j in CELLS if j != c
                         for r in D[('inner', c, j, d, a, rule)]]
             for c in CELLS for d in ('discharge', 'charge') for a in ARMS}
    T = {k: [] for k in ('full', 'paired', 'fixed', 'drift', 'counts', 'percell')}
    for d in ('discharge', 'charge'):
        tol = TOL[d]
        for tau in TAUS:
            # eligible outer rows per arm, keyed
            el = {a: {c: [r for r in outer[(c, d, a)] if keep(r, tau, key)] for c in CELLS}
                  for a in ARMS}
            eli = {a: {c: [r for r in inner[(c, d, a)] if keep(r, tau, key)] for c in CELLS}
                   for a in ARMS}
            common = None
            for a in ARMS:
                ks = {r['key'] for c in CELLS for r in el[a][c]}
                common = ks if common is None else common & ks
            common_in = {}
            for c in CELLS:
                ci = None
                for a in ARMS:
                    ks = {r['key'] for r in eli[a][c]}
                    ci = ks if ci is None else ci & ks
                common_in[c] = ci
            lam_fixed = {}
            for a in ARMS:
                # ---- counts
                raw = sum(1 for c in CELLS for r in outer[(c, d, a)] if r['tau'] == tau)
                st = {}
                for s in ('no_k', 'no_prior_drive', 'hull'):
                    st[s] = sum(1 for c in CELLS for r in outer[(c, d, a)]
                                if r['tau'] == tau and r['status'] == s)
                n_el = sum(len(el[a][c]) for c in CELLS)
                # ---- full (own eligible rows, nested calibration)
                recs, recs_sh = [], []
                for c in CELLS:
                    te = el[a][c]
                    if not te:
                        continue
                    tr = eli[a][c]
                    if len(tr) >= MIN_TRAIN:
                        lam = fit_lambda(np.array([r[key] for r in tr]),
                                         np.array([r['meas'] for r in tr]), tol)
                        recs.append(dict(cell=c, lam=lam, n_train=len(tr), **score(te, lam, tol, key)))
                    tr2 = [r for j in CELLS if j != c for r in el[a][j]]
                    if len(tr2) >= MIN_TRAIN:
                        lam2 = fit_lambda(np.array([r[key] for r in tr2]),
                                          np.array([r['meas'] for r in tr2]), tol)
                        recs_sh.append(dict(cell=c, lam=lam2, n_train=len(tr2), **score(te, lam2, tol, key)))
                for proto, rr in (('nested', recs), ('shipped', recs_sh)):
                    if rr:
                        g = agg(rr)
                        T['full'].append([LABEL[a], d, tau, proto, g['n_cells'], g['n'], g['exceed'],
                                          f"{g['cp95']:.3f}", f"{g['worst']:.3f}", f"{g['usable']:.2f}",
                                          f"{g['usable_worst']:.2f}", g['worst_cell'], f"{g['lam_min']:.4f}",
                                          f"{g['lam_med']:.4f}", f"{g['lam_max']:.4f}", f"{g['lo']:.2f}",
                                          f"{g['hi']:.2f}", g['cells_exceed'], g['per_cell']])
                        for r in rr:
                            T['percell'].append([LABEL[a], d, tau, proto, r['cell'], r['n'], r['n_train'],
                                                 f"{r['lam']:.4f}", r['exceed'], f"{r['worst']:.3f}",
                                                 f"{r['usable']:.2f}"])
                # ---- paired
                prec = []
                for c in CELLS:
                    te = [r for r in el[a][c] if r['key'] in common]
                    tr = [r for r in eli[a][c] if r['key'] in common_in[c]]
                    if not te or len(tr) < MIN_TRAIN:
                        continue
                    lam = fit_lambda(np.array([r[key] for r in tr]),
                                     np.array([r['meas'] for r in tr]), tol)
                    prec.append(dict(cell=c, lam=lam, n_train=len(tr), **score(te, lam, tol, key)))
                if prec:
                    g = agg(prec)
                    T['paired'].append([LABEL[a], d, tau, g['n_cells'], g['n'], g['exceed'], f"{g['cp95']:.3f}",
                                        f"{g['worst']:.3f}", f"{g['usable']:.2f}", f"{g['usable_worst']:.2f}",
                                        f"{g['lam_min']:.4f}", f"{g['lam_med']:.4f}", f"{g['lo']:.2f}",
                                        f"{g['hi']:.2f}", g['cells_exceed'], g['per_cell']])
                    if a == 'oo':
                        lam_fixed = {r['cell']: r['lam'] for r in prec}
                # ---- fixed lambda (from oo paired)
                if lam_fixed:
                    frec = []
                    for c in CELLS:
                        te = [r for r in el[a][c] if r['key'] in common]
                        if not te or c not in lam_fixed:
                            continue
                        frec.append(dict(cell=c, lam=lam_fixed[c], **score(te, lam_fixed[c], tol, key)))
                    if frec:
                        g = agg(frec)
                        T['fixed'].append([LABEL[a], d, tau, g['n_cells'], g['n'], g['exceed'], f"{g['cp95']:.3f}",
                                           f"{g['worst']:.3f}", f"{g['usable']:.2f}", f"{g['usable_worst']:.2f}",
                                           f"{g['lam_med']:.4f}", g['cells_exceed'], g['per_cell']])
                # ---- drift: rows outside common
                n_out = sum(1 for c in CELLS for r in el[a][c] if r['key'] not in common)
                drec, calibratable, scored = [], 0, 0
                for c in CELLS:
                    te = [r for r in el[a][c] if r['key'] not in common]
                    tr = [r for r in eli[a][c] if r['key'] not in common_in[c]]
                    if not te:
                        continue
                    if len(tr) < MIN_TRAIN:
                        continue
                    calibratable += 1
                    lam = fit_lambda(np.array([r[key] for r in tr]),
                                     np.array([r['meas'] for r in tr]), tol)
                    drec.append(dict(cell=c, lam=lam, **score(te, lam, tol, key)))
                    scored += len(te)
                k_out = sum(r['exceed'] for r in drec)
                T['drift'].append([LABEL[a], d, tau, n_el, len(common), n_out, scored, n_out - scored,
                                   k_out, f'{100 * k_out / scored:.2f}' if scored else '',
                                   ' '.join(f"{r['cell']}:{r['exceed']}/{r['n']}" for r in drec)])
                T['counts'].append([LABEL[a], d, tau, raw, st['no_k'], st['no_prior_drive'], st['hull'],
                                    n_el, len(common), n_out, calibratable, scored, n_out - scored])
    heads = {
        'full': ['condition', 'direction', 'tau_s', 'protocol', 'n_cells', 'n_rows', 'exceed', 'cp95_pct',
                 'worst_overshoot_A', 'usable_mean_pct', 'usable_worst_pct', 'worst_cell', 'lambda_min',
                 'lambda_median', 'lambda_max', 'boot_lo', 'boot_hi', 'cells_with_exceed', 'per_cell'],
        'percell': ['condition', 'direction', 'tau_s', 'protocol', 'cell', 'n', 'n_train', 'lambda',
                    'exceed', 'worst_A', 'usable_pct'],
        'paired': ['condition', 'direction', 'tau_s', 'n_cells', 'n_rows', 'exceed', 'cp95_pct',
                   'worst_overshoot_A', 'usable_mean_pct', 'usable_worst_pct', 'lambda_min',
                   'lambda_median', 'boot_lo', 'boot_hi', 'cells_with_exceed', 'per_cell'],
        'fixed': ['condition', 'direction', 'tau_s', 'n_cells', 'n_rows', 'exceed', 'cp95_pct',
                  'worst_overshoot_A', 'usable_mean_pct', 'usable_worst_pct', 'lambda_median_from_oo',
                  'cells_with_exceed', 'per_cell'],
        'drift': ['condition', 'direction', 'tau_s', 'n_eligible', 'n_common', 'n_outside', 'n_scored',
                  'n_unscored', 'exceed', 'exceed_rate_pct', 'per_cell'],
        'counts': ['condition', 'direction', 'tau_s', 'raw_label_rows', 'no_trim_k', 'no_prior_drive',
                   'solver_fail', 'eligible', 'common', 'outside', 'cells_calibratable_outside',
                   'scored_outside', 'unscored_outside'],
    }
    for k, rows in T.items():
        p = os.path.join(HERE, f'e2e_nested_{k}_{tag}_{key}.csv')
        with open(p, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(heads[k]); w.writerows(rows)
        print(f'  -> {os.path.basename(p)} ({len(rows)} rows)')


if __name__ == '__main__':
    main(*(sys.argv[1:]))
