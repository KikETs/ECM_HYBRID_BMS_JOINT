"""Build the paper's tables from the evaluation CSVs.

Produces:
  results/tables/safety.csv   safety factors and deployment metrics of the
                              adopted configuration
  results/tables/ladder.csv   the parameter ladder (section 32)
  results/tables/soh_cost.csv oracle SOH vs estimated SOH (section 29)

Every lambda is the median of the values set leaving one cell out.  The cell
being evaluated was not used to set its lambda.

The tolerance differs by direction — discharge 0.0 A, charge 0.5 A.  Charge's
0.5 A is the knee section 25 settled on; without it section 16's lambda cannot
be reproduced (31.1).
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, 'analysis', 'results', 'eval')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

EXTRAP_MAX = 1.5
TOL = {'discharge': 0.0, 'charge': 0.5}
TAUS = (10.0, 2.0)

LADDER = [('A0  no correction', 0, 'a8', 'ecm'),
          ('direct plug-in', 0, 'direct', 'hyb'),
          ('shrinkage coefficient', 2, 'shrink', 'hyb'),
          ('A8  dR_fast alone', 4, 'a8', 'hyb'),
          ('A3  12 features', 26, 'a3', 'hyb'),
          ('[upper bound] HPPC-RLS', 0, 'rls', 'hyb'),
          ('LSTM (seq, 5954)', 5954, 'lstm', 'hyb'),
          ('GRU (seq, 4482)', 4482, 'gru', 'hyb'),
          ('FFRLS adaptive ECM', 1, 'ffrls', 'hyb')]


def load(path):
    r = list(csv.DictReader(open(path, encoding='utf-8')))

    def g(k):
        return np.array([float(x[k]) if x[k] not in ('', 'nan') else np.nan
                         for x in r])
    return dict(meas=np.abs(g('I_meas_A')), hyb=np.abs(g('I_A3_A')),
                ecm=np.abs(g('I_A0_A')), extrap=g('extrap'), soh=g('SOH'),
                soc=g('SOC'), tau=g('tau_s'),
                cell=np.array([x['cell'] for x in r]))


def keep(d, tau, key='hyb'):
    return (np.isfinite(d['meas']) & np.isfinite(d[key]) & (d['meas'] > 0.5)
            & (d['extrap'] <= EXTRAP_MAX) & (np.round(d['tau'], 1) == tau))


def lam_loco(d, tau, tol, key='hyb', min_train=25):
    m = keep(d, tau, key)
    out = []
    for c in sorted(set(d['cell'][m])):
        tr = m & (d['cell'] != c)
        if tr.sum() < min_train:
            continue
        lo, hi = 0.02, 1.6
        for _ in range(70):
            mid = (lo + hi) / 2
            if np.max(mid * d[key][tr] - d['meas'][tr]) > tol:
                hi = mid
            else:
                lo = mid
        out.append(lo)
    return float(np.median(out)) if out else float('nan')


def metrics(path, tau, tol, key='hyb'):
    d = load(path)
    m = keep(d, tau, key)
    if m.sum() == 0:
        return None
    lam = lam_loco(d, tau, tol, key)
    p, y = d[key][m], d['meas'][m]
    o = lam * p - y
    return dict(n=int(m.sum()), lam=lam,
                optimism=float(np.mean(p > y) * 100),
                rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                util=float(np.median(lam * p / y) * 100),
                worst=float(max(o.max(), 0.0)))


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  -> {os.path.relpath(path, ROOT)}  ({len(rows)} rows)',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default=EVAL)
    a = ap.parse_args()
    ev = lambda n: os.path.join(a.eval_dir, f'{n}.csv')

    # --- adopted configuration ---------------------------------------
    rows = []
    for direction in ('discharge', 'charge'):
        for tau in TAUS:
            for arm in ('oracle', 'est'):
                p = ev(f'a8_{direction[:4]}_{arm}')
                if not os.path.exists(p):
                    continue
                r = metrics(p, tau, TOL[direction])
                if r:
                    rows.append([direction, tau, arm, r['n'],
                                 f"{r['lam']:.3f}", f"{r['optimism']:.1f}",
                                 f"{r['rmse']:.2f}", f"{r['util']:.1f}",
                                 f"{r['worst']:.2f}"])
    write(os.path.join(TABLES, 'safety.csv'),
          ['direction', 'tau_s', 'soh', 'n', 'lambda', 'optimism_pct',
           'rmse_A', 'usable_pct', 'worst_A'], rows)

    # --- parameter ladder ---------------------------------------------
    rows = []
    for direction in ('discharge', 'charge'):
        for tau in TAUS:
            for nm, par, tag, key in LADDER:
                p = ev(f'{tag}_{direction[:4]}_oracle')
                if not os.path.exists(p):
                    continue
                r = metrics(p, tau, TOL[direction], key)
                if r:
                    rows.append([direction, tau, nm, par, r['n'],
                                 f"{r['lam']:.3f}", f"{r['optimism']:.1f}",
                                 f"{r['rmse']:.2f}", f"{r['util']:.1f}"])
    write(os.path.join(TABLES, 'ladder.csv'),
          ['direction', 'tau_s', 'method', 'params', 'n', 'lambda',
           'optimism_pct', 'rmse_A', 'usable_pct'], rows)

    # --- the price of estimated SOH -----------------------------------
    rows = []
    for direction in ('discharge', 'charge'):
        for tau in TAUS:
            for tag in ('a8', 'a3'):
                ro = metrics(ev(f'{tag}_{direction[:4]}_oracle'), tau,
                             TOL[direction])
                pe = ev(f'{tag}_{direction[:4]}_est')
                re_ = metrics(pe, tau, TOL[direction]) if os.path.exists(pe) \
                    else None
                if ro and re_:
                    rows.append([direction, tau, tag,
                                 f"{ro['lam']:.3f}", f"{re_['lam']:.3f}",
                                 f"{ro['util']:.1f}", f"{re_['util']:.1f}",
                                 f"{re_['util'] - ro['util']:+.1f}"])
    write(os.path.join(TABLES, 'soh_cost.csv'),
          ['direction', 'tau_s', 'trim', 'lambda_oracle', 'lambda_est',
           'usable_oracle_pct', 'usable_est_pct', 'delta_pct_pt'], rows)


if __name__ == '__main__':
    main()
