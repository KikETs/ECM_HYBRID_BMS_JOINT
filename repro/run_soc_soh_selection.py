"""Was the SOC filter chosen under a condition it will never deploy in?

The adopted EKF configuration -- gate at 1 A, 30 s rest hold -- was picked on
soc_perturb_bench.py, which hands every filter its cell's TRUE SOH.  In
deployment SOH comes from the ridge estimator with an RMSE near 0.01, and
soc_est_soh.py re-scored the adopted configuration under that: +0.068 %p, of
which the R_volt schedule accounts for +0.003 and the surface axis the rest.

But re-scoring is not re-selecting.  soc_est_soh.py hard-codes i_gate=1.0 and
rest_hold_s=30.0, so it can say what the chosen filter costs under estimated
SOH and cannot say whether it is still the one that would be chosen.  Picking
a model under a condition the deployment does not have is the same defect as
scoring an external test against one surface out of six: the answer may be
right, but nothing in the repository has checked.

So the whole comparison is re-run under four SOH inputs:

    oracle        the true SOH, which is what the original selection saw
    estimated     the ridge estimate, which is what ships
    bias +0.02    a deliberate offset, about 2x the estimator's RMSE
    bias -0.02    the same the other way

Selection metric is the mean over the six sensor disturbances, the same one
the headline uses.  If the ranking is stable, the original choice stands and
the estimated-SOH number can be quoted with it.  If it is not, the adopted
configuration was selected on information it will not have.

A bias is not the same experiment as the estimator's error and both are here
on purpose.  The estimator's error varies by cell and correlates with cell
condition; a fixed offset does not, and it separates "the filter is sensitive
to SOH being wrong" from "the filter is sensitive to THIS estimator".
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

OUT = os.path.join(ANALYSIS, 'results', 'tables', 'soc_soh_selection.csv')
HEADER = ['soh_input', 'config', 'undisturbed_pct', 'mean_6_pct',
          'worst_of_6_pct', 'rank_by_mean6', 'delta_vs_oracle_pct']

SOH_MODES = [('oracle', None), ('estimated', 'est'),
             ('bias +0.02', +0.02), ('bias -0.02', -0.02)]

RUNS = None
SOHEST = None


def rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def soh_for(run_index, mode, true_soh):
    """The SOH this arm hands the filter.

    Deliberately NOT clipped.  The first version clamped to [0.70, 1.00] on
    the theory that a bias should not push the query outside the surfaces,
    and it silently changed 5 of the 36 estimated values -- reproducing
    1.573 %p where soc_est_soh.py publishes 1.592.  The true SOH minimum is
    0.6821, already below the surface range of (0.7009, 0.99952), so the
    ORACLE arm queries outside it too and the surfaces evidently handle it.
    Clipping one arm and not the others would have compared four different
    experiments.  Out-of-range queries are counted and reported instead.
    """
    if mode is None:
        return true_soh
    return float(SOHEST[run_index][1] if mode == 'est' else true_soh + mode)


def out_of_range(lo=0.7009, hi=0.99952):
    """How many (run, arm) queries fall outside the surface's SOH range."""
    n = {}
    for mname, mode in SOH_MODES:
        k = sum(1 for i, r in enumerate(RUNS)
                if not (lo <= soh_for(i, mode, float(r['soh'])) <= hi))
        n[mname] = k
    return n


def _one(job):
    from ekf_soc import run as ekf_run
    import soc_perturb_bench as B
    ci, pi, mi = job
    cname, ckw = B.CONFIGS[ci]
    pkw = B.PERTURB[pi][1]
    ckw = dict(ckw)
    open_loop = ckw.pop('_open', False)
    _, mode = SOH_MODES[mi]
    err = []
    for i, r in enumerate(RUNS):
        soh = soh_for(i, mode, float(r['soh']))
        rv = 1e4 if open_loop else rvolt(soh)
        If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
        s0 = min(max(float(r['soc'][0]) + pkw.get('dsoc', 0.0), 0.02), 0.98)
        e, _ = ekf_run(r['sd'], r['sc'], soh, If, r['V'], r['T'], s0, rv,
                       gamma=0.0, **ckw)
        err.append(float(np.sqrt(np.mean((e - r['soc']) ** 2))))
    return job, np.array(err)


def main():
    global RUNS, SOHEST
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--procs', type=int, default=14)
    a = ap.parse_args()

    import multiprocessing as mp
    import pickle
    import soc_perturb_bench as B
    import soc_est_soh as E

    os.chdir(ANALYSIS)
    RUNS = pickle.load(open('results/soc_runs.pkl', 'rb'))
    E.RUNS = RUNS
    SOHEST = E.build_map()
    d = np.array([e - r['soh'] for r, (_, e) in zip(RUNS, SOHEST)])
    print(f'  SOH estimate over {len(RUNS)} runs: RMSE '
          f'{np.sqrt((d ** 2).mean()):.4f}, bias {d.mean():+.4f}, '
          f'worst {np.abs(d).max():.4f}')

    jobs = [(c, p, m) for m in range(len(SOH_MODES))
            for c in range(len(B.CONFIGS)) for p in range(len(B.PERTURB))]
    oor = out_of_range()
    print('  queries outside the surface SOH range (0.7009-0.99952), of '
          f'{len(RUNS)}: '
          + ', '.join(f'{k} {v}' for k, v in oor.items())
          + '\n  Not clipped: the oracle arm is outside it too, and '
            'clipping one arm would compare\n  four different experiments.')
    print(f'  {len(jobs)} jobs = {len(B.CONFIGS)} configs x '
          f'{len(B.PERTURB)} perturbations x {len(SOH_MODES)} SOH inputs',
          flush=True)
    with mp.Pool(a.procs) as pool:
        res = dict(pool.map(_one, jobs))

    rows, oracle_mean = [], {}
    for mi, (mname, _) in enumerate(SOH_MODES):
        stats = []
        for ci, (cname, _) in enumerate(B.CONFIGS):
            und = res[(ci, 0, mi)].mean() * 100
            per = [res[(ci, p, mi)].mean() * 100
                   for p in range(1, len(B.PERTURB))]
            stats.append((cname, und, float(np.mean(per)), max(per)))
        order = sorted(range(len(stats)), key=lambda i: stats[i][2])
        rank = {order[k]: k + 1 for k in range(len(order))}
        print(f'\n  == SOH input: {mname}')
        print(f"    {'config':<28}{'undist':>9}{'mean6':>9}{'worst6':>9}"
              f"{'rank':>6}{'vs oracle':>11}")
        for ci, (cname, und, m6, w6) in enumerate(stats):
            if mi == 0:
                oracle_mean[cname] = m6
            dv = m6 - oracle_mean[cname]
            rows.append([mname, cname, f'{und:.3f}', f'{m6:.3f}',
                         f'{w6:.3f}', rank[ci],
                         f'{dv:+.3f}' if mi else '0.000'])
            print(f'    {cname:<28}{und:>9.3f}{m6:>9.3f}{w6:>9.3f}'
                  f'{rank[ci]:>6}' + (f'{dv:>+11.3f}' if mi else f'{"—":>11}'))

    by_mode = {}
    for r in rows:
        by_mode.setdefault(r[0], []).append(r)
    winners = {m: min(v, key=lambda r: float(r[3]))[1]
               for m, v in by_mode.items()}
    print('\n  best configuration by SOH input:')
    for m, w in winners.items():
        print(f'    {m:<14}{w}')
    # "The winner is stable" is the claim this experiment supports.  "The
    # ranking is stable" is not, and the first version printed it: the
    # configurations below first place do permute under a bias, so the
    # stronger sentence would have been an overstatement of exactly the kind
    # this audit keeps removing from the paper.
    moved = []
    for m, v in by_mode.items():
        if m == 'oracle':
            base = {r[1]: r[5] for r in v}
            continue
        for r in v:
            if base[r[1]] != r[5]:
                moved.append(f'{r[1]} {base[r[1]]}->{r[5]} under {m}')
    if len(set(winners.values())) == 1:
        print(f'\n  The WINNER is stable: {list(winners.values())[0]} places '
              f'first under every SOH input,\n  so the adopted configuration '
              f'is not an artefact of having been selected on oracle\n  SOH, '
              f'and the estimated-SOH row is the one to quote.')
        if moved:
            print(f'  The full ranking is NOT stable — {len(moved)} placement '
                  f'changes below first:\n    ' + '\n    '.join(moved))
    else:
        print('\n  THE WINNER MOVES.  The adopted configuration was selected '
              'on information\n  the deployment does not have; the choice '
              'has to be redone under estimated SOH.')

    os.chdir(ROOT)
    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(OUT, HEADER, rows, 'soc_soh_selection')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(OUT, ROOT)}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
