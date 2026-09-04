"""What does the SOC arm cost if it does NOT get to see its own cell?

The SOC numbers are a per-cell calibrated deployment: build_soc_runs.py hands
every run `ECMSurface(cell)`, that cell's own characterisation, so the filter
reads the surface of the cell it is driving (evidence_ledger,
soc_is_per_cell_calibrated).  That is a real limitation of the arm and the
contract states it.  It is not a limitation of the METHOD, and nothing had
measured the difference.

This does.  ecm_pool.surfaces(holdout) is the leave-one-cell-out pooled
surface the SOP and SOH arms already use -- built from the other five cells,
and existing precisely because using the held-out cell's own surface "imports
that cell's own R1 trajectory ... every held-out number computed that way
would be leaked" (analysis/ecm_pool.py).  Swap it in and the SOC arm becomes
leave-one-cell-out like the other two.

Exactly one thing changes between the two arms: the surface.  Same runs, same
adopted configuration (gate at 1 A, 30 s rest hold), same seven perturbations,
same R_volt schedule on true SOH.  The difference is therefore attributable.

soc_final_loco.py and soc_robust_loco.py already do leave-one-cell-out over
the CONFIGURATION -- which R_volt multiplier, which gate, which spread rule.
That is a different question and it left the surface alone.
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

OUT = os.path.join(ANALYSIS, 'results', 'tables', 'soc_loco.csv')
HEADER = ['surface', 'cell', 'n_runs', 'undisturbed_pct', 'mean_6_pct',
          'worst_of_6_pct']
RUNS = None
POOL = None


def rvolt(soh):
    return float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))


def _one(job):
    """One (perturbation, arm) over all runs.  arm 0 = per cell, 1 = LOCO."""
    from ekf_soc import run as ekf_run
    import soc_perturb_bench as B
    pi, arm = job
    pkw = B.PERTURB[pi][1]
    out = []
    for r in RUNS:
        sd, sc = (r['sd'], r['sc']) if arm == 0 else POOL[r['cell']]
        If = r['I'] * (1.0 + pkw.get('igain', 0.0)) + pkw.get('ibias', 0.0)
        s0 = min(max(float(r['soc'][0]) + pkw.get('dsoc', 0.0), 0.02), 0.98)
        e, _ = ekf_run(sd, sc, float(r['soh']), If, r['V'], r['T'], s0,
                       rvolt(float(r['soh'])), gamma=0.0,
                       i_gate=1.0, rest_hold_s=30.0)
        out.append(float(np.sqrt(np.mean((e - r['soc']) ** 2))))
    return job, np.array(out)


def main():
    global RUNS, POOL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--procs', type=int, default=7)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    import multiprocessing as mp
    import pickle
    import ecm_pool
    import soc_perturb_bench as B

    os.chdir(ANALYSIS)
    RUNS = pickle.load(open('results/soc_runs.pkl', 'rb'))
    cells = sorted({r['cell'] for r in RUNS})
    print(f'  building leave-one-cell-out pooled surfaces for {len(cells)} '
          f'cells', flush=True)
    POOL = {c: ecm_pool.surfaces(c) for c in cells}

    jobs = [(p, arm) for arm in (0, 1) for p in range(len(B.PERTURB))]
    with mp.Pool(a.procs) as pool:
        res = dict(pool.map(_one, jobs))

    cell = np.array([r['cell'] for r in RUNS])
    rows = []
    print(f"\n  {'surface':<22}{'cell':<20}{'undist':>9}{'mean6':>9}"
          f"{'worst6':>9}")
    print('  ' + '-' * 70)
    for arm, name in ((0, 'per-cell (published)'), (1, 'leave-one-cell-out')):
        means = []
        for c in cells + ['(mean)']:
            m = np.ones(len(RUNS), bool) if c == '(mean)' else (cell == c)
            und = res[(0, arm)][m].mean() * 100
            per = [res[(p, arm)][m].mean() * 100
                   for p in range(1, len(B.PERTURB))]
            rows.append([name, c, int(m.sum()), f'{und:.3f}',
                         f'{np.mean(per):.3f}', f'{max(per):.3f}'])
            if c == '(mean)':
                means.append(np.mean(per))
            print(f'  {name:<22}{c:<20}{und:>9.3f}{np.mean(per):>9.3f}'
                  f'{max(per):>9.3f}')
        print()

    pc = float([r[4] for r in rows if r[0].startswith('per-cell')
                and r[1] == '(mean)'][0])
    lo = float([r[4] for r in rows if r[0].startswith('leave')
                and r[1] == '(mean)'][0])
    print(f'  per-cell calibrated   {pc:.3f} %p')
    print(f'  leave-one-cell-out    {lo:.3f} %p'
          f'   ({lo - pc:+.3f}, x{lo / pc:.2f})')
    print('\n  One thing differs between the arms: the surface the filter '
          'reads.\n  Same runs, same adopted configuration, same '
          'perturbations, same R_volt.')

    os.chdir(ROOT)
    if a.check:
        from tablecheck import compare_or_fail
        return compare_or_fail(OUT, HEADER, rows, 'soc_loco')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(OUT, ROOT)}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
