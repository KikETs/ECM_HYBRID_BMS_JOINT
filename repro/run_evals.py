"""Run every trim version through the SOP inversion — the source of the
paper's tables.

Matrix:  2 directions x 5 trim versions x SOH input (oracle / estimated)

The estimated-SOH version is run only for the adopted trim (A8) and the A3
comparison.  There is no reason to run the literature comparisons (direct /
shrinkage / RLS) on estimated SOH as well — their conclusions are already
settled on oracle SOH, and estimated SOH stacks on top in the same direction.

Run from analysis/.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(os.path.dirname(HERE), 'analysis')
OUT = os.path.join(ANALYSIS, 'results', 'eval')
SOH_PRED = os.path.join(ANALYSIS, 'results', 'soh_pred.npz')

# (name, discharge trim dir, charge trim dir, also run estimated SOH)
TRIMS = [
    ('a8',     'runs_trim_a8',      'runs_trim_a8_chg',      True),
    ('a3',     'runs_trim_v2',      'runs_trim_chg_v2',      True),
    ('direct', 'runs_trim_direct',  'runs_trim_direct_chg',  False),
    ('shrink', 'runs_trim_shrink',  'runs_trim_shrink_chg',  False),
    ('rls',    'runs_trim_rls',     'runs_trim_rls_chg',     False),
]

# The adopted evaluation configuration.  31.1 confirmed it reproduces
# section 16's lambda.
#   discharge  --trim-agg max,  tolerance 0.0 A  -> lambda 0.679 / 0.462 (A3)
#   charge     --trim-agg max,  tolerance 0.5 A  -> lambda 0.567 / 0.544 (A3)
AGG = 'max'


def jobs():
    out = []
    for name, dis, chg, do_est in TRIMS:
        for direction, trim in (('discharge', dis), ('charge', chg)):
            out.append((f'{name}_{direction[:4]}_oracle', direction, trim, None))
            if do_est:
                out.append((f'{name}_{direction[:4]}_est', direction, trim,
                            SOH_PRED))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=7,
                    help='how many to run at once.  Each evaluation uses '
                         'one core.')
    ap.add_argument('--only', default=None,
                    help='only names containing this string')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    todo = [j for j in jobs() if not a.only or a.only in j[0]]
    print(f'  {len(todo)} evaluations, {a.jobs} at a time', flush=True)
    running = []
    failed = []

    def reap(block):
        while running and (block or len(running) >= a.jobs):
            nm, p = running.pop(0)
            rc = p.wait()
            print(f'    {"OK  " if rc == 0 else "FAIL"} {nm}', flush=True)
            if rc:
                failed.append(nm)

    for nm, direction, trim, soh in todo:
        if not os.path.isdir(os.path.join(ANALYSIS, trim)):
            print(f'    skipped {nm} — no {trim}', flush=True)
            continue
        cmd = [sys.executable, 'eval_sop_amps.py', '--direction', direction,
               '--trim', trim, '--trim-agg', AGG,
               '--out', os.path.join(OUT, f'{nm}.csv')]
        if soh:
            cmd += ['--soh-est', soh]
        reap(False)
        running.append((nm, subprocess.Popen(
            cmd, cwd=ANALYSIS, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)))
    reap(True)
    if failed:
        print(f'\n  {len(failed)} failed: {", ".join(failed)}', flush=True)
        return 1
    print(f'\n  -> {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
