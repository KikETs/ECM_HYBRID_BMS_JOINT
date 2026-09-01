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
#
# The estimated-SOH column was True for a8 and a3 only, so the six-method
# comparison ranked every method on ORACLE SOH -- and four of the six had
# never been evaluated under estimated SOH at all.  That makes the ranking a
# statement about a condition the deployment does not have.  The four
# baselines in method_comparison.csv are now True as well; 'rls' and 'direct'
# stay False because neither appears in that comparison.
TRIMS = [
    ('a8',     'runs_trim_a8',      'runs_trim_a8_chg',      True),
    ('a3',     'runs_trim_v2',      'runs_trim_chg_v2',      True),
    ('direct', 'runs_trim_direct',  'runs_trim_direct_chg',  False),
    ('shrink', 'runs_trim_shrink',  'runs_trim_shrink_chg',  True),
    ('rls',    'runs_trim_rls',     'runs_trim_rls_chg',     False),
    # Added by the audit: a sequence model over the same 12 causal drive
    # blocks, and a forgetting-factor RLS adaptive ECM.  Same splits, same
    # output head, same inversion.
    ('lstm',   'runs_trim_lstm',    'runs_trim_lstm_chg',    True),
    ('gru',    'runs_trim_gru',     'runs_trim_gru_chg',     True),
    ('ffrls',  'runs_trim_ffrls',   'runs_trim_ffrls_chg',   True),
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
    skipped = []

    def reap(block):
        while running and (block or len(running) >= a.jobs):
            nm, out_csv, p = running.pop(0)
            # communicate(), not wait(): a full stderr pipe deadlocks the
            # child, and the previous version never read the pipe at all, so
            # every error message was discarded.
            _, err = p.communicate()
            rc = p.returncode
            if rc == 0 and not os.path.exists(out_csv):
                rc = -1
                err = (err or '') + (
                    f'\n    exited 0 but wrote no {os.path.basename(out_csv)}')
            print(f'    {"OK  " if rc == 0 else "FAIL"} {nm}', flush=True)
            if rc:
                failed.append(nm)
                for line in (err or '').strip().splitlines()[-12:]:
                    print(f'        | {line}', flush=True)

    for nm, direction, trim, soh in todo:
        out_csv = os.path.join(OUT, f'{nm}.csv')
        if not os.path.isdir(os.path.join(ANALYSIS, trim)):
            # Not a skip to shrug at: a missing trim directory means an
            # upstream training stage never ran, and the evaluation matrix
            # this script exists to fill will come out with holes in it.
            print(f'    MISSING {nm} — no {trim}/ (run its training stage)',
                  flush=True)
            skipped.append(nm)
            continue
        cmd = [sys.executable, 'eval_sop_amps.py', '--direction', direction,
               '--trim', trim, '--trim-agg', AGG, '--out', out_csv]
        if soh:
            cmd += ['--soh-est', soh]
        reap(False)
        running.append((nm, out_csv, subprocess.Popen(
            cmd, cwd=ANALYSIS, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True)))
    reap(True)

    if failed or skipped:
        if failed:
            print(f'\n  {len(failed)} failed: {", ".join(failed)}', flush=True)
        if skipped:
            print(f'  {len(skipped)} not run for want of inputs: '
                  f'{", ".join(skipped)}', flush=True)
        print('  The evaluation matrix is incomplete — downstream tables '
              'built from it would be partial.', flush=True)
        return 1
    print(f'\n  -> {OUT}   ({len(todo)} evaluations, all complete)',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
