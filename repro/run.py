"""Runner for the reproduction pipeline.

    python3 repro/run.py --list                 show the stages and their state
    python3 repro/run.py --plan safety          what that stage needs
    python3 repro/run.py safety                 run that stage (upstream first)
    python3 repro/run.py --from 5               re-run tier 5 and above

By default it **does not rebuild outputs that already exist**.  If an upstream
is newer it marks the stage stale and, without --force, only reports it rather
than running.  Building the cache takes over three hours, so automatic re-runs
would be dangerous.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, HERE)
from stages import STAGES, EXPLORATORY          # noqa: E402

BY_ID = {s['id']: s for s in STAGES}


def path_of(p):
    return p if os.path.isabs(p) else os.path.join(ANALYSIS, p)


def newest(paths):
    """Newest mtime among those that exist.  None if none do."""
    ts = []
    for p in paths:
        q = path_of(p)
        if os.path.isdir(q):
            for root, _, fs in os.walk(q):
                ts += [os.path.getmtime(os.path.join(root, f)) for f in fs]
        elif os.path.exists(q):
            ts.append(os.path.getmtime(q))
    return max(ts) if ts else None


def oldest_output(s):
    ts = []
    for p in s['outputs']:
        q = path_of(p)
        if os.path.isdir(q):
            fs = [os.path.join(r, f) for r, _, g in os.walk(q) for f in g]
            if not fs:
                return None
            ts.append(min(os.path.getmtime(f) for f in fs))
        elif os.path.exists(q):
            ts.append(os.path.getmtime(q))
        else:
            return None
    return min(ts) if ts else None


def status(s):
    """Note: 'stale' is only an mtime comparison.

    The stages in this repository write their files one cell at a time, so
    running two stages alternately leaves the upstream's last cell newer than
    the downstream's first — identical content shows up as stale (measured:
    temp_factor was byte-identical when rebuilt).  So 'stale' should be read
    as "check the content", not as "re-run this".
    """
    out = oldest_output(s)
    if out is None:
        # A stage marked optional produces a comparison artifact nothing else
        # consumes.  Reporting it as 'absent' next to genuinely broken stages
        # says the pipeline is incomplete when it is not -- and the whole
        # point of this listing is that a missing output is a failure.
        return 'optional' if s.get('optional') else 'missing'
    inp = newest(s['inputs'])
    if inp is not None and inp > out + 1.0:
        return 'stale'
    return 'ok'


def upstream(target):
    """The stages target needs, in tier order."""
    want = {target}
    changed = True
    while changed:
        changed = False
        for s in STAGES:
            if s['id'] not in want:
                continue
            for i in s['inputs']:
                for t in STAGES:
                    if t['id'] in want:
                        continue
                    if any(os.path.normpath(o).startswith(os.path.normpath(i))
                           or os.path.normpath(i).startswith(os.path.normpath(o))
                           for o in t['outputs']):
                        want.add(t['id'])
                        changed = True
    return [s for s in STAGES if s['id'] in want]


def show(ss):
    mark = {'ok': 'OK  ', 'stale': 'mtime', 'missing': 'absent',
            'optional': 'opt '}
    print(f"  {'stage':<14}{'tier':>5}{'state':>8}{'min':>7}   what it does",
          flush=True)
    print('  ' + '-' * 92, flush=True)
    tot = 0
    for s in ss:
        st = status(s)
        if st != 'ok':
            tot += s['minutes']
        b = ' (board)' if s.get('board') else ''
        m = f"{s['minutes']}" + ('' if s.get('measured') else '~')
        print(f"  {s['id']:<14}{s['tier']:>5}{mark[st]:>8}{m:>7}{b}   "
              f"{s['why'].splitlines()[0][:52]}", flush=True)
    if tot:
        print(f"\n  About {tot} minutes to re-run "
              f"({tot/60:.1f} h).  ~ marks an estimate that was never timed.",
              flush=True)
        print("  'mtime' only means an upstream is newer, not that the "
              "content changed — see the comment in status().", flush=True)
    return tot


def run_one(s, dry, force=False):
    cmd = s['cmd'].replace('{py}', sys.executable)
    if force and s.get('force_flag'):
        # --force at the graph level does not reach a stage script that skips
        # existing outputs on its own.  Stages that need it declare the flag.
        cmd = ' && '.join(part.strip() + ' ' + s['force_flag']
                          for part in cmd.split('&&'))
    print(f"\n  == {s['id']}  (about {s['minutes']} min)\n     {cmd}",
          flush=True)
    if dry:
        return True
    if s.get('board'):
        print('     this stage needs the board — run it by hand', flush=True)
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True, cwd=ANALYSIS)
    dt = (time.time() - t0) / 60
    # Exit 0 is not completion.  The cache stage exited 0 while writing six
    # of its twelve declared outputs, because --part defaults to one
    # protocol; this reported "done" and the rebuild carried on with half a
    # cache.  A stage is done when the files it promised exist.
    missing = [o for o in s['outputs'] if not os.path.exists(path_of(o))]
    ok = rc == 0 and not missing
    print(f"     {'done' if ok else 'FAILED'}  took {dt:.1f} min "
          f"(recorded {s['minutes']} min)", flush=True)
    if rc == 0 and missing:
        print(f"     exited 0 but {len(missing)} of {len(s['outputs'])} "
              f"declared outputs are absent:", flush=True)
        for m in missing[:6]:
            print(f"       {m}", flush=True)
        if len(missing) > 6:
            print(f"       ... and {len(missing) - 6} more", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', nargs='?')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--plan', metavar='STAGE')
    ap.add_argument('--from', dest='from_tier', type=int)
    ap.add_argument('--to', dest='to_tier', type=int,
                    help='stop after this tier.  Use --to 5 for a raw-to-'
                         'result rebuild: tier 6 re-exports the MCU header '
                         'and would overwrite whatever is currently '
                         'deployed with a freshly rebuilt one.')
    ap.add_argument('--force', action='store_true',
                    help='re-run stages even when their state is ok')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--exploratory', action='store_true',
                    help='list the scripts off the critical path and why')
    a = ap.parse_args()

    if a.exploratory:
        print('  Off the critical path — not needed to reproduce\n')
        import textwrap
        for k, v in EXPLORATORY.items():
            print(f'  {k}')
            for line in textwrap.wrap(v, 70):
                print(f'      {line}')
            print()
        return 0

    if a.list or (not a.target and not a.plan and a.from_tier is None):
        show(STAGES)
        print('\n  python3 repro/run.py <stage>        run up to that stage',
              flush=True)
        print('  python3 repro/run.py --plan <stage>  only show what it needs',
              flush=True)
        return 0

    if a.plan:
        if a.plan not in BY_ID:
            print(f'  unknown stage: {a.plan}')
            return 1
        show(upstream(a.plan))
        return 0

    if a.from_tier is not None:
        todo = [s for s in STAGES if s['tier'] >= a.from_tier]
        if a.to_tier is not None:
            todo = [s for s in todo if s['tier'] <= a.to_tier]
    else:
        if a.target not in BY_ID:
            print(f'  unknown stage: {a.target}.  Check with --list')
            return 1
        todo = upstream(a.target)

    todo = [s for s in todo if a.force or status(s) != 'ok']
    if not todo:
        print('  Everything is up to date.  Use --force to run anyway.')
        return 0
    show(todo)
    for s in todo:
        if not run_one(s, a.dry_run, force=a.force):
            print(f'\n  stopping at {s["id"]}.', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
