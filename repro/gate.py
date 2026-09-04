"""Everything CI runs, plus everything a reviewer has caught, in one command.

Four review rounds found four defects that were sitting in the working tree
the whole time:

  * REPRODUCE.md not regenerated after stages.py changed  (CI caught it, on
    the server, after the push)
  * a stage reading a file no stage declared as an output, so a clean clone
    could not obtain it
  * README quoting a check count and a rebuild time that nothing verified
  * a table published from one of six surfaces, the most favourable one

Each was fixed, and a check was added for each, and the next round found the
next one.  The pattern is not that the checks are wrong; it is that they live
in four places and nobody ran all four before pushing.  This runs all of
them, in CI's order, and refuses to be optimistic:

  * --strict-clone additionally exports the committed tree to a temporary
    directory and runs the torch-free subset there, which is the only way to
    catch "works because of an untracked file".
  * every step's exit status is checked; nothing is `|| true` except the
    advisory lint CI itself marks advisory.

PLATFORM
    Developed and exercised on Linux, which is what CI runs (ubuntu-latest for
    both jobs).  It has never been run on Windows or macOS and is not claimed
    to work there.  Nothing in it is deliberately Linux-only -- paths go
    through os.path, commands are argv lists rather than shell strings, and
    the ruff lookup already tries Scripts/ruff.exe -- but "should port" is not
    "was tested", and this file exists precisely to stop that kind of
    assumption.  On Windows, run the individual CI steps or use WSL.

Usage:
    python3 repro/gate.py                 # what CI will do
    python3 repro/gate.py --strict-clone  # that, plus the clean-clone pass
    python3 repro/gate.py --fix           # regenerate derived docs first
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
# A tool CI runs must not be silently absent here.  ruff lives next to the
# interpreter in a conda env and is often not on PATH, which made the gate
# report "skipped" for the one step that blocks CI on lint.
def _ruff():
    """ruff next to the interpreter, whatever the platform calls it.

    conda puts it in the same directory as python on POSIX and in Scripts\\
    with a .exe on Windows.  The first version looked only for a bare 'ruff'
    beside sys.executable, which meant the lint step -- the one CI blocks on
    -- reported "skipped" anywhere else.
    """
    d = os.path.dirname(PY)
    for cand in (os.path.join(d, 'ruff'), os.path.join(d, 'ruff.exe'),
                 os.path.join(d, 'Scripts', 'ruff.exe'),
                 os.path.join(os.path.dirname(d), 'Scripts', 'ruff.exe')):
        if os.path.exists(cand):
            return cand
    return 'ruff'


RUFF = _ruff()

# What this gate can and cannot stand in for.  CI runs ubuntu-latest, and two
# things here are POSIX-shaped: run.py joins stage commands with && through
# shell=True, and the firmware stages need arm-none-eabi and STM32_Programmer_CLI.
# The checks in steps() are plain subprocess calls and portable; the clean-clone
# pass shells out to git.  Say so rather than let a green run on Windows imply
# more than it covers.
SUPPORTED = 'linux'


def steps(root):
    """CI's steps, in CI's order.  (label, argv, cwd, advisory)."""
    return [
        ('compile every python file',
         [PY, '-m', 'compileall', '-q', 'analysis', 'repro', 'mcu', 'env',
          'tests'], root, False),
        ('lint: real defects',
         [RUFF, 'check', 'repro', 'tests', 'env', '--select', 'E9,F'],
         root, False),
        ('lint: style',
         [RUFF, 'check', 'repro', 'tests', 'env'], root, True),
        ('stored-table verification',
         [PY, 'repro/verify.py'], root, False),
        ('producer: soh table',
         [PY, 'repro/run_soh_table.py', '--check'], root, False),
        ('producer: mcu table',
         [PY, 'repro/run_mcu_table.py', '--check'], root, False),
        ('producer: method comparison',
         [PY, 'repro/run_method_comparison.py', '--check'], root, False),
        ('producer: soh deploy tables',
         [PY, 'repro/run_soh_deploy_tables.py', '--check'], root, False),
        ('producer: soc per-cell',
         [PY, 'repro/run_soc_percell.py', '--check'], root, False),
        ('producer: external c-rate envelope',
         [PY, 'repro/run_external_crate.py', '--check'], root, False),
        ('producer: external c-rate surfaces',
         [PY, 'repro/run_external_crate.py', '--all-surfaces', '--check'],
         root, False),
        ('tests', [PY, '-m', 'pytest', 'tests', '-q'], root, False),
        ('C parity', [PY, 'repro/run_parity.py', '--n', '2000'], root, False),
        ('figures render: ladder', [PY, 'repro/fig_ladder.py'], root, False),
        ('figures render: soh traj', [PY, 'repro/fig_soh_traj.py'], root,
         False),
        ('REPRODUCE.md is regenerated, not hand-edited',
         [PY, 'repro/report.py'], root, False),
        ('  ... and matches the commit',
         ['git', 'diff', '--exit-code', 'REPRODUCE.md'], root, False),
        ('raw-data manifest parses',
         [PY, '-c', "import sys;sys.exit(0 if open('manifests/raw_data.yaml')"
          ".read().count('sha256')>20 else 1)"], root, False),
        ('qc: no stale, retracted or unbounded claim',
         [PY, 'repro/qc.py', '--fail-on-current'], root, False),
    ]


def run(label, argv, cwd, advisory, quiet):
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as e:
        print(f'  SKIP  {label}  ({e.filename} not installed)')
        return None  # counted as a failure unless --allow-skips
    dt = time.time() - t0
    ok = p.returncode == 0
    tag = 'ok  ' if ok else ('warn' if advisory else 'FAIL')
    print(f'  {tag}  {label}  ({dt:.1f}s)')
    if not ok and not quiet:
        for stream in (p.stdout, p.stderr):
            tail = [x for x in stream.strip().split('\n') if x.strip()][-14:]
            for line in tail:
                print(f'        {line}')
    return ok or advisory


def clean_clone_pass(quiet):
    """Run the torch-free subset against a fresh clone of HEAD.

    The working tree can pass on files that were never committed.  This is
    what CI does with actions/checkout, and it is the only step here that can
    catch that class.

    `git archive` was the first attempt, and it produced a directory with no
    .git in it.  tests/test_producers.py asks `git ls-files` whether a stage
    input is committed, could not answer without a repository, and reported a
    committed file as unobtainable -- a false failure that would have taught
    the next person to stop running --strict-clone.  A real clone it is.
    """
    tmp = tempfile.mkdtemp(prefix='gate-clone-')
    try:
        dst = os.path.join(tmp, 'repo')
        r = subprocess.run(
            ['git', 'clone', '--quiet', '--shared', '--no-checkout',
             ROOT, dst], capture_output=True, text=True)
        if r.returncode == 0:
            r = subprocess.run(['git', 'checkout', '--quiet', 'HEAD'],
                               cwd=dst, capture_output=True, text=True)
        if r.returncode:
            print('  FAIL  clean clone: git clone failed')
            print('        ' + (r.stderr or r.stdout).strip()[:400])
            return False
        print(f'\n  -- clean clone of HEAD in {dst}')
        good = True
        # The whole step list, not a hand-picked subset.  The subset version
        # ran verify and the tests but no producer --check, and CI failed on
        # exactly the step it skipped: run_soc_percell.py read a gitignored
        # npz, which the working tree had and a fresh checkout did not.
        # Choosing which checks the clone deserves is the optimism this file
        # exists to remove.
        for label, argv, cwd, advisory in steps(dst):
            if argv[:2] == ['git', 'diff']:
                continue        # nothing is edited in the clone
            if run('clone: ' + label, argv, cwd, advisory, quiet) is False:
                good = False
        if good:
            a = os.path.join(dst, 'REPRODUCE.md')
            b = os.path.join(ROOT, 'REPRODUCE.md')
            if open(a, encoding='utf-8').read() != open(
                    b, encoding='utf-8').read():
                print('  FAIL  clone: REPRODUCE.md regenerates differently '
                      'from a clean checkout')
                good = False
            else:
                print('  ok    clone: REPRODUCE.md identical')
        return good
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strict-clone', action='store_true',
                    help='also run the torch-free subset against git archive '
                         'of HEAD')
    ap.add_argument('--fix', action='store_true',
                    help='regenerate REPRODUCE.md before checking, instead '
                         'of failing on it')
    ap.add_argument('--allow-skips', action='store_true',
                    help='do not fail when a tool CI runs is missing here')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args()

    if a.fix:
        subprocess.run([PY, 'repro/report.py'], cwd=ROOT,
                       capture_output=True)
        print('  (--fix) REPRODUCE.md regenerated\n')

    import platform
    print(f'== submission gate   ({platform.system()} '
          f'{platform.python_version()})')
    if platform.system() != 'Linux':
        print('   NOTE: CI runs Linux and this gate has only been exercised '
              'there.\n   A pass here is weaker evidence than a pass on '
              'Linux.')
    print()
    if not sys.platform.startswith(SUPPORTED):
        print(f'  NOTE  running on {sys.platform!r}; CI runs ubuntu-latest.  '
              f'The checks below are\n        portable, but the stage runner '
              f'(repro/run.py) joins commands with && through\n        a '
              f'shell, and the firmware job needs a POSIX ARM toolchain.  A '
              f'green result here\n        is not evidence those work on '
              f'this platform.\n')
    results = [run(*s, a.quiet) for s in steps(ROOT)]
    if a.strict_clone:
        results.append(clean_clone_pass(a.quiet))

    ran = [r for r in results if r is not None]
    bad = [r for r in ran if r is False]
    skipped = len(results) - len(ran)
    print(f'\n  {len(ran) - len(bad)} of {len(ran)} checks passed, '
          f'{skipped} skipped')
    if skipped and not a.allow_skips:
        print('  A skipped step is a step CI will run and this did not.  '
              'Install the tool, or pass --allow-skips and say so.')
    if bad or (skipped and not a.allow_skips):
        print('  RESULT: NOT READY TO PUSH')
        return 1
    print('  RESULT: matches what CI will run'
          + ('' if a.strict_clone else
             '  (add --strict-clone before a release)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
