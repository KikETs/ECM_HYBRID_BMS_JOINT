"""repro/gate.py must run everything the CI `check` job runs.

The gate exists so a defect is found before the push, not by CI after it.
That only holds while the two agree, and nothing would notice them drifting:
a step added to ci.yml and forgotten here leaves the gate reporting a green
that CI will not honour, which is worse than having no gate.

So the CI job is parsed and every command in it has to appear in the gate's
step list.  The comparison is on the meaningful part of the command -- the
script and its flags -- because the gate calls the running interpreter by
absolute path while CI says `python`.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'repro'))

CI = os.path.join(ROOT, '.github', 'workflows', 'ci.yml')


def _normalise(cmd):
    """Reduce a shell command to (tool, script/args) for comparison.

    shlex, not split(): the manifest step passes a python one-liner as a
    single quoted argument, and splitting on whitespace turned it into eight
    tokens that could never match the gate's argv.
    """
    import shlex
    try:
        parts = [p for p in shlex.split(cmd.strip()) if p]
    except ValueError:
        parts = [p for p in cmd.strip().split() if p]
    if not parts:
        return None
    if parts[0] in ('python', 'python3') or parts[0].endswith('/python'):
        parts = parts[1:]
    elif parts[0].endswith('ruff') or parts[0] == 'ruff':
        parts = ['ruff'] + parts[1:]
    return tuple(p for p in parts if p not in ('-q',))


def ci_check_job_commands():
    """Every command line the `check` job runs, in order.

    Parsed as YAML rather than scraped with a regex: the first version of
    this read continuation lines out of the pip install block and the step
    names themselves, and a parser that returns junk makes the comparison
    below meaningless in whichever direction is convenient.
    """
    import yaml
    doc = yaml.safe_load(open(CI, encoding='utf-8'))
    out = []
    for step in doc['jobs']['check']['steps']:
        run = step.get('run')
        if not run:
            continue
        # Join backslash continuations before splitting into commands.
        run = run.replace('\\\n', ' ')
        for line in run.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for piece in line.split('&&'):
                piece = piece.replace('|| true', '').strip()
                n = _normalise(piece)
                if n:
                    out.append(n)
    return out


def gate_commands():
    import gate
    out = []
    for label, argv, cwd, advisory in gate.steps(ROOT):
        parts = list(argv)
        if parts and (parts[0] == sys.executable
                      or os.path.basename(parts[0]).startswith('python')):
            parts = parts[1:]
        elif parts and os.path.basename(parts[0]) == 'ruff':
            parts = ['ruff'] + parts[1:]
        out.append(tuple(p for p in parts if p not in ('-q',)))
    return out


# Commands CI runs that the gate deliberately does not: environment setup and
# installation, which the gate assumes is already true of the machine it runs
# on.  Anything else must be present in both.
EXEMPT_SUBSTRINGS = ('pip', 'install', 'env/verify_env.py', 'apt-get',
                     'actions/', 'make', 'arm-none-eabi')


def test_every_ci_command_is_in_the_gate():
    ci = ci_check_job_commands()
    have = gate_commands()
    missing = []
    for cmd in ci:
        joined = ' '.join(cmd)
        if any(x in joined for x in EXEMPT_SUBSTRINGS):
            continue
        if cmd not in have:
            missing.append(joined)
    assert not missing, (
        'ci.yml runs commands repro/gate.py does not, so the gate can report '
        'green for a push CI will fail:\n  ' + '\n  '.join(missing))


def test_the_gate_finds_at_least_the_core_checks():
    """A parser that silently matches nothing would make the test above pass."""
    ci = [' '.join(c) for c in ci_check_job_commands()]
    for needle in ('repro/verify.py', 'pytest', 'repro/report.py'):
        assert any(needle in c for c in ci), (
            f'the ci.yml parser found no {needle!r} command, so it is not '
            f'reading the workflow -- fix the parser, not the assertion')
