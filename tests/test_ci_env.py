"""The CI runner must be able to import everything CI runs.

This file exists because it did not.  PyYAML was used by four tests and was
absent from the workflow's install line, so `check` failed on every commit
with a ModuleNotFoundError that no local run could reproduce -- the author's
environment had yaml, the runner did not.  Comparing the two by hand is
exactly the check a human skips.
"""
import ast
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'ci.yml')

# Import name -> distribution name on PyPI, where they differ.
DIST = {
    'yaml': 'pyyaml',
    'sklearn': 'scikit-learn',
    'serial': 'pyserial',
    'PIL': 'pillow',
    'cv2': 'opencv-python',
}

# Modules CI is expected NOT to have: the workflow deliberately covers the
# torch-free path, and these are only imported behind a guard or in scripts
# CI does not run.
ALLOWED_ABSENT = {'torch'}


def _stdlib():
    names = set(getattr(sys, 'stdlib_module_names', ()))
    return names | {'__future__'}


def _top_level_imports(path):
    try:
        tree = ast.parse(open(path, encoding='utf-8').read())
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.level == 0 and n.module:
                out.add(n.module.split('.')[0])
    return out


def _local_modules():
    """Anything importable from the repo itself is not a dependency."""
    mods = set()
    for d in ('repro', 'tests', 'env', 'analysis', 'mcu'):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.endswith('.py'):
                mods.add(f[:-3])
    return mods


def _ci_installed():
    """Distribution names the workflow pip-installs."""
    if not os.path.exists(WORKFLOW):
        pytest.skip('no workflow file')
    src = open(WORKFLOW, encoding='utf-8').read()
    names = set()
    for m in re.finditer(r'pip install ((?:[^\n\\]|\\\n)*)', src):
        for tok in m.group(1).replace('\\\n', ' ').split():
            tok = tok.strip()
            if not tok or tok.startswith('-'):
                continue
            names.add(re.split(r'[=<>!~\[]', tok)[0].lower())
    return names


def _files_ci_runs():
    """Every .py under tests/ and repro/, plus env/, which CI imports."""
    out = []
    for d in ('tests', 'repro', 'env'):
        p = os.path.join(ROOT, d)
        for f in sorted(os.listdir(p)):
            if f.endswith('.py'):
                out.append(os.path.join(p, f))
    return out


def test_ci_installs_every_third_party_module_it_imports():
    std, local, installed = _stdlib(), _local_modules(), _ci_installed()
    missing = {}
    for path in _files_ci_runs():
        for mod in _top_level_imports(path):
            if mod in std or mod in local or mod in ALLOWED_ABSENT:
                continue
            if DIST.get(mod, mod).lower() in installed:
                continue
            missing.setdefault(DIST.get(mod, mod),
                               set()).add(os.path.relpath(path, ROOT))
    assert not missing, (
        'the CI workflow does not install: '
        + '; '.join(f'{k} (used by {sorted(v)})' for k, v in sorted(missing.items())))


def test_the_workflow_pins_what_it_installs():
    """An unpinned dependency makes a green run unreproducible."""
    src = open(WORKFLOW, encoding='utf-8').read()
    unpinned = []
    for m in re.finditer(r'pip install ((?:[^\n\\]|\\\n)*)', src):
        for tok in m.group(1).replace('\\\n', ' ').split():
            if not tok or tok.startswith('-') or tok in ('pip', 'ruff'):
                continue
            if '==' not in tok:
                unpinned.append(tok)
    assert not unpinned, f'unpinned in the CI install: {unpinned}'


def test_at_least_one_lint_gate_actually_blocks():
    """A workflow step that ends in `|| true` cannot fail, so it is not a gate.

    The lint step was advisory from the day it was added, which meant CI
    reported a clean lint on every commit regardless of the code.  At least
    one lint invocation must be able to fail the job.
    """
    src = open(WORKFLOW, encoding='utf-8').read()
    lint = [ln.strip() for ln in src.splitlines()
            if 'ruff check' in ln]
    assert lint, 'no lint step in the workflow'
    blocking = [ln for ln in lint if '|| true' not in ln]
    assert blocking, (
        'every ruff invocation ends in `|| true`; the lint step can never '
        'fail and is not a gate')
