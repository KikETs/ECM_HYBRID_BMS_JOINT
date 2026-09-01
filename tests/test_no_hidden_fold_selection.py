"""A published table must not depend on a fold nobody swept.

Twice now the same defect shipped.  run_external_crate.py and
run_chen2026_baseline.py both take `--holdout`, both default it to `CC`, and
both write tables that carry published checks.  RPCWBY is an external cell,
so nothing is held out for it and all six internal surfaces are equally
entitled to be used -- and CC turned out to be the most favourable of the six
in both cases.  The margin published from Test#8 was 1.655 when the worst
surface gives 1.351; the 0-40 C margin was 1.394 against a worst of 1.156.

A reviewer caught the first one.  The second was found by looking for other
scripts with the same shape, which is not a method that scales.

So: any script that selects a fold, cell, surface or arm by DEFAULT and
produces a table with published checks must also offer a sweep, and the sweep
must be registered in the stage that builds the table.  Either the selection
is swept, or it is declared here with a reason.
"""
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'repro'))

# Arguments that pick one member of a set the result could have been computed
# over.  --direction and --tau are axes the papers report separately, by
# design, and both are swept by their stages.
SELECTORS = ('--holdout', '--cell', '--fold', '--surface')

# A selection that is deliberate and does not need a sweep, with the reason.
EXEMPT = {
    # (script, argument): why
    ('fig_soc_traj.py', '--cell'):
        'a trajectory figure draws one cell by construction; it carries no '
        'published check',
}


def scripts_with_default_selection():
    out = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'repro'))):
        if not name.endswith('.py'):
            continue
        path = os.path.join(ROOT, 'repro', name)
        tree = ast.parse(open(path, encoding='utf-8').read(), path)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument'):
                continue
            if not (node.args and isinstance(node.args[0], ast.Constant)):
                continue
            flag = node.args[0].value
            if flag not in SELECTORS:
                continue
            has_default = any(
                k.arg == 'default' and isinstance(k.value, ast.Constant)
                and k.value.value is not None for k in node.keywords)
            if has_default:
                out.append((name, flag))
    return out


def test_no_published_table_rests_on_an_unswept_default():
    import stages
    spec = json.load(open(os.path.join(ROOT, 'repro', 'expected.json'),
                          encoding='utf-8'))
    checked_tables = {c['table'] for c in spec['checks']}

    # Which stage command mentions which script, and whether it sweeps.
    by_script = {}
    for s in stages.STAGES:
        for part in s['cmd'].split('&&'):
            for name in [x for x in part.split() if x.endswith('.py')]:
                base = os.path.basename(name)
                entry = by_script.setdefault(base, dict(sweeps=False,
                                                        tables=set()))
                if '--all-surfaces' in part or '--all-cells' in part:
                    entry['sweeps'] = True
                entry['tables'] |= {os.path.basename(o)
                                    for o in s['outputs']
                                    if o.endswith('.csv')}

    offenders = []
    for name, flag in scripts_with_default_selection():
        if (name, flag) in EXEMPT:
            continue
        info = by_script.get(name)
        if not info:
            continue                      # not on the critical path
        publishes = info['tables'] & checked_tables
        if publishes and not info['sweeps']:
            offenders.append(
                f'{name} defaults {flag} and its stage publishes '
                f'{sorted(publishes)} with no sweep')

    assert not offenders, (
        'a published table rests on one fold chosen by a default argument, '
        'with nothing showing what the other folds give:\n  '
        + '\n  '.join(offenders)
        + '\nAdd an --all-surfaces mode and register it in the stage, or '
          'add the pair to EXEMPT with a reason.')


def test_the_scanner_actually_finds_the_known_cases():
    """A scanner that finds nothing would make the test above vacuous."""
    found = {n for n, _ in scripts_with_default_selection()}
    for known in ('run_external_crate.py', 'run_chen2026_baseline.py'):
        assert known in found, (
            f'{known} takes a defaulted --holdout; the AST scan missed it, '
            f'so the guard above is not looking at anything')
