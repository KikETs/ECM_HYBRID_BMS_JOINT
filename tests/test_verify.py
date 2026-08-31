"""verify.py must fail loudly on every kind of broken artifact.

Each test corrupts a *copy* of the tables directory and runs verify.py
against it, so the repository's own tables are never touched.
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT, run

VERIFY = os.path.join('repro', 'verify.py')


def verify_against(tmp_path, tables_dir, extra=()):
    """Run verify.py with results/tables pointed at tables_dir."""
    stage = tmp_path / 'repo'
    if not stage.exists():
        stage.mkdir()
        (stage / 'repro').mkdir()
        for f in ('verify.py', 'expected.json'):
            shutil.copy(os.path.join(ROOT, 'repro', f), stage / 'repro' / f)
        (stage / 'analysis' / 'results').mkdir(parents=True)
    link = stage / 'analysis' / 'results' / 'tables'
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link)
    os.symlink(tables_dir, link)
    p = subprocess.run([sys.executable, VERIFY, *extra], cwd=stage,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_clean_tables_pass(tmp_path, sandbox_tables):
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 0, out
    assert 'RESULT: PASS' in out


def test_missing_table_fails(tmp_path, sandbox_tables):
    (sandbox_tables / 'safety.csv').unlink()
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 1, out
    assert 'ABSENT' in out
    assert 'RESULT: FAIL' in out


def test_missing_table_with_allow_missing_is_not_a_pass(tmp_path,
                                                        sandbox_tables):
    (sandbox_tables / 'safety.csv').unlink()
    rc, out = verify_against(tmp_path, sandbox_tables, ['--allow-missing'])
    # It may exit 0, but it must say in plain words that this is not a pass.
    assert 'NOT a verification pass' in out


def test_partial_table_fails(tmp_path, sandbox_tables):
    """A table with the right columns but too few rows is a partial run."""
    p = sandbox_tables / 'safety.csv'
    lines = p.read_text(encoding='utf-8').splitlines(True)
    p.write_text(''.join(lines[:4]), encoding='utf-8')
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 1, out
    assert 'partial run is not a pass' in out


def test_schema_mismatch_fails(tmp_path, sandbox_tables):
    p = sandbox_tables / 'safety.csv'
    r = list(csv.DictReader(open(p, encoding='utf-8')))
    for row in r:
        row['unexpected_column'] = '1'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(r[0].keys()))
        w.writeheader()
        w.writerows(r)
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 1, out
    assert 'column set differs' in out


def test_non_numeric_cell_fails(tmp_path, sandbox_tables):
    p = sandbox_tables / 'safety.csv'
    t = p.read_text(encoding='utf-8').splitlines()
    parts = t[1].split(',')
    parts[4] = 'n/a'                       # the lambda column
    t[1] = ','.join(parts)
    p.write_text('\n'.join(t) + '\n', encoding='utf-8')
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 1, out
    assert 'not numeric' in out


def test_numeric_mismatch_fails(tmp_path, sandbox_tables):
    p = sandbox_tables / 'safety.csv'
    t = p.read_text(encoding='utf-8').splitlines()
    parts = t[1].split(',')
    parts[4] = f'{float(parts[4]) + 0.5:.3f}'
    t[1] = ','.join(parts)
    p.write_text('\n'.join(t) + '\n', encoding='utf-8')
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 1, out
    assert 'MISMATCH' in out


def test_truncated_row_fails(tmp_path, sandbox_tables):
    p = sandbox_tables / 'safety.csv'
    t = p.read_text(encoding='utf-8').splitlines()
    t[1] = ','.join(t[1].split(',')[:3])
    p.write_text('\n'.join(t) + '\n', encoding='utf-8')
    rc, out = verify_against(tmp_path, sandbox_tables)
    assert rc == 1, out
    assert 'truncated' in out or 'fewer fields' in out


def test_only_narrows_without_touching_manifest(tmp_path, sandbox_tables):
    rc, out = verify_against(tmp_path, sandbox_tables, ['--only', 'soh'])
    assert rc == 0, out
    assert 'completeness not enforced' in out


def test_update_preserves_checks_that_did_not_run(tmp_path, sandbox_tables):
    """--only X --update must not drop or alter the other expected entries."""
    stage = tmp_path / 'repo'
    verify_against(tmp_path, sandbox_tables)          # build the stage dir
    exp = stage / 'repro' / 'expected.json'
    before = json.loads(exp.read_text(encoding='utf-8'))

    # Perturb one soh value so --update has something to rewrite.
    p = sandbox_tables / 'soh.csv'
    t = p.read_text(encoding='utf-8').splitlines()
    parts = t[-1].split(',')
    parts[2] = f'{float(parts[2]) + 0.001:.4f}'
    t[-1] = ','.join(parts)
    p.write_text('\n'.join(t) + '\n', encoding='utf-8')

    rc, out = verify_against(tmp_path, sandbox_tables,
                             ['--only', 'soh.rmse', '--update'])
    assert rc == 0, out
    after = json.loads(exp.read_text(encoding='utf-8'))

    assert len(after['checks']) == len(before['checks']), 'checks were dropped'
    b = {c['id']: c['value'] for c in before['checks']}
    a = {c['id']: c['value'] for c in after['checks']}
    assert set(a) == set(b), 'the id set changed'
    changed = [k for k in a if a[k] != b[k]]
    assert changed == ['soh.rmse'], f'unexpectedly rewrote {changed}'


def test_deterministic_rerun(tmp_path, sandbox_tables):
    """Two consecutive runs on identical inputs must produce identical text."""
    rc1, out1 = verify_against(tmp_path, sandbox_tables)
    rc2, out2 = verify_against(tmp_path, sandbox_tables)
    assert rc1 == rc2 == 0
    assert out1 == out2


def test_the_readme_number_of_checks_matches_expected_json():
    """README quotes how many values verify.py recomputes; it must be true.

    It said 43 while the file held 51, and 51 while it held 59.  A count in
    prose drifts every time a check is added, and nothing noticed.
    """
    import json
    import re
    n = len(json.load(open(os.path.join(ROOT, 'repro', 'expected.json'),
                           encoding='utf-8'))['checks'])
    readme = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    hits = re.findall(r'(\d+) published numbers', readme)
    assert hits, 'README no longer says how many published numbers verify.py checks'
    for h in hits:
        assert int(h) == n, (
            f'README says {h} published numbers, expected.json holds {n}')
