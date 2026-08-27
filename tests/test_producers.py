"""Every published table must be regenerable by a named command.

These are the checks that would have caught the two tables shipping with no
producer at all (soh.csv, mcu.csv) and the stage graph declaring inputs and
outputs that did not match what the scripts read and wrote.
"""
import csv
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT, TABLES, run

sys.path.insert(0, os.path.join(ROOT, 'repro'))


def expected():
    with open(os.path.join(ROOT, 'repro', 'expected.json'),
              encoding='utf-8') as f:
        return json.load(f)


def test_every_required_table_declares_a_producer():
    spec = expected()
    missing = [t for t, d in spec['tables'].items()
               if d.get('required') and d['producer'].startswith('[')]
    assert not missing, f'required tables with no producer: {missing}'


def test_every_checked_table_has_a_schema():
    spec = expected()
    need = {c['table'] for c in spec['checks']}
    assert not (need - set(spec['tables'])), \
        f'checked but unschemad: {need - set(spec["tables"])}'


def test_completeness_manifest_matches_checks():
    spec = expected()
    ids = sorted(c['id'] for c in spec['checks'])
    assert len(ids) == spec['completeness']['total_checks']
    assert ids == sorted(spec['completeness']['check_ids'])


def test_stage_outputs_exist_on_disk_or_are_declared_absent():
    """A stage that claims an output must not silently produce nothing."""
    from stages import STAGES
    for s in STAGES:
        for o in s['outputs']:
            if 'results/tables/' not in o:
                continue
            p = os.path.join(ROOT, 'analysis', o)
            assert os.path.exists(p), \
                f'stage {s["id"]} declares {o} but it is absent'


def test_stage_graph_inputs_resolve():
    """Every stage input is produced by another stage or exists as raw."""
    from stages import STAGES
    outs = set()
    for s in STAGES:
        outs.update(os.path.normpath(o) for o in s['outputs'])
    unresolved = []
    for s in STAGES:
        for i in s['inputs']:
            n = os.path.normpath(i)
            if any(n == o or n.startswith(o + os.sep) or o.startswith(n + os.sep)
                   for o in outs):
                continue
            if os.path.exists(os.path.join(ROOT, 'analysis', i)):
                continue
            unresolved.append((s['id'], i))
    assert not unresolved, f'inputs with no producer and no file: {unresolved}'


@pytest.mark.parametrize('script,table', [
    ('run_soh_table.py', 'soh.csv'),
    ('run_mcu_table.py', 'mcu.csv'),
])
def test_producer_reproduces_its_table(script, table):
    """--check must confirm the stored table equals a fresh computation."""
    rc, out, err = run([os.path.join('repro', script), '--check'])
    assert rc == 0, f'{script} --check failed:\n{out}\n{err}'


def test_soh_table_producer_detects_a_corrupted_table(tmp_path):
    """The --check path must actually be able to fail."""
    src = os.path.join(TABLES, 'soh.csv')
    dst = tmp_path / 'soh.csv'
    lines = open(src, encoding='utf-8').read().splitlines()
    parts = lines[1].split(',')
    parts[2] = f'{float(parts[2]) + 0.01:.4f}'
    lines[1] = ','.join(parts)
    dst.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    rc, out, err = run([os.path.join('repro', 'run_soh_table.py'),
                        '--check', '--out', str(dst)])
    assert rc == 1
    assert 'MISMATCH' in (out + err)


def test_no_table_is_empty():
    for f in sorted(os.listdir(TABLES)):
        if not f.endswith('.csv'):
            continue
        rows = list(csv.DictReader(open(os.path.join(TABLES, f),
                                        encoding='utf-8')))
        assert rows, f'{f} has a header but no rows'


def test_figures_survive_extra_ladder_rows():
    """Adding a method to ladder.csv must not break the figure.

    It did: fig_ladder indexed a dict by every method name in the table and
    raised KeyError the moment the audit added LSTM, GRU and FFRLS rows.
    """
    rc, out, err = run([os.path.join('repro', 'fig_ladder.py')])
    assert rc == 0, f'fig_ladder.py failed:\n{out}\n{err}'


def test_every_figure_renders():
    for f in ('fig_ladder.py', 'fig_soh_traj.py'):
        rc, out, err = run([os.path.join('repro', f)])
        assert rc == 0, f'{f} failed:\n{out}\n{err}'
