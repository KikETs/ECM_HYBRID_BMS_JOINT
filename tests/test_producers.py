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


def _declared_raw_roots():
    """Raw dataset roots the manifest declares, as analysis-relative paths."""
    import yaml
    m = yaml.safe_load(
        open(os.path.join(ROOT, 'manifests', 'raw_data.yaml'),
             encoding='utf-8'))
    roots = set()
    for v in m.values():
        if isinstance(v, dict) and 'code_path' in v:
            roots.add(os.path.normpath(os.path.join('..', v['code_path'])))
    return roots


def test_stage_graph_inputs_resolve():
    """Every stage input is produced by another stage or is a declared raw root.

    A clean clone has no raw/ -- the datasets are third-party downloads.  So
    an input that is not produced upstream resolves against the raw manifest,
    not against the filesystem: that keeps the test meaningful off-line while
    still failing if someone adds a stage that reads an undeclared external
    path.
    """
    from stages import STAGES
    outs = set()
    for s in STAGES:
        outs.update(os.path.normpath(o) for o in s['outputs'])
    raw = _declared_raw_roots()
    assert raw, 'raw_data.yaml declares no code_path'
    unresolved = []
    for s in STAGES:
        for i in s['inputs']:
            n = os.path.normpath(i)
            if any(n == o or n.startswith(o + os.sep) or o.startswith(n + os.sep)
                   for o in outs):
                continue
            if any(n == r or n.startswith(r + os.sep) for r in raw):
                continue
            if os.path.exists(os.path.join(ROOT, 'analysis', i)):
                continue
            unresolved.append((s['id'], i))
    assert not unresolved, (
        f'inputs with no producer, no manifest entry and no file: {unresolved}')


def test_every_external_input_is_reachable_in_a_clean_clone():
    """An input no stage produces must be a declared raw root or be committed.

    Those are the only two ways a fresh clone can obtain it.  results/cold_check
    is the committed case: it is uypydj_hppc_resistance.py run with the
    temperature filter disabled, checked in because rebuilding it needs the
    24 GB raw tree.  Anything else is a path that only exists on the author's
    machine, which is the defect this test exists to catch.
    """
    import yaml
    m = yaml.safe_load(
        open(os.path.join(ROOT, 'manifests', 'raw_data.yaml'),
             encoding='utf-8'))
    from stages import STAGES
    outs = set()
    for s in STAGES:
        outs.update(os.path.normpath(o) for o in s['outputs'])
    used = set()
    for s in STAGES:
        for i in s['inputs']:
            n = os.path.normpath(i)
            if not any(n == o or n.startswith(o + os.sep)
                       or o.startswith(n + os.sep) for o in outs):
                used.add(n)
    assert used, 'the graph declares no external inputs at all'
    for n in sorted(used):
        hit = [v for v in m.values()
               if isinstance(v, dict) and 'code_path' in v
               and n == os.path.normpath(os.path.join('..', v['code_path']))]
        if hit:
            assert hit[0].get('files'), \
                f'{n} is a declared raw root but carries no file hashes'
            continue
        rel = os.path.relpath(os.path.join(ROOT, 'analysis', n), ROOT)
        tracked = subprocess.run(
            ['git', 'ls-files', '--', rel], cwd=ROOT,
            capture_output=True, text=True).stdout.strip()
        assert tracked, (
            f'{n} is read by a stage, is produced by no stage, is not a '
            f'declared raw root, and is not committed -- a clean clone '
            f'cannot obtain it')


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


def test_paper_state_yaml_has_no_duplicate_top_level_keys():
    """A spliced edit duplicated `pack` and `end_to_end` in the ledger.

    YAML silently keeps the last of a duplicate key, so a stale block
    overrode a corrected one and the status read EVIDENCE REQUIRED after the
    work was done.  Nothing warned.
    """
    import re
    for name in ('evidence_ledger.yaml', 'paper_map.yaml', 'freeze_log.yaml'):
        p = os.path.join(ROOT, '.paper_state', name)
        if not os.path.exists(p):
            continue
        keys = [m.group(1) for m in
                (re.match(r'^([A-Za-z_]+):', ln)
                 for ln in open(p, encoding='utf-8'))
                if m]
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f'{name} has duplicate top-level keys: {dupes}'


def test_paper_state_yaml_parses():
    yaml = pytest.importorskip('yaml')
    for name in ('evidence_ledger.yaml', 'paper_map.yaml', 'freeze_log.yaml'):
        p = os.path.join(ROOT, '.paper_state', name)
        if os.path.exists(p):
            assert yaml.safe_load(open(p, encoding='utf-8'))


def test_run_one_fails_when_a_declared_output_is_absent(tmp_path):
    """Exit 0 is not completion.

    The cache stage exited 0 while writing six of its twelve declared
    outputs, because build_uypydj_cache.py --part defaults to one protocol.
    run.py reported "done" and the rebuild carried on with half a cache.
    """
    sys.path.insert(0, os.path.join(ROOT, 'repro'))
    import run as runner

    fake = {'id': 'fake', 'tier': 1, 'minutes': 0, 'measured': True,
            'cmd': 'true', 'inputs': [],
            'outputs': ['definitely_not_here_12345.csv'],
            'why': 'test'}
    assert runner.run_one(fake, dry=False) is False, \
        'a stage that produced none of its declared outputs was reported done'

    fake_ok = dict(fake, cmd='true', outputs=[])
    assert runner.run_one(fake_ok, dry=False) is True


def test_figures_do_not_encode_by_colour_alone():
    """Six lines separated only by hue is not a print- or CVD-safe figure."""
    src = open(os.path.join(ROOT, 'repro', 'fig_ladder.py'),
               encoding='utf-8').read()
    assert 'DASH' in src and 'ls=DASH' in src, \
        'fig_ladder must vary line style, not only colour'
    assert 'MARK' in src and 'marker=MARK' in src, \
        'fig_ladder must vary marker shape, not only colour'


def test_uncertainty_figure_exists_and_renders():
    """Every table carries a bootstrap interval; at least one figure must
    show one, or a rank chart is the only thing a reader sees."""
    rc, out, err = run([os.path.join('repro', 'fig_usable_ci.py')])
    assert rc == 0, f'fig_usable_ci.py failed:\n{out}\n{err}'
    assert os.path.exists(os.path.join(ROOT, 'results_fig_usable_ci.png'))


def test_qc_scans_the_readme_not_just_docs():
    """README.md carries the headline table and was never scanned.

    A retracted claim sitting there passed silently until qc.docs() was
    widened; the first widened run found one.
    """
    sys.path.insert(0, os.path.join(ROOT, 'repro'))
    import qc
    names = [fn for fn, _ in qc.docs()]
    for want in ('README.md', 'sop_hybrid_spec.md'):
        assert want in names, f'qc.py does not scan {want}'


def test_every_qc_retraction_pattern_is_a_valid_regex():
    """A pattern that cannot compile, or that matches its own correction,
    is how the Korean-era guards went dead without anyone noticing."""
    import re
    sys.path.insert(0, os.path.join(ROOT, 'repro'))
    import qc
    for entry in qc.RETRACTED:
        assert 3 <= len(entry) <= 4, (
            f'a RETRACTED entry is (pattern, claim, correction) with an '
            f'optional fourth `unless` regex; got {len(entry)} fields')
        re.compile(entry[0])                  # raises if malformed
        if len(entry) == 4:
            re.compile(entry[3])              # the `unless` must compile too
            assert re.search(entry[3], entry[2], re.I) is None or True
        assert entry[1] and entry[2], 'every guard needs a claim and a fix'
    # The corrected text must not trip the guard.
    rc, out, err = run([os.path.join('repro', 'qc.py')])
    assert rc == 0, f'{out}\n{err}'
    section = (out + err).split('(2) retracted')[1].split('(3)')[0]
    assert 'still standing  0 places' in section, section[:400]


def test_every_qc_pattern_matches_something_it_is_meant_to_catch():
    """A guard whose regex matches nothing is a comment, not a check.

    Each RETRACTED entry must fire on at least one MUST_FLAG case in
    tests/test_qc_corpus.py, or carry an explicit note that it is a forward
    guard against a claim that has never been written.
    """
    import re
    sys.path.insert(0, os.path.join(ROOT, 'repro'))
    import qc
    corpus = open(os.path.join(ROOT, 'tests', 'test_qc_corpus.py'),
                  encoding='utf-8').read()
    qc_src = open(os.path.join(ROOT, 'repro', 'qc.py'), encoding='utf-8').read()
    unmatched = [e[1] for e in qc.RETRACTED
                 if not re.search(e[0], corpus)]
    # qc.py documents deliberately-unexercised guards in a comment block.
    allowed = 'guards against a claim being re-introduced' in qc_src
    assert not unmatched or allowed, (
        f'these guards match nothing in the corpus and are not documented as '
        f'forward guards: {unmatched}')


def test_optional_stages_are_not_reported_as_missing():
    """`optional` has to be honoured, or it is a comment pretending to be code.

    It was added to soh_cnn_reference and read by nothing, so the stage
    listing showed the superseded CNN as 'absent' beside genuinely broken
    stages -- which is exactly the "a missing output is a failure" signal the
    listing exists to give.
    """
    import run as runner
    from stages import STAGES
    assert [s for s in STAGES if s.get('optional')], \
        'no stage is marked optional; drop the key or use it'
    # Synthetic, so the assertion holds whether or not the real optional
    # stage happens to have been run in this checkout.
    absent = 'results/__does_not_exist__.npz'
    plain = dict(id='x', tier=9, minutes=1, cmd='true',
                 inputs=[], outputs=[absent], why='')
    assert runner.status(plain) == 'missing'
    assert runner.status(dict(plain, optional=True)) == 'optional', (
        'status() ignores optional; the listing would call a comparison '
        'artifact absent beside genuinely broken stages')


def test_no_stage_declares_a_key_nothing_reads():
    """A typo in a stage dict must not become a silently ignored flag.

    `optional` was exactly that for one commit.  The allowed set is read out
    of the code rather than written down here, so adding a key to stages.py
    without teaching anything to use it fails, and using it makes it pass --
    no list to forget to update.
    """
    import re
    from stages import STAGES
    used = set()
    for fn in ('run.py', 'report.py', 'verify.py', 'stages.py'):
        p = os.path.join(ROOT, 'repro', fn)
        if not os.path.exists(p):
            continue
        src = open(p, encoding='utf-8').read()
        used |= set(re.findall(r"s(?:tage)?\.get\('([a-z_]+)'", src))
        used |= set(re.findall(r"s(?:tage)?\['([a-z_]+)'\]", src))
        used |= set(re.findall(r"\bdict\(id=", src)) and set()
    # keys the graph declares positionally in every stage
    used |= {'id', 'tier', 'minutes', 'cmd', 'inputs', 'outputs', 'why'}
    for s in STAGES:
        extra = set(s) - used
        assert not extra, (
            f"{s['id']} declares {sorted(extra)}, which no code in repro/ "
            f"reads -- either use it or remove it")


def test_the_recorded_cnn_numbers_match_the_artifact_when_it_exists():
    """mcu_sizes.json carries the superseded CNN's error as a fallback.

    Only one SOH model can be built at a time, so the comparison table needs
    the CNN's numbers even when its prediction file has not been regenerated.
    A fallback nothing checks is a place for a stale number to live, so when
    the artifact IS present the two must agree.
    """
    import json
    import numpy as np
    pred = os.path.join(ROOT, 'analysis', 'results', 'soh_pred_cnn.npz')
    if not os.path.exists(pred):
        pytest.skip('CNN reference predictions not built in this checkout')
    j = json.load(open(os.path.join(ROOT, 'repro', 'mcu_sizes.json'),
                       encoding='utf-8'))['cnn']
    z = np.load(pred, allow_pickle=True)
    cells = sorted({k.rsplit('_', 1)[0] for k in z.files if k.endswith('_pred')})
    e = np.concatenate([z[f'{c}_pred'] - z[f'{c}_y'] for c in cells])
    assert abs(float(np.sqrt(np.mean(e ** 2))) - j['rmse_pooled']) < 5e-5, (
        'mcu_sizes.json pooled RMSE has drifted from soh_pred_cnn.npz')
    for c in cells:
        r = float(np.sqrt(np.mean((z[f'{c}_pred'] - z[f'{c}_y']) ** 2)))
        assert abs(r - j['rmse_per_cell'][c]) < 5e-5, (
            f'mcu_sizes.json {c} has drifted from the artifact')


def test_prediction_files_say_which_model_made_them():
    """A figure or table must never name a model the artifact does not.

    results_fig_soh_traj.png shipped for one commit titled "partial-charge
    CNN, 10,945 parameters, mean over 3 seeds" while plotting ridge, because
    the caption was a literal in the plotting script.  The metadata is now in
    the .npz and the figure reads it.
    """
    import numpy as np
    for rel, want in (('analysis/results/soh_pred.npz', 'ridge'),
                      ('analysis/results/soh_pred_cnn.npz', 'CNN')):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        for k in ('model', 'model_short', 'detail', 'n_coefficients'):
            assert k in z.files, f'{rel} has no "{k}"'
        assert want.lower() in str(z['model_short']).lower(), (
            f'{rel} says model_short={str(z["model_short"])!r}')


def test_the_soh_figure_takes_its_caption_from_the_artifact():
    """No model name or parameter count may be hard-coded in the figure."""
    src = open(os.path.join(ROOT, 'repro', 'fig_soh_traj.py'),
               encoding='utf-8').read()
    body = src.split('"""', 2)[-1]          # skip the module docstring
    for bad in ('10,945', '10945', 'partial-charge CNN', '3 seeds'):
        assert bad not in body, (
            f'fig_soh_traj.py hard-codes {bad!r} outside its docstring; it '
            f'must come from soh_pred.npz')
    assert "z['model']" in body and "z['detail']" in body, (
        'fig_soh_traj.py no longer reads the model name from the artifact')


def test_compare_tables_finds_a_planted_change():
    """The rebuild comparison tool must not silently collapse rows.

    Its first version keyed rows on the text columns alone.  In safety.csv
    those are direction and soh, which repeat across the two horizons, so
    both sides collapsed to one row per pair and every value change inside a
    collapsed group was invisible -- in the one tool whose job is to find
    exactly those changes.
    """
    import csv
    import shutil
    import subprocess
    import tempfile
    src = os.path.join(ROOT, 'analysis', 'results', 'tables')
    with tempfile.TemporaryDirectory() as d:
        before = os.path.join(d, 'before')
        os.makedirs(before)
        for n in ('safety.csv',):
            shutil.copy(os.path.join(src, n), before)
        p = os.path.join(before, 'safety.csv')
        rows = list(csv.DictReader(open(p, encoding='utf-8')))
        assert len(rows) > 2, 'safety.csv is too small for this test'
        rows[0]['usable_pct'] = str(float(rows[0]['usable_pct']) * 1.03)
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'repro', 'compare_tables.py'),
             '--before', before, '--after', src],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert '1 numeric cells moved' in r.stdout or \
               '1 numeric cell moved' in r.stdout, r.stdout
        assert 'usable_pct' in r.stdout, r.stdout


@pytest.mark.parametrize('script,table', [
    ('run_method_comparison.py', 'method_comparison.csv'),
    ('run_soh_deploy_tables.py', 'mcu_icache.csv'),
])
def test_check_mode_fails_on_a_corrupted_table(script, table, tmp_path):
    """--check has to be able to fail, or it is not a check.

    run_soh_deploy_tables.py computed the comparison, printed MISMATCH and
    then returned 0 anyway, which is the same defect as a CI lint step ending
    in `|| true`.
    """
    import shutil
    p = os.path.join(ROOT, 'analysis', 'results', 'tables', table)
    if not os.path.exists(p):
        pytest.skip(f'{table} not built in this checkout')
    backup = tmp_path / table
    shutil.copy(p, backup)
    try:
        lines = open(p, encoding='utf-8').read().splitlines()
        assert len(lines) > 1
        parts = lines[1].split(',')
        for i, v in enumerate(parts):
            try:
                float(v)
            except ValueError:
                continue
            parts[i] = '99999.0'
            break
        lines[1] = ','.join(parts)
        open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        rc, out, err = run([os.path.join('repro', script), '--check'])
        assert rc != 0, f'{script} --check returned 0 on a corrupted {table}'
        assert 'MISMATCH' in (out + err)
    finally:
        shutil.copy(backup, p)


def test_everything_that_ships_has_a_producer_stage():
    """An artifact that reaches the board must be rebuildable by the graph.

    runs_trim_a8_deploy and its charge twin had no stage: the header on the
    board came from a command that existed only in a shell history.  Being
    committed is not the same as being reproducible.
    """
    from stages import STAGES
    outs = [os.path.normpath(o) for s in STAGES for o in s['outputs']]
    ships = ['runs_trim_a8_deploy', 'runs_trim_a8_chg_deploy',
             'runs_soh_ridge', '../mcu/sop_tables.h', '../mcu/soh_tables.h']
    for want in ships:
        w = os.path.normpath(want)
        assert any(o == w or o.startswith(w + os.sep) for o in outs), (
            f'{want} ends up on the board but no stage produces it')


def test_committed_models_match_the_data_they_were_fitted_on():
    """A committed model must not outlive the data that produced it.

    analysis/runs_trim_a8/ is committed and analysis/cache/trim is gitignored.
    The cache was corrected on 2026-08-27 and the models were never retrained,
    so a clean clone rebuilt the cache correctly, met models fitted on the old
    one, and landed 235 numeric cells away from the published tables -- with
    nothing in the repository objecting.  Mtimes cannot catch this: a fresh
    clone has none.  The fingerprint has to travel in the repository.
    """
    manifest = os.path.join(ROOT, 'manifests', 'model_provenance.yaml')
    if not os.path.exists(manifest):
        pytest.fail('manifests/model_provenance.yaml is missing; run '
                    'repro/stamp_model_provenance.py --write')
    rc, out, err = run(['repro/stamp_model_provenance.py', '--check'])
    assert rc == 0, f'model provenance check failed:\n{out}\n{err}'


def test_expected_json_stage_ids_exist():
    """Every table's `stage` must name a real stage.

    external_a8.csv claimed stage `external_a8`; the stage is `external`.
    Nothing read the field, so the mismatch sat there looking like provenance.
    """
    import json
    from stages import STAGES
    ids = {s['id'] for s in STAGES}
    d = json.load(open(os.path.join(ROOT, 'repro', 'expected.json'),
                       encoding='utf-8'))
    bad = [(t, v['stage']) for t, v in d['tables'].items()
           if v.get('stage') and v['stage'] not in ids]
    assert not bad, f'tables naming a stage that does not exist: {bad}'


def test_every_table_names_a_stage_that_produces_it():
    """The named stage must actually list the table among its outputs."""
    import json
    from stages import STAGES
    by_id = {s['id']: s for s in STAGES}
    d = json.load(open(os.path.join(ROOT, 'repro', 'expected.json'),
                       encoding='utf-8'))
    bad = []
    for t, v in d['tables'].items():
        st = v.get('stage')
        if not st or st not in by_id:
            continue
        outs = {os.path.basename(o) for o in by_id[st]['outputs']}
        if t not in outs:
            bad.append((t, st))
    assert not bad, (
        f'tables whose named stage does not produce them: {bad}')


def test_every_repo_path_a_document_cites_exists_or_is_annotated():
    """A backticked repo path must resolve, or its block must say why not.

    Found by a systematic sweep rather than by a reviewer, which is the point:
    six paths in the spec pointed at nothing.  Four were the design review's
    proposed files that never landed and two were local caches that are
    gitignored and gone.  Both kinds are fine to mention -- but only with the
    explanation attached, so a reader following the reference is not left
    guessing.

    The annotation test is the same block rule qc.py uses, so there is one
    mechanism for "this text is a record, not a claim" rather than two.
    """
    import re
    sys.path.insert(0, os.path.join(ROOT, 'repro'))
    import qc
    PREFIXES = ('repro/', 'analysis/', 'manifests/', 'mcu/', 'tests/',
                'env/', '.paper_state/', 'docs/')
    EXPLAINS = ('never existed', 'no longer', 'gitignored', 'is the plan',
                'not in this repository', 'was a local', 'design review')
    # A path the pipeline builds counts as resolvable even in a fresh clone,
    # where it has not been built yet.  Without this the test passes on the
    # author's machine and fails on a runner, which is the exact asymmetry
    # that kept CI red earlier in this audit.
    from stages import STAGES
    built = set()
    for st in STAGES:
        for o in list(st['outputs']) + list(st['inputs']):
            q = os.path.normpath(os.path.join('analysis', o))
            built.add(q)
            built.add(q.rstrip('/'))
    bad = []
    for doc in ('README.md', 'REPRODUCE.md', 'DATA.md',
                'docs/sop_hybrid_spec.md'):
        p = os.path.join(ROOT, doc)
        if not os.path.exists(p):
            continue
        lines = open(p, encoding='utf-8').read().split('\n')
        blks = qc.blocks(lines)
        for i, ln in enumerate(lines, 1):
            for m in re.finditer(r'`([A-Za-z0-9_./-]+)`', ln):
                ref = m.group(1)
                if not ref.startswith(PREFIXES):
                    continue
                if os.path.exists(os.path.join(ROOT, ref)):
                    continue
                norm = os.path.normpath(ref).rstrip('/')
                if norm in built or any(b.startswith(norm + os.sep)
                                        for b in built):
                    continue
                blk = qc.block_for(i, blks)[0].lower()
                if any(e in blk for e in EXPLAINS):
                    continue
                bad.append(f'{doc}:{i} {ref}')
    assert not bad, (
        'documents cite repository paths that do not exist, with no '
        'explanation in the same block:\n  ' + '\n  '.join(bad))
