"""The claim ledger has to be readable as a record, not just parseable.

paper_map.yaml is where a reader looks up "what does this paper still assert".
Four entries carried their current value under `note` instead of
`replacement`, so a scan for "claims with no current value" reported them as
open when they were not -- and, worse, two entries were VERIFIED while
describing a model the paper had replaced.  Neither is a YAML error; both are
the file failing at its one job.
"""
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, '.paper_state', 'paper_map.yaml')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

CURRENT = ('VERIFIED', 'VERIFIED_PROCESSED', 'NEW')
CORRECTION_KEYS = ('replacement', 'superseded_2026_08_31',
                   'wording_2026_08_31')


@pytest.fixture(scope='module')
def claims():
    return yaml.safe_load(open(MAP, encoding='utf-8'))['claims']


def test_every_claim_has_the_required_fields(claims):
    for c in claims:
        for k in ('id', 'text', 'status'):
            assert k in c and c[k], f'{c.get("id", c)} is missing {k}'


def test_claim_ids_are_unique(claims):
    ids = [c['id'] for c in claims]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f'duplicate claim ids: {sorted(dupes)}'


def test_a_claim_that_is_not_current_says_what_replaced_it(claims):
    """Anything not VERIFIED must carry the current statement.

    Otherwise a reader sees CONTRADICTED with no idea what the number is now,
    and an automated check cannot tell an open question from a closed one.
    """
    bad = [c['id'] for c in claims
           if c['status'] not in CURRENT
           and not any(k in c for k in CORRECTION_KEYS)]
    assert not bad, (
        'these claims are marked as not-current but never say what the '
        f'current value is: {bad}')


def test_a_current_claim_does_not_carry_a_replacement(claims):
    """VERIFIED plus a replacement is a contradiction in the file itself."""
    bad = [c['id'] for c in claims
           if c['status'] in ('VERIFIED', 'VERIFIED_PROCESSED')
           and any(k in c for k in CORRECTION_KEYS)]
    assert not bad, (
        f'these claims are VERIFIED yet carry a replacement, so the file '
        f'disagrees with itself about whether they still hold: {bad}')


def test_every_named_table_exists(claims):
    missing = []
    for c in claims:
        t = c.get('table')
        if not t or not isinstance(t, str):
            continue
        if '{' in t or '}' in t:
            # Brace expansion is ambiguous to a reader and to this parser --
            # 'safety_strict_{lstm,gru}_oracle.csv' split on commas leaves
            # 'gru}_oracle.csv', which is nobody's file.  Name the table.
            pytest.fail(f'{c["id"]}: name the table, not a brace pattern: {t}')
        for name in [x.strip() for x in t.replace('+', ',').split(',')]:
            if not name.endswith('.csv'):
                continue
            if not os.path.exists(os.path.join(TABLES, name)):
                missing.append((c['id'], name))
    assert not missing, f'claims naming tables that do not exist: {missing}'


def test_statuses_come_from_a_known_set(claims):
    KNOWN = {'VERIFIED', 'VERIFIED_PROCESSED', 'PARTIAL', 'CONTRADICTED',
             'REPLACED', 'CORRECTED', 'WITHDRAWN', 'SUPERSEDED', 'RESOLVED',
             'NEW'}
    bad = {c['status'] for c in claims} - KNOWN
    assert not bad, (
        f'unknown claim statuses {sorted(bad)} -- a typo here silently '
        f'changes which bucket a claim counts in')
