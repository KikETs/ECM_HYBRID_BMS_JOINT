"""A pooled surface must be able to prove which cells went into it.

The whole leave-one-cell-out claim -- for SOP, for SOH, and since 37.24 for
SOC as well -- rests on one property: the pool the evaluated cell is scored
against does not contain that cell.  The cache checked existence, then mtime.
Neither can say WHICH cells are in a file.  A pool built for the wrong holdout
carries a perfectly good timestamp and leaks the cell it was supposed to
exclude, and every number downstream would still look fine.

So each pool writes a sidecar naming the holdout, the members and the SHA-256
of its sources, and `_stale` rejects a pool whose sidecar disagrees.  These
tests pin that: the members are exactly the other cells, and tampering with
any part of the record forces a rebuild.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ANALYSIS, 'uypydj_ecm.csv')),
    reason='needs analysis/uypydj_ecm.csv')


def _mod():
    import ecm_pool
    return ecm_pool


def _paths(E, holdout='CC'):
    return (os.path.join(E.POOL_DIR, f'ecm_pool_{holdout}.csv'),
            E._provenance(E.POOL_DIR, holdout),
            [os.path.join(ANALYSIS, 'uypydj_ecm.csv'),
             os.path.join(ANALYSIS, 'uypydj_ocv.csv')])


def test_the_pool_excludes_its_holdout_and_says_so():
    E = _mod()
    cwd = os.getcwd()
    os.chdir(ANALYSIS)
    try:
        E.surfaces('CC')
        pe, pv, srcs = _paths(E)
        rec = json.load(open(pv, encoding='utf-8'))
        assert rec['holdout'] == 'CC'
        assert 'CC' not in rec['members'], (
            'the CC-holdout pool lists CC among its members -- the cell it '
            'exists to exclude')
        assert len(rec['members']) == 5, rec['members']
        assert set(rec['sources']) == {'uypydj_ecm.csv', 'uypydj_ocv.csv'}
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize('tamper', ['holdout', 'members', 'source_hash',
                                    'delete'])
def test_a_tampered_record_forces_a_rebuild(tamper):
    E = _mod()
    cwd = os.getcwd()
    os.chdir(ANALYSIS)
    try:
        E.surfaces('CC')
        pe, pv, srcs = _paths(E)
        keep = open(pv, encoding='utf-8').read()
        try:
            if tamper == 'delete':
                os.remove(pv)
            else:
                rec = json.loads(keep)
                if tamper == 'holdout':
                    rec['holdout'] = 'BOOST'
                elif tamper == 'members':
                    rec['members'] = rec['members'] + ['CC']
                else:
                    k = next(iter(rec['sources']))
                    rec['sources'][k] = '0' * 64
                json.dump(rec, open(pv, 'w', encoding='utf-8'))
            assert E._stale(pe, 'CC', E.POOL_DIR, srcs), (
                f'tampering with {tamper} left the pool looking fresh; the '
                f'sidecar is not being checked')
        finally:
            open(pv, 'w', encoding='utf-8').write(keep)
    finally:
        os.chdir(cwd)


def test_an_untouched_pool_is_not_rebuilt():
    """A check that always rebuilds is as useless as one that never does."""
    E = _mod()
    cwd = os.getcwd()
    os.chdir(ANALYSIS)
    try:
        E.surfaces('CC')
        pe, pv, srcs = _paths(E)
        assert not E._stale(pe, 'CC', E.POOL_DIR, srcs)
    finally:
        os.chdir(cwd)
