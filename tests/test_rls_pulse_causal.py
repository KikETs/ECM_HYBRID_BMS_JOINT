"""The HPPC-RLS baseline must not read the answer to the pulse it is predicting.

sop_trim_dataset.py pairs each measured pulse with twelve preceding drive
blocks, so one physical pulse is twelve rows with the SAME target.  The first
version of sop_rls.rls_cell walked rows -- predict, update, next row -- so rows
2..12 of a pulse were predicted by a theta already updated on that pulse's own
answer, up to twenty-two times.

The published effect was small (49.89 -> 50.02 mV) only because the shipped
forgetting factor is 1.0, where P shrinks monotonically and repeated updates on
one target add almost nothing.  At ff = 0.999 the same defect is worth
32.06 -> 44.14 mV.  A test that only pinned the shipped number would therefore
pass while the defect returned, so this tests the PROPERTY at a forgetting
factor where it bites.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

CACHE = os.path.join(ANALYSIS, 'cache', 'trim')
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CACHE, 'trim_CC.npz')),
    reason='needs analysis/cache/trim (the trim_data stage)')


def _cell(name='CC'):
    from sop_trim import load_cells
    return load_cells(CACHE)[name]


def _pulse_rows(d):
    """(cycle, SOC, rank) -> row indices.  One physical pulse each."""
    cyc = np.asarray(d['cycle']).astype(int)
    soc = np.round(np.asarray(d['SOC'], float), 6)
    rank = np.asarray(d['rank'])
    idx = {}
    for i in range(len(cyc)):
        idx.setdefault((cyc[i], soc[i], str(rank[i])), []).append(i)
    return idx


def test_one_pulse_really_is_many_rows_with_one_target():
    """The premise.  If this ever stops holding, the rest is testing nothing."""
    d = _cell()
    idx = _pulse_rows(d)
    Y = np.asarray(d['Y'], float)
    assert len(idx) < len(Y), 'no pulse is repeated across rows any more'
    counts = {len(v) for v in idx.values()}
    assert counts == {12}, f'expected 12 rows per pulse, saw {sorted(counts)}'
    for rows in idx.values():
        assert np.allclose(Y[rows], Y[rows[0]]), (
            'rows sharing a pulse key carry different targets; the key no '
            'longer identifies a physical pulse')


@pytest.mark.parametrize('ff', [1.0, 0.999])
def test_a_pulse_target_does_not_move_its_own_prediction(ff):
    """Perturb one pulse's Y; that pulse's own predictions must not move.

    Run at both forgetting factors: 1.0 is what ships, 0.999 is where the
    defect showed.  A property test rather than a value test, because the
    value at ff = 1.0 barely moved even while the leak was there.
    """
    from sop_rls import rls_cell
    d = _cell()
    idx = _pulse_rows(d)
    base, _ = rls_cell(d, ff=ff)
    rng = np.random.default_rng(1)
    keys = list(idx)
    worst = 0.0
    for k in [keys[i] for i in rng.choice(len(keys), 6, replace=False)]:
        d2 = {kk: (v.copy() if isinstance(v, np.ndarray) else v)
              for kk, v in d.items()}
        d2['Y'] = np.asarray(d2['Y'], float).copy()
        d2['Y'][idx[k]] += 0.05
        p2, _ = rls_cell(d2, ff=ff)
        worst = max(worst, float(np.abs(p2[idx[k]] - base[idx[k]]).max()))
    assert worst == 0.0, (
        f'at ff={ff} a pulse\'s own target moved its own prediction by '
        f'{worst:.3e} V.  rls_cell is updating on a pulse before it has '
        f'finished predicting it.')


def test_each_pulse_updates_the_filter_once():
    """The other half: predicting first must not mean never learning.

    With one update per pulse the filter still moves, so k varies across
    cycles.  A version that predicted everything and updated nothing would
    pass the leak test above.
    """
    from sop_rls import rls_cell
    d = _cell()
    _, K = rls_cell(d, ff=0.999)
    assert np.ptp(K[:, 0]) > 1e-6, 'k_f never moves; the filter is not learning'
    assert np.ptp(K[:, 1]) > 1e-6, 'k_s never moves; the filter is not learning'


def test_the_trajectory_does_not_depend_on_how_many_rows_a_pulse_has():
    """One physical pulse must move the filter once, not twelve times.

    Predicting a whole pulse before updating on it removes the leak but not
    this: updating twelve times on one observation weights it twelvefold and
    changes the effective learning rate for every pulse after it.  That is a
    different defect and the leak test cannot see it, so it is pinned
    separately.

    The invariant: thin the dataset to one row per pulse and the k trajectory
    at the surviving rows must be unchanged.  blocks_per_label is a feature
    choice; it must not be a filter hyperparameter.
    """
    from sop_rls import rls_cell
    d = _cell()
    idx = _pulse_rows(d)
    first = np.array(sorted(v[0] for v in idx.values()))

    _, K_full = rls_cell(d, ff=0.999)
    thin = {k: (v[first] if isinstance(v, np.ndarray) and len(v) == len(d['I'])
                else v) for k, v in d.items()}
    _, K_thin = rls_cell(thin, ff=0.999)

    assert np.allclose(K_full[first], K_thin, atol=1e-12), (
        'the k trajectory changes when the twelve rows of each pulse are '
        'thinned to one, so a pulse is updating the filter more than once and '
        'its observation is being weighted by however many feature blocks it '
        'happens to carry')
