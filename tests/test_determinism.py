"""Every script that fits a published model must be reproducible.

sop_trim.py called torch.manual_seed and nothing else.  Two runs on identical
data gave weights ~0.05 % apart, sop_baseline_fill.py's grid search over
np.linspace(0, 1, 51) turned that into a one-step jump in alpha, and eight
verified numbers moved on a rebuild as a result.  Seeding is not the same as
determinism, and these tests exist so the difference cannot be forgotten
again.
"""
import ast
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))

# Scripts that fit a model whose output reaches a published number.
TRAINERS = [
    'analysis/sop_trim.py',
    'analysis/soh_cnn.py',
    'repro/run_sop_seq_baselines.py',
    'repro/run_soh_ablations.py',
]


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


@pytest.mark.parametrize('rel', TRAINERS)
def test_trainer_enables_determinism(rel):
    src = _src(rel)
    assert 'determinism.enable(' in src, (
        f'{rel} does not call determinism.enable(); torch.manual_seed alone '
        f'does not fix cuDNN kernel choice or cuBLAS workspace')
    assert 'torch.manual_seed(' not in src, (
        f'{rel} still calls torch.manual_seed directly; determinism.enable '
        f'does that and the rest, and two seeding paths will diverge')


@pytest.mark.parametrize('rel', TRAINERS)
def test_determinism_is_imported_before_torch(rel):
    """CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle is created."""
    tree = ast.parse(_src(rel))
    order = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.name in ('torch', 'determinism'):
                    order.append((n.lineno, al.name))
        elif isinstance(n, ast.ImportFrom) and n.module == 'torch':
            order.append((n.lineno, 'torch'))
    order.sort()
    names = [x[1] for x in order]
    assert 'determinism' in names, f'{rel} never imports determinism'
    assert names.index('determinism') < names.index('torch'), (
        f'{rel} imports torch before determinism, so CUBLAS_WORKSPACE_CONFIG '
        f'may be set after the handle exists and have no effect')


def test_enable_refuses_without_the_cublas_variable():
    """A run that cannot be reproduced must fail, not warn."""
    import importlib
    import determinism
    old = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ''
        importlib.reload(determinism)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ''
        with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
            determinism.enable(0)
    finally:
        if old is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = old
        importlib.reload(determinism)


def test_enable_actually_sets_the_flags():
    pytest.importorskip('torch')   # the flags are torch's; skip without it
    import determinism
    determinism.enable(0)
    st = determinism.state()
    assert st['deterministic_algorithms'] is True
    assert st['cudnn_deterministic'] is True
    assert st['cudnn_benchmark'] is False
    assert st['tf32_matmul'] is False
    assert st['cublas_workspace'] == ':4096:8'


def test_the_grid_search_that_amplifies_noise_is_documented():
    """fit_alpha turns a 0.05 % prediction shift into a 2.5 % published move.

    Determinism removes the shift, but the amplifier is still there and the
    next person to change the fit needs to know.
    """
    src = _src('analysis/sop_baseline_fill.py')
    assert re.search(r'linspace\(\s*0\.?0?\s*,\s*1\.?0?\s*,\s*51\s*\)', src), \
        'fit_alpha no longer searches the 51-point grid; update this test'
    assert 'NOISE AMPLIFIER' in src, (
        'sop_baseline_fill.py no longer warns that fit_alpha is an argmin '
        'over a discrete grid and therefore a step function of its input; '
        'that is how a 0.05 % training difference became a 2.5 % published '
        'one, and the next person to touch it has to be told')
