"""generalization_scope.yaml must not mix data design with model performance.

The file answers "what does the data cover".  It had also accumulated what a
particular estimator measured on that data -- an SOP RMSE of 17.5 W at -10 C,
a margin of 1.30-1.43, a per-cell spread of 1.834 to 2.474 %p.  Those are true
of one model on one day.  A scope statement that cites them is no longer a
scope statement, and a manuscript quoting the axis text would inherit
performance claims it never meant to make.

So each axis keeps its design facts at the top level and everything measured
under `model_dependent`, which also has to name the table it came from.  This
test enforces that split by looking for the shapes measured results take.
"""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'manifests', 'generalization_scope.yaml')

# Shapes a measured model result takes in this repository.  Design facts use
# counts, temperatures and cycle ranges, which these deliberately do not match.
MEASURED = [
    (r'\bRMSE\b', 'an RMSE'),
    (r'\d+(?:\.\d+)?\s*%p\b', 'an error in %p'),
    (r'\bmargin\s+\d', 'a safety margin'),
    (r'\blambda\b\s*(?:of\s*)?\d|λ\s*=?\s*0\.\d', 'a fitted lambda'),
    (r'\bexceedance\s+(?:rate|of)\s*\d|\bzero exceedance\b', 'an exceedance result'),
    (r'\busable current\b', 'usable current'),
    (r'\d+(?:\.\d+)?\s*W\b', 'a power error in W'),
]


def load():
    return yaml.safe_load(open(PATH, encoding='utf-8'))


def scope_text(axis):
    """Everything in an axis EXCEPT model_dependent."""
    return ' '.join(str(v) for k, v in axis.items() if k != 'model_dependent')


def test_no_measured_result_sits_outside_model_dependent():
    bad = []
    for name, axis in load()['axes'].items():
        text = scope_text(axis)
        for pat, what in MEASURED:
            m = re.search(pat, text, re.I)
            if m:
                bad.append(f'{name}: {what} -- ...{text[max(0, m.start() - 60):m.end() + 40]}...')
    assert not bad, (
        'generalization_scope.yaml states model performance where a contract '
        'will read it as data scope.  Move it under that axis\'s '
        'model_dependent key with the table it came from:\n  '
        + '\n  '.join(bad))


def test_model_dependent_blocks_name_their_table():
    """A number with no table behind it is not checkable and not citable."""
    bad = []
    for name, axis in load()['axes'].items():
        md = axis.get('model_dependent')
        if not md:
            continue
        if not (md.get('table') or md.get('tables')):
            bad.append(name)
    assert not bad, (
        'model_dependent blocks without a table reference: ' + ', '.join(bad))


def test_the_policy_is_declared_in_the_file_itself():
    """A rule that lives only in a test is one a reader of the file never sees."""
    d = load()
    pol = d.get('field_policy')
    assert pol, 'generalization_scope.yaml no longer declares field_policy'
    assert 'model_dependent' in pol['not_citable_for_scope']
    for k in ('status', 'detail', 'claim_limit'):
        assert k in pol['citable_for_scope'], f'{k} dropped from the policy'


def test_the_detector_actually_matches_a_measured_sentence():
    """A pattern list that matches nothing would make the first test vacuous."""
    sample = ('SOP RMSE of 1.7-4.2 W at 0 C, margin 1.30-1.43, the per-cell '
              'mean spans 1.834 to 2.474 %p, usable current, zero exceedance')
    hits = [w for pat, w in MEASURED if re.search(pat, sample, re.I)]
    assert len(hits) >= 5, (
        f'the MEASURED patterns matched only {hits} in a sentence built from '
        f'real text this file used to contain')
