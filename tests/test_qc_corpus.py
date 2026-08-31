"""A corpus that pins qc's retraction check in both directions.

The exemption rule was widened three times in one session -- same line, then
one line back, then two, each time a correction paragraph wrapped differently.
Widening a filter until the false positives stop is how a check quietly stops
catching anything, and nothing in the repository would have noticed.

So the rule is fixed here instead of in prose.  MUST_FLAG holds every shape of
a genuine reintroduction; MUST_NOT_FLAG holds every shape of a legitimate
record, taken from what sections 34 to 37 actually contain.  A future change to
qc.py has to keep both columns green, which makes "just exempt it" impossible
to do by accident.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'repro'))

MUST_FLAG = [
    ('bare assertion',
     'A8 outperformed all baselines in every condition.'),
    ('bare assertion, mid-paragraph',
     'The trim is competitive on charge.\n'
     'A8 outperformed all baselines at both horizons.\n'
     'Bootstrap intervals overlap.'),
    ('quoted but not discussed',
     'Our headline: "A8 outperformed all baselines" across the sweep.'),
    ('quoted, discussion word present but the phrase is asserted outside quotes',
     'Earlier drafts claimed less. A8 outperformed all baselines.'),
    ('equivalence asserted',
     'The result is deployment-efficient equivalence under a safety-aware '
     'current utility.'),
    ('four-parameter model asserted',
     'The shipped estimator is a four-parameter model on a Cortex-M33.'),
    ('production framing asserted',
     'Hybrid SOP / SOH / SOC for a production BMS.'),
    ('onboard filter framing asserted',
     'A wrong SOC corrupts the filter that admits labels, so exceedance rises.'),
    ('pack validation asserted',
     'The pack validation confirms the margin survives to 192 cells.'),
    ('narrowed claim WITHOUT its qualifier is a reintroduction',
     'Across the sweep, exceedances are zero for every N from 1 to 192.'),
    ('specific true comparison must NOT be flagged is a MUST_NOT case; the '
     'universal one must be',
     'A8 outperformed all baselines at both horizons.'),
    ('a quoted line in the same block does not make the block a blockquote',
     '> Some material quoted from the review.\n'
     'A8 outperformed all baselines.'),
    ('one blockquote line among many does not exempt the rest',
     'Round three findings:\n'
     '> the reviewer asked for a margin\n'
     'The shipped estimator is a four-parameter model.'),
    ('marker word in a NEIGHBOURING block must not exempt',
     'This section was corrected in round three.\n'
     '\n'
     'A8 outperformed all baselines.'),
    # The three below are the leak the block rule exists to close: an
    # exemption that belongs to one block must not reach the next one.  Each
    # fails if block_for is ever widened past its own block, which is exactly
    # what the sliding-window version did.
    ('bracketed marker in a NEIGHBOURING block must not exempt',
     '[Corrected — 37.3] The equivalence claim is withdrawn.\n'
     '\n'
     'A8 outperformed all baselines at both horizons.'),
    ('blockquote correction next to a fresh assertion',
     '> **[Narrowed — 37.2]** The heading overstated the mechanism.\n'
     '\n'
     'A wrong SOC corrupts the filter that admits labels.'),
    ('discussion verb in a NEIGHBOURING block, quoted phrase here',
     'Round two called the framing into question.\n'
     '\n'
     'We conclude: "A8 outperformed all baselines" holds throughout.'),
    ('qc.py mentioned in a NEIGHBOURING block',
     'The guards live in `qc.py` and run on every commit.\n'
     '\n'
     'The shipped estimator is a four-parameter model.'),
]

MUST_NOT_FLAG = [
    ('bracketed audit marker in the same block',
     '[Corrected — 37.3] The paper claimed "deployment-efficient equivalence".\n'
     'That word needs a margin fixed before the data is seen.'),
    ('SUPERSEDED marker, wrapped bullet',
     '* `paper_map.yaml` still offered `24.3 %` and\n'
     '  "deployment-efficient equivalence" as **replacements** — the\n'
     '  corrections from round two. Marked `[SUPERSEDED]` with current values.'),
    ('blockquote correction',
     '> **[Narrowed — 37.2]** The heading said "a wrong SOC corrupts the '
     'filter",\n'
     '> which reads as a measured failure rate of an onboard filter.'),
    ('quoted with a discussion verb two lines up',
     '**And the drift finding has to be stated at its actual scope.** §35.2\n'
     'called\n'
     'it "a wrong SOC corrupts the filter". No onboard filter was tested.'),
    ('qc.py describing its own guards',
     '`qc.py` now fails if "equivalence", "outperformed", "four-parameter '
     'model",\n'
     'the onboard-filter framing or "production" reappear.'),
    ('strikethrough',
     'The claim ~~A8 outperformed all baselines~~ is withdrawn.'),
    ('specific comparison against a named baseline is true, not the claim',
     'A8 beats A0 at every extrapolation ceiling in both directions.'),
    ('negated pack validation',
     'It is not a pack validation; there is no pack hardware.'),
    ('narrowed claim WITH its qualifier is not a reintroduction',
     'At the lambda values set on a zero-exceedance criterion, exceedances\n'
     'are zero for every N from 1 to 192, and usable current improves.'),
    ('never-a phrasing',
     'Say "four effective deployed coefficients", never "a four-parameter '
     'model".'),
]


def _run(text):
    """Return the retraction hits qc would report for this text."""
    import importlib
    import qc
    importlib.reload(qc)
    lines = text.split('\n')
    blks = qc.blocks(lines)
    hits = []
    for i, ln in enumerate(lines, 1):
        # qc's own function, not a copy of its loop.  The earlier version
        # reimplemented the scan here, so mutating the real one left the whole
        # corpus green.
        for _i, what, _where, _txt in qc.retraction_hits(lines, blks, i, ln):
            hits.append((i, what))
    return hits


@pytest.mark.parametrize('name,text', MUST_FLAG, ids=[n for n, _ in MUST_FLAG])
def test_reintroduction_is_flagged(name, text):
    hits = _run(text)
    assert hits, (
        f'qc did not flag a genuine reintroduction ({name}).  Whatever '
        f'exemption was just widened has made the check blind:\n{text}')


@pytest.mark.parametrize('name,text', MUST_NOT_FLAG,
                         ids=[n for n, _ in MUST_NOT_FLAG])
def test_record_of_a_retraction_is_not_flagged(name, text):
    hits = _run(text)
    assert not hits, (
        f'qc flagged a legitimate record of a retraction ({name}).  Fix the '
        f'rule structurally rather than adding another marker word:\n'
        f'{text}\nhits: {hits}')


def test_the_repository_itself_is_clean():
    """Whatever the corpus says, the real documents must pass."""
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'repro', 'qc.py')],
                       capture_output=True, text=True, cwd=ROOT)
    line = [x for x in r.stdout.splitlines() if 'retracted or refuted' in x]
    assert line, r.stdout[-800:]
    assert ' 0 places' in line[0], line[0]
