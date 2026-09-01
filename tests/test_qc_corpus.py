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
    ('negation AFTER the phrase is still an assertion of it',
     'A8 outperformed all baselines, not just some of them.'),
    ('a denial in an earlier sentence does not license this one',
     'The trim is not superior. A8 outperformed all baselines.'),
    # The five words the fifth review named.
    ('superior asserted, naming the baselines',
     'A8 is superior to all baselines on safety-adjusted utility.'),
    # Without this the first alternative of the rule is dead weight: the case
    # above is caught by the second one, so neutering the first changed
    # nothing and the corpus stayed green.
    ('superior asserted without naming what it beats',
     'On charge at both horizons the trim is superior.'),
    ('statistical equivalence asserted',
     'The trim is statistically equivalent to the sequence baselines.'),
    ('pack-validated asserted',
     'The margin is pack-validated up to 192 cells in simulation.'),
    ('generalisation across protocols asserted',
     'The estimator is generalizable across cells and protocols.'),
    ('formal WCET asserted',
     'A formal WCET analysis gives 339.84 us per cycle.'),
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
    ('the safe form of the utility claim',
     'A8 provides competitive safety-adjusted utility against the tested '
     'baselines with a compact embedded implementation.'),
    ('superiority denied',
     'A8 is not superior to all baselines; three of twenty intervals '
     'separate.'),
    ('equivalence denied',
     'The trim is never statistically equivalent to the baselines without a '
     'pre-specified margin.'),
    ('a plain use of the word equivalent elsewhere',
     'AdaptiveAvgPool1d(8) is equivalent to AvgPool1d(4) on length-32 '
     'input, verified at max|delta| = 0.'),
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


# --- scope rules -----------------------------------------------------------
#
# SCOPED is the other half: not "is this retracted wording back" but "is this
# claim standing without the fact that bounds it".  A blacklist only catches
# the phrasing someone already used, so the first draft of these triggers
# matched on proximity -- and immediately flagged "10-14 %p of pack peak
# current" and "within 1.06x across cells", which are a cell-level result and
# a measured spread.  Both are pinned below.  Widening a trigger until real
# text stops tripping it is the same failure as widening an exemption.

SCOPE_MUST_FLAG = [
    ('pack exceedance with no mention of the simulation',
     'At the shipped lambdas the pack exceedance is 0.0 % at every string '
     'length.'),
    ('paraphrase that dodges the banned phrase',
     'The margin was confirmed on a 192-cell string in both directions.'),
    ('pack-level safety asserted',
     'These numbers establish pack-level safety at the shipped margin.'),
    ('lambda safe on a pack',
     'The shipped lambda is safe on a pack of any length we tried.'),
    ('cell generalisation asserted',
     'The estimator generalises across cells of this chemistry.'),
    ('part-number generalisation',
     'The result holds for INR21700-30T cells in general.'),
    ('transfer to all cells',
     'A single calibration transfers to all cells without adjustment.'),
    ('WCET stated flat',
     'The measured WCET of the full cycle is 339.84 us.'),
    ('binomial bound quoted as a risk figure',
     'External exceedance risk is bounded at 9.2 % with 95 % confidence.'),
    ('pooled bound with no denominator',
     'Pooling the sweep gives a one-sided 95 % upper bound of 6.1 %.'),
    # section_preamble deliberately reads only the section's opening lines.
    # A caveat thirty lines down is one a reader quoting the section never
    # sees, so it must not exempt.  Without this case, widening the window to
    # the whole section passes the corpus.
    ('caveat buried mid-section does not cover the section',
     '## 28. Pack level\n'
     '\n'
     + '\n'.join(f'Filler line {i} of the argument.' for i in range(30))
     + '\n'
     '\n'
     'It is only a resampling simulation, of course.\n'
     '\n'
     'The shipped lambda is safe on a pack of any length.'),
]

SCOPE_MUST_NOT_FLAG = [
    ('pack as a unit of current, not a system under test',
     'At the same zero-exceedance standard the trim buys back 10-14 %p of '
     'pack peak current.'),
    ('a measured spread across cells is not a generalisation claim',
     'R0 is unidentifiable at 1 Hz while being within 1.06x across cells '
     'anyway.'),
    ('pack claim carrying its qualifier',
     'In the resampling simulation the pack exceedance is 0.0 % at every '
     'string length; there is no pack hardware.'),
    ('cell claim carrying its population',
     'It generalises across the six cells under leave-one-cell-out, which is '
     'one model from one order.'),
    ('WCET named as a derived sum',
     'This is not a measured WCET: it is the sum of per-stage maxima.'),
    ('bound with its denominator as a fraction',
     'Internal holdout exceedance 3/651 = 0.46 %, 95 % upper bound 1.19 %.'),
    ('bound stated as conditional on the grid',
     'Zero in 48 in-hull points, 95 % upper bound 6.1 % over rows of one '
     'physical cell, conditional on the tested grid.'),
    ('the section preamble carries the caveat for the section',
     '## 28. Pack level\n'
     '\n'
     '> Everything in this section is a resampling simulation.  There is no\n'
     '> pack and no HIL bench.\n'
     '\n'
     'Using a cell-calibrated lambda directly on a pack raises the discharge\n'
     'exceedance rate more than threefold.'),
    ('pack-master is a part class, not a claim',
     'The S32K344 is a mainstream BMS pack-master class part.'),
    # The scope rules run through is_assertion for the same reason the
    # retraction rules do: a document that records what it withdrew has to be
    # able to write the withdrawn sentence down.
    ('a withdrawn pack claim, struck through',
     'The draft said ~~the pack exceedance is 0.0 % at every string '
     'length~~.'),
    ('a correction block quoting the unbounded claim',
     '[Corrected - 34.10] An earlier draft wrote "these numbers establish '
     'pack-level safety".\n'
     'They establish nothing of the kind.'),
    ('a correction block quoting an unbounded cell claim',
     '[Narrowed - 37.x] The manifest read COVERED, as if the result '
     'generalises across cells.\n'
     'It is now PARTIAL.'),
]


def _run_scope(text):
    import importlib
    import qc
    importlib.reload(qc)
    lines = text.split('\n')
    blks = qc.blocks(lines)
    hits = []
    for i, ln in enumerate(lines, 1):
        for _i, what, _why, _txt in qc.scope_hits(lines, blks, i, ln):
            hits.append((i, what))
    return hits


@pytest.mark.parametrize('name,text', SCOPE_MUST_FLAG,
                         ids=[n for n, _ in SCOPE_MUST_FLAG])
def test_unqualified_claim_is_flagged(name, text):
    assert _run_scope(text), (
        f'qc let an unbounded claim stand ({name}).  A trigger that was just '
        f'narrowed has made the scope check blind:\n{text}')


@pytest.mark.parametrize('name,text', SCOPE_MUST_NOT_FLAG,
                         ids=[n for n, _ in SCOPE_MUST_NOT_FLAG])
def test_qualified_claim_is_not_flagged(name, text):
    hits = _run_scope(text)
    assert not hits, (
        f'qc flagged legitimate text ({name}) as an unbounded claim: {hits}\n'
        f'{text}')
