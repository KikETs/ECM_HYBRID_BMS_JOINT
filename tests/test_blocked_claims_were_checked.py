"""An entry that says "blocked" has to say how that was established.

Twice in one session this repository recorded work as impossible without
looking.  raw/ was declared missing after checking ETRANS_AUDIT/raw instead of
the symlink at repo/raw, with 24 GB of it on disk.  Integrated loop timing was
recorded as needing a toolchain "that cannot be built on the audit host" while
that toolchain sat inside STM32CubeIDE and the board was plugged in — the gap
survived two review rounds and produced a partial workaround (37.21) for
something twenty lines of firmware could measure directly (37.22).

Both were cheap to check and expensive to leave wrong: a false "blocked" is a
claim about what the evidence CANNOT say, and it removes the work from anyone
else's view.

So every blocked_work entry must carry the check — `verified_absent_by`, a
command or an observation someone can repeat. Prose asserting absence is not
enough, because prose is exactly what was wrong both times.
"""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, '.paper_state', 'evidence_ledger.yaml')

# Fields that count as showing the check was made.
EVIDENCE_KEYS = ('verified_absent_by', 'what_would_unblock',
                 'command_when_data_exists', 'partly_answered_2026_09_01')


def test_a_decision_is_not_an_absence_claim():
    """Guards the exemption above from swallowing a real absence claim."""
    e = blocked_entries()['formal_equivalence_test']
    assert 'blocked_by' not in e, (
        'formal_equivalence_test now claims a material absence; it needs '
        'verified_absent_by like the others, or the blocked_by removed')
    assert e.get('action_taken') == 'deleted'


def load():
    return yaml.safe_load(open(LEDGER, encoding='utf-8'))


def blocked_entries():
    bw = load()['blocked_work']
    return {k: v for k, v in bw.items() if isinstance(v, dict)}


def test_every_blocked_entry_says_how_absence_was_established():
    """Only entries claiming something is ABSENT need the check.

    formal_equivalence_test is in blocked_work but claims no absence: the
    equivalence wording was deleted because a margin chosen now would be
    chosen after seeing the comparison.  That is a decision, and a decision is
    argued rather than verified.  `blocked_by` is what marks the other kind -
    "this does not exist on this machine" - and that is the kind that was
    wrong twice.
    """
    bad = []
    for name, entry in blocked_entries().items():
        if 'blocked_by' not in entry:
            continue
        if not any(k in entry for k in EVIDENCE_KEYS):
            bad.append(name)
    assert not bad, (
        'blocked_work entries asserting absence with nothing showing it was '
        'checked: ' + ', '.join(bad) + '\n'
        'Add verified_absent_by with the command or observation. Twice this '
        'session a "blocked" entry turned out to be wrong about a file that '
        'was on disk and a toolchain that was installed.')


def test_the_correction_is_kept():
    """The record of getting it wrong is the part most likely to be tidied away."""
    d = load()
    c = d.get('integrated_loop_was_never_blocked')
    assert c, (
        'the ledger no longer records that the integrated-loop gap was never '
        'actually blocked; without it the next reader sees only a clean '
        'history')
    assert c['status'] == 'CORRECTION_OF_THIS_LEDGER'
    for k in ('what_was_recorded', 'why_it_was_wrong', 'the_general_lesson'):
        assert k in c, f'{k} dropped from the correction'


def test_pack_and_hil_still_says_what_is_blocked_and_what_is_not():
    """Half-answered is the state most likely to be rounded to either end."""
    e = blocked_entries()['pack_and_hil']
    assert 'no pack hardware' in str(e.get('blocked_by', '')), (
        'pack_and_hil no longer states the hardware is absent')
    partly = str(e.get('partly_answered_2026_09_01', ''))
    assert 'NOT pack validation' in partly, (
        'the compute-cost result is recorded without the disclaimer that it '
        'is not pack validation, which is the sentence that keeps it honest')
