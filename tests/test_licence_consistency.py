"""Every place that states a licence has to state the same one.

The UYPYDJ licence was recorded three different ways in this repository over
five days -- "not stated", then CC BY 4.0 from the readme, then CC BY-SA 4.0
from the Borealis record -- and each time it was written into four files by
hand.  A licence that disagrees with itself across README, LICENSE-DATA, the
raw-data manifest and the evidence ledger is worse than one that is simply
wrong: a reader cannot tell which one the author meant.

So the four are pinned to each other here.  This does not check that the
licence is CORRECT -- only the depositor can settle that, and the deposit
currently contradicts itself -- it checks that the repository says one thing.
"""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The licence this repository applies to its own derived data, and the
# upstream licence that forces it.  Change these two and the test tells you
# every file that has to change with them.
DERIVED = 'CC BY-SA 4.0'
UYPYDJ = 'CC BY-SA 4.0'


def read(name):
    return open(os.path.join(ROOT, name), encoding='utf-8').read()


def test_license_data_declares_the_derived_licence():
    head = read('LICENSE-DATA').split('\n\n', 1)[0]
    assert 'Attribution-ShareAlike' in head and '4.0' in head, (
        f'LICENSE-DATA does not open by declaring {DERIVED}:\n{head}')


def test_readme_agrees_with_license_data():
    r = read('README.md')
    assert DERIVED in r, f'README.md does not state {DERIVED}'
    assert not re.search(r'documentation are\s*\n?\s*\[CC BY 4\.0\]', r), (
        'README.md still describes the derived data as CC BY 4.0')


def test_raw_manifest_records_the_upstream_licence():
    m = yaml.safe_load(read('manifests/raw_data.yaml'))
    entry = next((v for v in m.values()
                  if isinstance(v, dict) and 'UYPYDJ' in str(v.get('doi', ''))),
                 None)
    assert entry, 'no UYPYDJ entry in manifests/raw_data.yaml'
    assert UYPYDJ in entry['license'], (
        f'raw_data.yaml records {entry["license"]!r}, not {UYPYDJ}')


def test_the_upstream_conflict_is_recorded_not_hidden():
    """Picking one side quietly is the failure mode this guards.

    The deposit says CC BY-SA 4.0 in the Borealis record and CC BY 4.0 in its
    own readme.  Whichever the repository follows, both must remain written
    down -- otherwise the next person re-derives the question from scratch,
    which is how this field got three different values already.
    """
    m = yaml.safe_load(read('manifests/raw_data.yaml'))
    entry = next(v for v in m.values()
                 if isinstance(v, dict) and 'UYPYDJ' in str(v.get('doi', '')))
    # A dedicated field, not "the words appear somewhere in the entry".  The
    # looser version passed when the conflict field was deleted, because a
    # history note elsewhere still happened to mention CC BY 4.0.
    conflict = [k for k in entry if 'conflict' in k.lower()]
    assert conflict, (
        'the UYPYDJ entry in raw_data.yaml has no field recording the '
        'upstream conflict; deleting it makes the repository look like the '
        'licence is settled when the deposit still contradicts itself')
    text = ' '.join(str(entry[k]) for k in conflict)
    # The literal readme line, not the words "CC BY 4.0" somewhere in the
    # paragraph -- the prose says "could revert to CC BY 4.0" further down, so
    # a substring test passed even after the quoted evidence was deleted.
    # What must survive is the quotation a reader can check against the file.
    for side in ('CC BY-SA 4.0', 'Licenses/restrictions: CC BY 4.0'):
        assert side in text, (
            f'the conflict field no longer quotes {side!r}, which is the '
            f'evidence for that side of the conflict')

    led = yaml.safe_load(read('.paper_state/evidence_ledger.yaml'))['licence']
    assert led['status'] == 'CONFLICT_UNRESOLVED_UPSTREAM', (
        f'the ledger calls the licence {led["status"]}; it is not resolved '
        f'until the depositor reconciles the readme with the Borealis record')
    assert 'manuscript_risk' in led, (
        'the ledger no longer carries the ShareAlike-vs-publisher risk, which '
        'is the part that decides whether the paper can be submitted as is')


def test_the_code_licence_is_separate_and_unchanged():
    assert 'MIT' in read('LICENSE'), 'LICENSE is no longer MIT'
    assert 'MIT' in read('README.md'), 'README no longer states the code licence'
