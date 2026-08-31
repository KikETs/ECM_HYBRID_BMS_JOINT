"""MCU export and firmware-header consistency.

The audit found the firmware compiling A3 weights while the paper claimed
A8, and an exported header that was not valid C.  These pin both shut.
"""
import os
import re
import subprocess

import pytest

from conftest import ROOT, run

MCU = os.path.join(ROOT, 'mcu')
FW_INC = os.path.join(MCU, 'fw_sop', 'Inc')


@pytest.mark.parametrize('name', ['sop_tables.h', 'soh_tables.h',
                                  'soh_qparam.h', 'sop_core.h'])
def test_firmware_includes_the_exported_header(name):
    """fw_sop/Inc must not be an independent copy that can drift."""
    p = os.path.join(FW_INC, name)
    assert os.path.islink(p), \
        (f'{name} in fw_sop/Inc is a real file, not a symlink to mcu/{name}. '
         f'It can diverge from what the exporter writes, and it did: the '
         f'firmware carried A3 trim weights while the adopted model was A8.')
    assert os.path.realpath(p) == os.path.realpath(os.path.join(MCU, name))


def test_exported_header_is_valid_c():
    """`0f` is an integer constant with a float suffix; gcc rejects it."""
    src = open(os.path.join(MCU, 'sop_tables.h'), encoding='utf-8').read()
    # A C floating constant needs a '.' or an exponent.  Digits + 'f' alone
    # is an integer constant with a float suffix, which is a hard error.
    lits = re.findall(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?f\b', src)
    bad = [x for x in lits
           if '.' not in x and 'e' not in x.lower()]
    assert not bad, \
        (f'{len(bad)} integer-with-f literals such as {bad[:3]}. '
         f'A8 emits exact zeros, "%.7g" turns them into "0", and "0f" '
         f'does not compile.')


def test_header_compiles():
    out = os.path.join('/tmp', 'sop_core_test.o')
    p = subprocess.run(['gcc', '-O2', '-std=gnu11', '-I', MCU,
                        '-c', os.path.join(MCU, 'sop_core.c'), '-o', out],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[:2000]


def test_exporter_rejects_a_rung_that_does_not_match_the_directory():
    rc, out, err = run(['analysis/export_mcu_tables.py', '--rung', 'A3',
                        '--out', '/tmp/should_not_appear.h'])
    assert rc != 0
    assert 'A8' in (out + err) and 'A3' in (out + err)


def test_exporter_refuses_to_ship_a_leave_one_out_fold():
    rc, out, err = run(['analysis/export_mcu_tables.py', '--deployment',
                        '--out', '/tmp/should_not_appear.h'])
    assert rc != 0
    assert 'all-cell' in (out + err)


def test_manifest_header_hash_matches_the_committed_header():
    """The recorded hash must be the hash of the header actually in the tree.

    This file drifted once: the manifest went on describing the
    leave-one-cell-out fold for four days after the all-cell header replaced
    it, while the evidence ledger described the new one.  Two documents
    disagreeing about what is on the board is worse than either being wrong,
    because nothing in the repository objected.
    """
    import hashlib
    import yaml
    m = yaml.safe_load(open(os.path.join(ROOT, 'manifests', 'mcu_evidence.yaml'),
                            encoding='utf-8'))
    h = hashlib.sha256(
        open(os.path.join(ROOT, 'mcu', 'sop_tables.h'), 'rb').read()).hexdigest()
    assert m['model_on_board']['header_sha256'] == h, (
        f"manifests/mcu_evidence.yaml records "
        f"{m['model_on_board']['header_sha256'][:16]}... but mcu/sop_tables.h "
        f"hashes to {h[:16]}...")


def test_manifest_and_ledger_agree_on_the_deployed_weight():
    """The weight the manifest claims is flashed must be in the header."""
    import re
    import yaml
    m = yaml.safe_load(open(os.path.join(ROOT, 'manifests', 'mcu_evidence.yaml'),
                            encoding='utf-8'))
    led = yaml.safe_load(open(os.path.join(ROOT, '.paper_state',
                                           'evidence_ledger.yaml'),
                              encoding='utf-8'))
    hdr = open(os.path.join(ROOT, 'mcu', 'sop_tables.h'), encoding='utf-8').read()
    w0 = re.search(r'trim_w_dis\[\d+\] = \{\s*([-0-9.e+]+)f', hdr)
    assert w0, 'could not read trim_w_dis[0] out of the header'
    val = w0.group(1)
    ev = m['model_on_board']['binary_evidence']
    assert val in ev, (
        f'header ships trim_w_dis[0] = {val} but the manifest\'s binary '
        f'evidence does not mention it')
    # Only the blocks describing the CURRENT image, not the ones recording
    # the A3/A8 symlink defect the audit found -- those legitimately name the
    # old weights and must keep naming them.
    cur = [b for b in _walk_strings(led) if 'flashed image contains' in b]
    assert cur, 'the ledger no longer states what the flashed image contains'
    for block in cur:
        assert val in block, (
            f'the ledger says what the flashed image contains but does not '
            f'name {val}, which is the weight the header holds')


def _walk_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)


def test_soh_header_matches_the_deployed_ridge_fit():
    """The board's SOH weights must be the all-cell fit on disk.

    The SOP header had exactly this defect: the manifest, the ledger and the
    file disagreed for four days.  The SOH header is checked the same way,
    against the .npz the exporter reads, so a stale header cannot ship.
    """
    import re
    import numpy as np
    hdr = os.path.join(ROOT, 'mcu', 'soh_tables.h')
    fit = os.path.join(ROOT, 'analysis', 'runs_soh_ridge', 'soh_ALL.npz')
    if not os.path.exists(fit):
        pytest.skip('no all-cell ridge fit in this checkout')
    src = open(hdr, encoding='utf-8').read()
    assert '#define SOH_RIDGE 1' in src, 'the SOH header is not a ridge header'
    z = np.load(fit, allow_pickle=True)
    b = float(z['b'])
    m = re.search(r'#define SOH_B \(([-0-9.e+]+)f\)', src)
    assert m, 'no SOH_B in the header'
    assert abs(float(m.group(1)) - b) < 1e-6, (
        f'header intercept {m.group(1)} but soh_ALL.npz holds {b}')
    w = re.search(r'static const float soh_w\[(\d+)\]', src)
    assert w and int(w.group(1)) == len(z['w']), (
        f'header has {w.group(1) if w else "no"} weights, fit has {len(z["w"])}')


def test_the_ridge_build_refuses_the_integer_soh_opcode():
    """SOP_CMD_SOH_Q must NACK rather than answer with the float timing."""
    src = open(os.path.join(ROOT, 'mcu', 'fw_sop', 'Src', 'main.c'),
               encoding='utf-8').read()
    i = src.index('SOP_CMD_SOH_Q)\n    {')
    block = src[i:i + 700]
    assert '#if SOH_RIDGE' in block and 'SOP_NACK_UNKNOWN_CMD' in block, (
        'the ridge build no longer refuses the integer SOH opcode; it would '
        'report the float timing under the integer command')
