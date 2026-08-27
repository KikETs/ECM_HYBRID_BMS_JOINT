"""MCU export and firmware-header consistency.

The audit found the firmware compiling A3 weights while the paper claimed
A8, and an exported header that was not valid C.  These pin both shut.
"""
import os
import re
import subprocess
import sys

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
