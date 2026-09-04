"""Time the control cycle as one window, and at pack scale, on the board.

Two gaps this closes, both named by review.

337.10 relabelled 339.84 us a derived cycle-budget estimate rather than a
WCET, because the four stages of a control cycle -- feature update, SOC EKF,
trim, fixed-point inversion -- were only ever timed apart and added.  37.21
bounded the summation error over the 80 us the firmware could already pair
(SOP_CMD_FULL against TRIM + SOLVE) and found the sum sits 0.32 % ABOVE the
integrated maximum, i.e. not conservative pointwise.  It could not reach the
other 260 us because no command ran all four inside one DWT window.
SOP_CMD_CYCLE does.

And the pack question.  A series pack takes the min over cells, so a pack
master runs the per-cell estimate N times per cycle.  Whether that fits a
control budget is a compute question the board can answer, and it was
unanswered.  SOP_CMD_PACK runs the whole cycle for N cells with a separate
state per cell -- 124 B each, not shared -- and reduces by min.

WHAT THIS IS NOT.  It is not pack validation.  There is no second cell, no
pack, no HIL bench: the current, voltage and temperature handed to all N
cells are the same numbers.  It measures what the embedded implementation
costs at pack scale, not whether the estimate is right on a pack.  Those stay
blocked (evidence_ledger.blocked_work).

BUILD.  Both commands sit behind -DSOP_BENCH_PACK and are absent from the
default firmware, because SOP_CMD_PACK carries 24 KB of per-cell state and
would otherwise inflate the deployment footprint build_size.csv reports.

    cd mcu/fw_sop
    make MODEL_ID=pack_bench EXTRA_CFLAGS=-DSOP_BENCH_PACK
    STM32_Programmer_CLI -c port=SWD -w Build/pack_bench/sop_bench.elf -v -rst
    python3 repro/run_mcu_integrated.py --port /dev/ttyACM0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCU = os.path.join(ROOT, 'mcu')
sys.path.insert(0, MCU)
sys.path.insert(0, HERE)

OUT = os.path.join(ROOT, 'analysis', 'results', 'tables',
                   'mcu_integrated.csv')
HEADER = ['quantity', 'n_cells', 'n_points', 'median_us', 'max_us',
          'per_cell_us', 'pct_of_100hz_budget']
PACK_N = [1, 12, 48, 96, 192]
CMD_CYCLE, CMD_PACK = 0x70, 0x71


def operating_points():
    """The SAME points the existing stage bench used, from sop_mcu_bench.csv.

    A fresh random grid would make the integrated timing incomparable with the
    per-stage timings it is supposed to be checked against -- the comparison
    has to be paired on (SOC, SOH, tau), which is how 37.21 did the two-stage
    version.  FEAT_A8 rows are used because A8 is the adopted configuration
    and every stage in the published cycle total is measured over the same
    grid.
    """
    import csv as _csv
    path = os.path.join(MCU, 'sop_mcu_bench.csv')
    rows = [r for r in _csv.DictReader(open(path, encoding='utf-8'))
            if r['cmd'] == 'FEAT_A8']
    if not rows:
        raise SystemExit(f'no FEAT_A8 rows in {path}')
    return rows


def stage_sums(rows_by_cmd, keys):
    """Per-point sum of the stages a control cycle runs, paired on the key."""
    out = {}
    for k in keys:
        tot = 0.0
        for cmd in ('EKF', 'FEAT_A8', 'FULL'):
            r = rows_by_cmd[cmd].get(k)
            if r is None:
                return None
            tot += float(r['us'])
        out[k] = tot
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--n', type=int, default=200,
                    help='operating points per configuration; 0 = all')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    if a.check:
        # No fresh computation to compare against: reproducing this table
        # means re-flashing the board.  --check therefore only asserts the
        # table is present; verify.py holds the values.
        if not os.path.exists(OUT):
            print(f'  {os.path.relpath(OUT, ROOT)} missing; run on the board',
                  file=sys.stderr)
            return 1
        print('  board table, --check compares nothing: rerunning needs the '
              'hardware.  Verified through verify.py instead.')
        return 0

    import serial
    import struct
    import csv as _csv
    import bench_sop as B

    pts = operating_points()
    if a.n and a.n < len(pts):
        pts = pts[:a.n]
    keys = [(r['soc'], r['soh'], r['tau_s']) for r in pts]
    by_cmd = {}
    for r in _csv.DictReader(open(os.path.join(MCU, 'sop_mcu_bench.csv'),
                                  encoding='utf-8')):
        by_cmd.setdefault(r['cmd'], {})[(r['soc'], r['soh'], r['tau_s'])] = r
    summed = stage_sums(by_cmd, keys)
    x12 = [0.0] * B.NFEAT

    with serial.Serial(a.port, a.baud, timeout=2) as p:
        info = B.query(p)
        clk = info['clock_hz']
        print(f"  board up: {clk / 1e6:.0f} MHz, protocol v{info['version']}")
        print(f'  {len(pts)} operating points, the same grid as '
              f'sop_mcu_bench.csv')
        us = 1e6 / clk

        def timed(cmd_byte, dirn, r):
            p.write(bytes([cmd_byte]))
            p.flush()
            time.sleep(0.002)
            p.write(struct.pack(B.REQ, dirn, float(r['soc']), float(r['soh']),
                                3.0, 2.5, float(r['tau_s']), -20.0, *x12))
            p.flush()
            d = p.read(struct.calcsize(B.RES))
            if len(d) != struct.calcsize(B.RES):
                raise SystemExit(f'  short response for cmd {cmd_byte:#x} '
                                 f'({len(d)} B) — is the SOP_BENCH_PACK build '
                                 f'flashed?')
            m, cyc, it, hw, kf, ks, reff, ist = struct.unpack(B.RES, d)
            if m != B.MAGIC:
                raise SystemExit('  magic mismatch')
            return cyc * us

        rows = []
        integ = np.array([timed(CMD_CYCLE, 0, r) for r in pts])
        rows.append(['integrated cycle (1 solve)', 1, len(pts),
                     f'{np.median(integ):.3f}', f'{integ.max():.3f}',
                     f'{np.median(integ):.3f}',
                     f'{np.median(integ) / 10000 * 100:.3f}'])
        print(f'  integrated cycle    median {np.median(integ):8.3f}   '
              f'max {integ.max():8.3f} us')

        if summed:
            sm = np.array([summed[k] for k in keys])
            rows.append(['summed stages, same points', 1, len(pts),
                         f'{np.median(sm):.3f}', f'{sm.max():.3f}',
                         f'{np.median(sm):.3f}',
                         f'{np.median(sm) / 10000 * 100:.3f}'])
            d_med = (np.median(sm) - np.median(integ)) / np.median(integ) * 100
            d_max = (sm.max() - integ.max()) / integ.max() * 100
            larger = float((sm > integ).mean() * 100)
            rows.append(['summing error, paired', 1, len(pts),
                         f'{d_med:+.3f}', f'{d_max:+.3f}', '',
                         f'{larger:.1f}'])
            print(f'  summed, same points median {np.median(sm):8.3f}   '
                  f'max {sm.max():8.3f} us')
            print(f'  summing error       median {d_med:+7.3f} %   '
                  f'max {d_max:+7.3f} %   summing larger at {larger:.1f} % '
                  f'of points')

        for n in PACK_N:
            idx = PACK_N.index(n)
            dirn = 0 | (idx << 1)
            v = np.array([timed(CMD_PACK, dirn, r) for r in pts])
            rows.append(['pack min over N', n, len(pts),
                         f'{np.median(v):.3f}', f'{v.max():.3f}',
                         f'{np.median(v) / n:.3f}',
                         f'{np.median(v) / 10000 * 100:.3f}'])
            print(f'  pack N={n:>3}          median {np.median(v):9.3f}   '
                  f'max {v.max():9.3f} us   per cell {np.median(v) / n:6.3f}')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f'\n  -> {os.path.relpath(OUT, ROOT)}  ({len(rows)} rows)')
    print('  pct_of_100hz_budget is against a 10 ms control period.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
