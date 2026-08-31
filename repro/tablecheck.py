"""Shared --check helper: does the stored table equal a fresh computation?

verify.py compares a committed CSV against expected.json, which catches a
table being edited but not a table drifting from the code that made it.  The
producers that derive entirely from other committed files can close that gap
cheaply, so they all take --check and all report failure the same way.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compare_or_fail(out_path, header, rows, label=None):
    """0 if the stored table equals header+rows, 1 otherwise (with a diff)."""
    label = label or os.path.basename(out_path)
    if not os.path.exists(out_path):
        print(f'  missing table: {out_path}', file=sys.stderr)
        return 1
    with open(out_path, encoding='utf-8', newline='') as f:
        old = list(csv.reader(f))
    new = [list(map(str, header))] + [[str(x) for x in r] for r in rows]
    if old == new:
        print(f'  {label} matches a fresh computation ({len(rows)} rows)')
        return 0
    print(f'  MISMATCH between stored {label} and a fresh computation',
          file=sys.stderr)
    if len(old) != len(new):
        print(f'    stored {len(old)} lines, recomputed {len(new)}',
              file=sys.stderr)
    shown = 0
    for i, (o, n) in enumerate(zip(old, new)):
        if o != n:
            print(f'    line {i}: stored={o}', file=sys.stderr)
            print(f'    line {i}:   fresh={n}', file=sys.stderr)
            shown += 1
            if shown >= 5:
                print('    ...', file=sys.stderr)
                break
    return 1
