"""Cell-by-cell diff of two sets of result tables.

Written for the defect-excluded rebuild, which changes numbers across many
tables at once.  Eyeballing a diff of ten CSVs is how a real movement gets
filed as noise, so this reports every changed numeric cell with its relative
change, groups them by table and column, and names the largest movers.

    python3 repro/compare_tables.py --before .preserved/<dir> --after analysis/results/tables

It refuses to call a table unchanged when its shape changed: added or removed
rows and columns are reported separately from moved values, because those are
different kinds of event and only one of them is a measurement.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read(path):
    with open(path, encoding='utf-8', newline='') as f:
        r = list(csv.DictReader(f))
    return r


def keyof(row, cols):
    """Identify a row by its non-numeric columns."""
    return tuple(row[c] for c in cols)


def choose_key(a, b, ca, cb, txt, num):
    """Smallest set of shared columns that identifies a row uniquely in both.

    The first version of this keyed on the text columns alone.  In safety.csv
    those are direction and soh, which repeat across the two horizons, so both
    tables collapsed to one row per pair and every value change inside a
    collapsed group was invisible.  A comparison tool that silently drops rows
    is worse than no tool, so the key grows until it is unique on both sides
    and the columns spent on the key are excluded from the value comparison.
    """
    shared = [c for c in ca if c in cb]
    ordered = [c for c in txt if c in shared] + [c for c in shared if c in num]
    key = []
    for c in ordered:
        key.append(c)
        ka = [keyof(r, key) for r in a]
        kb = [keyof(r, key) for r in b]
        if len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
            return key, [c2 for c2 in num if c2 not in key]
    return [], num


def numeric_cols(rows):
    num, txt = [], []
    for c in rows[0]:
        vals = [r[c] for r in rows if r[c] not in ('', 'nan', 'None')]
        if not vals:
            txt.append(c)
            continue
        try:
            [float(v) for v in vals]
            num.append(c)
        except ValueError:
            txt.append(c)
    return num, txt


def compare(before, after, name):
    a, b = read(before), read(after)
    if not a or not b:
        return dict(table=name, note='one side is empty', cells=0)
    ca, cb = list(a[0]), list(b[0])
    out = dict(table=name, rows_before=len(a), rows_after=len(b),
               cols_added=[c for c in cb if c not in ca],
               cols_removed=[c for c in ca if c not in cb],
               cells=0, moves=[])
    num, txt = numeric_cols(b)
    key_cols, num = choose_key(a, b, ca, cb, txt, num)
    out['key_cols'] = key_cols
    if not key_cols:
        out['note'] = 'no unique row key; not compared'
        return out
    ia = {keyof(r, key_cols): r for r in a}
    ib = {keyof(r, key_cols): r for r in b}
    out['rows_added'] = len(set(ib) - set(ia))
    out['rows_removed'] = len(set(ia) - set(ib))
    for k in sorted(set(ia) & set(ib)):
        for c in num:
            if c not in ca:
                continue
            va, vb = ia[k][c], ib[k][c]
            if va in ('', 'nan') or vb in ('', 'nan'):
                continue
            fa, fb = float(va), float(vb)
            if fa == fb:
                continue
            out['cells'] += 1
            rel = abs(fb - fa) / abs(fa) * 100 if fa else float('inf')
            out['moves'].append((rel, name, '|'.join(k), c, fa, fb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True)
    ap.add_argument('--after',
                    default=os.path.join(ROOT, 'analysis', 'results', 'tables'))
    ap.add_argument('--top', type=int, default=12)
    a = ap.parse_args()

    names = sorted(f for f in os.listdir(a.before) if f.endswith('.csv'))
    total, changed, all_moves = 0, [], []
    print(f"  {'table':<38}{'rows':>10}{'cells moved':>13}{'max rel %':>11}")
    print('  ' + '-' * 74)
    for n in names:
        pb = os.path.join(a.after, n)
        if not os.path.exists(pb):
            print(f'  {n:<38}{"":>10}{"gone from after":>13}')
            continue
        r = compare(os.path.join(a.before, n), pb, n)
        if r.get('note'):
            print(f'  {n:<38}{r["note"]}')
            continue
        total += r['cells']
        mx = max((m[0] for m in r['moves']), default=0.0)
        shape = ''
        if r['rows_before'] != r['rows_after']:
            shape = f"{r['rows_before']}->{r['rows_after']}"
        elif r['cols_added'] or r['cols_removed']:
            shape = 'cols'
        if r['cells'] or shape:
            changed.append(n)
            print(f'  {n:<38}{shape:>10}{r["cells"]:>13}{mx:>11.3f}')
        all_moves += r['moves']

    print('  ' + '-' * 74)
    print(f'  {len(changed)} of {len(names)} tables changed, '
          f'{total} numeric cells moved')
    if all_moves:
        rels = sorted(m[0] for m in all_moves)

        def pct(q):
            # linear interpolation, so a two-element list does not report a
            # p95 below its own median
            if len(rels) == 1:
                return rels[0]
            i = q * (len(rels) - 1)
            lo = int(i)
            hi = min(lo + 1, len(rels) - 1)
            return rels[lo] + (rels[hi] - rels[lo]) * (i - lo)

        print(f'  relative change: median {pct(0.5):.4f} %, '
              f'p95 {pct(0.95):.4f} %, max {rels[-1]:.4f} %')
        print(f'\n  largest {a.top} movers')
        for rel, tbl, key, col, fa, fb in sorted(all_moves, reverse=True)[:a.top]:
            print(f'    {rel:>8.3f} %  {tbl:<30}{key:<34}{col:<22}'
                  f'{fa:>12g} -> {fb:<12g}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
