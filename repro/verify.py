"""Check that the numbers cited in the paper come out of the pipeline as it is.

    python3 repro/verify.py              check everything
    python3 repro/verify.py --only sop   only ids containing 'sop'

expected.json holds (value, tolerance, source section).  The tolerance
defaults to 0 — the only randomness in this pipeline is the pack simulation
(fixed seed) and trim training (fixed seed), so reproduction should be exact
in principle.  Any entry with a non-zero tolerance carries the reason why.

**If this script fails, the number in the paper is what needs fixing.**  Not
the other way round.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')
EXPECTED = os.path.join(HERE, 'expected.json')


def read(name):
    p = os.path.join(TABLES, name)
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p, encoding='utf-8')))


def pick(rows, where, col):
    """Take col from the one row matching every column in `where`."""
    hit = [r for r in rows
           if all(str(r.get(k, '')) == str(v) for k, v in where.items())]
    if len(hit) != 1:
        raise LookupError(f'{len(hit)} rows matched (expected 1): {where}')
    return float(hit[0][col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--update', action='store_true',
                    help='rewrite expected.json with the current values. '
                         'Only use this when you know why a value changed.')
    a = ap.parse_args()

    spec = json.load(open(EXPECTED, encoding='utf-8'))
    checks = [c for c in spec['checks']
              if not a.only or a.only in c['id']]

    cache = {}
    ok = bad = skip = 0
    out = []
    print(f"  {'check':<34}{'expected':>10}{'actual':>10}{'tol':>8}  source",
          flush=True)
    print('  ' + '-' * 78, flush=True)
    for c in checks:
        rows = cache.get(c['table'])
        if rows is None:
            rows = cache[c['table']] = read(c['table'])
        if rows is None:
            print(f"  {c['id']:<34}{'':>10}{'no table':>10}{'':>8}  "
                  f"{c['source']}", flush=True)
            skip += 1
            out.append(c)
            continue
        try:
            got = pick(rows, c['where'], c['column'])
        except LookupError as e:
            print(f"  {c['id']:<34}  {e}", flush=True)
            bad += 1
            out.append(c)
            continue
        tol = c.get('tol', 0.0)
        good = abs(got - c['value']) <= tol
        mark = '' if good else '   <-- MISMATCH'
        print(f"  {c['id']:<34}{c['value']:>10.3f}{got:>10.3f}{tol:>8.3f}  "
              f"{c['source']}{mark}", flush=True)
        ok += good
        bad += not good
        d = dict(c)
        if a.update:
            d['value'] = round(got, 4)
        out.append(d)

    if a.update:
        spec['checks'] = out
        json.dump(spec, open(EXPECTED, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        print(f'\n  expected.json updated ({len(out)} entries)', flush=True)
        return 0

    print(f'\n  matched {ok}   mismatched {bad}   skipped {skip}', flush=True)
    if skip:
        print('  A skip means that stage has not been run yet — '
              'see repro/run.py', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
