"""Check that the numbers cited in the paper come out of the pipeline as it is.

    python3 repro/verify.py                 verify every stored table
    python3 repro/verify.py --only sop      only ids containing 'sop'
    python3 repro/verify.py --allow-missing tolerate absent tables (dev only)

This verifies **stored tables**.  It does not rebuild anything from the raw
data — that is `repro/run.py`, and the two are deliberately separate: a
stored-table pass says "the paper matches the artifacts in this repository",
not "the artifacts regenerate from the raw archives".  Never quote a
verify.py pass as raw-to-result reproduction.

What makes it fail (exit 1):

  * a required table is absent, empty, or unreadable
  * a table's column set, row count or numeric dtypes do not match the
    schema declared in expected.json
  * a check's lookup matches zero or more than one row
  * a value differs by more than its declared tolerance
  * the set of checks that ran does not match the completeness manifest
    (unless --only was given, which narrows deliberately)

A skipped check is a failure, not a pass.  Earlier versions returned 0 with
skips outstanding, so an unbuilt stage read as success.

**If this script fails, the number in the paper is what needs fixing.**  Not
the other way round.
"""
import argparse
import csv
import io
import json
import os
import sys

# Table text is UTF-8 regardless of the caller's locale.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')
EXPECTED = os.path.join(HERE, 'expected.json')


def read(name):
    """Return the rows of a table, or None if it is absent."""
    p = os.path.join(TABLES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def check_schema(name, rows, decl):
    """Structural problems with a table.  Empty list means it is sound."""
    bad = []
    if not rows:
        return [f'{name}: no data rows']
    cols = list(rows[0].keys())
    if cols != decl['columns']:
        miss = [c for c in decl['columns'] if c not in cols]
        extra = [c for c in cols if c not in decl['columns']]
        bad.append(f'{name}: column set differs'
                   + (f'; missing {miss}' if miss else '')
                   + (f'; unexpected {extra}' if extra else '')
                   + (f'; order changed' if not miss and not extra else ''))
    if len(rows) != decl['rows']:
        bad.append(f'{name}: {len(rows)} rows, schema says {decl["rows"]} '
                   f'— a partial run is not a pass')
    for c in decl['numeric']:
        if c not in cols:
            continue
        for i, r in enumerate(rows):
            v = r.get(c, '')
            try:
                float(v)
            except (TypeError, ValueError):
                bad.append(f'{name}: row {i} column {c!r} is {v!r}, '
                           f'not numeric')
                break
    for r in rows:
        if any(v is None for v in r.values()):
            bad.append(f'{name}: a row has fewer fields than the header '
                       f'— the file is truncated')
            break
    return bad


def pick(rows, where, col, name):
    """Take col from the one row matching every column in `where`."""
    for k in where:
        if rows and k not in rows[0]:
            raise LookupError(f'{name} has no column {k!r} '
                              f'(has {list(rows[0])})')
    hit = [r for r in rows
           if all(str(r.get(k, '')) == str(v) for k, v in where.items())]
    if len(hit) != 1:
        raise LookupError(f'{len(hit)} rows matched (expected 1): {where}')
    if col not in hit[0]:
        raise LookupError(f'{name} has no column {col!r}')
    return float(hit[0][col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    help='run only checks whose id contains this string')
    ap.add_argument('--allow-missing', action='store_true',
                    help='do not fail on an absent table.  Development only '
                         '— never use this to report a verification pass.')
    ap.add_argument('--update', action='store_true',
                    help='rewrite expected.json with the current values for '
                         'the checks that ran.  Every other entry is left '
                         'byte-for-byte alone.  Only use this when you know '
                         'why a value changed.')
    a = ap.parse_args()

    with open(EXPECTED, encoding='utf-8') as f:
        spec = json.load(f)
    decls = spec.get('tables', {})
    manifest = spec.get('completeness', {})
    all_checks = spec['checks']
    checks = [c for c in all_checks if not a.only or a.only in c['id']]

    if not checks:
        print(f'  no check id contains {a.only!r}', file=sys.stderr)
        return 1

    # ---- structural pass over every table the selected checks touch -------
    wanted = sorted({c['table'] for c in checks})
    cache, struct_bad, absent = {}, [], []
    for t in wanted:
        rows = cache[t] = read(t)
        decl = decls.get(t)
        if rows is None:
            absent.append(t)
            continue
        if decl is None:
            struct_bad.append(f'{t}: no schema declared in expected.json')
            continue
        struct_bad += check_schema(t, rows, decl)

    if absent:
        for t in absent:
            d = decls.get(t, {})
            print(f'  ABSENT  {t}   producer: {d.get("producer", "unknown")}',
                  file=sys.stderr)
    for m in struct_bad:
        print(f'  SCHEMA  {m}', file=sys.stderr)

    # ---- value pass -------------------------------------------------------
    ok = bad = skipped = 0
    ran_ids = []
    updated = {}
    print(f"  {'check':<34}{'expected':>10}{'actual':>10}{'tol':>8}  source")
    print('  ' + '-' * 78)
    for c in checks:
        rows = cache.get(c['table'])
        if rows is None:
            print(f"  {c['id']:<34}{'':>10}{'ABSENT':>10}{'':>8}  "
                  f"{c['source']}")
            skipped += 1
            continue
        try:
            got = pick(rows, c['where'], c['column'], c['table'])
        except LookupError as e:
            print(f"  {c['id']:<34}  LOOKUP FAILED: {e}")
            bad += 1
            continue
        tol = c.get('tol', 0.0)
        good = abs(got - c['value']) <= tol
        mark = '' if good else '   <-- MISMATCH'
        print(f"  {c['id']:<34}{c['value']:>10.3f}{got:>10.3f}{tol:>8.3f}  "
              f"{c['source']}{mark}")
        ok += good
        bad += not good
        ran_ids.append(c['id'])
        updated[c['id']] = round(got, 4)

    # ---- completeness -----------------------------------------------------
    comp_bad = []
    if not a.only:
        want_n = manifest.get('total_checks')
        if want_n is not None and len(all_checks) != want_n:
            comp_bad.append(f'expected.json holds {len(all_checks)} checks, '
                            f'the manifest declares {want_n}')
        want_ids = manifest.get('check_ids')
        if want_ids is not None:
            have = sorted(c['id'] for c in all_checks)
            if have != sorted(want_ids):
                lost = sorted(set(want_ids) - set(have))
                new = sorted(set(have) - set(want_ids))
                comp_bad.append(
                    'check id list drifted from the manifest'
                    + (f'; lost {lost}' if lost else '')
                    + (f'; added {new}' if new else ''))
    for m in comp_bad:
        print(f'  MANIFEST  {m}', file=sys.stderr)

    # ---- update, preserving everything that did not run -------------------
    if a.update:
        n = 0
        for c in all_checks:                    # iterate the FULL list
            if c['id'] in updated:
                c['value'] = updated[c['id']]
                n += 1
        with open(EXPECTED, 'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(f'\n  expected.json updated: {n} of {len(all_checks)} values '
              f'rewritten, {len(all_checks) - n} left untouched')
        return 0

    print(f'\n  matched {ok}   mismatched {bad}   absent {skipped}')
    if a.only:
        print(f"  (--only {a.only!r}: {len(checks)} of {len(all_checks)} "
              f"checks ran; completeness not enforced)")

    fail = bad > 0 or struct_bad or comp_bad
    if skipped:
        if a.allow_missing:
            print('  --allow-missing: absent tables did not fail the run.  '
                  'This is NOT a verification pass.')
        else:
            fail = True
            print('  An absent table is a failure, not a skip — run the '
                  'stage that produces it (repro/run.py --list).')
    print('  RESULT: ' + ('FAIL' if fail else 'PASS'))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
