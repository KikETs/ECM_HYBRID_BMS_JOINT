"""Record which training data each committed model was fitted on.

The defect this exists for: analysis/runs_trim_a8/ is committed, cache/trim is
gitignored, and the two drifted apart.  The cache was corrected, the models
were not retrained, and nothing objected -- so a clean clone rebuilt the cache
correctly, met models fitted on the old one, and landed 235 numeric cells away
from the published tables.  Mtimes cannot catch this because a fresh clone has
none, and the cache is not in git to diff against.

So the fingerprint travels with the repository instead.  --write records the
SHA-256 of every training file behind each committed run directory; --check
recomputes them and fails on a mismatch.  In a clone that has not built the
cache yet there is nothing to compare and the check reports that rather than
passing quietly.

    python3 repro/stamp_model_provenance.py --write
    python3 repro/stamp_model_provenance.py --check
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
OUT = os.path.join(ROOT, 'manifests', 'model_provenance.yaml')

# run directory -> the cache directory it is fitted on
RUNS = {
    'runs_trim_a8': 'cache/trim',
    'runs_trim_a8_chg': 'cache/trim_chg',
    'runs_trim_v2': 'cache/trim',
    'runs_trim_chg_v2': 'cache/trim_chg',
    'runs_trim_a8_deploy': 'cache/trim',
    'runs_trim_a8_chg_deploy': 'cache/trim_chg',
    'runs_soh_ridge': 'cache/soh_charge.npz',
    'runs_soh_cnn': 'cache/soh_charge.npz',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(rel):
    """{filename: sha256} for a cache directory, or a single file."""
    p = os.path.join(ANALYSIS, rel)
    if os.path.isfile(p):
        return {os.path.basename(p): sha(p)}
    if not os.path.isdir(p):
        return None
    return {f: sha(os.path.join(p, f))
            for f in sorted(os.listdir(p)) if f.endswith('.npz')}


def model_files(run):
    """{filename: sha256} for the checkpoints in a run directory.

    Recorded as well as the input hashes because the pipeline is now
    bit-reproducible: with training deterministic, a committed model that
    hashes differently from a fresh fit means something upstream moved, and
    that is worth failing on rather than discovering through a table diff.
    """
    d = os.path.join(ANALYSIS, run)
    if not os.path.isdir(d):
        return {}
    return {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d))
            if f.endswith(('.pt', '.npz'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.write == a.check:
        print('  pass exactly one of --write / --check', file=sys.stderr)
        return 2

    import yaml
    if a.write:
        doc = {'note': ('SHA-256 of the training data behind each committed '
                        'model directory.  Written by '
                        'repro/stamp_model_provenance.py --write after '
                        'retraining; checked by --check and by the test '
                        'suite.  The cache itself is gitignored, so this is '
                        'the only link between a committed model and the '
                        'data it was fitted on.'),
               'models': {}}
        for run, cache in RUNS.items():
            if not os.path.isdir(os.path.join(ANALYSIS, run)):
                continue
            fp = fingerprint(cache)
            if fp is None:
                print(f'  {run}: {cache} not present, skipped',
                      file=sys.stderr)
                continue
            mf = model_files(run)
            doc['models'][run] = {'trained_on': cache, 'files': fp,
                                  'model_files': mf}
            print(f'  {run:<26} {cache:<22} {len(fp)} inputs, '
                  f'{len(mf)} model files')
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
        print(f'  -> {os.path.relpath(OUT, ROOT)}')
        return 0

    if not os.path.exists(OUT):
        print(f'  {os.path.relpath(OUT, ROOT)} does not exist; run --write',
              file=sys.stderr)
        return 1
    doc = yaml.safe_load(open(OUT, encoding='utf-8'))
    bad, skipped = [], []
    for run, rec in doc['models'].items():
        fp = fingerprint(rec['trained_on'])
        if fp is None:
            skipped.append(run)
            continue
        for name, h in (rec.get('model_files') or {}).items():
            q = os.path.join(ANALYSIS, run, name)
            if not os.path.exists(q):
                bad.append(f'{run}: checkpoint {name} is missing')
            elif sha(q) != h:
                bad.append(f'{run}: checkpoint {name} differs from the '
                           f'recorded fit -- training is deterministic, so '
                           f'something upstream moved')
        want = rec['files']
        for name, h in want.items():
            if name not in fp:
                bad.append(f'{run}: {name} missing from {rec["trained_on"]}')
            elif fp[name] != h:
                bad.append(f'{run}: {name} changed since the model was fitted')
        for name in fp:
            if name not in want:
                bad.append(f'{run}: {name} is new in {rec["trained_on"]}')
    if skipped:
        print(f'  {len(skipped)} run dirs skipped: their cache is not built '
              f'here ({", ".join(sorted(skipped))})')
    if bad:
        print(f'  MODEL PROVENANCE MISMATCH ({len(bad)})', file=sys.stderr)
        for b in bad[:12]:
            print(f'    {b}', file=sys.stderr)
        if len(bad) > 12:
            print(f'    ... and {len(bad) - 12} more', file=sys.stderr)
        print('  The committed models were fitted on data this repository no '
              'longer produces.  Retrain, or the published numbers cannot be '
              'reached from a clean clone.', file=sys.stderr)
        return 1
    checked = len(doc['models']) - len(skipped)
    print(f'  {checked} model directories match the data they were fitted on')
    return 0


if __name__ == '__main__':
    sys.exit(main())
