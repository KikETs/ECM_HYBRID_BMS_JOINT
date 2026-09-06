"""Compare the tables recomputed by `reproduce.sh score` with bundle/final/."""
import csv
import os
import sys

def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


B = os.environ['RCSOP_BUNDLE']; W = os.environ['RCSOP_WORK']
pairs = []
for fn in os.listdir(os.path.join(B, 'final', 'arms')):
    pairs.append((os.path.join(B, 'final', 'arms', fn), os.path.join(W, 'arms', fn)))
for fn in os.listdir(os.path.join(B, 'final', 'methods')):
    pairs.append((os.path.join(B, 'final', 'methods', fn), os.path.join(W, 'methods', fn)))
bad = 0
for ref, new in pairs:
    if not os.path.exists(new):
        print(f'  MISSING {os.path.basename(new)}'); bad += 1; continue
    a = list(csv.reader(open(ref))); b = list(csv.reader(open(new)))
    same = a == b
    if not same:
        # numeric tolerance 1e-3 on floats, exact on the rest
        same = len(a) == len(b) and all(len(x) == len(y) and all(
            (abs(float(u) - float(v)) <= 1e-3 if _isnum(u) and _isnum(v) else u == v) for u, v in zip(x, y)) for x, y in zip(a, b))
    print(f'  {"same" if same else "DIFF"}  {os.path.basename(ref)}')
    bad += 0 if same else 1
print(f'  {len(pairs)} tables compared, {bad} differ')
sys.exit(1 if bad else 0)
