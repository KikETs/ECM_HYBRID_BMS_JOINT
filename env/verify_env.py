"""Import smoke test and determinism check.

    python3 env/verify_env.py            # imports + versions + determinism
    python3 env/verify_env.py --strict   # also fail if torch is absent

Exits non-zero on any missing requirement, so CI can gate on it.  torch is
optional by default: every stored-table verification and every figure runs
without it; only the two training stages need it.
"""
import argparse
import importlib
import platform
import sys

# (module, pip name, required-without-torch)
REQUIRED = [
    ('numpy', 'numpy', True),
    ('scipy', 'scipy', True),
    ('pandas', 'pandas', True),
    ('matplotlib', 'matplotlib', True),
    ('openpyxl', 'openpyxl', True),
    ('serial', 'pyserial', True),
    ('torch', 'torch', False),
    ('pytest', 'pytest', False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true',
                    help='fail when an optional package (torch) is absent')
    a = ap.parse_args()

    print(f'  python   {platform.python_version()}  ({sys.executable})')
    if sys.version_info[:2] != (3, 12):
        print(f'  WARNING  results were produced on 3.12, this is '
              f'{platform.python_version()}')

    missing = []
    for mod, pipname, hard in REQUIRED:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, '__version__', '?')
            print(f'  {mod:<12} {v}')
        except ImportError:
            print(f'  {mod:<12} MISSING  (pip install {pipname})')
            if hard or a.strict:
                missing.append(pipname)

    try:
        import torch
        print(f'  torch.cuda  available={torch.cuda.is_available()} '
              f'devices={torch.cuda.device_count()}')
    except ImportError:
        print('  torch.cuda  n/a (torch not installed)')

    # Determinism: the pipeline's only randomness is the pack simulation and
    # trim/CNN training, both seeded.  This checks the seeding actually
    # reproduces on this interpreter.
    import numpy as np
    a1 = np.random.default_rng(0).normal(size=5)
    a2 = np.random.default_rng(0).normal(size=5)
    det_np = bool(np.array_equal(a1, a2))
    print(f'  numpy default_rng(0) reproducible: {det_np}')
    if not det_np:
        missing.append('numpy-determinism')

    det_t = None
    try:
        import torch
        torch.manual_seed(0)
        t1 = torch.randn(5)
        torch.manual_seed(0)
        t2 = torch.randn(5)
        det_t = bool(torch.equal(t1, t2))
        print(f'  torch manual_seed(0) reproducible: {det_t}')
        if not det_t:
            missing.append('torch-determinism')
    except ImportError:
        pass

    if missing:
        print(f'\n  FAIL: {len(missing)} problem(s): {", ".join(missing)}')
        return 1
    print('\n  OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
