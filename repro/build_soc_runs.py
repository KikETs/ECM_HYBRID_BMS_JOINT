"""Build the bundle of drive runs the SOC benchmark uses.

Why this is a separate stage: section 30's SOC experiments originally read
/tmp/soc_runs.pkl.  That was convenient while exploring, but a reproduction
package must not depend on /tmp — it disappears on a reboot and carries no
record of what made it.

Six drive-cycle files are picked at even spacing from each of the six cells,
giving 36 runs.  The picking rule is np.linspace, so it is deterministic.
Each run uses only the first 20,000 samples (about 5.5 hours at 1 Hz).

Note: BOOST_NEGPULSE_1S has few files, so linspace picks the same file twice
(recorded in 30.11).  That cell has 5 effective runs.  It is left unfixed
because every number so far was produced on this configuration.
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(os.path.dirname(HERE), 'analysis')
sys.path.insert(0, ANALYSIS)

CELLS = ['CC', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_REST', 'CC_CELL2',
         'BOOST_NEGPULSE_1S']
NRUN_PER_CELL = 6
NMAX = 20000
MIN_VALID = 2000


def build(cache_dir):
    from ecm_surface import ECMSurface
    runs = []
    for cell in CELLS:
        sd = ECMSurface(cell, 'discharge')
        sc = ECMSurface(cell, 'charge')
        z = np.load(os.path.join(cache_dir,
                                 f'uypydj_{cell}_Fifteen_Drive_Cycles.npz'))
        lens = z['lens']
        off = np.concatenate([[0], np.cumsum(lens)])
        for k in np.linspace(0, len(lens) - 1, NRUN_PER_CELL).astype(int):
            sl = slice(off[k], off[k] + lens[k])
            soc, V, I, SOH, T = (z[x][sl] for x in
                                 ('SOC', 'V', 'I', 'SOH', 'T'))
            ok = (np.isfinite(soc) & np.isfinite(V) & np.isfinite(I)
                  & np.isfinite(T))
            if ok.sum() < MIN_VALID:
                continue
            runs.append(dict(cell=cell, cyc=int(k), sd=sd, sc=sc,
                             soc=soc[ok][:NMAX], V=V[ok][:NMAX],
                             I=I[ok][:NMAX], T=T[ok][:NMAX],
                             soh=float(np.nanmedian(SOH))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(ANALYSIS, 'cache_t'))
    ap.add_argument('--out',
                    default=os.path.join(ANALYSIS, 'results', 'soc_runs.pkl'))
    ap.add_argument('--check-against', default=None,
                    help='only check that it matches an existing pkl')
    a = ap.parse_args()

    runs = build(a.cache)
    print(f'  {len(runs)} runs  ({len(set(r["cell"] for r in runs))} cells)',
          flush=True)
    for c in CELLS:
        rs = [r for r in runs if r['cell'] == c]
        sohs = sorted({round(r['soh'], 4) for r in rs})
        note = '  <- duplicate' if len(sohs) < len(rs) else ''
        print(f'    {c:<20} {len(rs)} runs, {len(sohs)} distinct SOH{note}',
              flush=True)

    if a.check_against:
        old = pickle.load(open(a.check_against, 'rb'))
        same = len(old) == len(runs) and all(
            o['cell'] == n['cell'] and abs(o['soh'] - n['soh']) < 1e-12
            and len(o['soc']) == len(n['soc'])
            and np.array_equal(o['soc'], n['soc'])
            and np.array_equal(o['I'], n['I'])
            for o, n in zip(old, runs))
        print(f"  \n  {'identical to' if same else 'DIFFERS from'} "
              f"{a.check_against}", flush=True)
        return 0 if same else 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(runs, f)
    print(f'\n  -> {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
