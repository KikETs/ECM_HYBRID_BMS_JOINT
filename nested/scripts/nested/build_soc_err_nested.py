"""SOC-error assets for the state-input arms under the nested protocol.

Three things the shipped/previous assets did not do together:
  1. key = the drive record's actual END cycle (from the file name), not the
     file index.  A pulse at cycle N uses the nearest record with end < N.
  2. the drive record picked by linspace is checked for metadata consistency
     (cycle range vs. recording date vs. neighbours).  BOOST_NEGPULSE_1S file
     index 79 carries the cycle/SOH/CAP of the May record (1149-1163) but was
     recorded on 2023-08-18 between the 1507-1521 and 1544-1558 records.  Its
     SOC column is computed from that wrong CAP, so its reference SOC is not
     usable.  Rule: replace with the adjacent file (index-1) whose metadata is
     consistent ('replace'); the 'exclude' variant drops it instead.
  3. inner level: for outer target c and calibration cell j != c, the EKF for
     cell j runs on surfaces pooled from the four cells other than {c, j}
     (innerpool/no<c>/pool, holdout=j).  Outer level: surfaces pooled from
     the five cells other than j (repo cache/pool), as shipped (LOCO).

EKF configuration is the adopted one (run_end_to_end.build_soc_errors):
true SOH of the record, R_volt interp at SOH, gamma 0, i_gate 1 A,
rest_hold 30 s, first recorded SOC as the initial state, terminal-quarter
mean error.  Nothing in the repository is written.
"""
import csv
import datetime as dt
import multiprocessing as mp
import os
import re
import sys

import numpy as np

REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
RA = os.path.join(os.environ['RCSOP_BUNDLE'], 'assets')
INNERPOOL = os.environ.get('INNERPOOL', os.path.join(RA, 'innerpool'))
OUTTAG = os.environ.get('OUTTAG', '')
HERE = os.environ.get('OUTDIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, A)

CELLS = ['CC', 'BOOST', 'BOOST_NEGPULSE', 'BOOST_REST', 'CC_CELL2',
         'BOOST_NEGPULSE_1S']
NRUN, NMAX, MIN_VALID = 6, 20000, 2000


def parse(fname):
    s = str(fname)
    m = re.search(r'cycle_#(\d+)_to_(\d+)', s)
    d = dt.datetime.strptime(s.split('|')[1][:14], '%m-%d-%y_%H.%M')
    return int(m.group(1)), int(m.group(2)), d


def file_table(cell):
    z = np.load(os.path.join(A, 'cache_t',
                             f'uypydj_{cell}_Fifteen_Drive_Cycles.npz'),
                allow_pickle=True)
    rows = []
    for i, f in enumerate(z['files']):
        a, b, d = parse(f)
        rows.append(dict(idx=i, start=a, end=b, date=d, file=str(f).split('|')[1]))
    # metadata consistency: cycle order must equal date order
    by_date = sorted(rows, key=lambda r: r['date'])
    flagged = set()
    for k, r in enumerate(by_date):
        before = [x for x in by_date[:k] if x['start'] > r['start']]
        after = [x for x in by_date[k + 1:] if x['start'] < r['start']]
        # a file is the odd one out if it breaks monotonicity against many
        if len(before) + len(after) >= 2:
            flagged.add(r['idx'])
    return z, rows, flagged


def pick(cell, rule):
    z, rows, flagged = file_table(cell)
    idx = list(np.linspace(0, len(rows) - 1, NRUN).astype(int))
    note = []
    out = []
    for i in idx:
        if i in flagged:
            if rule == 'exclude':
                note.append(f'{cell} idx {i} excluded (inconsistent metadata)')
                continue
            j = i - 1 if (i - 1) not in flagged and (i - 1) not in idx else i + 1
            note.append(f'{cell} idx {i} -> {j} (inconsistent metadata: '
                        f'{rows[i]["file"]} recorded {rows[i]["date"]:%Y-%m-%d})')
            i = j
        out.append(i)
    return z, rows, out, note, flagged


def run_cell(args):
    level, c, j, rule = args
    import ecm_pool
    from ekf_soc import run as ekf_run
    if level == 'outer':
        sd, sc = ecm_pool.surfaces(j)
    else:
        pooldir = os.path.join(INNERPOOL, f'no{c}')
        sd, sc = ecm_pool.surfaces(j, outdir=os.path.join(pooldir, 'pool'),
                                   ecm_csv=os.path.join(pooldir, 'uypydj_ecm.csv'),
                                   ocv_csv=os.path.join(pooldir, 'uypydj_ocv.csv'))
    z, rows, idx, note, flagged = pick(j, rule)
    lens = z['lens']
    off = np.concatenate([[0], np.cumsum(lens)])
    recs = []
    for k in idx:
        sl = slice(off[k], off[k] + lens[k])
        soc, V, I, SOH, T = (z[x][sl] for x in ('SOC', 'V', 'I', 'SOH', 'T'))
        ok = np.isfinite(soc) & np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        if ok.sum() < MIN_VALID:
            continue
        soc, V, I, T = soc[ok][:NMAX], V[ok][:NMAX], I[ok][:NMAX], T[ok][:NMAX]
        soh = float(np.nanmedian(SOH))
        rv = float(np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015]))
        est, _ = ekf_run(sd, sc, soh, I, V, T, float(soc[0]), rv, gamma=0.0,
                         i_gate=1.0, rest_hold_s=30.0)
        q = max(1, len(est) // 4)
        err = float(np.mean(est[-q:] - soc[-q:]))
        recs.append(dict(level=level, outer=c, cell=j, idx=int(k),
                         file=rows[k]['file'], start=rows[k]['start'],
                         end=rows[k]['end'], date=rows[k]['date'].isoformat(),
                         soh=soh, n=len(est), err=err, rule=rule))
    return recs, note


def main():
    rules = sys.argv[1:] or ['replace', 'exclude']
    tasks = []
    for rule in rules:
        for j in CELLS:
            tasks.append(('outer', 'NONE', j, rule))
        for c in CELLS:
            for j in CELLS:
                if j != c:
                    tasks.append(('inner', c, j, rule))
    print(f'  {len(tasks)} EKF tasks x {NRUN} runs', flush=True)
    with mp.Pool(int(os.environ.get('NPROC', '12'))) as pool:
        results = pool.map(run_cell, tasks)
    allrecs, notes = [], set()
    for recs, note in results:
        allrecs += recs
        notes.update(note)
    for n in sorted(notes):
        print('  NOTE', n)
    with open(os.path.join(HERE, f'soc_err_runs{OUTTAG}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(allrecs[0].keys()))
        w.writeheader(); w.writerows(allrecs)
    for rule in rules:
        # outer
        d = {}
        for j in CELLS:
            rs = sorted([r for r in allrecs if r['rule'] == rule
                         and r['level'] == 'outer' and r['cell'] == j],
                        key=lambda r: r['end'])
            d[f'{j}_cycle'] = np.array([r['end'] for r in rs], int)
            d[f'{j}_err'] = np.array([r['err'] for r in rs], float)
        np.savez(os.path.join(HERE, f'soc_err_outer_{rule}{OUTTAG}.npz'), **d)
        for c in CELLS:
            d = {}
            for j in CELLS:
                if j == c:
                    continue
                rs = sorted([r for r in allrecs if r['rule'] == rule
                             and r['level'] == 'inner' and r['outer'] == c
                             and r['cell'] == j], key=lambda r: r['end'])
                d[f'{j}_cycle'] = np.array([r['end'] for r in rs], int)
                d[f'{j}_err'] = np.array([r['err'] for r in rs], float)
            np.savez(os.path.join(HERE, f'soc_err_inner_no{c}_{rule}{OUTTAG}.npz'), **d)
    print('  written', len(allrecs), 'runs', flush=True)


if __name__ == '__main__':
    main()
