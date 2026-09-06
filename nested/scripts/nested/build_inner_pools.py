"""Correctly nested inner pools.  ecm_pool.surfaces() does not forward
ecm_csv/ocv_csv to build(), so every 'inner' pool the previous audit built
came from the full six-cell CSV (byte-identical to the shipped pools).  Here
build() is called directly with the five-cell CSVs, and the provenance sidecar
is written for those CSVs so that a later surfaces(j, outdir, ecm_csv, ocv_csv)
call finds the pool fresh and never rebuilds it from the defaults."""
import csv, hashlib, json, os, shutil, sys
import numpy as np
REPO = os.environ['RCSOP_REPO']; A = os.path.join(REPO, 'analysis')
RA = os.path.join(os.environ['RCSOP_BUNDLE'], 'assets', 'innerpool')
HERE = os.environ.get('OUTDIR', os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(HERE, 'innerpool')
sys.path.insert(0, A)
import ecm_pool
CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
log = []
for c in CELLS:
    d = os.path.join(OUT, f'no{c}'); os.makedirs(os.path.join(d, 'pool'), exist_ok=True)
    for fn in ('uypydj_ecm.csv', 'uypydj_ocv.csv'):
        # rebuild the five-cell CSV from the repo source (do not trust a copy)
        src = os.path.join(A, fn)
        rows = [r for r in csv.DictReader(open(src, encoding='utf-8')) if r['cell'] != c]
        assert {r['cell'] for r in rows} == set(CELLS) - {c}
        with open(os.path.join(d, fn), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ecmf, ocvf = os.path.join(d, 'uypydj_ecm.csv'), os.path.join(d, 'uypydj_ocv.csv')
    for j in CELLS:
        if j == c: continue
        pe, po, ne, no = ecm_pool.build(j, os.path.join(d, 'pool'), ecm_csv=ecmf, ocv_csv=ocvf)
        members = sorted(set(CELLS) - {c, j})
        ecm_pool._write_provenance(os.path.join(d, 'pool'), j, members, [ecmf, ocvf])
        same = sha(pe) == sha(os.path.join(A, 'cache', 'pool', f'ecm_pool_{j}.csv'))
        # content check: the pool must not contain any node whose members include c -> compare theta with the repo pool
        sd_new, _ = ecm_pool.surfaces(j, os.path.join(d, 'pool'), ecm_csv=ecmf, ocv_csv=ocvf)
        sd_old, _ = ecm_pool.surfaces(j)
        diffs = []
        for soc in (0.3, 0.5, 0.7, 0.9):
            for soh in (0.72, 0.8, 0.9, 0.98):
                a, b = sd_new.theta(soc, soh, -10.0), sd_old.theta(soc, soh, -10.0)
                if a['in_hull'][0] and b['in_hull'][0]:
                    diffs.append(abs(float(a['R0'][0]) - float(b['R0'][0])) / float(b['R0'][0]))
        log.append(dict(outer=c, holdout=j, members=','.join(members), n_ecm_rows=ne, n_ocv_rows=no,
                        identical_to_repo_pool=int(same), median_rel_R0_diff=float(np.median(diffs)) if diffs else float('nan'),
                        max_rel_R0_diff=float(np.max(diffs)) if diffs else float('nan'), sha_ecm=sha(pe)[:12]))
        print(f'  no{c} holdout={j} members={members} identical_to_repo={same} relR0 med {log[-1]["median_rel_R0_diff"]:.4f} max {log[-1]["max_rel_R0_diff"]:.4f}', flush=True)
with open(os.path.join(HERE, 'inner_pool_check.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
assert not any(r['identical_to_repo_pool'] for r in log), 'a pool is still identical to the shipped pool'
print('  all 30 inner pools differ from the shipped pools; provenance written for the five-cell CSVs')
