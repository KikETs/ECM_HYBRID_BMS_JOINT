"""Mechanical check that every level-2 inner asset excludes the outer target
(and, where it is a per-cell asset, the calibration cell too).  Checks the
CONTENT, not only sidecars: pool CSVs must differ from the shipped pools and
contain no node whose provenance members include c; trim datasets must exist
for exactly the five other cells; inner trim predictions likewise; inner SOH
and SOC-error npz must carry no key for c.  Exit code 1 on any failure."""
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

X = os.path.dirname(os.path.abspath(__file__))
F = os.path.dirname(X)
REPO = os.environ['RCSOP_REPO']
A = os.path.join(REPO, 'analysis')
CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST', 'CC', 'CC_CELL2']
SUB = {'discharge': 'trim', 'charge': 'trim_chg'}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
fail, checked = [], 0


def chk(cond, msg):
    global checked
    checked += 1
    if not cond:
        fail.append(msg)


for c in CELLS:
    others = [j for j in CELLS if j != c]
    pd_ = os.path.join(X, 'innerpool', f'no{c}')
    for fn in ('uypydj_ecm.csv', 'uypydj_ocv.csv'):
        cells = {r['cell'] for r in csv.DictReader(open(os.path.join(pd_, fn), encoding='utf-8'))}
        chk(cells == set(others), f'{c}: {fn} cells {sorted(cells)}')
    for j in others:
        pe = os.path.join(pd_, 'pool', f'ecm_pool_{j}.csv'); po = os.path.join(pd_, 'pool', f'ocv_pool_{j}.csv')
        chk(os.path.exists(pe) and os.path.exists(po), f'{c}/{j}: pool missing')
        chk(sha(pe) != sha(os.path.join(A, 'cache', 'pool', f'ecm_pool_{j}.csv')), f'{c}/{j}: ecm pool identical to shipped')
        chk(sha(po) != sha(os.path.join(A, 'cache', 'pool', f'ocv_pool_{j}.csv')), f'{c}/{j}: ocv pool identical to shipped')
        pv = json.load(open(os.path.join(pd_, 'pool', f'pool_{j}.provenance.json')))
        chk(pv['members'] == sorted(set(others) - {j}), f'{c}/{j}: provenance members {pv["members"]}')
        chk(pv['sources']['uypydj_ecm.csv'] == sha(os.path.join(pd_, 'uypydj_ecm.csv')), f'{c}/{j}: provenance source hash != five-cell CSV')
        chk(not os.access(pe, os.W_OK), f'{c}/{j}: pool not read-only')
    # inner SOH
    sp = os.path.join(F, 'raudit_innersoh', f'soh_pred_no{c}.npz')
    keys = set(np.load(sp).files)
    chk(f'{c}_pred' not in keys and len([k for k in keys if k.endswith('_pred')]) == 5, f'{c}: inner SOH keys {sorted(keys)}')
    # inner SOC error
    se = os.path.join(F, '20_stateinput_nested', f'soc_err_inner_no{c}_replace_fixpool.npz')
    if chk(os.path.exists(se), f'{c}: inner soc_err missing') is None and os.path.exists(se):
        keys = set(np.load(se).files)
        chk(f'{c}_err' not in keys and {k[:-4] for k in keys if k.endswith('_err')} == set(others), f'{c}: inner soc_err keys {sorted(keys)}')
    for d, sub in SUB.items():
        dd = os.path.join(X, 'innerdata', f'{sub}_no{c}')
        have = {os.path.basename(x)[5:-4] for x in glob.glob(f'{dd}/trim_*.npz')}
        chk(have == set(others), f'{c}/{d}: inner dataset cells {sorted(have)}')
        for j in have:
            rp = os.path.join(A, 'cache', sub, f'trim_{j}.npz')
            chk(sha(os.path.join(dd, f'trim_{j}.npz')) != sha(rp), f'{c}/{d}/{j}: inner dataset identical to shipped')
        td = os.path.join(X, 'innertrim', f'{sub}_no{c}')
        preds = {os.path.basename(x).split('pred_A8_')[1][:-4] for x in glob.glob(f'{td}/pred_A8_*.npz')}
        chk(preds == set(others), f'{c}/{d}: inner A8 preds {sorted(preds)}')
        bt = os.path.join(X, 'basetrees', f'no{c}', 'analysis')
        sfx = '' if d == 'discharge' else '_chg'
        for tag in ('runs_trim_v2', 'runs_trim_shrink', 'runs_trim_lstm', 'runs_trim_gru', 'runs_trim_ffrls'):
            preds = {os.path.basename(x).split('pred_A3_')[1][:-4] for x in glob.glob(f'{bt}/{tag}{sfx}/pred_A3_*.npz')}
            chk(preds == set(others), f'{c}/{d}: {tag}{sfx} preds {sorted(preds)}')
print(f'  {checked} checks, {len(fail)} failures')
for m in fail[:60]:
    print('   -', m)
sys.exit(1 if fail else 0)
