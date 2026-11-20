"""E5 step 1: row-level predictions for all comparison methods (same nested protocol as score_methods_nested.py)
plus a conventional HPPC pulse-resistance lookup baseline.

HPPC baseline: R(SOC, SOH) = measured pulse resistance R_mOhm at the highest-rate rank for the same direction and tau,
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
from the training cells only (outer: five cells != c; inner: four cells != c, j), linear in (SOC, SOH) with
nearest-neighbour fill outside the convex hull. I_lim = |V_lim - V_pre| / R. SOH is the same estimated SOH the other
methods use; SOC is the reference SOC (primary arm).
Output: out/E5/method_rows.pkl  {(method, level, c, j, direction): [row dicts]}
"""
import os, sys, csv, pickle, multiprocessing as mp
import numpy as np

os.environ.update(RCSOP_REPO=REPO_ROOT, RCSOP_BUNDLE=os.path.join(REPO_ROOT, 'nested'),
                  SOLVER='hull', OUTDIR=os.path.join(R2, 'out', 'E5'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'nested', 'scripts', 'nested'))
import score_methods_nested as SM                                   # noqa: E402
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator  # noqa: E402

OUT = os.path.join(R2, 'out', 'E5'); os.makedirs(OUT, exist_ok=True)
CELLS = SM.CELLS
A = SM.A

HP = list(csv.DictReader(open(os.path.join(A, 'uypydj_hppc_resistance.csv'), encoding='utf-8')))


def r_table(direction, tau, cells):
    sel = [r for r in HP if r['direction'] == direction and float(r['tau_s']) == tau and r['protocol'] in cells]
    top = {}
    for r in sel:
        k = (r['protocol'], r['cycle'], r['soc_group'])
        if k not in top or abs(float(r['I_A'])) > abs(float(top[k]['I_A'])):
            top[k] = r
    P = np.array([[float(r['SOC']), float(r['SOH'])] for r in top.values()])
    V = np.array([float(r['R_mOhm']) for r in top.values()])
    lin, near = LinearNDInterpolator(P, V), NearestNDInterpolator(P, V)
    def f(soc, soh):
        v = float(lin(soc, soh))
        return v if np.isfinite(v) else float(near(soc, soh))
    return f


def hppc_rows(level, c, j, direction):
    charge = direction == 'charge'
    lab = SM.EV.LABEL_CHG if charge else SM.EV.LABEL
    rows = [r for r in csv.DictReader(open(lab, encoding='utf-8')) if r['cell'] == j]
    train = [k for k in CELLS if k != c] if level == 'outer' else [k for k in CELLS if k not in (c, j)]
    soh_map = dict(np.load(os.path.join(A, 'results', 'soh_pred.npz'))) if level == 'outer' else \
        dict(np.load(os.path.join(SM.INNERSOH, f'soh_pred_no{c}.npz')))
    tabs = {tau: r_table(direction, tau, set(train)) for tau in (2.0, 10.0)}
    out = []
    for r in rows:
        if r.get('I_star_lin4_A') in (None, '', 'nan'):
            continue
        cyc = int(r['cycle']); soc = float(r['SOC']); tau = float(r['tau_s'])
        cc, pp = soh_map[f'{j}_cycle'], soh_map[f'{j}_pred']
        prior = cc <= cyc
        soh = float(pp[prior][int(np.argmax(cc[prior]))]) if prior.any() else float(pp[int(np.argmin(cc))])
        R = tabs[tau](soc, soh) / 1000.0
        if not (np.isfinite(R) and R > 0):
            continue
        I = abs(float(r['V_min_V']) - float(r['V_pre_V'])) / R
        out.append(dict(cell=j, meas=abs(float(r['I_star_lin4_A'])), hyb=I, ecm=np.nan, extrap=float(r['extrap']), tau=tau,
                        cycle=cyc, soc=soc, soh=soh, V_pre=r['V_pre_V']))
    return ('hppc', level, c, j, direction), out


def method_rows(m, level, c, j, direction):
    key, rows = SM.rows_for(m, level, c, j, direction)
    return key, rows


if __name__ == '__main__':
    tasks = []
    for m in ('a8', 'a3', 'shrink', 'lstm', 'gru', 'ffrls'):
        for d in ('discharge', 'charge'):
            for c in CELLS:
                tasks.append((m, 'outer', c, c, d))
                tasks += [(m, 'inner', c, j, d) for j in CELLS if j != c]
    htasks = [(lvl, c, j, d) for d in ('discharge', 'charge') for c in CELLS
              for (lvl, j) in [('outer', c)] + [('inner', j) for j in CELLS if j != c]]
    # shipped five-cell pools must exist before workers start (reproduce.sh prepare already did this)
    with mp.Pool(int(os.environ.get('NPROC', '16'))) as pool:
        res = dict(pool.starmap(method_rows, tasks))
        res.update(dict(pool.starmap(hppc_rows, htasks)))
    pickle.dump(res, open(os.path.join(OUT, 'method_rows.pkl'), 'wb'))
    print('rows cached:', len(res), sum(len(v) for v in res.values()))
