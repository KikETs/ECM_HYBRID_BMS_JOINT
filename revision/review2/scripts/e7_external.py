"""E7 (R2-9, R1-2): why the frozen RC-SOP over-predicts CHARGE power on the two external cells.

Inputs: frozen-model outputs already computed in review1 (RPCWBY Test#1 / Test#2, six LOCO folds, constant-power SOP
search with Test#2 limits 2.55-4.15 V, 30 A / 15 A) and the external pulse table rpcwby_resistance.csv
(pre-pulse voltage and measured R at 10 s for the same cycles/SOC).
1. A0 vs A8: is the over-prediction there without the residual trim?
2. where: ratio by SOC, temperature, model limit type
3. attribution at 25 C with the voltage-limited power approximation P ~ V_max (V_max - V0) / R:
     log(P_pred/P_ext) = log[(V_max - V0_int)/(V_max - V0_ext)] - log[R_int/R_ext]
   V0_int = internal OCV - hysteresis half-width at (SOC, SOH); V0_ext = measured pre-pulse voltage;
   R_int = internal R_eff(10 s) with the fold's k_f, k_s; R_ext = measured R(10 s). Each term's share of the mean.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, csv
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import REPO, CELLS, write_csv, out_dir
sys.path.insert(0, os.path.join(REPO, 'analysis'))
from ecm_surface import ECMSurface        # noqa: E402
import eval_sop_amps as EV                # noqa: E402

OUT = out_dir('E7')
B1 = REV1
V_MAX, I_CHG = 4.15, 15.0
files = {('Test1', 'RPC_CC'): {c: os.path.join(B1, 'followup', 'external', f'Test1_{c}.csv') for c in CELLS},
         ('Test2', 'RPC_US06'): {c: os.path.join(B1, 'external', f'{c}.csv') for c in CELLS}}
ext = defaultdict(list)
for r in csv.DictReader(open(os.path.join(REPO, 'analysis', 'rpcwby_resistance.csv'), encoding='utf-8')):
    if r['direction'] == 'charge' and abs(float(r['tau_s']) - 10) < .01:
        ext[r['cell'], int(r['cycle'])].append(r)

rows = []
surfs = {}
for (test, xcell), fmap in files.items():
    for fold, path in fmap.items():
        if not os.path.exists(path):
            continue
        if fold not in surfs:
            surfs[fold] = ECMSurface(fold, 'charge')
        S = surfs[fold]
        for r in csv.DictReader(open(path, encoding='utf-8')):
            if r['direction'] != 'charge' or not r['SOP_A8_W'] or not r['SOP_A0_W']:
                continue
            soc, soh, T = float(r['SOC']), float(r['SOH']), int(r['temp_C'])
            meas, p8, p0 = float(r['SOP_meas_W']), float(r['SOP_A8_W']), float(r['SOP_A0_W'])
            kf, ks = float(r['k_f']), float(r['k_s'])
            rec = dict(test=test, fold=fold, cycle=int(r['cycle']), soc=soc, soh=soh, temp_C=T, P_meas=meas, P_A0=p0, P_A8=p8,
                       ratio_A0=p0 / meas, ratio_A8=p8 / meas, limit_hit=r['limit_hit'], kf=kf, ks=ks)
            # matched external pulse at the same cycle, nearest SOC, same temperature band
            def fnum(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return np.nan
            cand = [e for e in ext[xcell, int(r['cycle'])] if abs(fnum(e['SOC']) - soc) <= 0.03
                    and (abs(fnum(e['T_C']) - T) <= 6 or (not np.isfinite(fnum(e['T_C'])) and T == 25))
                    and np.isfinite(fnum(e['R_mOhm'])) and np.isfinite(fnum(e['V_pre_V']))]
            if cand:
                e = min(cand, key=lambda e: (abs(float(e['SOC']) - soc), -abs(float(e['I_A']))))
                I = abs(float(e['I_A']))
                v0e = float(e['V_pre_V']); Re = float(e['R_mOhm']) / 1000
                ocv, _ = S.ocv(soc, soh); m, _ = S.hyst_M(soc, soh)
                v0i = float(np.atleast_1d(ocv)[0]) - float(np.atleast_1d(m)[0])
                Ri = EV.r_eff(S, soc, soh, I, 10.0, kf, ks)
                Ri0 = EV.r_eff(S, soc, soh, I, 10.0, 1.0, 1.0)
                Pext = min(V_MAX * (V_MAX - v0e) / Re, I_CHG * V_MAX) if Re > 0 else np.nan
                rec.update(ext_soc=float(e['SOC']), ext_I=I, ext_rest_s=float(e['rest_before_s']), V0_ext=v0e, V0_int=v0i,
                           dV0_mV=1000 * (v0i - v0e), R_ext_mOhm=1000 * Re, R_int_mOhm=1000 * Ri if np.isfinite(Ri) else np.nan,
                           R_int_A0_mOhm=1000 * Ri0 if np.isfinite(Ri0) else np.nan, P_ext_analytic=Pext,
                           ratio_ext_analytic=Pext / meas,
                           term_V0=np.log((V_MAX - v0i) / (V_MAX - v0e)) if v0e < V_MAX and v0i < V_MAX else np.nan,
                           term_R=-np.log(Ri / Re) if np.isfinite(Ri) and Ri > 0 else np.nan)
            rows.append(rec)
write_csv(os.path.join(OUT, 'E7_charge_rows.csv'), rows)

import pandas as pd
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
summ = D.groupby(['test', 'temp_C']).agg(n=('P_meas', 'size'), ratio_A0_med=('ratio_A0', 'median'), ratio_A8_med=('ratio_A8', 'median'),
                                         frac_A0_over=('ratio_A0', lambda s: (s > 1).mean()), frac_A8_over=('ratio_A8', lambda s: (s > 1).mean()),
                                         lam_needed_A0=('ratio_A0', lambda s: 1 / s.max()), lam_needed_A8=('ratio_A8', lambda s: 1 / s.max()))
print(summ.round(3).to_string()); summ.to_csv(os.path.join(OUT, 'E7_A0_vs_A8.csv'))
D['soc_bin'] = pd.cut(D.soc, [0, .3, .5, .7, .8, .9, 1.01], right=False).astype(str)
bys = D.groupby(['test', 'soc_bin']).agg(n=('P_meas', 'size'), ratio_A8_med=('ratio_A8', 'median'), ratio_A8_max=('ratio_A8', 'max'),
                                          P_meas_med=('P_meas', 'median'))
print(bys.round(3).to_string()); bys.to_csv(os.path.join(OUT, 'E7_by_soc.csv'))
print(D.groupby(['test', 'limit_hit']).ratio_A8.agg(['size', 'median', 'max']).round(3).to_string())
M = D.dropna(subset=['term_V0', 'term_R'])
M = M[M.temp_C == 25]
att = M.groupby('test').agg(n=('term_V0', 'size'), log_ratio_A8_mean=('ratio_A8', lambda s: np.log(s).mean()),
                            log_ratio_extanalytic_mean=('ratio_ext_analytic', lambda s: np.log(s).mean()),
                            term_V0_mean=('term_V0', 'mean'), term_R_mean=('term_R', 'mean'),
                            dV0_mV_med=('dV0_mV', 'median'), dV0_mV_p95=('dV0_mV', lambda s: s.quantile(.95)),
                            R_int_over_ext_med=('R_int_mOhm', lambda s: (s / M.loc[s.index, 'R_ext_mOhm']).median()),
                            ext_rest_s_med=('ext_rest_s', 'median'))
print(att.round(3).to_string()); att.to_csv(os.path.join(OUT, 'E7_attribution_25C.csv'))
