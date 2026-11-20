import os, pandas as pd, numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2
O = _os.path.join(R2, 'out', 'E5b') + '/'
pd.set_option('display.width', 250)
S = pd.read_csv(O + 'E5b_summary.csv')
print(S[(S.direction == 'charge') & (S.tau_s == 2.0) & (S.band_mV == 10.0)].round(3).to_string(index=False))
D = pd.read_csv(O + 'E5b_rows.csv')
g = D[(D.method == 'a8') & (D.tol == 0.0) & (D.dist_mV <= 10)].drop_duplicates(['cell', 'cycle', 'group', 'direction', 'tau_s'])
g = g.assign(ref_minus_Ib=g.I_ref - g.I_b, corr=g.I_b - g.I_obs)
t = g.groupby(['direction', 'tau_s']).agg(n=('cell', 'size'), cells=('cell', 'nunique'), ref_minus_Ib_median=('ref_minus_Ib', 'median'),
                                           ref_minus_Ib_max=('ref_minus_Ib', 'max'), ref_minus_Ib_min=('ref_minus_Ib', 'min'),
                                           corr_abs_max=('corr', lambda s: s.abs().max()), soc_min=('soc', 'min'), soc_max=('soc', 'max'),
                                           extrap_median=('extrap', 'median'))
print(t.round(3).to_string())
t.to_csv(O + 'E5b_reference_vs_measured.csv')
# paired A8 vs HPPC on the same pulses (ratio difference), per direction/tau, tol 0
P = D[(D.dist_mV <= 10) & (D.tol == 0.0) & D.method.isin(['a8', 'hppc'])].pivot_table(index=['direction', 'tau_s', 'cell', 'cycle', 'group'], columns='method', values='allowed')
P = P.dropna().reset_index().merge(g[['direction', 'tau_s', 'cell', 'cycle', 'group', 'I_b']], on=['direction', 'tau_s', 'cell', 'cycle', 'group'])
P['diff_pp'] = 100 * (P.hppc - P.a8) / P.I_b
print(P.groupby(['direction', 'tau_s']).diff_pp.agg(['size', 'median', 'min', 'max', lambda s: (s > 0).sum()]).round(2).to_string())
