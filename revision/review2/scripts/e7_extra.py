import os, numpy as np, pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2
O = _os.path.join(R2, 'out', 'E7') + '/'
pd.set_option('display.width', 250)
D = pd.read_csv(O + 'E7_charge_rows.csv')
M = D.dropna(subset=['P_ext_analytic'])
M = M[M.temp_C == 25].copy()
M['soc_bin'] = pd.cut(M.soc, [0, .5, .7, .8, .9, 1.01], right=False).astype(str)
M['I_sop_meas'] = M.P_meas / 4.15
t = M.groupby(['test', 'soc_bin']).agg(n=('P_meas', 'size'), ratio_A8=('ratio_A8', 'median'), ratio_A0=('ratio_A0', 'median'),
                                        ratio_ext_analytic=('ratio_ext_analytic', 'median'), pulse_I_med=('ext_I', 'median'),
                                        sop_I_meas_med=('I_sop_meas', 'median'), R_int_over_ext=('R_int_mOhm', lambda s: (s / M.loc[s.index, 'R_ext_mOhm']).median()),
                                        dV0_mV=('dV0_mV', 'median'))
print(t.round(3).to_string())
t.to_csv(O + 'E7_by_soc_25C.csv')
