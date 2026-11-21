"""E13 summary (R1-3, R2-8): estimated-state replay on every eligible HPPC cycle, strict-fold factors, no refitting.
Reads out/E13/estimated_state_preceding_drive.csv and fixed_lambda_summary.csv written by e13_replay_drive.py."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import read_csv, write_csv, out_dir   # noqa: E402

OUT = out_dir('E13')
rows = read_csv(os.path.join(OUT, 'estimated_state_preceding_drive.csv'))
fl = read_csv(os.path.join(OUT, 'fixed_lambda_summary.csv'))
summ = []
for d in ('discharge', 'charge'):
    for t in ('10.0', '2.0'):
        sub = [r for r in rows if r['direction'] == d and r['tau_s'] == t]
        st = {k: sum(r['paired_status'] == k for r in sub) for k in
              ('ok', 'solver_exclusion', 'initial_rest_or_history_inadequate', 'no_timestamp_confirmed_history')}
        ok = [r for r in sub if r['paired_status'] == 'ok']
        soc_err = np.array([abs(float(r['SOC_est']) - float(r['SOC_ref'])) for r in ok]) if ok else np.array([np.nan])
        rec = dict(direction=d, tau_s=float(t), rows=len(sub), cells=len({r['cell'] for r in sub}),
                   cycles=len({(r['cell'], r['cycle']) for r in sub}), scored=st['ok'], solver_exclusion=st['solver_exclusion'],
                   history_inadequate=st['initial_rest_or_history_inadequate'], no_confirmed_history=st['no_timestamp_confirmed_history'],
                   soc_abs_err_median=float(np.median(soc_err)), soc_abs_err_p95=float(np.quantile(soc_err, .95)))
        for m in ('hyb', 'ecm'):
            f = [r for r in fl if r['direction'] == d and r['tau_s'] == t and r['model'] == m]
            n = np.array([int(r['n']) for r in f]); u = np.array([float(r['usable_pct']) for r in f])
            rec[f'{m}_rows'] = int(n.sum()); rec[f'{m}_cells'] = len(f)
            rec[f'{m}_exceed'] = int(sum(int(r['n_exceed']) for r in f))
            rec[f'{m}_usable_pct'] = float(np.average(u, weights=n)) if len(f) else float('nan')
        summ.append(rec)
write_csv(os.path.join(OUT, 'E13_summary.csv'), summ)
for s in summ:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
cyc = {}
for r in rows:
    cyc.setdefault(r['cell'], set()).add(r['cycle'])
print('cycles per cell', {c: len(v) for c, v in cyc.items()})
