"""E2 (R2-5, R1-6): anatomy of discharge rows with no accepted root, and fallback policies.

Inputs (read-only): nested/results/solver_diagnostics.csv (primary arm, outer rows, A0 and A8),
baseline/arms/arm_rows_replace_level2_hull2.pkl (SOC/SOH per row), e2e_nested_percell_*_hyb.csv (lambda_c).
Outputs: $OUT/E2_*.csv
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK as R2, REPO as REPO_ROOT, R1 as REV1
import os, sys, pickle, re
import numpy as np, pandas as pd

OUT = os.path.join(R2, 'out', 'E2'); os.makedirs(OUT, exist_ok=True)
diag = pd.read_csv(os.path.join(REPO_ROOT, 'nested/results/solver_diagnostics.csv'))
arms = pickle.load(open(os.path.join(R2, 'baseline/arms/arm_rows_replace_level2_hull2.pkl'), 'rb'))
pc = pd.read_csv(os.path.join(R2, 'baseline/arms/e2e_nested_percell_replace_level2_hull2_hyb.csv'))
print('percell columns', pc.columns.tolist()); print(pc.head(3).to_string())

rows = pd.DataFrame([r for k, v in arms.items() if k[0] == 'outer' and k[4] == 'eo' for r in v])
rows['extrap_r'] = rows.extrap.astype(float).round(3)
diag['extrap_r'] = diag.extrap.astype(float).round(3)
rows['iref_r'] = rows.meas.astype(float).abs().round(2)
diag['iref_r'] = diag.I_ref.astype(float).abs().round(2)
key = ['cell', 'cycle', 'tau', 'extrap_r', 'iref_r', 'direction']
diag = diag.rename(columns={'tau_s': 'tau'})
dup = rows.duplicated(key).sum(); print('duplicate join keys in arm rows:', dup)
rows = rows.drop_duplicates(key)

def valid_edge(seg):
    """upper |I| of the contiguous valid domain starting at 0.1 A, and whether any valid segment lies above an outside one."""
    parts = [p.split(':') for p in seg.split(';')]
    edge, broken, gap_valid = 0.1, False, False
    for rng, lab in parts:
        lo, hi = map(float, rng.split('..'))
        if lab == 'valid' and not broken:
            edge = hi
        elif lab != 'valid':
            broken = True
        elif broken and lab == 'valid':
            gap_valid = True
    return edge, gap_valid

out = []
for (direction, tau), g in diag[diag.eligible_by_label == 1].groupby(['direction', 'tau']):
    a0 = g[g.model == 'A0'].drop_duplicates(['cell', 'cycle', 'extrap_r', 'iref_r']).set_index(['cell', 'cycle', 'extrap_r', 'iref_r'])
    a8 = g[g.model == 'A8'].drop_duplicates(['cell', 'cycle', 'extrap_r', 'iref_r']).set_index(['cell', 'cycle', 'extrap_r', 'iref_r'])
    print(direction, tau, 'A8 rows', (g.model == 'A8').sum(), 'unique', len(a8))
    j = a8[['v2_status', 'segments', 'v2_root', 'I_ref']].join(a0[['v2_status', 'v2_root']], lsuffix='_a8', rsuffix='_a0').reset_index()
    j['direction'], j['tau'] = direction, tau
    out.append(j)
J = pd.concat(out, ignore_index=True)
rows['iref_est'] = rows.I_meas.astype(float).abs() * rows.extrap.astype(float)
k5 = ['cell', 'cycle', 'tau', 'extrap_r', 'direction']
J['_id'] = np.arange(len(J))
M = J[k5 + ['_id', 'I_ref']].merge(rows[k5 + ['iref_est', 'soc_ref', 'soh_ref', 'soh_in']], on=k5, how='left')
M['_dist'] = (M.iref_est - M.I_ref.abs()).abs()
ncand = M.groupby('_id').size()
M = M.sort_values('_dist').drop_duplicates('_id')
amb = M.set_index('_id').loc[ncand[ncand > 1].index]
print('ids with >1 candidate:', len(amb), ' their SOC spread unresolved if dist large; dist p50/p95/max:',
      amb._dist.quantile([.5, .95, 1]).round(3).tolist())
J = J.merge(M[['_id', 'soc_ref', 'soh_ref', 'soh_in']], on='_id', how='left')
print('rows without SOC after join:', J.soc_ref.isna().sum(), 'of', len(J))

def klass(r):
    a8, a0 = r.v2_status_a8, r.v2_status_a0
    if a8 in ('ok', 'multi') and a0 in ('ok', 'multi'):
        return 'scored'
    if a8 == 'hull' or a0 == 'hull':
        return 'no valid segment (outside hull at all currents)'
    if a8 == 'noroot' and a0 == 'noroot':
        return 'both models: crossing only beyond hull edge'
    if a8 in ('ok', 'multi') and a0 == 'noroot':
        return 'A8 root, A0 crossing beyond hull edge (paired exclusion)'
    if a8 == 'tangent':
        return 'A8 tangent (flagged failure)'
    return f'other a8={a8} a0={a0}'
J['class'] = J.apply(klass, axis=1)
e = J.segments.apply(valid_edge)
J['I_edge_A'] = [x[0] for x in e]; J['valid_above_gap'] = [x[1] for x in e]
J['I_ref_abs'] = J.I_ref.abs()
J.to_csv(os.path.join(OUT, 'E2_eligible_rows.csv'), index=False)

t1 = J.groupby(['direction', 'tau', 'class']).size().unstack(fill_value=0)
t1['eligible'] = t1.sum(axis=1)
t1['scored_pct'] = (100 * t1.get('scored', 0) / t1.eligible).round(1)
t1.to_csv(os.path.join(OUT, 'E2_class_counts.csv')); print(t1.to_string())

D = J[J.direction == 'discharge'].copy()
D['soc_bin'] = pd.cut(D.soc_ref, [0, .15, .3, .5, .7, .9, 1.01], right=False)
D['soh_bin'] = pd.cut(D.soh_ref, [0, .8, .85, .9, .95, 1.01], right=False)
D['fail'] = D['class'] != 'scored'
for by in ('soc_bin', 'soh_bin', 'cell'):
    t = D.groupby(['tau', by], observed=True).agg(eligible=('fail', 'size'), not_scored=('fail', 'sum'))
    t['not_scored_pct'] = (100 * t.not_scored / t.eligible).round(1)
    t.to_csv(os.path.join(OUT, f'E2_by_{by}.csv')); print(t.to_string())

# I_edge vs I_ref on beyond-hull rows: where does the hull end relative to the reference?
B = D[D['class'] == 'both models: crossing only beyond hull edge']
s = B.groupby('tau').apply(lambda g: pd.Series({'n': len(g), 'edge_lt_ref': int((g.I_edge_A < g.I_ref_abs).sum()),
                                               'edge_over_ref_median': float((g.I_edge_A / g.I_ref_abs).median()),
                                               'edge_over_ref_p05': float((g.I_edge_A / g.I_ref_abs).quantile(.05)),
                                               'edge_over_ref_p95': float((g.I_edge_A / g.I_ref_abs).quantile(.95))}))
s.to_csv(os.path.join(OUT, 'E2_edge_vs_ref.csv')); print(s.to_string())

# Fallback policies on all eligible discharge rows (primary arm = estimated SOH + oracle SOC, A8, nested lambda_c)
lam = pc[(pc.condition == 'estimated SOH + oracle SOC') & (pc.protocol == 'nested')][['direction', 'tau_s', 'cell', 'lambda', 'n', 'exceed', 'usable_pct']]
lam = lam.rename(columns={'tau_s': 'tau'})
print(lam[lam.direction == 'discharge'].to_string())
D = D.merge(lam[['direction', 'tau', 'cell', 'lambda']], on=['direction', 'tau', 'cell'], how='left')
# cells with no scored outer row have no reported lambda_c: use the condition median and flag it
D['lambda_imputed'] = D['lambda'].isna()
D['lambda'] = D.groupby('tau')['lambda'].transform(lambda s: s.fillna(s.dropna().median()))
print('rows with imputed lambda:', D.groupby('tau').lambda_imputed.sum().to_dict())
D['I_pred_abs'] = D.v2_root_a8.astype(float).abs()
def paper_usable(ratio, cell):
    """manuscript definition: per-cell median of lambda*|I_pred|/|I_ref|, averaged with row-count weights"""
    t = pd.DataFrame({'r': ratio, 'c': cell}).groupby('c').r.agg(['median', 'size'])
    return 100 * float((t['median'] * t['size']).sum() / t['size'].sum())

res = []
for tau, g in D.groupby('tau'):
    sc = g['class'] == 'scored'
    base = g['lambda'] * g.I_pred_abs
    ratio_sc = (base[sc] / g.I_ref_abs[sc])
    res.append(dict(tau=tau, policy='scored rows only (submitted)', n=int(sc.sum()), exceed=int((base[sc] > g.I_ref_abs[sc]).sum()),
                    usable_paper_pct=paper_usable(ratio_sc, g.cell[sc]),
                    usable_mean_pct=100 * ratio_sc.mean(), usable_median_pct=100 * ratio_sc.median()))
    for pol in ('zero current (firmware)', 'hull-edge current', 'lambda x hull-edge current'):
        fb = ~sc & g.I_edge_A.gt(0.1)
        if pol == 'zero current (firmware)':
            alt = pd.Series(0.0, index=g.index)
        elif pol == 'hull-edge current':
            alt = g.I_edge_A
        else:
            alt = g['lambda'] * g.I_edge_A
        I_all = base.where(sc, alt.where(fb, 0.0))
        r_all = I_all / g.I_ref_abs
        r_fb = (alt / g.I_ref_abs)[~sc]
        res.append(dict(tau=tau, policy=pol, n=len(g), exceed=int((I_all > g.I_ref_abs).sum()),
                        usable_paper_pct=paper_usable(r_all, g.cell),
                        usable_mean_pct=100 * r_all.mean(), usable_median_pct=100 * r_all.median(),
                        fallback_rows=int((~sc).sum()), fallback_exceed=int(((alt > g.I_ref_abs) & ~sc).sum()),
                        fallback_usable_median_pct=100 * r_fb.median()))
R = pd.DataFrame(res).round(2)
R.to_csv(os.path.join(OUT, 'E2_fallback_policies.csv'), index=False); print(R.to_string())
