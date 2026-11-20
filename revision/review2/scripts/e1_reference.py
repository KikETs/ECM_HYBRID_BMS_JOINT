"""E1 (R1-1, R2-1, minor 6): uncertainty of the HPPC-fitted reference current and its effect on the reported results.

1. extrapolation-ratio distribution of the scored rows
2. alternative reference fits (all-rank linear = published, top-3 linear, top-2 linear, all-rank quadratic) and
   eligibility caps (extrap <= 1.25 / 1.5 / 2.0): lambda recalibrated, A0 and A8 rescored
3. bracketed-only subset (reference voltage bracketed by measured pulses at the horizon)
4. lambda held fixed while the 'true' boundary is lowered by b A or by p %: exceedance counts
"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import *

OUT = out_dir('E1')
arms = load_arms()
A = os.path.join(REPO, 'analysis')

# ---- reference alternatives from the HPPC rank points
groups = defaultdict(dict)
for r in read_csv(os.path.join(A, 'uypydj_hppc_resistance.csv')):
    groups[r['direction'], r['protocol'], r['cycle'], r['soc_group'], float(r['tau_s'])][int(r['rate_rank'])] = r


def lin(I, V, lim):
    b, a = np.polyfit(I, V, 1)
    return (lim - a) / b if b > 1e-6 else np.nan


def quad(I, V, lim, sign):
    c2, c1, c0 = np.polyfit(I, V, 2)
    roots = np.roots([c2, c1, c0 - lim])
    roots = [x.real for x in roots if abs(x.imag) < 1e-9 and x.real * sign > 0]
    return min(roots, key=abs) if roots else np.nan


ref = {}           # (direction, key) -> dict of alternative references (abs A) + bracket flag
for direction, fname in (('discharge', 'sop_label_measured.csv'), ('charge', 'sop_label_charge.csv')):
    sign = -1 if direction == 'discharge' else 1
    for r in read_csv(os.path.join(A, fname)):
        tau = float(r['tau_s'])
        key = f"{r['cell']}|{r['cycle']}|{tau:.1f}|{r['V_pre_V']}|{round(float(r['I_star_lin4_A']), 3)}"
        g = sorted(groups[direction, r['cell'], r['cycle'], r['soc_group'], tau].values(), key=lambda q: -abs(float(q['I_A'])))
        I = np.array([float(z['I_A']) for z in g]); V = np.array([float(z['V_tau_V']) for z in g])
        t = np.array([float(z['tau_actual_s']) for z in g]); lim = float(r['V_min_V'])
        trusted = abs(t - tau) <= 0.15
        alt = {'lin_all': lin(I, V, lim),
               'lin_top3': lin(I[:3], V[:3], lim) if len(I) >= 3 else np.nan,
               'lin_top2': lin(I[:2], V[:2], lim) if len(I) >= 2 else np.nan,
               'quad_all': quad(I, V, lim, sign) if len(I) >= 3 else np.nan}
        alt = {k: (abs(v) if np.isfinite(v) and v * sign > 0 else np.nan) for k, v in alt.items()}
        alt['bracketed'] = bool(trusted.any() and V[trusted].min() <= lim <= V[trusted].max())
        alt['n_ranks'] = len(I)
        ref[direction, key] = alt

rows_out, ext_rows = [], []
# ---- 1. extrapolation ratio of scored rows (A8, published reference)
for d in DIRECTIONS:
    for tau in TAUS:
        recs = score_condition(arms, d, tau)
        ex = np.array([r['extrap'] for rec in recs for r in rec['rows']])
        br = np.array([ref.get((d, r['key']), {}).get('bracketed', False) for rec in recs for r in rec['rows']])
        q = np.quantile(ex, [0, .05, .25, .5, .75, .95, 1])
        ext_rows.append(dict(direction=d, tau_s=tau, n=len(ex), frac_extrap_le_1=float((ex <= 1).mean()),
                             frac_bracketed=float(br.mean()), **{f'q{int(p*100)}': v for p, v in zip([0, .05, .25, .5, .75, .95, 1], q)}))
        for rec in recs:
            for r in rec['rows']:
                rows_out.append(dict(direction=d, tau_s=tau, cell=rec['cell'], key=r['key'], extrap=r['extrap'],
                                     bracketed=int(ref.get((d, r['key']), {}).get('bracketed', False))))
write_csv(os.path.join(OUT, 'E1_extrap_distribution.csv'), ext_rows)
write_csv(os.path.join(OUT, 'E1_scored_rows_extrap.csv'), rows_out)

# ---- 2/3. alternative references and caps
sens = []
for d in DIRECTIONS:
    tols = [TOL[d]] + ([0.0] if d == 'charge' else [])
    for tau in TAUS:
        for tol in tols:
            for model in ('ecm', 'hyb'):
                for refname in ('lin_all', 'lin_top3', 'lin_top2', 'quad_all'):
                    for cap in (1.25, 1.5, 2.0):
                        yf = (lambda r, rn=refname, dd=d: ref.get((dd, r['key']), {}).get(rn))
                        s = summarize(score_condition(arms, d, tau, model=model, tol=tol, yfun=yf, extrap_max=cap))
                        if s:
                            sens.append(dict(direction=d, tau_s=tau, tol=tol, model=model, reference=refname, extrap_cap=cap, subset='all', **s))
                yf = (lambda r, dd=d: r['meas'])
                flt = (lambda r, dd=d: ref.get((dd, r['key']), {}).get('bracketed', False))
                s = summarize(score_condition(arms, d, tau, model=model, tol=tol, yfun=yf, rowfilter=flt))
                if s:
                    sens.append(dict(direction=d, tau_s=tau, tol=tol, model=model, reference='lin_all', extrap_cap=1.5, subset='bracketed_only', **s))
write_csv(os.path.join(OUT, 'E1_reference_sensitivity.csv'), sens)

# ---- 4. fixed lambda, lowered true boundary
shift = []
for d in DIRECTIONS:
    for tau in TAUS:
        for model in ('ecm', 'hyb'):
            base = score_condition(arms, d, tau, model=model)
            lam = {r['cell']: r['lam'] for r in base}
            for kind, vals in (('abs_A', (0.5, 1.0, 1.97, 3.0, 5.0)), ('rel_pct', (2, 5, 10))):
                for v in vals:
                    yf = (lambda r, v=v, k=kind: r['meas'] - v if k == 'abs_A' else r['meas'] * (1 - v / 100))
                    recs = score_condition(arms, d, tau, model=model, yfun=yf, lam_override=lam)
                    s = summarize(recs)
                    shift.append(dict(direction=d, tau_s=tau, model=model, shift_kind=kind, shift=v,
                                      n_rows=s['n_rows'], exceed_gt_tol=s['exceed'], exceed_gt0=s['exceed_gt0'],
                                      exceed_pct=100 * s['exceed_gt0'] / s['n_rows'], max_pos_A=s['max_pos_A']))
write_csv(os.path.join(OUT, 'E1_fixed_lambda_lowered_boundary.csv'), shift)
print('E1 done:', len(sens), 'sensitivity rows,', len(shift), 'shift rows')
