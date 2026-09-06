"""solver_diagnostics.csv — one row per (label row, model) on the outer assets:
exact segment enumeration, grid-v1 verdict, segment-aware v2 verdict, the
reference-start fixed point, and the firmware host result where available."""
import csv
import os

import numpy as np

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = list(csv.DictReader(open(f'{V}/solver/exact_roots_outer.csv')))
V2 = {(r['direction'], r['cell'], r['cycle'], r['tau'], round(float(r['meas']), 3)): r
      for r in csv.DictReader(open(f'{V}/solver/validate_v2.csv'))}
FW = {}
for r in csv.DictReader(open(f'{V}/firmware/fw_root_diagnostics.csv')):
    FW[(r['direction'], r['cell'], r['cycle'], r['tau'], r['model'])] = r
try:
    import pickle
    lost = pickle.load(open(os.environ['RCSOP_LOST19'], 'rb'))   # optional: the 19 rows of the earlier check
    L19 = {(r['cell'], int(r['cycle']), float(r['tau']), round(r['meas'], 3)) for r in lost}
except Exception:
    L19 = set()
out = []
for r in EX:
    k = (r['direction'], r['cell'], r['cycle'], r['tau'], round(float(r['meas']), 3))
    v2 = V2.get(k, {})
    for key, model in (('A0', 'A0'), ('A3', 'A8')):
        fw = FW.get((r['direction'], r['cell'], r['cycle'], r['tau'], model), {})
        ne = int(r[f'{key}_n_roots_exact']); fp = r[f'{key}_fp_meas_start']
        out.append(dict(
            direction=r['direction'], cell=r['cell'], cycle=r['cycle'], tau_s=r['tau'], extrap=r['extrap'], I_ref=r['meas'], model=model,
            eligible_by_label=int(float(r['extrap']) <= 1.5 and abs(float(r['meas'])) > 0.5),
            in_previous_19=int((r['cell'], int(r['cycle']), float(r['tau']), round(float(r['meas']), 3)) in L19),
            segments=r[f'{key}_segments'], exact_n_roots=ne, exact_roots=r[f'{key}_roots_exact'],
            exact_min_root=r[f'{key}_exact_min_root'], tangent_candidates=r[f'{key}_tangent_candidates'], exact_notes=r[f'{key}_notes'],
            grid_v1_n_roots=r[f'{key}_n_roots_grid'], grid_v1_status=r[f'{key}_grid_status'], grid_v1_root=r[f'{key}_grid_root'],
            grid_v1_vs_exact=r[f'{key}_grid_vs_exact'],
            v2_n_roots=v2.get(f'{key}_v2_n', ''), v2_status=v2.get(f'{key}_v2_status', ''), v2_root=v2.get(f'{key}_v2_root', ''),
            v2_vs_exact=v2.get(f'{key}_v2_vs_exact', ''),
            fp_reference_start=fp, fp_nearest_root_dist=r[f'{key}_fp_nearest_root_dist'], fp_is_min_root=r[f'{key}_fp_is_min_root'],
            fp_classification=('no_root_exact' if (fp not in ('', 'nan') and ne == 0) else
                               ('missed_existing_root' if (fp in ('', 'nan') and ne >= 1) else
                                ('larger_root' if r[f'{key}_fp_is_min_root'] == '0' else
                                 ('ok' if fp not in ('', 'nan') else 'no_root')))),
            fw_n_roots=fw.get('fw_n_roots', ''), fw_sop_solve=fw.get('fw_solve', ''), fw_iters=fw.get('fw_iters', ''),
            fw_capped24=fw.get('fw_capped', ''), fw_solve_nearest_root_dist=fw.get('fw_solve_nearest_root_dist', '')))
with open(f'{V}/solver_diagnostics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
import collections
print(len(out), 'rows -> solver_diagnostics.csv')
for model in ('A0', 'A8'):
    for d in ('discharge', 'charge'):
        g = [r for r in out if r['model'] == model and r['direction'] == d and r['eligible_by_label']]
        print(model, d, 'eligible', len(g), 'exact n_roots', dict(collections.Counter(r['exact_n_roots'] for r in g)),
              'grid_v1 DIFFER', sum(1 for r in g if r['grid_v1_vs_exact'] == 'DIFFER'), 'v2 DIFFER', sum(1 for r in g if r['v2_vs_exact'] == 'DIFFER'),
              'fp classes', dict(collections.Counter(r['fp_classification'] for r in g)), 'tangent cand', sum(1 for r in g if r['tangent_candidates']))
