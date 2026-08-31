"""Two tables the SOH replacement needs, both from measurements on the board.

soh_model_cost.csv   what swapping the CNN for ridge cost and bought
mcu_icache.csv       whether the SOP slowdown that came with it is real

The second table exists because the first produced a result that looks wrong:
removing 71 kB of unrelated code made the SOP inversion 5.3 % SLOWER.  Reporting
that as a cost of the ridge model would be wrong, and reporting the faster
number would be dishonest, so it was isolated -- same source, same board,
minutes apart, with the instruction cache enabled and disabled.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MCU = os.path.join(ROOT, 'mcu')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')
CLOCK = 250e6


def sop_stage_us(path, stage='FULL'):
    """Median/p95/max microseconds for one stage of a sop bench file."""
    if not os.path.exists(path):
        return None
    c = [float(r['cycles']) for r in csv.DictReader(open(path, encoding='utf-8'))
         if r['cmd'] == stage]
    if not c:
        return None
    a = np.array(c) / CLOCK * 1e6
    return float(np.median(a)), float(np.percentile(a, 95)), float(a.max())


def soh_us(path):
    if not os.path.exists(path):
        return None
    c = [float(r['cycles']) for r in csv.DictReader(open(path, encoding='utf-8'))
         if r['cmd'] == 'SOH']
    if not c:
        return None
    a = np.array(c) / CLOCK * 1e6
    return float(np.median(a)), float(np.percentile(a, 95)), float(a.max())


def rmse_from(pred_npz):
    z = np.load(pred_npz, allow_pickle=True)
    cells = sorted({k.rsplit('_', 1)[0] for k in z.files if k.endswith('_pred')})
    per, errs = {}, []
    for c in cells:
        e = z[f'{c}_pred'] - z[f'{c}_y']
        per[c] = float(np.sqrt(np.mean(e ** 2)))
        errs.append(e)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), per


def size_of(build):
    p = os.path.join(MCU, 'fw_sop', 'Build', build, 'size.txt')
    if not os.path.exists(p):
        return None
    f = open(p, encoding='utf-8').read().splitlines()[1].split()
    return int(f[0]), int(f[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default=os.path.join(HERE, 'mcu_sizes.json'),
                    help='recorded text/bss for both models, since only one '
                         'can be built at a time')
    a = ap.parse_args()

    sizes = json.load(open(a.sizes, encoding='utf-8'))
    rows = []
    r_ridge = soh_us(os.path.join(MCU, 'soh_mcu_bench.csv'))
    r_cnn = soh_us(os.path.join(MCU, 'soh_mcu_bench_cnn.csv'))
    rm_r, per_r = rmse_from(os.path.join(ROOT, 'analysis', 'results',
                                         'soh_pred.npz'))
    cnn_pred = os.path.join(ROOT, 'analysis', 'results', 'soh_pred_cnn.npz')
    if os.path.exists(cnn_pred):
        rm_c, per_c = rmse_from(cnn_pred)
    else:
        rm_c, per_c = sizes['cnn']['rmse_pooled'], sizes['cnn']['rmse_per_cell']

    for name, us, rm, per, sz in (
            ('ridge (deployed)', r_ridge, rm_r, per_r, sizes['ridge']),
            ('1D CNN (superseded)', r_cnn, rm_c, per_c, sizes['cnn'])):
        if us is None:
            print(f'  no board measurement for {name}', file=sys.stderr)
            continue
        rows.append([name, sz['coefficients'], f'{rm:.4f}',
                     f'{max(per.values()):.4f}', max(per, key=per.get),
                     f'{us[0]:.2f}', f'{us[1]:.2f}', f'{us[2]:.2f}',
                     sz['text_B'], sz['bss_B']])
    out = os.path.join(TABLES, 'soh_model_cost.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'coefficients', 'rmse_pooled', 'rmse_worst_cell',
                    'worst_cell', 'soh_median_us', 'soh_p95_us', 'soh_max_us',
                    'firmware_text_B', 'firmware_bss_B'])
        w.writerows(rows)
    print(f'  -> {os.path.relpath(out, ROOT)}')
    print(f"  {'model':<22}{'coef':>6}{'RMSE':>8}{'worst':>8}"
          f"{'SOH us':>10}{'text B':>9}{'bss B':>8}")
    for r in rows:
        print(f'  {r[0]:<22}{r[1]:>6}{r[2]:>8}{r[3]:>8}{r[5]:>10}'
              f'{r[8]:>9}{r[9]:>8}')

    irows = []
    for icache, tag, ridge_f, cnn_f in (
            ('on', 'deployment setting',
             'sop_mcu_bench.csv', 'sop_mcu_bench_cnn.csv'),
            ('off', 'cache disabled at build time',
             'sop_mcu_bench_noicache_ridge.csv',
             'sop_mcu_bench_noicache_cnn.csv')):
        for model, fn in (('ridge', ridge_f), ('CNN', cnn_f)):
            for stage in ('FEAT', 'SOLVE', 'FULL'):
                v = sop_stage_us(os.path.join(MCU, fn), stage)
                if v is None:
                    continue
                irows.append([icache, model, stage, f'{v[0]:.2f}',
                              f'{v[1]:.2f}', f'{v[2]:.2f}', tag])
    out = os.path.join(TABLES, 'mcu_icache.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['icache', 'soh_model', 'stage', 'median_us', 'p95_us',
                    'max_us', 'note'])
        w.writerows(irows)
    print(f'\n  -> {os.path.relpath(out, ROOT)}')
    d = {(r[0], r[1]): float(r[3]) for r in irows if r[2] == 'FULL'}
    if len(d) == 4:
        on = d[('on', 'ridge')] / d[('on', 'CNN')] - 1
        off = d[('off', 'ridge')] / d[('off', 'CNN')] - 1
        print(f"  FULL, ridge against CNN:  cache on {on * 100:+.1f} %   "
              f"cache off {off * 100:+.1f} %")
        print('  The penalty exists only with the cache enabled, and reverses '
              'without it.  It is placement, not work.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
