"""E5 step 3: cost table inputs. Parameter counts from the model definitions, HPPC table size from the data."""
import os, sys, csv, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import REPO, write_csv, out_dir
A = os.path.join(REPO, 'analysis')
hp = list(csv.DictReader(open(os.path.join(A, 'uypydj_hppc_resistance.csv'), encoding='utf-8')))
pts = {}
for r in hp:
    pts.setdefault((r['direction'], r['tau_s']), set()).add((r['protocol'], r['cycle'], r['soc_group']))
soc_groups = sorted({r['soc_group'] for r in hp}, key=int)
n_in, hid = 12, 32
lstm = 4 * hid * (n_in + hid) + 8 * hid + (hid * 2 + 2)
gru = 3 * hid * (n_in + hid) + 6 * hid + (hid * 2 + 2)
rows = [
    dict(method='A8 (RC-SOP)', params='4 effective (2 weights + 2 biases on dR_fast); 26 stored', state='2 EW sums (e_ir, e_ii)',
         per_call='2 MAC + 2 exp/tanh + hull-aware root search on 2RC surface', note='board-measured: inversion 53.21 us median, text 70,796 B, ECM table 32,768 B, OCV 4,096 B'),
    dict(method='A0 (nominal 2RC)', params='0', state='-', per_call='root search on 2RC surface', note='same surface tables as A8'),
    dict(method='A3 (linear, 12 features)', params='26', state='12 feature EW states', per_call='24 MAC + root search', note='not built for board'),
    dict(method='LSTM (hid 32, 12 blocks)', params=str(lstm), state='12x12 feature buffer or 64 hidden floats', per_call=f'{12 * 4 * hid * (n_in + hid):,} MAC + root search', note='estimate, not built for board'),
    dict(method='GRU (hid 32, 12 blocks)', params=str(gru), state='12x12 feature buffer or 32 hidden floats', per_call=f'{12 * 3 * hid * (n_in + hid):,} MAC + root search', note='estimate, not built for board'),
    dict(method='FFRLS (block level)', params='1 forgetting factor', state='2 recursive means + gain', per_call='~10 flops per 600 s block + root search', note='estimate'),
    dict(method='HPPC resistance lookup', params=f'table: {len(soc_groups)} SOC groups x SOH grid x 2 directions x 2 horizons',
         state='-', per_call='2-D interpolation + 1 division, no root search',
         note='training points per (direction, tau): ' + '; '.join(f'{k[0]} {k[1]} s: {len(v)}' for k, v in sorted(pts.items()))),
]
write_csv(os.path.join(out_dir('E5'), 'E5_cost.csv'), rows)
print(rows)
