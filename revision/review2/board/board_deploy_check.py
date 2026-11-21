"""Board check of the deployment image (flagged solver in SOP_CMD_SOLVE / SOP_CMD_FULL, -DSOP_A8_ONLY build).
Grid of host_review_test.c with the multiplier grid replaced by three dR_fast inputs (trim on the board).
For every point: board SOLVE (status in r_eff, current in i_star, cycles = solver only) and FULL (trim + solver),
compared with a host build of the same sources. Writes rows and a summary to OUT."""
import csv, struct, subprocess, sys, time
from pathlib import Path
import numpy as np
import serial

OUT = Path(sys.argv[1]); REF = Path(sys.argv[2]); MCU = Path(sys.argv[3])
sys.path.insert(0, str(MCU))
import bench_sop as board

pts = []
for d in (0, 1):
    for soc in np.arange(0.05, 1.0001, 0.05):
        for soh in np.arange(0.72, 1.0001, 0.04):
            for dv in np.arange(0.0, 0.30001, 0.05):
                for tau in (2.0, 10.0):
                    for x0 in (15.0, 30.0, 60.0):
                        vlim = 4.2 if d else 2.5
                        vpre = 3.5 + dv * 2.0 if d else 3.4 + dv * 2.5
                        pts.append((d, float(np.float32(soc)), float(np.float32(soh)), float(np.float32(vpre)), vlim, tau, x0))
inp = '\n'.join(f'{d} {s!r} {h!r} {v!r} {l!r} {t!r} {x!r}' + ' 0' * 11 for d, s, h, v, l, t, x in pts) + '\n'
ref = subprocess.run([str(REF)], input=inp, capture_output=True, text=True, check=True).stdout.split('\n')
ref = [r.split() for r in ref if r.strip()]
assert len(ref) == len(pts)

rows = []
with serial.Serial('/dev/ttyACM0', 921600, timeout=5, exclusive=True) as port:
    port.reset_input_buffer()
    info = board.query(port)
    print('board', info, flush=True)
    assert info['clock_hz'] == 250000000
    for k, ((d, s, h, v, l, t, x0), r) in enumerate(zip(pts, ref)):
        x12 = [x0] + [0.0] * 11
        a = board.call(port, 'SOLVE', d, s, h, v, l, t, 0.0, x12)
        b = board.call(port, 'FULL', d, s, h, v, l, t, 0.0, x12)
        rkf, rks, rst, ra, rit = float(r[0]), float(r[1]), int(r[2]), float(r[3]), int(r[4])
        rows.append(dict(dir=d, soc=s, soh=h, v_pre=v, v_lim=l, tau=t, x0=x0,
                         host_status=rst, host_A=ra, host_iters=rit, host_kf=rkf, host_ks=rks,
                         solve_status=int(round(a['r_eff'])), solve_A=a['i_star'], solve_iters=a['iters'], solve_cycles=a['cycles'],
                         full_status=int(round(b['r_eff'])), full_A=b['i_star'], full_iters=b['iters'], full_cycles=b['cycles'],
                         full_kf=b['kf'], full_ks=b['ks']))
        if k % 2000 == 0:
            print('point', k, flush=True)
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'board_deploy_rows.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

R = rows
ok = [r for r in R if r['host_status'] == 0]
fl = [r for r in R if r['host_status'] != 0]
st_match = sum(r['solve_status'] == r['host_status'] and r['full_status'] == r['host_status'] for r in R)
dA = [max(abs(r['solve_A'] - r['host_A']), abs(r['full_A'] - r['host_A'])) for r in ok if r['solve_status'] == 0 and r['full_status'] == 0]
flag_nonzero = sum((r['solve_status'] != 0 and r['solve_A'] != 0.0) or (r['full_status'] != 0 and r['full_A'] != 0.0) for r in R)
us = lambda c: c / 250.0
def stat(v):
    v = np.array(v, float); return dict(n=len(v), median_us=float(np.median(v)) if len(v) else '', max_us=float(v.max()) if len(v) else '')
summ = dict(board_version=info['version'], points=len(R), host_ok=len(ok), host_flagged=len(fl), status_match=st_match,
            ok_max_abs_diff_A=float(max(dA)) if dA else '', ok_over_1e3A=int(sum(x > 1e-3 for x in dA)),
            flagged_nonzero=flag_nonzero, max_iters=max(r['full_iters'] for r in R))
for name, sel in (('solve_ok', [us(r['solve_cycles']) for r in R if r['solve_status'] == 0]),
                  ('solve_flagged', [us(r['solve_cycles']) for r in R if r['solve_status'] != 0]),
                  ('full_ok', [us(r['full_cycles']) for r in R if r['full_status'] == 0]),
                  ('full_flagged', [us(r['full_cycles']) for r in R if r['full_status'] != 0]),
                  ('full_all', [us(r['full_cycles']) for r in R])):
    for k2, v2 in stat(sel).items():
        summ[f'{name}_{k2}'] = v2
cnt = {}
for r in R:
    cnt[r['solve_status']] = cnt.get(r['solve_status'], 0) + 1
summ['status_counts_board'] = ' '.join(f'{k}:{v}' for k, v in sorted(cnt.items()))
with (OUT / 'board_deploy_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summ)); w.writeheader(); w.writerow(summ)
for k2, v2 in summ.items():
    print(k2, v2)
