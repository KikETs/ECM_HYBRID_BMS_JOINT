import csv
import itertools
from pathlib import Path
from collections import Counter
import math
import sys
import numpy as np
import serial

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "repo" / "mcu"))
import bench_sop as board

board.CMD["REVIEW4"] = 0x72
out = RUN / "mcu"
rng = np.random.default_rng(12345)
points = [
    ("grid", float(soc), float(soh), float(v), float(i))
    for soc, soh, v, i in itertools.product(
        np.linspace(.20, .96, 13), np.linspace(.73, .99, 9), (2.85, 3.45, 3.85, 4.10), (-10., -0.2, 0., 0.2, 5.))]
points += [("invalid_soh", .75, .60, 3.71, -2.),
           ("nonfinite_current", .75, .90, 3.71, float("nan")),
           ("at_discharge_cutoff", .65, .88, 2.49, 0.),
           ("at_charge_cutoff", .65, .88, 4.21, 0.)]
rows = []
with serial.Serial("/dev/ttyACM0", 921600, timeout=3, exclusive=True) as port:
    port.reset_input_buffer()
    identity = board.query(port)
    assert identity["clock_hz"] == 250_000_000
    assert identity["version"] == 5
    for kind, soc, soh, v, current in points:
        key = ("dis2", "dis10", "chg2", "chg10")
        r = board.call(port, "REVIEW4", 128, soc, soh, v, 0., 0., current, [0.] * 12)
        coded = r["iters"]
        max_iter = coded >> 16
        assert 0 < r["cycles"] < 2**30 and max_iter <= 24
        statuses = [(coded >> (3*k)) & 7 for k in range(4)]
        limits = [r[k] for k in ("kf", "ks", "r_eff", "i_star")]
        for k, (status, limit) in enumerate(zip(statuses, limits)):
            assert status in {0, 1, 2, 3, 4, 5} and math.isfinite(limit)
            assert limit == 0. if status else (limit < 0. if k < 2 else limit > 0.)
        if kind == "invalid_soh":
            assert statuses == [2] * 4
        if kind == "nonfinite_current":
            assert statuses == [1] * 4
        d = dict(kind=kind, soc=soc, soh=soh, v_pre=v, current=current, max_iters=max_iter,
                 cycles=r["cycles"], time_us=r["cycles"] / 250, stack_bytes=r["stack"])
        d.update({f"status_{k}": n for k,n in zip(key, statuses)})
        d.update({f"limit_A_{k}": n for k,n in zip(key, limits)})
        rows.append(d)
    assert any(row["max_iters"] == 24 for row in rows)
    assert any(all(row[f"status_{k}"] == 0 for k in key) for row in rows)
    assert any(any(row[f"status_{k}"] == 4 for k in key) for row in rows)
with (out / "coupled_points.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for group in ("all_grid", "all_success", "convergence_cap_exercised"):
    chosen = [r for r in rows if r["kind"] == "grid" and
              (group == "all_grid" or (group == "all_success" and all(r[f"status_{k}"] == 0 for k in key))
               or (group == "convergence_cap_exercised" and r["max_iters"] == 24))]
    d = np.array([r["time_us"] for r in chosen])
    print(group, len(chosen), "median/p95/max us", np.median(d), np.quantile(d,.95), max(d))
print("solver statuses", dict(Counter(r[f"status_{k}"] for r in rows for k in key)))
print("board", identity)
