"""Time the SOH inference on the board, and check its answer against Python.

The SOP bench never covered SOH -- the ledger recorded it as "not in the
control cycle, runs once per charge, unmeasured".  That was tolerable while
SOH was a fixed cost nobody compared; it is not tolerable when the point is
that one model is cheaper than another.  This sends real dQ/dV curves from
cache/soh_charge.npz, so the timing is on inputs the model will actually see
and the returned SOH can be checked against the same fit in NumPy.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAGIC = 0x43454D41
CMD = dict(QUERY=0x60, SOH=0x68, SOH_Q=0x69)
RES = "<IIII4f"
NACK_UNKNOWN = 0xFFFFFFFE


def one(p, cmd, x64):
    p.write(bytes([CMD[cmd]]))
    p.write(np.asarray(x64, np.float32).tobytes())
    p.flush()
    raw = p.read(struct.calcsize(RES))
    if len(raw) != struct.calcsize(RES):
        raise RuntimeError(f"short reply for {cmd}: {len(raw)} B")
    magic, cycles, iters, hw, kf, ks, r_eff, soh = struct.unpack(RES, raw)
    if magic != MAGIC:
        raise RuntimeError(f"bad magic {magic:#x}")
    return cycles, iters, hw, soh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--clock", type=float, default=250e6)
    ap.add_argument("--data", default=os.path.join(ROOT, "analysis", "cache",
                                                   "soh_charge.npz"))
    ap.add_argument("--ridge-fit", default=os.path.join(
        ROOT, "analysis", "runs_soh_ridge", "soh_ALL.npz"),
        help="if present, the board's answer is checked against it")
    ap.add_argument("--out", default=os.path.join(HERE, "soh_mcu_bench.csv"))
    a = ap.parse_args()

    z = np.load(a.data, allow_pickle=True)
    X = z["X"]
    idx = np.arange(min(a.n, len(X)))

    ref = None
    if os.path.exists(a.ridge_fit):
        ck = np.load(a.ridge_fit, allow_pickle=True)
        w, b = np.asarray(ck["w"], float), float(ck["b"])
        mu, sd = np.asarray(ck["mu"], float), np.asarray(ck["sd"], float)
        ref = lambda x: float(((x - mu) / sd) @ w + b)      # noqa: E731

    rows, bad = [], 0
    with serial.Serial(a.port, a.baud, timeout=3) as p:
        p.reset_input_buffer()
        for cmd in ("SOH", "SOH_Q"):
            got = []
            for i in idx:
                try:
                    cyc, it, hw, soh = one(p, cmd, X[i])
                except RuntimeError as e:
                    print(f"  {cmd}: {e}", file=sys.stderr)
                    break
                if it == NACK_UNKNOWN:
                    print(f"  {cmd}: refused by the firmware "
                          f"(no integer path in this build)")
                    got = []
                    break
                got.append((cyc, soh))
                rows.append([cmd, int(i), cyc, cyc / a.clock * 1e6, soh])
                if ref is not None and cmd == "SOH":
                    if abs(soh - ref(X[i])) > 2e-4:
                        bad += 1
            if got:
                c = np.array([g[0] for g in got], float)
                us = c / a.clock * 1e6
                print(f"  {cmd:<7} n={len(got):<4} median {np.median(us):8.2f} us"
                      f"   p95 {np.percentile(us, 95):7.2f}"
                      f"   max {us.max():7.2f}"
                      f"   median {int(np.median(c)):>8} cyc")

    if ref is not None:
        n_soh = sum(1 for r in rows if r[0] == "SOH")
        print(f"\n  board vs NumPy on the same fit: {n_soh - bad}/{n_soh} "
              f"within 2e-4" + ("" if not bad else f"   {bad} DISAGREE"))
        if bad:
            raise SystemExit("  the board is not computing the exported model")

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write("cmd,sample,cycles,us,soh\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.6f}\n")
    print(f"  -> {os.path.relpath(a.out, ROOT)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
