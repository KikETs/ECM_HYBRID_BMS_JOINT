"""Evaluate a trained voltage LSTM the way the reference paper reports it.

The paper's headline number is an average over 24 test conditions - four named
drive cycles at six temperatures - and its central claim is about WHERE the
error sits, not just its mean: a model trained on drive cycles alone is said to
fail at -10 and -20 degC because the cell self-heats during a drive cycle and
never actually spends time cold. A single averaged RMSE cannot show that, so
this script always breaks the error down by temperature and by cycle.

Reference: average voltage RMSE 21.54 mV for their LSTM across those 24
conditions (their Table 4 discussion).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import (DRIVE_CYCLES, load_meas_mat, mendeley_files,  # noqa: E402
                     resample_1hz)
from lstm_voltage import VoltageLSTM, WINDOW  # noqa: E402

NAMED_CYCLES = ("UDDS", "HWFET", "LA92", "US06")


def apply_scale(a, lo, hi):
    rng = np.where(np.array(hi) - np.array(lo) == 0, 1.0, np.array(hi) - np.array(lo))
    return ((a - np.array(lo)) / rng).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "raw", "Mendeley"))
    ap.add_argument("--chunk", type=int, default=2000)
    # Evaluation is inference only and usually runs while a training job holds
    # most of the card, so it defaults to CPU rather than fighting for VRAM.
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = VoltageLSTM().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    sx, sy = ck["sx"], ck["sy"]
    print(f"checkpoint: {args.ckpt}   (trained split: {ck.get('split')})")

    files = mendeley_files(args.root, kinds=NAMED_CYCLES)
    per = {}
    for path, info in files:
        d = resample_1hz(load_meas_mat(path))
        feats = np.stack([d["SOC"], d["T"], d["P"]], axis=1).astype(np.float32)
        volts = d["V"].astype(np.float32)
        n = len(volts) - WINDOW + 1
        if n <= 0:
            continue
        X = np.stack([feats[i:i + WINDOW] for i in range(n)])
        Xs = torch.from_numpy(apply_scale(X, sx["lo"], sx["hi"]))
        with torch.no_grad():
            pred = []
            for k in range(0, len(Xs), args.chunk):
                pred.append(model(Xs[k:k + args.chunk].to(dev)).cpu().numpy())
        pv = np.concatenate(pred) * (sy["hi"][0] - sy["lo"][0]) + sy["lo"][0]
        # Label at the last input sample - see build_windows in lstm_voltage.py.
        yv = volts[WINDOW - 1:WINDOW - 1 + n]
        per[(info["temp_C"], info["tag"])] = {
            "rmse_mV": float(np.sqrt(np.mean((pv - yv) ** 2)) * 1000),
            "max_mV": float(np.max(np.abs(pv - yv)) * 1000),
            "n": n,
        }

    temps = sorted({t for t, _ in per})
    print(f"\n{'T[C]':>5} " + "".join(f"{c:>10}" for c in NAMED_CYCLES) + f"{'mean':>10}")
    col_acc = {c: [] for c in NAMED_CYCLES}
    all_r = []
    for t in temps:
        row, vals = [], []
        for c in NAMED_CYCLES:
            r = per.get((t, c))
            if r:
                row.append(f"{r['rmse_mV']:10.2f}")
                vals.append(r["rmse_mV"])
                col_acc[c].append(r["rmse_mV"])
                all_r.append(r["rmse_mV"])
            else:
                row.append(f"{'-':>10}")
        print(f"{t:>5} " + "".join(row) + f"{np.mean(vals):10.2f}")
    print(f"{'mean':>5} " + "".join(f"{np.mean(col_acc[c]):10.2f}" for c in NAMED_CYCLES)
          + f"{np.mean(all_r):10.2f}")
    print(f"\noverall mean RMSE over {len(all_r)} conditions: {np.mean(all_r):.2f} mV")
    print("reference paper (LSTM, same 24 conditions): 21.54 mV")

    worst = sorted(per.items(), key=lambda kv: -kv[1]["rmse_mV"])[:5]
    print("\nworst conditions")
    for (t, c), r in worst:
        print(f"  {t:>4}C {c:<6} RMSE {r['rmse_mV']:8.2f} mV   max |err| {r['max_mV']:8.1f} mV")


if __name__ == "__main__":
    main()
