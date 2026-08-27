"""Per-condition breakdown of the reproduction, in the reference paper's layout.

WHY THIS IS THE RIGHT SET TO BREAK DOWN
    The reproduction's split is mixed->named: it trains on the eight Mixed
    profiles and validates on UDDS, HWFET, LA92 and US06. With six chamber
    temperatures that is exactly the 4 x 6 = 24 conditions the paper's Figure
    tabulates, and every one of them is held out - none of the named cycles is
    ever seen in training. So the numbers below can be put beside the paper's
    without any caveat about train contamination.

WHAT IS AND IS NOT COMPARABLE
    The paper's figure carries three series: an equivalent-circuit model, an
    electrochemical model, and their LSTM. Only the LSTM has a counterpart here.
    This project has an ECM, but it is fitted on the UYPYDJ aging cells at 25 C,
    not on the Mendeley cell across six temperatures, so putting it on this axis
    would compare two different cells and call the gap "model type". It is left
    out rather than drawn misleadingly.

    The paper's own LSTM values are read off its published figure and plotted for
    reference; they are transcribed from a raster image, so they carry whatever
    error that implies and are marked as such.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import load_meas_mat, mendeley_files, resample_1hz  # noqa: E402
from lstm_voltage import WINDOW, MinMax, VoltageLSTM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "runs", "lstm_ratio05_e1000.pt")
ROOT = os.path.join(HERE, "..", "raw", "Mendeley")
NAMED = ("UDDS", "HWFET", "LA92", "US06")
TEMPS = (-20, -10, 0, 10, 25, 40)
# Read off the paper's published figure (panel b). Raster transcription.
PAPER_LSTM_B = {-20: 30.0, -10: 25.5, 0: 26.6, 10: 22.7, 25: 15.1, 40: 13.4}
PAPER_ECM_B = {-20: 77.5, -10: 64.8, 0: 52.7, 10: 42.5, 25: 25.1 if False else 34.2,
               40: 25.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(HERE, "repro_conditions.npz"))
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = VoltageLSTM().to(dev); m.load_state_dict(ck["model"]); m.eval()
    sx, sy = ck["sx"], ck["sy"]
    W = int(ck["window"])
    print(f"  {args.ckpt}  split={ck['split']}  window={W}  {dev}")

    files = [(p, i) for p, i in mendeley_files(args.root, kinds=NAMED)]
    res, traces = {}, {}
    for path, info in sorted(files, key=lambda x: (x[1]["temp_C"], x[1]["tag"])):
        d = resample_1hz(load_meas_mat(path))
        feats = np.stack([d["SOC"], d["T"], d["P"]], 1).astype(np.float32)
        volts = d["V"].astype(np.float32)
        n = len(volts) - W + 1
        if n <= 0:
            continue
        lo = np.asarray(sx["lo"], np.float32); hi = np.asarray(sx["hi"], np.float32)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        f = (feats - lo) / rng
        out = np.empty(n, np.float32)
        with torch.no_grad():
            for s in range(0, n, args.chunk):
                idx = np.arange(s, min(s + args.chunk, n))
                x = np.stack([f[i:i + W] for i in idx])
                p = m(torch.from_numpy(x).to(dev)).reshape(-1).cpu().numpy()
                out[idx - 0] = p
        vlo = float(np.asarray(sy["lo"])); vhi = float(np.asarray(sy["hi"]))
        pred = out * (vhi - vlo) + vlo
        true = volts[np.arange(n) + W - 1]
        e = (pred - true) * 1000.0
        key = (info["temp_C"], info["tag"])
        res[key] = float(np.sqrt(np.mean(e ** 2)))
        if key == (25, "US06"):
            traces[key] = (d["t"][W - 1:W - 1 + n], true, pred)
        print(f"  {info['temp_C']:>4} C  {info['tag']:<7} n={n:>7,}  "
              f"RMSE {res[key]:>7.2f} mV")

    np.savez(args.out,
             keys=np.array([f"{a}|{b}" for a, b in res]),
             rmse=np.array(list(res.values())),
             **({f"trace_{a}_{b}_{k}": v for (a, b), tr in traces.items()
                 for k, v in zip(("t", "true", "pred"), tr)}))
    print(f"\n  -> {args.out}")
    print(f"\n  {'온도':>5} " + "".join(f"{c:>9}" for c in NAMED) + f"{'평균':>9}"
          f"{'논문 LSTM':>10}")
    for T in TEMPS:
        v = [res.get((T, c), np.nan) for c in NAMED]
        print(f"  {T:>4}C " + "".join(f"{x:>8.1f}m" for x in v)
              + f"{np.nanmean(v):>8.1f}m {PAPER_LSTM_B[T]:>9.1f}m")
    allv = np.array(list(res.values()))
    print(f"  {'전체':>5} {'':>36} {np.sqrt(np.mean(allv**2)):>8.1f}m")


if __name__ == "__main__":
    main()
