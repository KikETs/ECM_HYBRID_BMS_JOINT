"""LSTM voltage model for the Samsung 30T, built to the reference paper's spec.

REFERENCE
    Chen, Yao, Kollmeyer, Vidal, Naguib, Panchal, Emadi,
    "Battery state-of-power estimation: A machine learning battery model with
    numerical searching approach", J. Energy Storage 175 (2026) 123023.
    Dataset: doi:10.5683/SP3/RPCWBY plus the Mendeley 30T drive cycles.

    Hyperparameters below are the ones the paper reports as selected (its
    Table 1), not values tuned here. The point of this script is to REPRODUCE
    first; retuning would make any later comparison meaningless.

        inputs   sequence of (SOC, temperature, power) -> single voltage
        LSTM     2 layers x 256 hidden
        output   2 fully-connected layers, 256 then 16
        window   200 s at 1 Hz  (200 samples)
        optim    Adam, lr 1e-4, MultiStepLR gamma 0.1 at every 20 % of epochs
        batch    5000
        loss     RMSE
        scaling  min-max on both inputs and output
        seed     5

WHY VOLTAGE AND NOT SOP DIRECTLY
    SOP labels are sparse (a few points per temperature) while drive cycles give
    millions of voltage samples. The paper learns voltage, then finds SOP by
    binary-searching the power that drives predicted voltage to its limit. The
    search is model-agnostic, so this file only has to produce a good voltage
    model.

SPLIT DISCIPLINE
    Splitting drive-cycle samples at random would leak: consecutive 1 Hz samples
    inside one 200 s window are nearly the same state. Splits are therefore by
    FILE, and the held-out temperature option exists because generalising to an
    unseen temperature is the thing that actually gets claimed later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import (DRIVE_CYCLES, load_meas_mat, mendeley_files,  # noqa: E402
                     parse_name, resample_1hz, test3_windows_source)

SEED = 5
WINDOW = 200
HIDDEN = 256
LSTM_LAYERS = 2
# THE PAPER'S TWO STATEMENTS ABOUT THIS DISAGREE, AND THE PARAMETER COUNT WINS.
# Table 1 lists the output-layer hidden units as 2^8, 2^4 = (256, 16). Table 4
# lists parameter counts, and subtracting the recurrent part from those leaves a
# head of ~262k against the 69,921 that (256, 16) produces. A head of (512, 256)
# reproduces all three of the paper's counts EXACTLY:
#     RNN   461,057   GRU   858,369   LSTM 1,054,721
# Three exact 6-7 digit matches are not coincidence, so the head is (512, 256)
# and Table 1's "2^4" is the erroneous statement. The first reproduction run
# here used (256, 16) and plateaued at 35.45 mV against the paper's 21.54.
FC_SIZES = (512, 256)
LR = 1e-4
BATCH = 5000
GAMMA = 0.1
# Validation chunk. nn.LSTM materialises the FULL sequence output, so a chunk of
# C windows costs C x 200 x 256 floats per layer - 20k windows asked for 19.5 GiB
# and died. 2k keeps it near 2 GB, and the box is shared, so the whole card is
# not ours to take.
VAL_CHUNK = 2000


class VoltageLSTM(nn.Module):
    def __init__(self, n_in=3, hidden=HIDDEN, layers=LSTM_LAYERS, fc=FC_SIZES):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True)
        seq, prev = [], hidden
        for h in fc:
            seq += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        seq += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*seq)

    def forward(self, x):                      # x: (B, W, n_in)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_windows(files, window=WINDOW):
    """Turn each file into (window, 3) input blocks and the voltage AT THE LAST
    INPUT SAMPLE.

    THE TARGET IS V AT THE END OF THE WINDOW, NOT ONE STEP PAST IT. The paper is
    explicit: "at each time step k ... the model then estimates the battery
    voltage by integrating information from both the PRESENT measured inputs and
    memory states". So the label belongs to the same instant as the last input.

    The first reproduction here used volts[i+window] - one step ahead - which is
    a different and much harder problem, because terminal voltage responds to
    power almost instantly through the ohmic drop and a one-step-ahead model
    cannot see the power step that causes the voltage step. Measured on this
    dataset, that cost almost exactly the unpredictable part: per-cycle RMSE came
    out at 0.72-0.94x the RMS of the one-second voltage change, correlation 0.93,
    with the cycle ranking (US06 > LA92 > UDDS > HWFET) identical in both. The
    error was not a training failure, it was the task being wrong.

    Windows are cut inside a file only - never across a file boundary, which
    would splice two different thermal states into one sequence.
    """
    X, Y, meta = [], [], []
    for path, info in files:
        d = resample_1hz(load_meas_mat(path))
        feats = np.stack([d["SOC"], d["T"], d["P"]], axis=1).astype(np.float32)
        volts = d["V"].astype(np.float32)
        n = len(volts) - window + 1
        if n <= 0:
            continue
        idx = np.arange(n)
        X.append(np.stack([feats[i:i + window] for i in idx]))
        Y.append(volts[idx + window - 1])
        meta += [(info["temp_C"], info["tag"])] * n
    return np.concatenate(X), np.concatenate(Y), meta


def windows_from_series(series, window=WINDOW):
    """Cut (SOC, T, P)->V windows out of already-1 Hz Test#3 runs.

    Kept separate from build_windows() because these arrive as plain arrays
    rather than files, and because they must never be resampled again - the
    Test#3 CSVs are logged at 1 Hz already.
    """
    X, Y, meta = [], [], []
    for s_ in series:
        feats = np.stack([s_["SOC"], s_["T"], s_["P"]], axis=1).astype(np.float32)
        volts = s_["V"].astype(np.float32)
        # Same alignment as build_windows: the label is the voltage at the LAST
        # input sample, not one step past it.
        n = len(volts) - window + 1
        if n <= 0:
            continue
        idx = np.arange(n)
        X.append(np.stack([feats[i:i + window] for i in idx]))
        Y.append(volts[idx + window - 1])
        m = s_["meta"]
        meta += [(m["temp_C"], f"SOP{m['pulse_s']}s")] * n
    if not X:
        return (np.empty((0, window, 3), np.float32), np.empty((0,), np.float32), [])
    return np.concatenate(X), np.concatenate(Y), meta


class MinMax:
    def __init__(self, a):
        self.lo = a.reshape(-1, a.shape[-1]).min(0)
        self.hi = a.reshape(-1, a.shape[-1]).max(0)
        self.rng = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)

    def __call__(self, a):
        return ((a - self.lo) / self.rng).astype(np.float32)

    def to_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "raw", "Mendeley"))
    ap.add_argument("--holdout-temp", type=int, default=None,
                    help="temperature (degC) kept entirely out of training")
    # THE SCHEDULE IS TIED TO --epochs, SO SHORTENING THE RUN SHORTENS THE
    # LEARNING RATE. The paper decays at every 20 % of TOTAL epochs over a 5000
    # epoch budget, i.e. it holds lr = 1e-4 for the first 1000 epochs. Running
    # the same recipe with --epochs 300 decays at 60/120/180/240 and reaches
    # 1e-8 by epoch 240 - the first run here froze at 44 mV that way, which is
    # a dead optimiser, not convergence.
    ap.add_argument("--epochs", type=int, default=1000)
    # Patience counts VALIDATION CHECKS, not epochs, and checks happen every
    # --val-every epochs. The paper's 100 means 1000 epochs of no improvement,
    # which can never fire inside a 300 epoch budget. Scale it to the budget.
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs"))
    # Adding the SOP measurement runs is the difference between the paper's
    # 'LSTM without SOP' control and its actual model. Drive cycles barely visit
    # the SOP current limits (1.6 % of samples in the 25-40 A band, measured);
    # these runs are built to sit on them.
    ap.add_argument("--with-sop", action="store_true",
                    help="add Test#3 2 s and 30 s SOP runs to training")
    ap.add_argument("--test3-zip",
                    default=os.path.join(os.path.dirname(__file__), "..", "raw",
                                         "RPCWBY", "3_Test_3.zip"))
    ap.add_argument("--tag", default=None)
    # SOP runs outnumber drive-cycle windows 6.4:1 when added whole. Measured
    # effect of that imbalance: cold error improved 19-21 % while 25/40 degC got
    # 27-43 % WORSE, leaving the mean almost unchanged. --sop-ratio caps the SOP
    # share so the warm end is not crowded out. 1.0 = as many SOP windows as
    # drive-cycle windows; 0 = none.
    ap.add_argument("--sop-ratio", type=float, default=None,
                    help="SOP windows per drive-cycle window (None = use all)")
    # Speed work here is deliberately limited to changes that do not alter the
    # arithmetic: where the training tensors live, and how often the host reads
    # the loss. Measured at the real training size (978k windows, 196 batches per
    # epoch), those two took 18.4 -> 14.9 s/epoch.
    #
    # bf16 autocast would take it to 8.1 s/epoch, but it CHANGES THE NUMBERS and
    # the fp32-vs-bf16 equivalence run was cut short before it produced a verdict,
    # so it stays off by default at the user's instruction. Do not flip this
    # default without finishing that comparison - a reproduction that silently
    # ran in a different precision than the paper is not a reproduction.
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"],
                    help="training-step precision (bf16 UNVERIFIED); "
                         "validation is always fp32")
    ap.add_argument("--data-on-gpu", default="auto",
                    choices=["auto", "yes", "no"],
                    help="keep the training tensors in VRAM (auto: if they fit)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32 IS NOT LEFT ON HERE. It was switched on while profiling and carried
    # into the script by mistake; it has a 10-bit mantissa, so it is a precision
    # change like bf16, not a free speed-up. Measured cost of the mistake: epoch
    # 1 validation moved 207.86 -> 219.23 mV against the fp32 reference run. The
    # backend defaults are left exactly as the earlier runs saw them.

    files = mendeley_files(args.root, kinds=DRIVE_CYCLES)
    if args.holdout_temp is None:
        # validation = the four named cycles; training = the eight mixed ones.
        tr = [(p, i) for p, i in files if i["tag"].startswith("Mixed")]
        va = [(p, i) for p, i in files if not i["tag"].startswith("Mixed")]
        split = "mixed->named"
    else:
        tr = [(p, i) for p, i in files if i["temp_C"] != args.holdout_temp]
        va = [(p, i) for p, i in files if i["temp_C"] == args.holdout_temp]
        split = f"holdout {args.holdout_temp}C"

    print(f"split: {split}   train {len(tr)} files / val {len(va)} files")
    t0 = time.time()
    Xtr, Ytr, _ = build_windows(tr)
    Xva, Yva, mva = build_windows(va)
    if args.with_sop:
        # 2 s and 30 s only. The 10 s runs are the held-out test set in the
        # reference protocol and must not touch training.
        sop = test3_windows_source(args.test3_zip, pulse_lengths=(2, 30))
        Xs, Ys, _ = windows_from_series(sop)
        print(f"  + Test#3 SOP runs: {len(sop)} runs -> {Xs.shape[0]:,} windows")
        if args.sop_ratio is not None:
            keep = int(len(Xtr) * args.sop_ratio)
            if keep < len(Xs):
                rng = np.random.default_rng(SEED)
                sel = rng.choice(len(Xs), size=keep, replace=False)
                sel.sort()                      # keep file order for locality
                Xs, Ys = Xs[sel], Ys[sel]
                print(f"    subsampled to ratio {args.sop_ratio}: {len(Xs):,} windows")
        Xtr = np.concatenate([Xtr, Xs])
        Ytr = np.concatenate([Ytr, Ys])
    print(f"windows: train {Xtr.shape}  val {Xva.shape}   ({time.time()-t0:.0f}s)")

    sx, sy = MinMax(Xtr), MinMax(Ytr.reshape(-1, 1))
    Xtr_t = torch.from_numpy(sx(Xtr))
    Ytr_t = torch.from_numpy(sy(Ytr.reshape(-1, 1)).ravel())
    Xva_t = torch.from_numpy(sx(Xva))   # kept on CPU; moved per chunk
    Yva_np = Yva

    # Residency decision. The training tensor is a 200x duplication of a few MB
    # of raw signal, so it is big (2.35 GB with the SOP runs) but far smaller
    # than the card. Leaving it in host memory made every step gather 5000 random
    # rows out of it on one CPU thread and then copy them over PCIe unpinned,
    # which is what the profile above is mostly measuring.
    need = (Xtr_t.numel() + Ytr_t.numel()) * 4
    if args.data_on_gpu == "no" or dev == "cpu":
        resident = False
    elif args.data_on_gpu == "yes":
        resident = True
    else:
        free, _ = torch.cuda.mem_get_info()
        resident = need < free * 0.5          # leave room for activations
    if resident:
        Xtr_t, Ytr_t = Xtr_t.to(dev), Ytr_t.to(dev)
    print(f"train tensors: {need/1e9:.2f} GB, "
          f"{'resident in VRAM' if resident else 'in host memory'}")

    model = VoltageLSTM().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam:,} trainable parameters on {dev}, "
          f"train step in {args.precision}, validation in fp32")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    miles = [int(args.epochs * f) for f in (0.2, 0.4, 0.6, 0.8)]
    print(f"schedule: lr {LR:g} until epoch {miles[0]}, then x{GAMMA} at {miles}")
    print(f"early stop: {args.patience} checks x {args.val_every} epochs "
          f"= {args.patience * args.val_every} epochs without improvement")
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=GAMMA)
    lossf = nn.MSELoss()

    best, best_ep, bad = float("inf"), -1, 0
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (f"holdout{args.holdout_temp}" if args.holdout_temp is not None else "mixed")
    if args.with_sop and not args.tag:
        tag += "_sop"
        if args.sop_ratio is not None:
            tag += f"{args.sop_ratio:g}".replace(".", "p")
    ckpt = os.path.join(args.out, f"lstm_{tag}.pt")

    n = len(Xtr_t)
    amp = args.precision == "bf16" and dev == "cuda"
    for ep in range(1, args.epochs + 1):
        model.train()
        # Drawn on the CPU generator and then moved, NOT generated on the device.
        # torch.randperm(n, device="cuda") uses the CUDA generator and returns a
        # different permutation for the same seed, which would silently change
        # batch composition and make this run incomparable with earlier ones.
        perm = torch.randperm(n).to(Xtr_t.device, non_blocking=True)
        # Accumulated on the device: reading the loss every step forced 196 host
        # syncs per epoch, each one stalling the queue for a number only ever
        # printed as an epoch mean.
        tot = torch.zeros((), device=dev)
        for k in range(0, n, BATCH):
            b = perm[k:k + BATCH]
            xb, yb = Xtr_t[b], Ytr_t[b]
            if not resident:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mse = lossf(model(xb), yb)
                loss = torch.sqrt(mse.float())
            else:
                loss = torch.sqrt(lossf(model(xb), yb))
            loss.backward()
            opt.step()
            tot += loss.detach() * len(b)
        tot = tot.item()
        sched.step()

        if ep % args.val_every == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                pred = []
                for k in range(0, len(Xva_t), VAL_CHUNK):
                    pred.append(model(Xva_t[k:k + VAL_CHUNK].to(dev)).cpu().numpy())
                pv = np.concatenate(pred) * sy.rng[0] + sy.lo[0]
            v = rmse(pv, Yva_np) * 1000.0            # mV
            flag = ""
            if v < best - 1e-6:
                best, best_ep, bad = v, ep, 0
                torch.save({"model": model.state_dict(),
                            "sx": sx.to_dict(), "sy": sy.to_dict(),
                            "window": WINDOW, "split": split}, ckpt)
                flag = "  *best"
            else:
                bad += 1
            print(f"  ep {ep:4d}  train {tot/n:.5f}  val RMSE {v:7.2f} mV"
                  f"  lr {sched.get_last_lr()[0]:.1e}{flag}")
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}")
                break

    print(f"\nbest val RMSE {best:.2f} mV at epoch {best_ep}  -> {ckpt}")
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump({"split": split, "best_val_rmse_mV": best, "best_epoch": best_ep,
                   "params": nparam, "train_files": len(tr), "val_files": len(va),
                   "train_windows": int(Xtr.shape[0]), "val_windows": int(Xva.shape[0])}, f, indent=2)


if __name__ == "__main__":
    main()
