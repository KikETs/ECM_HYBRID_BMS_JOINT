"""Train one conditioning variant with a whole CELL held out.

THE SPLIT IS THE POINT OF THIS SCRIPT
    One aging protocol is one physical cell spread over ~100 runs. Holding out
    runs instead of cells would let the model meet the held-out cell's own
    resistance trajectory during training - and that trajectory is exactly what
    differs: two cells aged under the SAME protocol diverge 1.58x in 10 s pulse
    resistance by SOH 0.75 (findings.md section 4.1). A run-level split would
    report a generalisation number no new cell would ever see.

    Five cells train, the sixth validates, nothing shared.

WHAT GOES IN
    Drive cycles carry the aging axis but never reach the SOP region: measured
    on the cached campaign, drive-cycle power tops out at 86.9 W and NO sample
    exceeds 100 W, while HPPC runs spend 2.6 % of their samples past it
    (design section 3.3). Training on drive cycles alone would ask the model to
    extrapolate exactly where SOP lives, so HPPC runs are mixed in. In the
    reproduction the analogous imbalance (SOP windows outnumbering drive-cycle
    windows 6.4:1) improved cold error and made warm error worse, so the mix is
    a knob, --hppc-ratio, not a constant.

THE SCALER IS FIT ON TRAINING WINDOWS ONLY
    Both sets share one scaler - they feed one model - but its min/max come from
    the five training cells. Fitting on everything would let the held-out cell's
    range leak into the transform, which is a small leak but a real one, and the
    whole point of this split is to not have any.

SCHEDULE
    MultiStepLR at 20 % of TOTAL epochs, as in the paper. That makes the
    schedule collapse if --epochs is reduced - the reproduction lost a full run
    to it - so the schedule and the early-stopping condition print at startup.
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
from models_soh import build  # noqa: E402
from windows import load_uypydj_cells  # noqa: E402

SEED = 5
WINDOW = 200
CTX_LEN = 200
BATCH = 5000
LR = 1e-4
GAMMA = 0.1
VAL_CHUNK = 2000
CACHE = os.path.join(os.path.dirname(__file__), "cache")


def cells_available(cache_dir, part="Fifteen_Drive_Cycles"):
    import glob
    return sorted(os.path.basename(p)[len("uypydj_"):-len(f"_{part}.npz")]
                  for p in glob.glob(os.path.join(cache_dir, f"uypydj_*_{part}.npz")))


def fit_shared_scaler(sets):
    """One scaler over the TRAINING windows of every part."""
    los, his, vlos, vhis = [], [], [], []
    for ws in sets:
        sc = ws.fit_scaler()
        los.append(sc["lo"]); his.append(sc["lo"] + sc["rng"])
        vlos.append(sc["vlo"]); vhis.append(sc["vlo"] + sc["vrng"])
    lo = torch.stack(los).min(0).values
    hi = torch.stack(his).max(0).values
    rng = torch.where(hi - lo == 0, torch.ones_like(hi), hi - lo)
    vlo = torch.stack(vlos).min()
    vhi = torch.stack(vhis).max()
    return {"lo": lo, "rng": rng, "vlo": vlo,
            "vrng": vhi - vlo if vhi > vlo else torch.ones_like(vhi)}


def scale_ctx(c, sc_ctx):
    return (c - sc_ctx["lo"]) / sc_ctx["rng"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["M0", "M1", "M2", "M3"])
    ap.add_argument("--holdout-cell", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--feats", default="SOC,T,P",
                    help="input channels. The reference model uses P, which is "
                         "recorded as V*I and therefore carries the target "
                         "voltage; 'SOC,T,I' is the same model asked a current "
                         "question, which is the only currency in which it can "
                         "be compared against the hybrid arm on equal "
                         "information (sop_hybrid_spec.md 7.6).")
    ap.add_argument("--resume", action="store_true",
                    help="continue from {tag}.resume.pt if it exists")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--hppc-stride", type=int, default=10)
    ap.add_argument("--hppc-ratio", type=float, default=1.0,
                    help="HPPC windows per drive-cycle window (0 disables)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-every", type=int, default=10)
    # Sizes are arguments, not constants: this box shares one card, and a run
    # started next to another one has to fit in what is left. The first smoke
    # test of this script died because the batch and validation chunk were
    # baked in at full size while the reproduction held 24 of 31 GiB.
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--val-chunk", type=int, default=VAL_CHUNK)
    # Deployment size axis: a BMS MCU has single-digit MB of flash for
    # everything, and the 256x2 network is 4.2 MB in fp32 (1.06 MB int8).
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs_soh"))
    args = ap.parse_args()

    avail = cells_available(args.cache)
    if args.holdout_cell not in avail:
        sys.exit(f"unknown cell {args.holdout_cell}; have {avail}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    need_ctx = args.variant in ("M2", "M3")
    need_soh = args.variant in ("M1", "M3")

    # ---- data -------------------------------------------------------------
    t0 = time.time()
    parts = [("Fifteen_Drive_Cycles", args.stride)]
    if args.hppc_ratio > 0:
        parts.append(("HPPC", args.hppc_stride))
    tr, va = [], []
    for part, stride in parts:
        ws = load_uypydj_cells(args.cache, part=part, window=WINDOW, stride=stride,
                               feat_keys=tuple(args.feats.split(",")),
                               ctx_keys=("V", "I", "T") if need_ctx else None,
                               ctx_len=CTX_LEN if need_ctx else 0,
                               with_soh=need_soh)
        hi = ws.group_names.index(args.holdout_cell)
        rest = [i for i in range(len(ws.group_names)) if i != hi]
        tr.append(ws.by_group(rest)); va.append(ws.by_group([hi]))
        print(f"  {part:<22} train {len(tr[-1]):>9,}  val {len(va[-1]):>9,}")

    sc = fit_shared_scaler(tr)
    sc_ctx = None
    for s in tr + va:
        s.apply_scaler(sc)
    if need_ctx:
        # Context channels are (V, I, T) - a different set from the model inputs
        # - so they need their own range, also taken from training only.
        c = torch.cat([s.ctx[s.starts.min():s.starts.max()] for s in tr])
        clo, chi = c.min(0).values, c.max(0).values
        sc_ctx = {"lo": clo, "rng": torch.where(chi - clo == 0,
                                                torch.ones_like(chi), chi - clo)}
        del c
    print(f"load+scale {time.time()-t0:.0f}s   holdout {args.holdout_cell} "
          f"of {avail}")

    for s in tr + va:
        s.to(dev)
    if sc_ctx:
        sc_ctx = {k: v.to(dev) for k, v in sc_ctx.items()}

    # Sampling plan: HPPC windows are drawn at --hppc-ratio per drive-cycle
    # window, resampled fresh each epoch so the choice is not frozen at startup.
    n_dc = len(tr[0])
    n_hp = int(min(len(tr[1]), n_dc * args.hppc_ratio)) if len(tr) > 1 else 0
    print(f"per epoch: {n_dc:,} drive-cycle + {n_hp:,} HPPC windows")

    # ---- model ------------------------------------------------------------
    model = build(args.variant, hidden=args.hidden, layers=args.layers).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    miles = [int(args.epochs * f) for f in (0.2, 0.4, 0.6, 0.8)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=GAMMA)
    lossf = nn.MSELoss()
    print(f"variant {args.variant}: {nparam:,} params "
          f"({nparam*4/1e6:.2f} MB fp32 / {nparam/1e6:.2f} MB int8), "
          f"hidden {args.hidden}x{args.layers}, on {dev}")
    print(f"schedule: lr {LR:g} until epoch {miles[0]}, then x{GAMMA} at {miles}")
    print(f"early stop: {args.patience} checks x {args.val_every} epochs "
          f"= {args.patience * args.val_every} epochs without improvement")

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{args.variant}_hold{args.holdout_cell}"
    ckpt = os.path.join(args.out, f"{tag}.pt")

    def run_batch(ws, sel, train):
        got = ws.batch(sel, with_ctx=need_ctx, with_aux=need_soh)
        x, y = got[0], got[1]
        c = scale_ctx(got[2], sc_ctx) if need_ctx else None
        soh = got[2 + int(need_ctx)] if need_soh else None
        pred = model(x, ctx=c, soh=soh)
        return torch.sqrt(lossf(pred, y)), len(sel), pred, y

    def validate():
        model.eval()
        se = n = 0.0
        with torch.no_grad():
            for ws in va:
                for k in range(0, len(ws), args.val_chunk):
                    sel = torch.arange(k, min(k + args.val_chunk, len(ws)),
                                       device=dev)
                    _, _, p, y = run_batch(ws, sel, False)
                    se += float(((p - y) ** 2).sum()); n += len(sel)
        return float(np.sqrt(se / n)) * float(sc["vrng"]) * 1000.0   # mV

    # ---- resume -----------------------------------------------------------
    # The BEST checkpoint carries weights and scalers and nothing else, because
    # eval_a13.py and a13_psweep.py read it and must not have to know about
    # training state. Resume state therefore lives in its own file. Losing a
    # 2.6 h fold to an interrupted queue is avoidable and was avoided once too
    # late.
    #
    # The per-epoch sampling plan is redrawn from the RNG rather than restored,
    # so a resumed run is not bit-identical to an uninterrupted one. That is the
    # same nondeterminism the plan already has between epochs, not a new one.
    rs_path = os.path.join(args.out, f"{tag}.resume.pt")
    best, best_ep, bad, ep0 = float("inf"), -1, 0, 1
    if args.resume and os.path.exists(rs_path):
        r = torch.load(rs_path, map_location=dev, weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"])
        best, best_ep, bad, ep0 = r["best"], r["best_ep"], r["bad"], r["ep"] + 1
        print(f"resume: epoch {ep0} 부터, 현재 best {best:.2f} mV (ep {best_ep})")

    def save_resume(ep):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ep": ep, "best": best,
                    "best_ep": best_ep, "bad": bad}, rs_path + ".tmp")
        os.replace(rs_path + ".tmp", rs_path)      # atomic; a kill mid-write
                                                   # must not leave a torn file

    for ep in range(ep0, args.epochs + 1):
        model.train()
        plan = [(0, torch.randperm(len(tr[0]), device=dev))]
        if n_hp:
            plan.append((1, torch.randperm(len(tr[1]), device=dev)[:n_hp]))
        order = torch.cat([torch.full((len(p),), i, device=dev, dtype=torch.long)
                           for i, p in plan])
        order = order[torch.randperm(len(order), device=dev)]
        cursor = [0, 0]
        tot = torch.zeros((), device=dev)
        seen = 0
        for k in range(0, len(order), args.batch):
            chunk = order[k:k + args.batch]
            loss_sum = 0.0
            opt.zero_grad(set_to_none=True)
            for i, perm in plan:
                m = int((chunk == i).sum())
                if m == 0:
                    continue
                sel = perm[cursor[i]:cursor[i] + m]; cursor[i] += m
                if len(sel) == 0:
                    continue
                l, nb, _, _ = run_batch(tr[i], sel, True)
                (l * nb / len(chunk)).backward()
                loss_sum += float(l.detach()) * nb
            opt.step()
            tot += loss_sum; seen += len(chunk)
        sched.step()

        if ep % args.val_every == 0 or ep == 1:
            v = validate()
            flag = ""
            if v < best - 1e-6:
                best, best_ep, bad = v, ep, 0
                torch.save({"model": model.state_dict(), "variant": args.variant,
                            "holdout": args.holdout_cell, "sc": sc,
                            "sc_ctx": sc_ctx, "window": WINDOW,
                            "feat_keys": tuple(args.feats.split(",")),
                            "ctx_len": CTX_LEN if need_ctx else 0}, ckpt)
                flag = "  *best"
            else:
                bad += 1
            print(f"  ep {ep:4d}  train {float(tot)/max(seen,1):.5f}  "
                  f"val RMSE {v:7.2f} mV  lr {sched.get_last_lr()[0]:.1e}{flag}",
                  flush=True)
            save_resume(ep)
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}")
                break

    print(f"\nbest val RMSE {best:.2f} mV at epoch {best_ep}  -> {ckpt}")
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump({"variant": args.variant, "holdout": args.holdout_cell,
                   "best_val_rmse_mV": best, "best_epoch": best_ep,
                   "params": nparam, "hppc_ratio": args.hppc_ratio,
                   "feats": args.feats,
                   "train_windows_dc": n_dc, "train_windows_hppc": n_hp,
                   "val_windows": sum(len(s) for s in va)}, f, indent=2)
    if os.path.exists(rs_path):
        os.remove(rs_path)          # the summary is the completion marker; a
                                    # leftover resume file would make a finished
                                    # fold look interrupted


if __name__ == "__main__":
    main()
