"""Does the reference model answer a POWER query, or does it echo the power?

WHY THE dV NUMBERS ALONE COULD NOT SETTLE THIS
    eval_a13.py showed the model's pulse accuracy collapses when the whole pulse
    P trajectory is replaced by a self-consistent constant (32.6 -> 162.6 mV) but
    survives when only the target sample is replaced (32.6 -> 30.7 mV). Those two
    facts are compatible with two opposite stories:

      (a) the model reads V out of P over the pulse, and the constant-V
          substitution took the answer away, or
      (b) the model needs the pulse SHAPE, and a constant-V substitution is a
          crude trajectory that would hurt any honest model.

    RMSE cannot separate them, because both stories predict a large number.

SCALING THE POWER CHANNEL ALONE PROVES NOTHING, AND THE FIRST RUN SHOWED IT
    Scaling only P leaves the SOC channel reporting the ORIGINAL current. The
    input then says "same current, s times the power", whose only consistent
    reading is "s times the voltage" - so a positive slope is the CORRECT answer
    to that question and says nothing about whether the model can run an SOP
    search. The first sweep returned +0.19 to +1.08 V/s and had to be discarded
    for this reason.

    The fix is to move both channels together. SOC over the pulse is advanced
    with s * I, so the SOC slope reports the scaled current, while P is scaled by
    the same s. Now the input is a coherent description of a pulse drawing s
    times the current at the same terminal voltage, and the only physical answer
    is a LOWER voltage.

THE SWEEP SEPARATES THEM BY SIGN, NOT BY MAGNITUDE
    A power-based SOP search asks the model the same question repeatedly with a
    different proposed power. So scale the recorded pulse power by s and watch
    the returned voltage.

      A cell:   dV(tau) = I * R_eff(tau), so scaling I by s scales the drop by s
                and dV/ds = dV_measured - a NEGATIVE number, and one this data
                already contains per pulse, so the expectation needs no fitting.

      An echo:  V = P / I_inferred, with I taken from the SOC history, which does
                not move when the proposed power moves. Then V = s * V_measured,
                i.e. dV/ds > 0 and about 3.5x larger.

    Opposite signs. No calibration, no threshold, no argument about what counts
    as "close" - the model either bends the right way or it does not. A model
    that bends the wrong way cannot run a binary search for SOP at all, whatever
    its RMSE at s = 1 says.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_a13 import (RUNS_SOH, hybrid_pulses, load_model,  # noqa: E402
                      pulse_index, soc_group_map)

SCALES = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)


def sweep_cell(cell, var, tag, dev, chunk, max_pulses, scale_soc=True,
               last_only=False):
    """last_only: vary the power at the FINAL sample alone.

    A binary search that proposes an instantaneous power - rather than a whole
    horizon - would move exactly this one number. If the model is insensitive to
    it the search has no gradient to descend and terminates wherever it started,
    which is a different failure from bending the wrong way and has to be
    measured separately.
    """
    want = hybrid_pulses(cell)
    gmap, _ = soc_group_map(cell)
    idx, cols = pulse_index(cell)
    valid = cols["valid"].astype(bool)
    ck_path = os.path.join(RUNS_SOH, f"{tag}_{cell}_{var}.pt")
    if not os.path.exists(ck_path):
        return None
    model, ck = load_model(ck_path, dev)
    W = int(ck["window"]); L = int(ck["ctx_len"])
    sc, scc = ck["sc"], ck["sc_ctx"]
    lo = sc["lo"].cpu().numpy(); rng = sc["rng"].cpu().numpy()
    vlo = float(sc["vlo"]); vrng = float(sc["vrng"])
    need_ctx = bool(model.use_z)
    if need_ctx:
        clo = scc["lo"].cpu().numpy(); crng = scc["rng"].cpu().numpy()

    fk = tuple(ck.get("feat_keys") or ("SOC", "T", "P"))
    feats = np.stack([cols[k] for k in fk], 1).astype(np.float32)
    dem = fk.index("P") if "P" in fk else fk.index("I")   # the demand channel
    ctxa = np.stack([cols["V"], cols["I"], cols["T"]], 1).astype(np.float32)
    SOH = cols["SOH"].astype(np.float32)
    V = cols["V"].astype(np.float64)

    jobs = []
    for (cyc, soc, rank) in want:
        g = gmap.get((cyc, soc, rank))
        if g is None:
            continue
        hit = idx.get((cyc, g, rank, "discharge"))
        if hit is None:
            continue
        base, a, ks, end, ip = hit
        if 10.0 not in ks:
            continue
        j = base + a + ks[10.0]
        need = j - W - L + 1
        if need < base or j >= end or not valid[need:j + 1].all():
            continue
        jobs.append((j, base + a, float(V[base + a - 1]), float(V[j])))
    if max_pulses:
        jobs = jobs[:max_pulses]
    if not jobs:
        return None

    out = np.empty((len(jobs), len(SCALES)), np.float32)
    with torch.no_grad():
        for s0 in range(0, len(jobs), chunk):
            blk = jobs[s0:s0 + chunk]
            X0 = np.empty((len(blk), W, 3), np.float32)
            C = np.empty((len(blk), L, 3), np.float32) if need_ctx else None
            A = np.empty(len(blk), np.float32)
            k0s = np.empty(len(blk), np.int64)
            for q, (j, a, _vp, _vm) in enumerate(blk):
                w0 = j - W + 1
                X0[q] = feats[w0:j + 1]
                k0s[q] = max(a, w0) - w0
                if need_ctx:
                    C[q] = ctxa[w0 - L:w0]
                A[q] = SOH[j]
            c = torch.from_numpy((C - clo) / crng).to(dev) if need_ctx else None
            aux = torch.from_numpy(A).to(dev)
            for si, s in enumerate(SCALES):
                X = X0.copy()
                for q in range(len(blk)):
                    k0 = int(k0s[q])
                    if k0 > W - 1:
                        continue
                    if last_only:
                        X[q, W - 1, dem] *= s
                        continue
                    X[q, k0:, dem] *= s                    # demand channel
                    if scale_soc:
                        # SOC advances with s*I from the pre-pulse value, so the
                        # SOC slope reports the scaled current.
                        soc0 = X0[q, k0 - 1, 0] if k0 >= 1 else X0[q, k0, 0]
                        X[q, k0:, 0] = soc0 + (X0[q, k0:, 0] - soc0) * s
                x = torch.from_numpy((X - lo) / rng).to(dev)
                p = model(x, ctx=c, soh=aux if model.use_soh else None)
                out[s0:s0 + len(blk), si] = \
                    p.reshape(-1).cpu().numpy() * vrng + vlo
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.array(jobs, dtype=object), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="M1,M2")
    ap.add_argument("--cells", default="BOOST,BOOST_NEGPULSE,BOOST_NEGPULSE_1S,"
                                       "BOOST_REST,CC,CC_CELL2")
    ap.add_argument("--tag", default="f")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-pulses", type=int, default=400)
    ap.add_argument("--last-only", action="store_true",
                    help="vary the power at the final sample alone")
    ap.add_argument("--no-scale-soc", action="store_true",
                    help="scale only the power channel (the discarded first run)")
    ap.add_argument("--out", default=os.path.join(RUNS_SOH, "a13_psweep.json"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    i1 = SCALES.index(1.0)
    rep = {}
    print(f"  전력 배율 {SCALES}   (s=1.0 이 실제 펄스)")
    print(f"  {'셀':<20} {'변형':>4} {'펄스':>6} {'dV/ds 실측':>12} "
          f"{'물리 기대':>11} {'비율':>8} {'음수비':>8} {'에코비':>8}")
    for cell in args.cells.split(","):
        for var in args.variants.split(","):
            r = sweep_cell(cell, var, args.tag, dev, args.chunk, args.max_pulses,
                           scale_soc=not args.no_scale_soc,
                           last_only=args.last_only)
            if r is None:
                print(f"  {cell:<20} {var:>4}  체크포인트/펄스 없음"); continue
            jobs, Vs = r
            v_meas = np.array([j[3] for j in jobs], np.float64)
            v_pre = np.array([j[2] for j in jobs], np.float64)
            dv_meas = v_meas - v_pre          # the ECM's expected dV/ds
            # slope in V per unit s, from the two points bracketing s = 1
            d = (Vs[:, i1 + 1] - Vs[:, i1 - 1]) / (SCALES[i1+1] - SCALES[i1-1])
            echo = v_meas                     # d/ds of (s * V_meas) is V_meas
            neg = float(np.mean(d < 0))
            rep.setdefault(cell, {})[var] = {
                "n": len(jobs), "slope_V_per_s": float(np.median(d)),
                "frac_negative": neg,
                "expected_slope_V_per_s": float(np.median(dv_meas)),
                "ratio_to_physical": float(np.median(d / dv_meas)),
                "ratio_to_echo": float(np.median(d / echo)),
                "V_at_scales": [float(np.median(Vs[:, k]))
                                for k in range(len(SCALES))],
            }
            print(f"  {cell:<20} {var:>4} {len(jobs):>6,} "
                  f"{np.median(d):>+11.3f}V {np.median(dv_meas):>+10.3f}V "
                  f"{np.median(d/dv_meas):>+7.2f} {neg*100:>7.1f}% "
                  f"{np.median(d/echo):>+7.2f}")
    with open(args.out, "w") as f:
        json.dump({"scales": list(SCALES), "cells": rep}, f, indent=2)
    print("\n  물리 기대 = 측정된 dV (음수). 비율 +1.0 이면 셀처럼 굽고, "
          "에코비 +1.0 이면 P를 되읽는 것.")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
