"""A13 — put the two arms on ONE axis: measured pulse dV, identical pulse set.

WHY THIS FILE HAD TO EXIST BEFORE THE ARMS COULD BE COMPARED
    The full-AI arm was scored on drive-cycle voltage RMSE and the hybrid arm on
    HPPC pulse dV RMSE. Those are different numbers about different events, and
    reporting them side by side would let a reader conclude something neither
    measurement supports. Here the reference LSTM is evaluated at exactly the
    pulses the hybrid was evaluated at, with exactly the hybrid's quantity:

        dV(tau) = V(t0 + tau) - V(t0-),   tau in {2 s, 10 s}

    The LSTM does not predict dV. It predicts terminal voltage at the last sample
    of a 200-sample window, so it is placed at three samples per pulse - t0-,
    t0+2 s, t0+10 s - and the differences are taken. Differencing also cancels
    whatever constant bias the LSTM carries into that operating point, which is
    the same courtesy the hybrid's parameterisation gets for free.

THE INDEX MAP IS VERIFIED, NOT ASSUMED
    Pulse positions are recovered from the cache with the SAME find_pulses /
    rank_pulses used to write uypydj_hppc_resistance.csv, then checked against
    that CSV's V_pre_V and V_tau_V. Agreement is 0.24 uV, i.e. float32 storage
    precision. The key must carry direction: charge and discharge pulses are
    ranked in separate counters, so (cycle, soc_group, rank, tau) alone collides
    and silently pairs a discharge label with a charge sample - which showed up
    as a 1.7 V mismatch the first time.

THE CONTEXT WINDOW IS CLEAR OF THE PULSE, WITH 99 SAMPLES TO SPARE
    Checked, not assumed. HPPC pulses are logged at 10 Hz, so tau = 10 s is 100
    samples (measured: median 100, max 101 over 28,855 pulses), not the 10 it
    would be at the 1 Hz median of the record as a whole. The context ends 200
    samples before the target, i.e. at a - 100, so it stops short of the pulse
    start - but by 99 samples, not by a comfortable margin. A record logged
    faster than 20 Hz would break this and the differencing would start reading
    its own answer.

THE COMPARISON SET IS THE INTERSECTION, AND BOTH ARMS ARE RESCORED ON IT
    A pulse enters only if the hybrid kept it AND the LSTM has 400 clean samples
    of history for all three of its evaluation points (200 window + 200 context).
    Both arms are then rescored on that surviving set - the hybrid's published
    numbers are not carried over, because a subset comparison against a
    full-set number is not a comparison.

THE TWO ARMS ARE NOT ASKED THE SAME QUESTION UNLESS THE POWER IS SOLVED FOR
    The reference model is a POWER->VOLTAGE map: its input is P and its output is
    the V that results. The hybrid is a CURRENT->VOLTAGE map: its input is I. On a
    recorded pulse, P = V*I contains the answer, so handing the LSTM the measured
    P and the hybrid only I compares an estimator against an estimator that was
    told more. Mode `isolve` removes the gap by running the query a BMS actually
    runs: iterate

        V <- f(P = V * I_measured),    V initialised at V_pre

    until it stops moving. Now both arms receive the current and neither receives
    the voltage. If this converges to what `full` produced, the model was solving
    the intersection of the demand hyperbola with the cell characteristic rather
    than reading V off the input, and `full` was a fair number after all. If it
    does not, `full` was inflated. Either outcome is informative, which is why it
    is worth the eight extra forward passes.

THE P CHANNEL CARRIES THE MEASURED VOLTAGE, AND THAT IS MEASURED HERE
    The reference model's inputs are (SOC, T, P) with P = V*I as recorded, so the
    target voltage is present in the input at the very sample being predicted.
    Recovering it algebraically fails on this data - estimating I from the SOC
    channel and forming P/I lands 33-232 mV from the truth, worse than the model
    itself - but "the naive route is closed" is not "the channel is inert".
    Variant `pnom` replaces P over the pulse with V_pre * I, the power a BMS
    believes it is demanding when it has only the pre-pulse voltage. The gap
    between `full` and `pnom` is an upper bound on how much of the LSTM's pulse
    accuracy is reading rather than predicting; it is an upper bound and not an
    estimate, because the substitution also moves the input off-distribution.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_soh import ConditionedVoltageLSTM  # noqa: E402
from uypydj_hppc_resistance import (Q_RATED_AH, TAU_TOL_S, TAUS,  # noqa: E402
                                    find_pulses, rank_pulses)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_t")
RES_CSV = os.path.join(HERE, "uypydj_hppc_resistance.csv")
TRIM_DIR = os.path.join(HERE, "cache", "trim")
RUNS_SOH = os.path.join(HERE, "runs_soh")
RUNS_TRIM = os.path.join(HERE, "runs_trim")
MIN_PULSE_S = 5.0


# -- the pulse set the hybrid actually used --------------------------------
def hybrid_pulses(cell, trim_dir=TRIM_DIR):
    """Unique pulses in the hybrid's dataset, with its measured dV labels.

    The npz holds 12 feature blocks per pulse; the label is identical across
    them, so the pulse set is the unique key set.
    """
    z = np.load(os.path.join(trim_dir, f"trim_{cell}.npz"), allow_pickle=True)
    cyc = z["m_cycle"].astype(int)
    soc = np.round(z["m_SOC"].astype(float), 4)
    rnk = z["m_rank"].astype(str)
    out = {}
    for i in range(len(cyc)):
        out.setdefault((int(cyc[i]), float(soc[i]), str(rnk[i])),
                       (float(z["Y"][i, 0]), float(z["Y"][i, 1])))
    return out


def soc_group_map(cell, res_csv=RES_CSV):
    """(cycle, SOC, rank) -> soc_group, for discharge rows only."""
    m, dup = {}, 0
    for r in csv.DictReader(open(res_csv, encoding="utf-8")):
        if r["protocol"] != cell or r["direction"] != "discharge":
            continue
        k = (int(r["cycle"]), round(float(r["SOC"]), 4), r["rate_rank"])
        g = int(r["soc_group"])
        if k in m and m[k] != g:
            dup += 1
        m[k] = g
    return m, dup


def pulse_index(cell, cache=CACHE):
    """(cycle, soc_group, rank, direction) -> (base, a, {tau: k}, run_end)."""
    z = np.load(os.path.join(cache, f"uypydj_{cell}_HPPC.npz"), allow_pickle=True)
    lens, files = z["lens"], z["files"]
    cols = {k: z[k] for k in ("t", "V", "I", "SOC", "T", "P", "SOH", "valid")}
    off = np.concatenate([[0], np.cumsum(lens)])
    idx = {}
    for kk in range(len(lens)):
        mm = re.match(r"(\d+)\|", str(files[kk]))
        if not mm:
            continue
        cyc = int(mm.group(1)); base = int(off[kk]); n = int(lens[kk])
        sl = slice(base, base + n)
        tt, II, ss = cols["t"][sl], cols["I"][sl], cols["SOC"][sl]
        Ah = (ss - 1.0) * Q_RATED_AH
        for a, b, grp, rank in rank_pulses(find_pulses(tt, II), tt, II, Ah):
            if tt[b - 1] - tt[a] < MIN_PULSE_S or a == 0:
                continue
            ip = float(np.median(II[a:b]))
            if abs(ip) < 1.0:
                continue
            rel = tt[a:b] - tt[a]
            ks = {}
            for tau in TAUS:
                k = int(np.searchsorted(rel, tau, side="right")) - 1
                if k < 0 or rel[k] < tau - TAU_TOL_S:
                    continue
                ks[tau] = k
            d = "discharge" if ip < 0 else "charge"
            idx[(cyc, grp, str(rank), d)] = (base, int(a), ks, base + n, ip)
    return idx, cols


# -- the LSTM, placed at those samples -------------------------------------
def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    w = ck["model"]
    hidden = w["lstm.weight_hh_l0"].shape[1]
    layers = sum(1 for k in w if re.match(r"lstm\.weight_hh_l\d+$", k))
    fc = tuple(w[k].shape[0] for k in w
               if re.match(r"head\.\d+\.weight$", k))[:-1]
    m = ConditionedVoltageLSTM(ck["variant"], hidden=hidden, layers=layers,
                               fc=fc).to(dev)
    m.load_state_dict(w); m.eval()
    return m, ck


def predict_at(model, ck, cols, jobs, dev, chunk=128, mode="full", iters=8):
    """jobs: list of (j, a, v_pre, i_pulse). Returns predicted V at each j.

    mode "full"   - P as recorded (the model is handed the true delivered power)
    mode "pnom"   - P over the pulse replaced by V_pre * I, one shot
    mode "isolve" - P over the pulse solved for self-consistency, V <- f(V*I)
    mode "isolve1"- ONLY the target sample's P is solved; every other sample of
                    the window keeps its recorded P. This separates "the model
                    reads its own target" from "the model needs the pulse shape":
                    isolve replaces ~100 samples and so also perturbs the shape,
                    while isolve1 replaces exactly one number - the one that
                    equals V(j)*I(j).
    """
    W = int(ck["window"]); L = int(ck["ctx_len"])
    sc, scc = ck["sc"], ck["sc_ctx"]
    lo = sc["lo"].cpu().numpy(); rng = sc["rng"].cpu().numpy()
    vlo = float(sc["vlo"]); vrng = float(sc["vrng"])
    # M1 carries no context branch, so it has no context scaler. The history
    # gate below still uses the SAME 400-sample requirement for every variant so
    # that M1 and M2 are scored on one pulse set rather than on two.
    need_ctx = bool(model.use_z)
    if need_ctx:
        clo = scc["lo"].cpu().numpy(); crng = scc["rng"].cpu().numpy()

    fk = tuple(ck.get("feat_keys") or ("SOC", "T", "P"))
    feats = np.stack([cols[k] for k in fk], 1).astype(np.float32)
    dem = fk.index("P") if "P" in fk else fk.index("I")   # the demand channel
    ctxa = np.stack([cols["V"], cols["I"], cols["T"]], 1).astype(np.float32)
    SOH = cols["SOH"].astype(np.float32)
    Ifull = cols["I"].astype(np.float32)

    out = np.empty(len(jobs), np.float32)
    moved = []                                   # last-iteration movement, mV
    n_it = iters if mode in ("isolve", "isolve1") else 1
    with torch.no_grad():
        for s in range(0, len(jobs), chunk):
            blk = jobs[s:s + chunk]
            X0 = np.empty((len(blk), W, 3), np.float32)
            C = np.empty((len(blk), L, 3), np.float32) if need_ctx else None
            A = np.empty(len(blk), np.float32)
            p0s = np.empty(len(blk), np.int64)
            for q, (j, a, v_pre, _ip) in enumerate(blk):
                w0 = j - W + 1
                if w0 - L < 0 or j >= len(feats):
                    # main() gates on this, but a caller that skips the gate
                    # would otherwise get an opaque broadcast error - or worse,
                    # a silently truncated window if numpy ever allowed it.
                    raise ValueError(
                        f"표본 {j} 는 창 {W} + 문맥 {L} 만큼의 이력이 없다")
                X0[q] = feats[w0:j + 1]
                p0s[q] = max(a, w0) - w0          # first in-pulse slot, or > W-1
                if need_ctx:
                    C[q] = ctxa[w0 - L:w0]
                A[q] = SOH[j]
            c = torch.from_numpy((C - clo) / crng).to(dev) if need_ctx else None
            aux = torch.from_numpy(A).to(dev)
            v_est = np.array([b[2] for b in blk], np.float32)   # start at V_pre
            prev = v_est.copy()
            for it in range(n_it):
                X = X0.copy()
                if mode == "isolve1":
                    for q, (j, a, _v, _ip) in enumerate(blk):
                        if int(p0s[q]) <= W - 1:      # target lies in the pulse
                            X[q, W - 1, dem] = v_est[q] * Ifull[j]
                elif mode in ("pnom", "isolve"):
                    for q, (j, a, _v, _ip) in enumerate(blk):
                        w0 = j - W + 1
                        k0 = int(p0s[q])
                        if k0 <= W - 1:
                            X[q, k0:, dem] = v_est[q] * Ifull[w0 + k0:j + 1]
                x = torch.from_numpy((X - lo) / rng).to(dev)
                p = model(x, ctx=c, soh=aux if model.use_soh else None)
                vnew = p.reshape(-1).cpu().numpy() * vrng + vlo
                if mode in ("isolve", "isolve1"):
                    prev = v_est.copy()
                    v_est = vnew.astype(np.float32)
            out[s:s + len(blk)] = v_est if mode in ("isolve", "isolve1") else vnew
            if mode in ("isolve", "isolve1"):
                moved.append(np.abs(v_est - prev) * 1000.0)
    if mode in ("isolve", "isolve1"):
        m = np.concatenate(moved)
        return out, float(np.median(m)), float(np.percentile(m, 99))
    return out


def rmse_mv(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="M1,M2")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--tag", default="f")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--save-pred", action="store_true",
                    help="dump per-pulse predictions for trajectory plots")
    ap.add_argument("--modes", default="full,pnom,isolve,isolve1")
    ap.add_argument("--iters", type=int, default=8,
                    help="fixed-point iterations for mode isolve")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(RUNS_SOH, "a13.json"))
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cells = (args.cells.split(",") if args.cells else
             sorted(os.path.basename(f)[5:-4]
                    for f in os.listdir(TRIM_DIR) if f.startswith("trim_")))
    variants = args.variants.split(",")

    report = {}
    print(f"장치 {dev}   셀 {cells}   변형 {variants}")
    for cell in cells:
        want = hybrid_pulses(cell)
        gmap, dup = soc_group_map(cell)
        idx, cols = pulse_index(cell)
        valid = cols["valid"].astype(bool)
        V = cols["V"].astype(np.float64)

        W_need = 0
        kept, dropped = [], collections.Counter()
        for (cyc, soc, rank), (dv2, dv10) in want.items():
            g = gmap.get((cyc, soc, rank))
            if g is None:
                dropped["soc_group 없음"] += 1; continue
            hit = idx.get((cyc, g, rank, "discharge"))
            if hit is None:
                dropped["캐시에 펄스 없음"] += 1; continue
            base, a, ks, end, ip = hit
            if 2.0 not in ks or 10.0 not in ks:
                dropped["지평 없음"] += 1; continue
            js = [base + a - 1, base + a + ks[2.0], base + a + ks[10.0]]
            need = min(js) - 200 - 200 + 1
            if need < base or max(js) >= end:
                dropped["이력 부족"] += 1; continue
            if not valid[need:max(js) + 1].all():
                dropped["유효하지 않은 구간"] += 1; continue
            kept.append({"key": (cyc, soc, rank), "a": base + a, "js": js,
                         "dv": (dv2, dv10), "I": ip,
                         "v_pre": float(V[base + a - 1])})
        if not kept:
            print(f"  {cell}: 남은 펄스 0  {dict(dropped)}")
            continue

        # sanity: the measured dV recovered from the cache must equal the label
        chk = max(abs(float(V[k["js"][1]]) - k["v_pre"] - k["dv"][0])
                  for k in kept)
        print(f"  {cell:<20} 하이브리드 펄스 {len(want):>4}  "
              f"A13 대상 {len(kept):>4}  라벨 재현 오차 {chk*1e6:.2f} uV")
        if dropped:
            print(f"  {'':<20} 제외: {dict(dropped)}"
                  + (f"  (soc_group 충돌 {dup})" if dup else ""))

        Y = np.array([k["dv"] for k in kept])
        report.setdefault(cell, {"n": len(kept)})
        for var in variants:
            ck_path = os.path.join(RUNS_SOH, f"{args.tag}_{cell}_{var}.pt")
            if not os.path.exists(ck_path):
                print(f"  {'':<20} {var}: 체크포인트 없음 — 건너뜀"); continue
            model, ck = load_model(ck_path, dev)
            fk_ck = tuple(ck.get("feat_keys") or ("SOC", "T", "P"))
            # A current-input model receives no voltage, so there is nothing to
            # make self-consistent; `full` is already the equal-information
            # query and the solve modes would only add noise.
            modes = [m for m in args.modes.split(",")
                     if m == "full" or "P" in fk_ck]
            for mode in modes:
                jobs = [(j, k["a"], k["v_pre"], k["I"])
                        for k in kept for j in k["js"]]
                res = predict_at(model, ck, cols, jobs, dev, args.chunk,
                                 mode=mode, iters=args.iters)
                extra = ""
                if mode in ("isolve", "isolve1"):
                    res, mv50, mv99 = res
                    extra = f"   마지막 반복 이동 중앙 {mv50:.2f} mV / 99% {mv99:.2f} mV"
                    report[cell][f"{var}_{mode}_move"] = mv50
                vh = res.reshape(-1, 3)
                P = np.stack([vh[:, 1] - vh[:, 0], vh[:, 2] - vh[:, 0]], 1)
                r = rmse_mv(P, Y)
                report[cell][f"{var}_{mode}"] = r
                if args.save_pred:
                    # Per-pulse predictions, so a trajectory can be drawn later
                    # without paying for the forward passes again.
                    np.savez(os.path.join(RUNS_SOH,
                                          f"pred_a13_{args.tag}_{var}_{mode}_{cell}.npz"),
                             pred=P, Y=Y, v_pre=vh[:, 0],
                             cycle=np.array([k["key"][0] for k in kept]),
                             SOC=np.array([k["key"][1] for k in kept]),
                             rank=np.array([k["key"][2] for k in kept]),
                             I=np.array([k["I"] for k in kept]))
                print(f"  {'':<20} {var} {mode:<6} dV RMSE {r:8.2f} mV{extra}")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

        # both arms on THIS set
        for rung in ("A0", "A3", "A4"):
            f = os.path.join(RUNS_TRIM, f"pred_{rung}_{cell}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            key = list(zip(z["cycle"].astype(int).tolist(),
                           np.round(z["SOC"].astype(float), 4).tolist(),
                           z["rank"].astype(str).tolist()))
            last = {}
            for i, k in enumerate(key):
                last[k] = i                      # rows are in block order
            sel = [last[k["key"]] for k in kept if k["key"] in last]
            if len(sel) != len(kept):
                print(f"  {'':<20} {rung}: 짝지음 {len(sel)}/{len(kept)}")
            if not sel:
                continue
            r = rmse_mv(z["pred"][sel], z["Y"][sel])
            report[cell][rung] = r
            print(f"  {'':<20} {rung:<10} dV RMSE {r:8.2f} mV")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
