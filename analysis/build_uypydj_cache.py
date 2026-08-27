"""Flatten the UYPYDJ aging campaign into one .npz per cell, once.

WHY A CACHE
    The drive-cycle half of the campaign is 583 .mat files inside nine multi-GB
    zips, ~55.8 M samples. Decompressing and parsing that costs minutes every
    time, and the leave-one-cell-out study reads it six times per model variant.
    The parsed form is only 670 MB as plain series (docs/soh_extension_design.md
    section 3.1), so it is cached as float32 arrays and memory-mapped afterwards.

WHAT GOES IN, AND ON WHICH AXIS
    SOC is the RATED axis, converted from the published aged-capacity column by
    data30t.load_uypydj_mat. Using the published column would erase the aging
    signal this project exists to model - see findings.md section 2.1.

ONE FILE PER CELL, NOT PER PROTOCOL FILE
    Held-out splits are per CELL. Keeping each cell in its own archive makes a
    fold a choice of files to open rather than a filter applied after loading.

    Note the naming: CC and CC_CELL2 are the SAME charging protocol on DIFFERENT
    cells. They are separate groups here because they are separate cells.

STORED PER SAMPLE
    SOC, T, P, V  - the model's inputs and target
    I             - measured current, for the context encoder only
    SOH, CAP      - per-file constants, broadcast so a window can be labelled
                    without a second lookup
    file_id       - index into the archive's file table, so a window can be
                    traced back to its source run
    valid         - per-sample usability flag, see below

TEN OF THE 583 DRIVE-CYCLE FILES HAVE A BROKEN TEMPERATURE CHANNEL
    Found by auditing every file rather than sampling the first few, which is
    how the first pass missed all of it:

      BOOST_NEGPULSE_1S  cycles 7, 24        T stored as a 0-d NaN scalar,
                                             not a series - this crashed the
                                             first cache build
      CC                 cycles 1501, 1518   T oscillates about zero for the
      BOOST              cycles 1463, 1480   WHOLE file (718 distinct values in
                                             -0.57..0.43) - a disconnected
                                             thermocouple, not a cold cell
      BOOST_NEGPULSE     cycles 488, 505     same, but only the first ~20 % of
                                             the file; the rest reads 25-33 C
      CC                 cycles 1388, 1405   T entirely NaN

    Voltage and current are normal in all of them, so the runs are real; only
    the temperature channel is lost. Temperature is a model input, so those
    samples cannot be used.

    THE RULE, NOT A FILE LIST. A sample is valid when T is finite and inside
    T_MIN..T_MAX. A file is dropped outright when too little of it survives;
    otherwise the mask is stored and window enumeration skips any window that
    touches an invalid sample.

    A PLAIN THRESHOLD IS NOT ENOUGH, because a dying thermocouple does not read
    zero - it wanders. In BOOST_NEGPULSE cycle 488 the dead stretch oscillates
    between 9.71 and 10.36 C, so a 10 C floor cut through the middle of the
    defect and passed 398 of its samples through as valid. Tightening the number
    only moves the cut: the genuine minimum across all six cells is 16.4 C
    (CC_CELL2), leaving almost no room, and only BOOST_NEGPULSE has any sample
    at all below 16 C.

    So the mask is DILATED by DILATE_S seconds either side of every invalid
    sample. Values recorded next to a sensor failure are not trustworthy
    whatever number they happen to show, and this removes the transition without
    having to guess where the defect ends. Cost is about 0.6 % of one cell.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data30t import Q_RATED_AH, load_uypydj_mat, uypydj_members  # noqa: E402
from uypydj_hppc_resistance import PROTOCOLS, soh_anchors  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "UYPYDJ")
CACHE = os.path.join(os.path.dirname(__file__), "cache")

T_MIN, T_MAX = 10.0, 60.0     # plausible cell temperature in a 25 C chamber
MIN_VALID_FRAC = 0.5          # below this the file is dropped entirely
DILATE_S = 300                # invalidate this far either side of a bad sample


def _dilate(bad, radius):
    """True wherever any sample within +/-radius is bad. O(n), file-local."""
    n = len(bad)
    c = np.concatenate([[0], np.cumsum(bad)])
    lo = np.maximum(0, np.arange(n) - radius)
    hi = np.minimum(n, np.arange(n) + radius + 1)
    return (c[hi] - c[lo]) > 0


def _hppc_soc(r, cap):
    """Rated SOC for an HPPC run, from its own Ah counter.

    HPPC files carry no CAP/SOH/SOC, so the published-column problem does not
    arise here - but neither does a ready-made axis. Verified across all 287
    HPPC runs: every one starts at Ah = 0.0000 (cell full) and ends between
    -3.01 Ah when fresh and -2.10 Ah when aged, so the counter is a depth from
    full and SOC = 1 + Ah/3.0 applies directly.

    The rising floor is the physics, not an error: an aged cell hits its voltage
    limit at rated SOC ~0.30 rather than ~0. That is the same gap measured in
    findings.md section 4.3, and it must NOT be normalised away.
    """
    ah = np.asarray(r["Ah"], dtype=float)
    if ah.ndim == 0 or len(ah) < 100 or abs(ah[0]) > 0.05:
        return None
    return 1.0 + ah / Q_RATED_AH


def build_cell(key, zips, part, out_path):
    # Current is stored even though the voltage model does not take it: the
    # context encoder reads measured (V, I, T) to infer how this cell is
    # responding, and deriving I from P/V is wrong wherever the cell is resting.
    # TIME IS STORED. It was not, and that was a real defect rather than an
    # omission: HPPC sampling is non-uniform - about 101 points inside a 10 s
    # pulse, 1 Hz between them, 60 s inside long rests - so a "200-sample"
    # window is 200 s on a drive cycle and anywhere from 20 s to hours on an
    # HPPC record. Any exponentially-weighted feature, any dt in a filter, and
    # any claim that a context window covers N seconds needs the real timestamps.
    cols = {k: [] for k in ("t", "SOC", "T", "P", "V", "I", "SOH", "CAP",
                            "file_id", "valid")}
    files, skipped = [], []
    # HPPC runs have no SOH of their own; it is joined from THIS cell's
    # drive-cycle series by cycle number, never extrapolated past its ends.
    is_hppc = "HPPC" in part
    if is_hppc:
        acyc, asoh, acap = soh_anchors(zips)
        if len(acyc) < 2:
            return None, [(key, "SOH 앵커 부족")]
    for zp in zips:
        if not os.path.exists(zp):
            skipped.append((os.path.basename(zp), "없음"))
            continue
        for member, info in uypydj_members(zp, part=part):
            try:
                r = load_uypydj_mat(zp, member)
            except Exception as e:                      # noqa: BLE001
                skipped.append((info["file"], type(e).__name__))
                continue
            soc = r["SOC"]
            soh, cap = r["SOH"], r["CAP_Ah"]
            if is_hppc:
                c0 = info["cycle"]
                if c0 is None or c0 < acyc.min() or c0 > acyc.max():
                    skipped.append((info["file"], f"cycle {c0} 앵커 범위 밖"))
                    continue
                soc = _hppc_soc(r, float(np.interp(c0, acyc, acap)))
                if soc is None:
                    skipped.append((info["file"], "Ah가 만충에서 시작하지 않음"))
                    continue
                c = info["cycle"]
                if c is None or c < acyc.min() or c > acyc.max():
                    skipped.append((info["file"], f"cycle {c} 앵커 범위 밖"))
                    continue
                soh = float(np.interp(c, acyc, asoh))
                cap = float(np.interp(c, acyc, acap))
            if soc is None:
                skipped.append((info["file"], "SOC 없음"))
                continue
            n = len(r["V"])

            # T may be a 0-d scalar instead of a series in a few files.
            T = np.asarray(r["T"], dtype=float)
            if T.shape == ():
                skipped.append((info["file"], f"T가 스칼라({T:.3g})"))
                continue
            if len(T) != n:
                skipped.append((info["file"], f"T 길이 불일치 {len(T)}!={n}"))
                continue
            ok = np.isfinite(T) & (T > T_MIN) & (T < T_MAX)
            if not ok.all():
                ok &= ~_dilate(~ok, DILATE_S)
            frac = ok.mean()
            if frac < MIN_VALID_FRAC:
                skipped.append((info["file"], f"온도채널 사망 (유효 {frac*100:.0f}%)"))
                continue
            if frac < 1.0:
                skipped.append((info["file"], f"온도 일부 무효 ({(~ok).sum():,}샘플)"))
            fid = len(files)
            files.append(f"{info['cycle']}|{info['file']}")
            cols["t"].append(np.asarray(r["t"], dtype=float))
            cols["SOC"].append(soc); cols["T"].append(r["T"])
            cols["P"].append(r["P"]); cols["V"].append(r["V"])
            cols["I"].append(r["I"])
            cols["SOH"].append(np.full(n, soh if soh is not None else np.nan))
            cols["CAP"].append(np.full(n, cap if cap is not None else np.nan))
            cols["file_id"].append(np.full(n, fid, dtype=np.int32))
            cols["valid"].append(ok)
    if not files:
        return None, skipped
    # Time is float64: within a run it reaches 1e5 s and float32 would quantise
    # to ~8 ms there, which is coarser than the 0.1 s HPPC pulse sampling.
    dt = {"file_id": np.int32, "valid": bool, "t": np.float64}
    arr = {k: np.concatenate(v).astype(dt.get(k, np.float32))
           for k, v in cols.items()}

    # Seven of the 575 kept runs carry no SOH/CAP at all (1.16 % of samples).
    # Every one of them sits inside the cell's own anchor range and several share
    # a cycle number with a run that DOES carry it, so the value is filled by
    # interpolating this cell's cycle -> SOH curve. Never across cells: their
    # aging trajectories differ by up to 1.58x in resistance at equal SOH
    # (findings.md section 4.1), so a neighbouring cell is not evidence here.
    cyc = np.array([int(f.split("|")[0]) for f in files], dtype=float)
    filled = []
    for name, col in (("SOH", "SOH"), ("CAP", "CAP")):
        per = np.array([np.nanmedian(arr[col][arr["file_id"] == k])
                        if np.isfinite(arr[col][arr["file_id"] == k]).any() else np.nan
                        for k in range(len(files))])
        ok = np.isfinite(per)
        if ok.all() or ok.sum() < 2:
            continue
        per[~ok] = np.interp(cyc[~ok], cyc[ok], per[ok])
        for k in np.flatnonzero(~ok):
            arr[col][arr["file_id"] == k] = per[k]
            filled.append((name, files[k].split("|")[0], round(float(per[k]), 4)))
    if filled:
        skipped.append(("(SOH/CAP 보간)", f"{len(filled)}건: {filled[:4]}"))
    # Sample counts per file, so window enumeration can respect file boundaries
    # without re-deriving them from file_id at load time.
    lens = np.bincount(arr["file_id"], minlength=len(files)).astype(np.int64)
    np.savez(out_path, files=np.array(files), lens=lens, **arr)
    return arr, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--part", default="Fifteen_Drive_Cycles")
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even when the output exists.  Without this "
                         "the script skips, so repro/run.py --force does not "
                         "actually force anything here - a 'forced' rebuild "
                         "would silently reuse whatever was already on disk.")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    keys = [args.only] if args.only else list(PROTOCOLS)
    total = 0
    for key in keys:
        out = os.path.join(args.cache, f"uypydj_{key}_{args.part}.npz")
        if os.path.exists(out) and not args.force:
            print(f"{key}: 이미 있음 - 건너뜀 ({os.path.getsize(out)/1e6:.0f} MB)"
                  f"  [--force 로 다시 짓는다]")
            continue
        t0 = time.time()
        zips = [os.path.join(args.raw, z) for z in PROTOCOLS[key]]
        arr, skipped = build_cell(key, zips, args.part, out)
        if arr is None:
            print(f"{key}: 파일 없음 - 건너뜀 ({skipped[:2]})")
            continue
        n = len(arr["V"])
        total += n
        soh = arr["SOH"][np.isfinite(arr["SOH"])]
        print(f"{key}: {len(np.unique(arr['file_id']))}개 파일, {n:,} 샘플, "
              f"{os.path.getsize(out)/1e6:.0f} MB, {time.time()-t0:.0f}s")
        vt = arr["T"][arr["valid"]]
        print(f"    SOC {arr['SOC'].min():.3f}~{arr['SOC'].max():.3f}   "
              f"SOH {soh.min():.3f}~{soh.max():.3f}   "
              f"T {vt.min():.1f}~{vt.max():.1f}C   "
              f"유효 {arr['valid'].mean()*100:.2f}%")
        for f, why in skipped:
            print(f"    - {f[:58]}: {why}")
    if total:
        print(f"\n합계 {total:,} 샘플")


if __name__ == "__main__":
    main()
