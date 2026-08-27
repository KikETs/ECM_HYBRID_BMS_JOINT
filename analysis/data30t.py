"""Loaders for the Samsung INR21700-30T sources, on a single agreed SOC axis.

SOC DEFINITION - the one decision everything else rests on
    SOC(t) = 1.0 + Ah(t) / Q_RATED,  Q_RATED = 3.0 Ah, FIXED.

    Not the per-temperature or per-age measured capacity. A cold cell really
    does deliver less (2.35 Ah at -20 degC against 2.945 at 25 degC, from the
    Mendeley CC-discharge table), and an aged cell less again - normalising by
    the achieved capacity would fold temperature and state-of-health INTO the
    SOC axis and make them unrecoverable. The BYD half of this project was bitten
    by exactly that: the published SOC column divided by per-test achieved
    capacity, two nominally identical files sat on axes 20 % apart, and their OCV
    curves looked 106.8 mV different until the axis was rebuilt against rated.

    Consequence, stated rather than hidden: the SOC RANGE COVERED differs by
    temperature. A 25 degC drive cycle ends near SOC 0.07, a -20 degC one near
    0.26, because the protocol stops at 95 % of that temperature's 1C capacity.
    That is a fact about the cell, not an artifact to normalise away.

SIGN CONVENTION (verified in every source, not assumed)
    Negative current = discharge; Ah decreases on discharge. Same in the
    Mendeley .mat files, the RPCWBY CSVs and the UYPYDJ .mat files.

SAMPLING
    Raw drive cycles are logged at ~0.1 s. The reference work operates at a 1 Hz
    timestep (its missing-sample study is defined per 1 Hz sample) and uses a
    200 s sequence window, which is 200 samples at 1 Hz but 2000 at 0.1 s.
    resample_1hz() therefore downsamples before windowing.

KNOWN DEFECT
    meas.Wh in the Mendeley files is stored as uint8 with values 44/45 - it
    cannot be watt-hours. The field is never read here. Integrate Power over
    Time if energy is needed.
"""
from __future__ import annotations

import glob
import io
import os
import re

import numpy as np
import scipy.io as sio

Q_RATED_AH = 3.0
MENDELEY_TEMPS = (-20, -10, 0, 10, 25, 40)
DRIVE_CYCLES = ("UDDS", "HWFET", "LA92", "US06",
                "Mixed1", "Mixed2", "Mixed3", "Mixed4",
                "Mixed5", "Mixed6", "Mixed7", "Mixed8")


def _field(struct, *names):
    """Fetch the first field that exists - UYPYDJ spells it Battery_temp_DegC
    while Mendeley spells it Battery_Temp_degC, and a KeyError here would be a
    silent temperature drop-out rather than an obvious failure."""
    for n in names:
        if n in struct._fieldnames:
            return np.asarray(getattr(struct, n), dtype=float)
    raise KeyError(f"none of {names} in {struct._fieldnames}")


def load_meas_mat(path: str) -> dict:
    """Read one meas-schema .mat into plain arrays with SOC attached."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    s = m[key]
    t = _field(s, "Time")
    ah = _field(s, "Ah")
    out = {
        "t": t,
        "V": _field(s, "Voltage"),
        "I": _field(s, "Current"),
        "P": _field(s, "Power"),
        "T": _field(s, "Battery_Temp_degC", "Battery_temp_DegC", "Battery_Temp_DegC"),
        "Ah": ah,
        "SOC": 1.0 + ah / Q_RATED_AH,
        "path": path,
    }
    return out


def resample_1hz(d: dict) -> dict:
    """Downsample to a uniform 1 s grid by interpolation.

    The raw grid is non-uniform (median 0.1 s, occasional multi-second gaps), so
    a plain stride would alias. Interpolating onto an even grid also makes the
    sequence window a fixed wall-clock length, which is what the window-length
    hyperparameter is supposed to mean.
    """
    t = d["t"]
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0)
    out = {"t": grid, "path": d["path"]}
    for k in ("V", "I", "P", "T", "Ah", "SOC"):
        out[k] = np.interp(grid, t, d[k])
    return out


def parse_name(path: str) -> dict:
    """Pull temperature and test tag out of the Kollmeyer filename convention."""
    base = os.path.basename(path)
    mt = re.search(r"(-?\d+)degC", base)
    tag = re.sub(r"^\d\d-\d\d-\d\d_[\d.]+\s*\d*_", "", base)
    tag = re.sub(r"_-?\d+degC_IN21700_30T\.mat$", "", tag)
    return {"temp_C": int(mt.group(1)) if mt else None, "tag": tag, "file": base}


def mendeley_files(root: str, kinds=DRIVE_CYCLES, temps=MENDELEY_TEMPS) -> list:
    """All Mendeley .mat paths matching the requested test kinds/temperatures."""
    found = []
    for temp in temps:
        for p in sorted(glob.glob(os.path.join(root, f"{temp}degC", "*.mat"))):
            info = parse_name(p)
            if any(info["tag"] == k for k in kinds):
                found.append((p, info))
    return found


def summarise(d: dict) -> str:
    return (f"{len(d['t']):6d}pt  {d['t'][-1]/3600:5.2f}h  "
            f"V {d['V'].min():.3f}-{d['V'].max():.3f}  "
            f"I {d['I'].min():+7.2f}/{d['I'].max():+6.2f}A  "
            f"SOC {d['SOC'].min():.3f}-{d['SOC'].max():.3f}  "
            f"T {d['T'].min():.1f}-{d['T'].max():.1f}C")


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "raw", "Mendeley")
    files = mendeley_files(root)
    print(f"drive-cycle files: {len(files)}")
    bad = 0
    for p, info in files:
        try:
            d = resample_1hz(load_meas_mat(p))
        except Exception as e:                      # noqa: BLE001
            print(f"  FAIL {info['file']}: {e}")
            bad += 1
            continue
        print(f"  {info['temp_C']:>4}C {info['tag']:<8} {summarise(d)}")
    print(f"\n{len(files)-bad} ok, {bad} failed")


# ---------------------------------------------------------------------------
# RPCWBY Test#3 - SOP measurement runs (2 / 10 / 30 s pulses, six temperatures)
# ---------------------------------------------------------------------------
import csv as _csv
import hashlib as _hashlib
import zipfile as _zipfile

TEST3_ZIP = "3_Test_3.zip"


def _t3_meta(name: str) -> dict:
    """Temperature and pulse length from a Test#3 filename.

    The archive spells negative temperatures BOTH ways - 'n10degC' and
    '-10degC' - and ships one pair of each as byte-identical duplicates, so the
    tag has to be normalised and the content de-duplicated or those runs get
    counted twice.
    """
    base = name.split("/")[-1]
    mt = re.search(r"(n?-?\d+)degC", base)
    temp = None
    if mt:
        temp = int(mt.group(1).replace("n", "-").replace("--", "-"))
    ml = re.search(r"_(\d+)s[_.]", base)
    mc = re.search(r"Channel_(\d+)", base)
    return {"file": base, "temp_C": temp,
            "pulse_s": int(ml.group(1)) if ml else None,
            "channel": mc.group(1) if mc else None}


def test3_index(zip_path: str) -> list:
    """List the unique Test#3 CSVs with their metadata, duplicates removed."""
    z = _zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist()
             if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
    seen, out = {}, []
    for n in sorted(names):
        h = _hashlib.sha256(z.read(n)).hexdigest()
        if h in seen:                       # byte-identical duplicate
            continue
        seen[h] = n
        out.append((n, _t3_meta(n)))
    return out


def load_test3_csv(zip_path: str, member: str) -> dict:
    """Read one Test#3 run. Its SOC column was VERIFIED against the documented
    formula - back-solving gives Q = 3.0000 Ah at R^2 = 1.000000, i.e. rated,
    matching this module's own SOC axis. The BYD half of the project found the
    same field normalised by achieved capacity instead, so this was checked
    rather than trusted."""
    z = _zipfile.ZipFile(zip_path)
    with z.open(member) as f:
        rd = _csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
        rows = list(rd)

    bad_cells = []

    def col(*names):
        """Numeric column, tolerating stray non-numeric cells.

        Not hypothetical: SOP_30T_July_14_-20degC_10s has the literal string 's'
        in the SOC column at one row out of 38173. A bare float() kills the whole
        run over one cell, and a silent NaN hides a data defect - so the offender
        is recorded and reported instead.
        """
        for k in names:
            if rows and k in rows[0]:
                vals = np.empty(len(rows))
                for i, r in enumerate(rows):
                    v = r.get(k)
                    if v in ("", None):
                        vals[i] = np.nan
                        continue
                    try:
                        vals[i] = float(v)
                    except (TypeError, ValueError):
                        vals[i] = np.nan
                        bad_cells.append((k, i, str(v)[:16]))
                return vals
        return None

    tcol = next((k for k in rows[0] if k.startswith("Aux_Temperature")), None)
    return {
        "t": col("Test_Time(s)"),
        "V": col("Voltage(V)"),
        "I": col("Current(A)"),
        "P": col("Power(W)"),
        "T": col(tcol) if tcol else None,
        "SOC": col("SOC"),
        "step": col("Step_Index"),
        "ud": {k: col(f"MetaCode_MV_UD{k}") for k in range(1, 17)},
        "path": member,
        "bad_cells": bad_cells,
    }


def test3_windows_source(zip_path: str, pulse_lengths=(2, 30), channel=None):
    """Test#3 runs selected by pulse length, as (SOC, T, P) -> V material.

    The reference work splits by PULSE LENGTH to avoid leakage: it trains on the
    2 s and 30 s SOP measurements and keeps the 10 s ones entirely for test. The
    default here follows that. Runs are already logged at 1 Hz, so no resampling.

    Why these runs matter at all: a drive cycle almost never visits the SOP
    current limits, while a SOP measurement is built to sit on them. Measured
    here at 25 degC, only 1.6 % of a Test#3 run's samples lie in the 25-40 A
    discharge band - but that is 1.6 % more than the drive cycles provide.
    """
    out = []
    for member, meta in test3_index(zip_path):
        if meta["pulse_s"] not in pulse_lengths:
            continue
        if channel is not None and meta["channel"] != channel:
            continue
        d = load_test3_csv(zip_path, member)
        if d["T"] is None or d["SOC"] is None:
            continue
        ok = np.isfinite(d["SOC"]) & np.isfinite(d["T"]) & np.isfinite(d["P"]) & np.isfinite(d["V"])
        if ok.sum() < 500:
            continue
        out.append({
            "SOC": d["SOC"][ok], "T": d["T"][ok], "P": d["P"][ok], "V": d["V"][ok],
            "meta": meta,
        })
    return out


# ---------------------------------------------------------------------------
# UYPYDJ - fast-charge aging campaign (six protocols, cells run down to 70 % SOH)
# ---------------------------------------------------------------------------
"""THE SOC COLUMN IN THIS SOURCE IS NOT THE ONE THIS MODULE USES.

The published SOC is normalised by the cell's CURRENT AGED CAPACITY, which the
readme states and the data confirms: back-solving SOC against Ah returns Q equal
to the file's own CAP field (2.9717 / 2.7386 / 2.5976 Ah at cycles 7 / 505 / 993)
at R^2 = 1.000000, never the 3.0 Ah rating.

Why that is fatal HERE specifically, rather than merely untidy: the aging
protocol cycles between 10 % and 80 % OF THE AGED CAPACITY. Measured over 54
drive-cycle files spanning SOH 1.000 -> 0.874, the top-of-charge on the
published axis moves by -0.06 %p (r = 0.033) - the cell looks like it still
charges to 80 % when it no longer holds the charge to do so. Rebuilt against
rated, the same top-of-charge falls 10.03 %p (r = 0.987). The published axis
does not merely obscure state-of-health, it CANCELS it, and this project exists
to model SOH-dependent power.

Conversion, exact and cheap - the empty point is set by the voltage cut-off and
so is the same physical state at any age, which makes the aged fraction and the
rated fraction differ by the capacity ratio alone:

    SOC_rated = (SOC_published / 100) * CAP / Q_RATED

Ah cannot be used directly the way it is for the Mendeley files: the readme
warns the Ah counter is reset after each charge or test, so it is a within-file
relative count, not a depth from full.

TEMPERATURE, a structural limit worth stating before it is discovered late:
this campaign ran in a 25 degC chamber. Cell temperature spans 22-42 degC only
because of self-heating under fast charge. So SOH variation and TEMPERATURE
variation live in DISJOINT sources - Mendeley/RPCWBY give six temperatures at
one age, UYPYDJ gives many ages at one ambient. No file supplies both, and any
claim about cold behaviour of an aged cell is therefore an extrapolation.
"""

UYPYDJ_PROTOCOLS = ("CONSTANT CURRENT", "BOOST CHARGING",
                    "BOOST CHARGING WITH REST",
                    "BOOST CHARGING WITH NEGATIVE PULSES",
                    "BOOST CHARGING WITH NEGATIVE PULSES_1s_PERIOD",
                    "CONSTANT CURRENT SECOND CELL")


def uypydj_members(zip_path: str, part: str = "Fifteen_Drive_Cycles") -> list:
    """Members of one UYPYDJ zip belonging to a schedule part.

    Useful part names: Fifteen_Drive_Cycles (drive cycles, carries SOH/CAP/SOC),
    HPPC (pulse characterisation, every 30 cycles), OCV_0.05C (every 60 cycles),
    halfC / OneC / TwoC (the 0.5C, 1C and 2C capacity discharges).
    """
    z = _zipfile.ZipFile(zip_path)
    out = []
    for n in sorted(z.namelist()):
        if not n.endswith(".mat") or "__MACOSX" in n or part not in n:
            continue
        mc = re.search(r"cycle_#(\d+)", n)
        out.append((n, {"file": n.split("/")[-1],
                        "cycle": int(mc.group(1)) if mc else None}))
    return out


def load_uypydj_mat(zip_path: str, member: str) -> dict:
    """Read one UYPYDJ .mat onto THIS MODULE'S rated SOC axis.

    Returns SOC on the rated axis and keeps the published column as SOC_aged so
    the substitution stays auditable rather than silent. Files that carry no CAP
    field (the pure-characterisation parts) get SOC = None instead of a guess.
    """
    z = _zipfile.ZipFile(zip_path)
    m = sio.loadmat(io.BytesIO(z.read(member)), squeeze_me=True,
                    struct_as_record=False)
    s = m[next(k for k in m if not k.startswith("__"))]
    have = set(s._fieldnames)
    g = lambda f: np.asarray(getattr(s, f), dtype=float)   # noqa: E731

    cap = float(np.unique(g("CAP"))[0]) if "CAP" in have else None
    soh = float(np.unique(g("SOH"))[0]) if "SOH" in have else None
    soc_aged = g("SOC") if "SOC" in have else None
    # B AXIS: anchored at FULL, so SOC = 1 + (charge below full)/3.0.
    #   published SOC is % of the AGED capacity measured up from empty, so the
    #   charge still below full is cap*(1 - SOC_pub/100).
    # The empty-anchored alternative, SOC = SOC_pub/100 * cap/3, is equally self
    # consistent but needs the capacity to place EVERY file - including the
    # 6-temperature Mendeley HPPC, where the capacity is temperature dependent
    # (2.35 Ah at -20 C against 2.945 at 25 C). Dividing by that would fold
    # TEMPERATURE into the SOC axis, which is the exact trap this module's
    # header warns about. Anchored at full, a cold cell simply stops early -
    # the limitation shows up as a restricted range, not a rescaled axis.
    soc = (1.0 - cap * (1.0 - soc_aged / 100.0) / Q_RATED_AH
           if soc_aged is not None and cap else None)

    return {
        "t": _field(s, "Time"),
        "V": _field(s, "Voltage"),
        "I": _field(s, "Current"),
        "P": _field(s, "Power"),
        "T": _field(s, "Battery_Temp_degC", "Battery_temp_DegC", "Battery_Temp_DegC"),
        "Ah": _field(s, "Ah"),          # within-file relative count - see note above
        "SOC": soc,                     # rated axis, this module's convention
        "SOC_aged": soc_aged,           # as published, kept for audit only
        "CAP_Ah": cap,
        "SOH": soh,
        "path": member,
    }
