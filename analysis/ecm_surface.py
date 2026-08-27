"""Continuous ECM parameter surfaces theta(SOC, SOH, |I|) for the Kalman filter.

WHY A SURFACE AND NOT A LOOKUP
    uypydj_ecm.csv holds fitted parameters at the SOC/SOH/current points the HPPC
    protocol happened to visit. A filter needs them at whatever state it is in,
    every timestep, so the scattered fits have to become a callable field.

PER CELL, NEVER AVERAGED ACROSS CELLS
    At SOH 0.75 the six cells' charge-transfer resistance R1 spans 4.79-26.75
    mOhm - a 5.58x spread - while their ohmic R0 agrees within 1.06x
    (findings.md section 4.3.1). A surface built from the pooled six would sit
    between cells and match none of them. Each cell gets its own surface; the
    dual filter's parameter branch is what adapts to an unseen cell online.

CURRENT IS A REAL AXIS, NOT A NUISANCE
    R1 and R2 fall to about 0.70x and 0.66x going from 2.6 A to 29.6 A - the
    Butler-Volmer nonlinearity - while R0 moves 1.02x, i.e. not at all. So the
    surface is built per rate rank and interpolated in |I| between ranks. Using
    a single low-rate R would overstate resistance at the currents SOP cares
    about.

TEMPERATURE COMES FROM A SECOND DATASET, AS A MULTIPLIER
    The UYPYDJ campaign ran at one chamber temperature, so its theta carries no
    temperature axis. Mendeley's HPPC ran at six temperatures on a fresh cell,
    and ecm_temp_factor.csv reduces it to g(SOC, |I|, T) = R(T)/R(25 degC).
    Applied here as

        R_k(SOC, SOH, T, I) = R_k,25(SOC, SOH, I) * g(SOC, |I|, T)

    to R0, R1 and R2 alike. Time constants are not corrected - measured tau moves
    non-monotonically and weakly with temperature, so there is no defensible
    trend to apply.

    THIS ASSUMES g DOES NOT DEPEND ON SOH, which is an assumption imported from a
    fresh cell and applied to an aged one. It is not left as an assumption:
    RPCWBY Test#1/#2 measure the same ratio at 10 and 25 degC across the whole
    aging range, and ecm_validate_temp.py checks the prediction against them.

EXTRAPOLATION IS FLAGGED, NOT HIDDEN
    Two regions have no data: SOH below each cell's own end point, and low SOC at
    low SOH (HPPC stops at SOC 0.29 by SOH 0.70, findings.md section 4.4) - which
    is exactly where SOP is tightest. Queries there return the nearest-neighbour
    value AND in_hull=False, so a caller can refuse to report them.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy.interpolate import (LinearNDInterpolator,  # noqa: E402
                               NearestNDInterpolator)

HERE = os.path.dirname(os.path.abspath(__file__))
ECM_CSV = os.path.join(HERE, "uypydj_ecm.csv")
OCV_CSV = os.path.join(HERE, "uypydj_ocv.csv")
TEMP_CSV = os.path.join(HERE, "ecm_temp_factor.csv")
T_REF_C = 25.0

PARAMS = ("R0_mOhm", "R1_mOhm", "tau1_s", "R2_mOhm", "tau2_s")


class _Field:
    """Linear interpolation over (SOC, SOH) with a nearest fallback + hull flag."""

    def __init__(self, pts, vals):
        self.lin = LinearNDInterpolator(pts, vals)
        self.near = NearestNDInterpolator(pts, vals)

    def __call__(self, soc, soh):
        q = np.column_stack([np.atleast_1d(soc), np.atleast_1d(soh)])
        v = self.lin(q)
        inside = np.isfinite(v).all(axis=-1) if v.ndim > 1 else np.isfinite(v)
        if not inside.all():
            v = np.where(np.isfinite(v), v, self.near(q))
        return v, inside


class TempFactor:
    """g(SOC, |I|, T) from the measured table, linear in T between measurements.

    Outside the measured temperature span the nearest measured factor is held
    and out_of_range is set - an Arrhenius extrapolation is available in the
    table's Ea column but is deliberately not applied silently, because Ea
    itself moves from 28.7 kJ/mol over -20..0 degC to 14.3 over 10..40.
    """

    def __init__(self, path=TEMP_CSV):
        self.rows = list(csv.DictReader(open(path, encoding="utf-8")))
        for r in self.rows:
            for k in ("SOC_lo", "SOC_hi", "I_lo", "I_hi", "T_cell_C", "g"):
                r[k] = float(r[k])

    def __call__(self, soc, current_A, T_cell_C):
        i = abs(float(current_A))
        cell = [r for r in self.rows
                if r["SOC_lo"] <= soc < r["SOC_hi"] and r["I_lo"] <= i < r["I_hi"]]
        if not cell:                      # fall back on the SOC bin at any current
            cell = [r for r in self.rows if r["SOC_lo"] <= soc < r["SOC_hi"]]
        if not cell:
            return 1.0, False
        cell.sort(key=lambda r: r["T_cell_C"])
        ts = np.array([r["T_cell_C"] for r in cell])
        gs = np.array([r["g"] for r in cell])
        inside = ts.min() <= T_cell_C <= ts.max()
        return float(np.interp(T_cell_C, ts, gs)), inside


class ECMSurface:
    """theta(SOC, SOH, |I|) and OCV(SOC, SOH) for ONE cell."""

    def __init__(self, cell, direction="discharge",
                 ecm_csv=ECM_CSV, ocv_csv=OCV_CSV, temp_csv=TEMP_CSV):
        self.cell = cell
        self.direction = direction
        self.gfac = TempFactor(temp_csv) if os.path.exists(temp_csv) else None

        rows = [r for r in csv.DictReader(open(ecm_csv, encoding="utf-8"))
                if r["cell"] == cell and r["direction"] == direction]
        if not rows:
            raise ValueError(f"no {direction} ECM rows for {cell}")

        # One field per rate rank, plus that rank's representative current.
        self.ranks, self.rank_I = [], []
        for rk in sorted({r["rate_rank"] for r in rows}, key=int):
            sub = [r for r in rows if r["rate_rank"] == rk]
            if len(sub) < 8:
                continue
            pts = np.array([[float(r["SOC"]), float(r["SOH"])] for r in sub])
            vals = np.array([[float(r[p]) for p in PARAMS] for r in sub])
            self.ranks.append(_Field(pts, vals))
            self.rank_I.append(float(np.median([abs(float(r["I_A"])) for r in sub])))
        self.rank_I = np.array(self.rank_I)

        # OCV: curves at discrete SOH, interpolated in both axes.
        o = [r for r in csv.DictReader(open(ocv_csv, encoding="utf-8"))
             if r["cell"] == cell]
        if not o:
            raise ValueError(f"no OCV rows for {cell}")
        pts = np.array([[float(r["SOC"]), float(r["SOH"])] for r in o])
        vals = np.array([[float(r["OCV_V"])] for r in o])
        self.ocv_field = _Field(pts, vals)

        # HYSTERESIS MAGNITUDE, which the pseudo-OCV construction deliberately
        # removed. OCV(SOC) here is the AVERAGE of the charge and discharge
        # sweeps, so half their gap is how far the real curve sits either side
        # of it. That gap is not a nuisance: measured on this cell it grows from
        # 12.8 mV at SOH 0.99 to 76.3 mV at 0.76, which is the same size as the
        # open-loop model error at those ages. An RC branch cannot represent it,
        # because hysteresis persists at zero current while an RC decays.
        hv = np.array([[abs(float(r["hyst_mV"])) / 2000.0] for r in o])
        self.hyst_field = _Field(pts, hv)

        # Capacity against SOH, from the same cell's own CAP column.
        cap = sorted({(float(r["SOH"]), float(r["CAP_Ah"])) for r in rows})
        a = np.array(cap)
        self._soh, self._cap = a[:, 0], a[:, 1]

        sohs = [float(r["SOH"]) for r in rows]
        self.soh_range = (min(sohs), max(sohs))

    # -- queries -------------------------------------------------------------
    def theta(self, soc, soh, current_A, T_cell_C=T_REF_C):
        """{R0, R1, tau1, R2, tau2} in ohms/seconds, plus in_hull.

        Resistances are returned in OHMS - the csv stores mOhm and forgetting
        that factor is a 1000x error that still looks plausible on a plot.
        """
        i = float(np.abs(current_A))
        vs, ins = [], []
        for fld in self.ranks:
            v, ok = fld(soc, soh)
            vs.append(np.atleast_2d(v)); ins.append(ok)
        V = np.stack(vs)                      # (rank, N, 5)
        # Interpolate across ranks at |I|, clamped to the measured ladder.
        w = np.interp(i, self.rank_I, np.arange(len(self.rank_I)))
        lo, hi = int(np.floor(w)), int(np.ceil(w))
        frac = w - lo
        M = V[lo] * (1 - frac) + V[hi] * frac
        out = {p: M[:, k] for k, p in enumerate(PARAMS)}
        g, g_ok = 1.0, True
        if self.gfac is not None and abs(T_cell_C - T_REF_C) > 0.05:
            g, g_ok = self.gfac(float(np.atleast_1d(soc)[0]), i, T_cell_C)
        return {
            "R0": out["R0_mOhm"] / 1000.0 * g,
            "R1": out["R1_mOhm"] / 1000.0 * g,
            "tau1": out["tau1_s"],
            "R2": out["R2_mOhm"] / 1000.0 * g,
            "tau2": out["tau2_s"],
            "g_temp": g,
            "temp_in_range": g_ok,
            "in_hull": np.logical_and(ins[lo], ins[hi]),
            "rate_clamped": bool(i < self.rank_I.min() or i > self.rank_I.max()),
        }

    def d_tau(self, soc, soh, current_A, taus, T_cell_C=T_REF_C):
        """등가저항 D(tau) = R0 + R1(1-e^-tau/t1) + R2(1-e^-tau/t2).

        WHY THIS EXISTS INSTEAD OF LETTING THE CALLER COMBINE theta()
            theta() 는 다섯 파라미터를 rank(전류) 축에서 각각 선형 보간한 뒤
            돌려준다. 그것을 지수에 넣어 D 를 만들면 물리적으로 없는 값이 나온다.

            측정된 예: SOC 0.95, SOH 0.72 의 충전 면에서 rank 0 은 (R2 23.3 mOhm,
            tau2 5.71 s), rank 1 은 (R2 460 mOhm, tau2 3000 s) 다. rank 1 의
            적합은 **발산한 것**이다 — tau2 가 상한에 붙으면 10 s 창에서
            1-e^(-10/3000) = 0.0033 이라 R2 가 460 이든 0 이든 D10 에 기여하지
            않는다. R2 와 tau2 가 서로를 상쇄하며 값이 정해지지 않는다.

            그 둘을 따로 보간하면 1.6 A 에서 (64 mOhm, 285 s) 라는, 어느 rank 에도
            없고 근거도 없는 조합이 나온다. 두 발산한 적합의 중간이 발산하지 않은
            값처럼 보이는 것뿐이다. 그 결과 D10 이 49.0 -> 32.8 mOhm 으로 꺼지고,
            전류가 커질 때 저항이 **오르는** 비물리적 구간이 생긴다(측정은 52.2 ->
            41.3 -> 29.6 mOhm 으로 단조 감소).

            D 는 각 rank 에서 잘 정의된 양이다. rank 마다 D 를 먼저 만들고 그것을
            보간하면 발산은 그 rank 안에 갇힌다.
        """
        i = float(np.abs(current_A))
        taus = np.atleast_1d(np.asarray(taus, float))
        Ds, ins = [], []
        for fld in self.ranks:
            v, ok = fld(soc, soh)
            v = np.atleast_2d(v)
            R0, R1, t1, R2, t2 = (v[:, k] for k in range(5))
            Ds.append(np.stack([R0 + R1 * (1 - np.exp(-t / t1))
                                + R2 * (1 - np.exp(-t / t2)) for t in taus], -1))
            ins.append(ok)
        D = np.stack(Ds)                      # (rank, N, tau)
        w = np.interp(i, self.rank_I, np.arange(len(self.rank_I)))
        lo, hi = int(np.floor(w)), int(np.ceil(w))
        frac = w - lo
        out = (D[lo] * (1 - frac) + D[hi] * frac) / 1000.0
        g = 1.0
        if self.gfac is not None and abs(T_cell_C - T_REF_C) > 0.05:
            g, _ = self.gfac(float(np.atleast_1d(soc)[0]), i, T_cell_C)
        return out * g, np.logical_and(ins[lo], ins[hi])

    def ocv(self, soc, soh):
        v, ok = self.ocv_field(soc, soh)
        return np.atleast_2d(v)[:, 0], ok

    def hyst_M(self, soc, soh):
        """Half the charge-discharge gap, in volts."""
        v, ok = self.hyst_field(soc, soh)
        return float(np.atleast_2d(v)[0, 0]), ok

    def capacity(self, soh):
        return float(np.interp(soh, self._soh, self._cap))


def available_cells(ecm_csv=ECM_CSV):
    return sorted({r["cell"] for r in csv.DictReader(open(ecm_csv, encoding="utf-8"))})


if __name__ == "__main__":
    cells = available_cells()
    print(f"셀: {cells}\n")
    s = ECMSurface("CC")
    print(f"  rank 대표 전류: {s.rank_I.round(1)} A\n")
    print(f"  {'SOC':>5} {'SOH':>6} {'|I|':>5} {'T':>5} {'R0':>8} {'R1':>8} "
          f"{'R2':>8} {'g':>6} {'OCV':>7} {'hull':>6}")
    for soc, soh, cur, T in ((0.5, 0.99, 3, 25), (0.5, 0.99, 30, 25),
                             (0.5, 0.80, 30, 25), (0.5, 0.80, 30, 10),
                             (0.5, 0.80, 30, -10), (0.5, 0.80, 30, 40)):
        t = s.theta(soc, soh, cur, T)
        v, _ = s.ocv(soc, soh)
        print(f"  {soc:>5.2f} {soh:>6.2f} {cur:>5.0f} {T:>5.0f} "
              f"{t['R0'][0]*1e3:>7.2f}m {t['R1'][0]*1e3:>7.2f}m "
              f"{t['R2'][0]*1e3:>7.2f}m {t['g_temp']:>6.3f} {v[0]:>7.4f} "
              f"{str(bool(t['in_hull'][0])):>6}")
    print(f"\n  용량: SOH 1.00 -> {s.capacity(1.0):.3f} Ah, "
          f"SOH 0.75 -> {s.capacity(0.75):.3f} Ah")
