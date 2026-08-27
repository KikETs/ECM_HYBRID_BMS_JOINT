"""Extended Kalman filter for SOC on the ECM built from the three datasets.

STATE AND MEASUREMENT
    x = [SOC, V1, V2]

    SOC_{k+1} = SOC_k + I_k dt / (3600 Q_RATED)
    V1_{k+1}  = a1 V1_k + R1 (1 - a1) I_k        a1 = exp(-dt/tau1)
    V2_{k+1}  = a2 V2_k + R2 (1 - a2) I_k
    y_k       = OCV(SOC_k, SOH) + I_k R0 + V1_k + V2_k

    Q_RATED, not the aged capacity: this project's SOC axis is anchored at FULL
    and divided by 3.0 Ah, so an aged cell simply cannot reach SOC 0. Dividing
    by the aged capacity here would silently move the axis - see data30t.py.

    Sign convention: negative current is discharge, so I R0 pulls the terminal
    voltage below OCV without any sign flip in the equations.

WHY THE MEASUREMENT NOISE IS SET SO LARGE
    R is not the voltage sensor's noise, which is under a millivolt. It has to
    absorb the MODEL error, and that error is known and grows with age: open
    loop, this ECM predicts drive-cycle voltage to 13.7 mV at SOH 1.00 and
    108 mV at 0.70 (ecm_simulate.py). Four alternative fits and a drive-cycle
    refit all failed to reduce it - it is a 2RC structural limit, not a bad
    parameter. Setting R optimistically would make the filter trust a model that
    is wrong by 100 mV and drag SOC with it, so R is scaled to the measured
    open-loop error at the cell's SOH.

HYSTERESIS IS PART OF THE MEASUREMENT, AND LEAVING IT OUT BIASES SOC
    The open-loop model this filter is built on carries a Plett one-state
    hysteresis term, and adding it cut drive-cycle error from 50.62 to 37.13 mV
    (ecm_simulate.py). The first version of this filter omitted it, and the
    omission is visible in the residual: its correlation with the hysteresis
    state runs +0.18 at SOH 1.00 and +0.80 at SOH 0.69. The Kalman gain has to
    put that residual somewhere and it puts it into SOC.

    The damage is a bias, not scatter - measured median |error| equals |bias| to
    three digits in every SOC band - and it is worst at SOC 0.3-0.4 (-0.034),
    which is exactly where the pooled R_eff falls 26 % and where the SOP answer
    is most sensitive to SOC. Sign is consistently negative: during discharge
    h < 0 pulls the true voltage below a model that has no h, and the filter
    reads the gap as "less charge than I thought".

WHAT THE TEST IS
    Not "does it track when started correctly" - coulomb counting does that. The
    filter is started from a DELIBERATELY WRONG SOC and has to pull itself in
    using voltage alone. Convergence time and residual error are what matter.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecm_surface import ECMSurface  # noqa: E402

Q_RATED_AH = 3.0


class EKF:
    """State [SOC, V1, V2] or [SOC, V1, V2, h] when gamma > 0.

    h is bounded in [-1, 1] and propagates deterministically from the current, so
    its process noise is small - it is in the state to be CORRECTED at start-up
    (where the true h is unknown and the filter assumes 0), not because it
    wanders. Its initial covariance is 1.0 for the same reason: h could be
    anywhere in its range when the key turns.
    """

    def __init__(self, sd, sc, soh, R_volt=0.02, q_soc=1e-9, q_v=1e-7, dt=1.0,
                 gamma=0.0, q_h=1e-6, estimate_h=False, k_f=1.0, k_s=1.0,
                 i_gate=None, q_b=None, b_on_load=False, q_ib=None, p0_ib=0.25, ib_clip=2.0,
                 tau_h_rest=None,
                 m_cap=None, rest_hold_s=0.0,
                 huber_c=None, r_adapt=None, r_floor_k=None, r_max_mult=None,
                 r_var_k=None, ew_rate=0.01):
        self.sd, self.sc, self.soh, self.dt = sd, sc, soh, dt
        self.gamma = float(gamma)
        # h is FULLY DETERMINED by the current history, so by default it is
        # propagated, not estimated: giving the filter a fourth free state lets
        # it explain residuals that are not hysteresis at all. Estimating it
        # helped at SOC < 0.6 (+30 %) but hurt above 0.6 and diverged on one run
        # (SOC error 0.055 -> 0.314, residual 336 mV). Only the INITIAL value of
        # h is unknown, and under load it forgets that in seconds.
        # The SOP arm already estimates how far this cell's resistance sits from
        # the pooled surface, and the filter has been using the uncorrected
        # surface. That is why R_volt has to be scheduled up to 110 mV at SOH 0.70
        # - the filter is told to distrust a model it could instead be given
        # corrected. k_f scales the fast branch (R0 and R1), k_s the slow one,
        # exactly as in the pulse model.
        self.k_f, self.k_s = float(k_f), float(k_s)
        # The measurement model's error enters through I*R, so it is smallest
        # where the current is smallest - at rest, y is OCV + M*h and carries no
        # resistance error at all. Three separate tests said the aged-cell
        # residual is structural rather than a wrong resistance (own-OCV and
        # own-R substitution both inconsistent, lowering R_volt always worse, the
        # trim's k worth only 5-7 %), so gating the UPDATE to samples the model
        # can be trusted on is the remaining lever. None means never gate.
        self.i_gate = i_gate
        # A current threshold alone is not a rest. Most |I| < 1 A samples in a
        # drive cycle are brief pauses, and with tau2 ~ 8 s the RC branches still
        # hold the previous load's voltage - measured, the residual against
        # OCV + M*h reaches -115 mV at mid SOC on an aged cell, and that is
        # leftover RC, not model error. Requiring the low current to have been
        # HELD lets the branches decay first: 60 s is 7.5 time constants.
        self.rest_hold_s = float(rest_hold_s)
        self.rest_s = 0.0
        # 성공한 세 번은 전부 "언제 믿을지" 를 고른 것이고, 셋 다 **한 문턱을
        # 넘으면 믿고 못 넘으면 버리는** 이진 규칙이다. 아직 안 해본 것은 버리는
        # 대신 **잔차 크기에 따라 신뢰를 연속으로 낮추는** 것이다.
        #   huber_c   잔차가 c 를 넘으면 이득을 |r|/c 로 나눈다 (M-추정)
        #   r_adapt   최근 잔차 분산으로 R_volt 를 갱신 (적응 필터)
        #   r_floor_k R_volt 하한을 잔차 이동평균의 k 배로
        self.huber_c = huber_c
        self.r_adapt = r_adapt
        self.r_floor_k = r_floor_k
        #   r_max_mult  부풀린 R 의 상한을 R_volt 의 몇 배로 (보정이 완전히
        #               멎는 것을 막는다)
        self.r_max_mult = r_max_mult
        #   r_var_k   R 하한을 잔차의 *퍼짐* 의 k 배로.  r_floor_k 와 달리
        #             잔차 이동평균(편향)을 먼저 빼고 남은 흔들림만 본다.
        #             한쪽으로 쏠린 잔차는 모델이 틀렸다는 뜻이므로 R 을
        #             키우면 안 되고, 평균 0 으로 흔들리는 잔차만 키운다.
        self.r_var_k = r_var_k
        self.ew_rate = float(ew_rate)
        self._r_ew = None
        self._i_ew = None
        self._v_ew = 0.0
        self.estimate_h = bool(estimate_h) and self.gamma > 0
        # A constant measurement bias. Four separate substitutions failed to
        # locate the aged-cell error in any single term - resistance, OCV,
        # hysteresis, the label - which leaves "the model is offset" as the
        # remaining description. b states that explicitly instead of letting the
        # gain push it into SOC.
        #
        # b and SOC are NOT separable at one operating point: both move y. They
        # separate as SOC sweeps, because OCV changes with SOC and b does not,
        # and this data's drive cycles run SOC 0.85 -> 0.35. q_b sets how fast b
        # is allowed to move and is the whole risk - too large and it absorbs the
        # SOC information it is meant to protect.
        # b is unobservable where SOC does not move, and the gate leaves exactly
        # that: updates only at rest. Estimating b there let it absorb the
        # deliberate 0.20 initial SOC error and run to its bound (-209 mV, 42 %p
        # on a FRESH cell). Splitting the regimes fixes the observability - under
        # load SOC sweeps 0.85 -> 0.35 so a constant offset separates from it,
        # and at rest the resistance term is gone so SOC is clean.
        #   loaded : update b only
        #   at rest: update SOC, V1, V2 only
        # Plett's h changes only with current, so at rest it FREEZES at whatever
        # the last load left it. A real cell relaxes. That matters far more with
        # age than it does fresh: M grows from 14 mV at SOH 1.00 to 171 mV at
        # 0.70, and the initial value of h alone moves the SOC error by 3.7 %p on
        # an aged cell against 0.4 %p on a fresh one - comparable to the entire
        # residual error. tau_h_rest lets h decay toward zero while the cell is
        # not being driven; None keeps the original frozen behaviour.
        # The stored hysteresis width is not hysteresis at low SOC and low SOH.
        # uypydj_ocv.csv reaches 666 mV, with a median of 184 mV in the
        # SOH 0.66-0.75 band and 310 mV below SOC 0.35 - an NCA 21700 carries
        # 10-30 mV. The OCV test runs at C/20, so IR accounts for 3 mV; what
        # remains is the charge and discharge sweeps sitting on slightly
        # different SOC axes, which the steep low-SOC slope turns into hundreds
        # of millivolts, and which grows with age as the capacity mismatch does.
        # A term that size dominates the measurement model, so it is capped.
        self.m_cap = m_cap
        self.tau_h_rest = tau_h_rest
        self.b_on_load = bool(b_on_load)
        self.estimate_b = q_b is not None
        # q_ib 는 전류 센서 옵셋을 상태로 추정한다.  q_b 가 예측 *전압* 에
        # 더해지는 상수인 것과 달리, 이쪽은 측정 전류에서 빼는 상수다:
        # I_true = I_meas - ib.  쿨롱 적분과 전압 모델 양쪽에 들어가므로
        # 옵셋이 만드는 SOC 표류가 OCV 오차로 드러나 관측된다.
        self.estimate_ib = q_ib is not None
        # 실제 전류 센서의 옵셋은 만스케일의 1 % 를 넘지 않는다.  제한을
        # 현실적인 크기로 조이면 상태가 참 SOC 오차를 대신 흡수하는 것을 막는다.
        self.ib_clip = float(ib_clip)
        self.n = (3 + int(self.estimate_h) + int(self.estimate_b)
                  + int(self.estimate_ib))
        self.ib = 3 + int(self.estimate_h) if self.estimate_b else None
        self.iib = (3 + int(self.estimate_h) + int(self.estimate_b)
                    if self.estimate_ib else None)
        self.h_det = 0.0
        self.R = R_volt ** 2
        q = [q_soc, q_v, q_v] + ([q_h] if self.estimate_h else []) \
            + ([float(q_b)] if self.estimate_b else []) \
            + ([float(q_ib)] if self.estimate_ib else [])
        p0 = [1e-2, 1e-4, 1e-4] + ([1.0] if self.estimate_h else []) \
             + ([1e-3] if self.estimate_b else []) \
             + ([float(p0_ib)] if self.estimate_ib else [])
        self.Q = np.diag(q)
        self.P = np.diag(p0)
        self.x = np.zeros(self.n)
        # The SOC range the OCV table actually covers for this cell, so the
        # estimate is never pushed somewhere the measurement cannot see it.
        pts = sd.ocv_field.near.points
        self.soc_span = (float(pts[:, 0].min()), float(pts[:, 0].max()))

    def _theta(self, soc, I, T):
        s = self.sc if I > 0 else self.sd
        th = s.theta(soc, self.soh, I, T)
        o, _ = s.ocv(soc, self.soh)
        return (float(th["R0"][0]) * self.k_f, float(th["R1"][0]) * self.k_f,
                float(th["tau1"][0]), float(th["R2"][0]) * self.k_s,
                float(th["tau2"][0]), float(o[0]), s)

    def dhyst(self, soc, s, d=0.01):
        """dM/dSOC. Small next to dOCV/dSOC but free to include, and it is the
        only place the hysteresis magnitude's SOC dependence enters."""
        lo, hi = self.soc_span
        c = min(max(soc, lo + d), hi - d)
        # hyst_M returns a scalar where ocv returns an array; atleast_1d makes
        # the two call sites read the same and survives either.
        a, _ = s.hyst_M(c + d, self.soh)
        b, _ = s.hyst_M(c - d, self.soh)
        v = (float(np.atleast_1d(a)[0]) - float(np.atleast_1d(b)[0])) / (2 * d)
        return v if np.isfinite(v) else 0.0

    def docv(self, soc, s, h=0.01):
        """dOCV/dSOC, never allowed to reach zero - see dekf_soh._docv.

        Outside the OCV table a centred difference returns exactly zero, which
        drops SOC out of the measurement Jacobian and freezes the estimate for
        good. Evaluation is pulled inside the table's span and the slope floored.
        """
        lo, hi = self.soc_span
        c = min(max(soc, lo + h), hi - h)
        a, _ = s.ocv(c + h, self.soh)
        b, _ = s.ocv(c - h, self.soh)
        d = (float(a[0]) - float(b[0])) / (2 * h)
        return d if abs(d) > 1e-3 else 1e-3

    def step(self, I, V_meas, T=25.0, dt=None):
        # The drive-cycle records are 1 Hz, but HPPC is 10 Hz inside pulses and
        # sparser between them, so a fixed dt would mis-integrate the coulomb
        # count exactly where the pulses are. Clipped because a logging pause is
        # not a ten-hour time constant.
        dt = self.dt if dt is None else float(min(max(dt, 1e-3), 60.0))
        soc, v1, v2 = self.x[0], self.x[1], self.x[2]
        hy = self.x[3] if self.estimate_h else self.h_det
        # 센서가 읽은 전류에서 추정 옵셋을 뺀 값이 참 전류다
        Ic = I - self.x[self.iib] if self.estimate_ib else I
        R0, R1, t1, R2, t2, _, s = self._theta(soc, Ic, T)
        a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)

        # predict
        soc_p = soc + Ic * dt / 3600.0 / Q_RATED_AH
        soc_p = min(max(soc_p, self.soc_span[0]), self.soc_span[1])
        xp = [soc_p, a1 * v1 + R1 * (1 - a1) * Ic, a2 * v2 + R2 * (1 - a2) * Ic]
        diag = [1.0, a1, a2]
        if self.gamma > 0:
            ah = np.exp(-abs(self.gamma * Ic * dt / 3600.0 / Q_RATED_AH))
            h_new = ah * hy + (1 - ah) * np.sign(Ic)
            if self.tau_h_rest is not None and abs(Ic) < 0.5:
                h_new *= np.exp(-dt / self.tau_h_rest)
            if self.estimate_h:
                xp.append(h_new); diag.append(ah)
            else:
                self.h_det = h_new
        if self.estimate_b:
            xp.append(self.x[self.ib]); diag.append(1.0)
        if self.estimate_ib:
            xp.append(self.x[self.iib]); diag.append(1.0)
        xp = np.array(xp)
        F = np.diag(diag)
        if self.estimate_ib:
            # 옵셋이 1 A 커지면 참 전류가 1 A 작아진다 -> 부호가 음
            F[0, self.iib] = -dt / 3600.0 / Q_RATED_AH
            F[1, self.iib] = -R1 * (1 - a1)
            F[2, self.iib] = -R2 * (1 - a2)
        P = F @ self.P @ F.T + self.Q

        # update
        Ic = I - xp[self.iib] if self.estimate_ib else I
        R0, R1, t1, R2, t2, ocv, s = self._theta(xp[0], Ic, T)
        y = ocv + Ic * R0 + xp[1] + xp[2]
        H = [self.docv(xp[0], s), 1.0, 1.0]
        if self.gamma > 0:
            M, _ = s.hyst_M(xp[0], self.soh)
            M = float(np.atleast_1d(M)[0])
            if not np.isfinite(M):
                M = 0.0
            if self.m_cap is not None:
                M = float(np.clip(M, -self.m_cap, self.m_cap))
            h_now = xp[3] if self.estimate_h else self.h_det
            y += M * h_now
            H[0] += self.dhyst(xp[0], s) * h_now
            if self.estimate_h:
                H.append(M)
        if self.estimate_b:
            y += xp[self.ib]
            H.append(1.0)
        if self.estimate_ib:
            H.append(-R0)
        H = np.array(H)
        if self.i_gate is not None:
            self.rest_s = self.rest_s + dt if abs(I) <= self.i_gate else 0.0
        if self.i_gate is not None and (abs(I) > self.i_gate
                                        or self.rest_s < self.rest_hold_s):
            if self.estimate_b and self.b_on_load:
                # project the innovation onto b alone
                Hb = np.zeros(self.n); Hb[self.ib] = 1.0
                S = Hb @ P @ Hb + self.R
                K = P @ Hb / S
                self.x = xp + K * (V_meas - y)
                self.x[self.ib] = min(max(self.x[self.ib], -0.25), 0.25)
                self.P = (np.eye(self.n) - np.outer(K, Hb)) @ P
            else:
                self.x = xp                # predict only; the model is not
                self.P = P                 # trustworthy at this current
            if self.estimate_h:
                self.x[3] = min(max(self.x[3], -1.0), 1.0)
            if self.estimate_ib:
                self.x[self.iib] = min(max(self.x[self.iib], -self.ib_clip),
                                   self.ib_clip)
            return self.x[0], y
        innov = V_meas - y
        R_eff = self.R
        # 잔차의 지수가중 크기 — 적응·하한 규칙이 함께 쓴다
        a = np.abs(innov)
        w = self.ew_rate
        self._r_ew = a if self._r_ew is None else ((1 - w) * self._r_ew + w * a)
        if self._i_ew is None:
            self._i_ew, self._v_ew = innov, 0.0
        else:
            self._i_ew += w * (innov - self._i_ew)
            self._v_ew += w * ((innov - self._i_ew) ** 2 - self._v_ew)
        if self.r_adapt is not None:
            # 적응 필터: 최근 잔차 분산이 R 을 정한다. 모델이 그 구간에서
            # 실제로 얼마나 틀리는지를 필터가 스스로 재는 셈이다.
            R_eff = max(self.R, float(self.r_adapt) * self._r_ew ** 2)
        elif self.r_floor_k is not None:
            R_eff = max(self.R, (float(self.r_floor_k) * self._r_ew) ** 2)
        elif self.r_var_k is not None:
            R_eff = max(self.R, float(self.r_var_k) ** 2 * self._v_ew)
        if self.r_max_mult is not None:
            R_eff = min(R_eff, float(self.r_max_mult) * self.R)
        S = H @ P @ H + R_eff
        K = P @ H / S
        if self.huber_c is not None:
            # M-추정: 잔차가 c 를 넘으면 이득을 |r|/c 로 나눈다. 큰 잔차는
            # 모델 오차이지 정보가 아니므로 그만큼 덜 믿는다.
            c = float(self.huber_c)
            if a > c:
                K = K * (c / a)
        self.x = xp + K * innov
        self.x[0] = min(max(self.x[0], self.soc_span[0]), self.soc_span[1])
        if self.estimate_h:
            self.x[3] = min(max(self.x[3], -1.0), 1.0)   # h is bounded by design
        if self.estimate_b:
            self.x[self.ib] = min(max(self.x[self.ib], -0.25), 0.25)
        if self.estimate_ib:
            self.x[self.iib] = min(max(self.x[self.iib], -self.ib_clip),
                                   self.ib_clip)
        self.P = (np.eye(self.n) - np.outer(K, H)) @ P
        return self.x[0], y


def run(sd, sc, soh, I, V, T, soc0, R_volt, gamma=0.0, estimate_h=False, t=None,
        k_f=1.0, k_s=1.0, i_gate=None, q_b=None, q_ib=None, p0_ib=0.25, ib_clip=2.0,
        tau_h_rest=None,
        m_cap=None,
        rest_hold_s=0.0, huber_c=None, r_adapt=None, r_floor_k=None,
        r_max_mult=None, r_var_k=None, ew_rate=0.01):
    f = EKF(sd, sc, soh, R_volt=R_volt, gamma=gamma, estimate_h=estimate_h,
            k_f=k_f, k_s=k_s, i_gate=i_gate, q_b=q_b, q_ib=q_ib, p0_ib=p0_ib, ib_clip=ib_clip,
            tau_h_rest=tau_h_rest,
            m_cap=m_cap, rest_hold_s=rest_hold_s, huber_c=huber_c,
            r_adapt=r_adapt, r_floor_k=r_floor_k, r_max_mult=r_max_mult,
            r_var_k=r_var_k, ew_rate=ew_rate)
    f.x = np.zeros(f.n)
    f.x[0] = soc0
    est = np.empty(len(I)); pred = np.empty(len(I))
    dts = None if t is None else np.diff(np.asarray(t, float), prepend=float(t[0]) - 1.0)
    for k in range(len(I)):
        est[k], pred[k] = f.step(I[k], V[k], T[k],
                                 None if dts is None else dts[k])
    return est, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="CC")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--soc-error", type=float, default=0.20,
                    help="initial SOC offset the filter must remove")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="Plett hysteresis rate; 0 disables the h state")
    args = ap.parse_args()

    sd = ECMSurface(args.cell, "discharge")
    sc = ECMSurface(args.cell, "charge")
    z = np.load(os.path.join(args.cache,
                             f"uypydj_{args.cell}_Fifteen_Drive_Cycles.npz"))
    lens = z["lens"]; off = np.concatenate([[0], np.cumsum(lens)])
    idx = np.linspace(0, len(lens) - 1, args.runs).astype(int)

    print(f"cell {args.cell}   초기 SOC를 {args.soc_error:+.2f} 틀리게 주고 시작\n")
    print(f"{'run':>4} {'SOH':>6} {'R_volt':>7} {'SOC RMSE':>9} {'말단 오차':>9} "
          f"{'수렴 t':>8} {'전압 RMSE':>10}")
    for k in idx:
        sl = slice(off[k], off[k] + lens[k])
        soc, V, I, SOH, T = (z[x][sl] for x in ("SOC", "V", "I", "SOH", "T"))
        ok = np.isfinite(soc) & np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        if ok.sum() < 2000:
            continue
        soc, V, I, T = (x[ok][:args.max_samples] for x in (soc, V, I, T))
        soh = float(np.nanmedian(SOH))
        # Measurement noise tracks the known open-loop model error at this SOH.
        rv = np.interp(soh, [0.70, 0.90, 1.00], [0.110, 0.035, 0.015])
        est, pred = run(sd, sc, soh, I, V, T, float(soc[0]) + args.soc_error, rv,
                        gamma=args.gamma)
        err = est - soc
        conv = np.flatnonzero(np.abs(err) < 0.02)
        tconv = conv[0] if len(conv) else -1
        print(f"{k:>4} {soh:>6.3f} {rv*1000:>6.0f}m {np.sqrt(np.mean(err**2)):>9.4f} "
              f"{err[-1]:>+9.4f} {tconv if tconv>=0 else -1:>7}s "
              f"{np.sqrt(np.mean((pred-V)**2))*1000:>9.1f}m")


if __name__ == "__main__":
    main()
