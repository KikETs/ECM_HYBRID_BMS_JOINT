"""Twelve O(1) statistics that identify how far a cell sits from the pooled ECM.

WHY TWELVE SCALARS AND NOT A SEQUENCE WINDOW
    The quantity being estimated is a pair of resistance multipliers. It is
    slowly varying and effectively flat in SOC and rate, so a sequence encoder
    would spend its capacity re-deriving an average. The load-bearing statistic
    is a current-weighted regression of the residual on current - one number -
    and everything else here is context for interpreting it.

    The practical argument is stronger than the statistical one. A 200-sample
    ring buffer cannot emit anything for its first 200 s after key-on, and on
    HPPC records "200 samples" is anywhere from 20 s to hours because the logger
    is non-uniform. An exponentially-weighted recursion has no dead window, no
    buffer, and takes the real dt.

THE RESIDUAL IS AGAINST THE UNCORRECTED NOMINAL
    r = V - V_hat where V_hat comes from the pooled surface at unit multipliers.
    Deliberately not the corrected model: if the residual were taken against the
    corrected prediction, the feature would chase its own output and the map from
    feature to multiplier would drift as the multiplier moved. One propagation
    serves both.

WHY I-SQUARED WEIGHTING IS THE RATE-TRANSFER ANSWER
    dR_fast = EW{I*r} / EW{I*I} is the least-squares slope of residual on
    current over the exponential window. The I^2 denominator means high-current
    samples dominate automatically - which matters because the correction has to
    transfer to the ~30 A regime where SOP lives, while a drive cycle spends
    most of its time under 5 A. No explicit gating is needed to focus the
    statistic where it must be valid.

STATE, AND WHAT HAPPENS AT KEY-ON
    All EW states start at zero. On this data that is accidentally correct
    because every run begins at rest, which is exactly why the lab data cannot
    detect a bad cold-start rule. `restore()` exists so a vehicle can carry the
    statistics across a key cycle in 64 B of NVM; `age_s` reports how long the
    window has been accumulating so a caller can refuse to trust a cold one.
"""
from __future__ import annotations

import numpy as np

TAU_EW = 600.0          # main window
TAU_I = 8.0             # slow-current filter, matched to measured tau2 ~ 7.95 s
TAU_DUTY = 300.0        # duty window
I_REST = 0.5            # A, below which the cell counts as resting
I_DUTY = 5.0            # A, above which the sample counts as loaded
I_HI = 10.0             # A, the knee for the high-rate excitation channel
EPS = 1e-9

N_FEATURES = 12
NAMES = ("dR_fast", "dR_slow", "log_exc", "I_hi", "f_rest", "duty",
         "SOC", "SOH", "T", "I_rms", "R_fast_nom", "R_slow_nom")

# --- diagnostic variants, carried alongside the twelve --------------------
# Measured in-sample, a linear map of the twelve explains log k_f with R2 = 0.860
# (dR_fast alone correlates +0.89) but log k_s with only 0.481, and dR_slow - the
# channel built for it - is not even among its strongest terms. Two things could
# be wrong with it, and both are cheap to carry in the same pass:
#
#   the filter constant   TAU_I is pinned at 8 s, matched to the measured tau2
#                         median of 7.95 s. But tau2 falls to about half that in
#                         aged cells (sop_hybrid_spec.md 12.4), so a fixed 8 s
#                         is wrong exactly where the error is largest.
#   the sampling regime   a drive cycle's current turns over in seconds, so the
#                         slow branch rarely reaches steady state and the
#                         residual is read where it carries least. Gating on
#                         SUSTAINED load reads it where it has developed.
#
# These are extra state, not extra surface lookups, so they cost almost nothing -
# the interpolator dominates the loop.
TAU_I_VARIANTS = (2.0, 4.0, 8.0, 16.0, 32.0)
TAU_EW_LONG = 1800.0
LOAD_HOLD_S = 3.0
N_EXTRA = len(TAU_I_VARIANTS) + 4
NAMES_EXTRA = tuple(f"dR_slow_t{int(t)}" for t in TAU_I_VARIANTS) + (
    "dR_fast_long", "dR_slow_long", "dR_fast_load", "dR_slow_load")


def _ew(state, x, dt, tau):
    """One-pole exponential average with the REAL dt."""
    a = 1.0 - np.exp(-dt / tau)
    return state + a * (x - state)


class TrimFeatures:
    """Streaming features against a pooled ECM nominal.

    surf_dis / surf_chg are the POOLED surfaces from ecm_pool.surfaces(holdout).
    Passing a per-cell surface here would leak the held-out cell's own
    resistance into the feature that is supposed to discover it.
    """

    def __init__(self, surf_dis, surf_chg, gamma=20.0, q_rated=3.0):
        self.sd, self.sc = surf_dis, surf_chg
        self.gamma, self.q = gamma, q_rated
        self.reset()

    def reset(self):
        self.v1n = self.v2n = 0.0        # nominal RC states, unit multipliers
        self.h = 0.0                     # nominal hysteresis state
        self.e_ir = self.e_ii = 0.0      # EW{I r}, EW{I I}
        self.e_sr = self.e_ss = 0.0      # EW{Itil r}, EW{Itil Itil}
        self.i_slow = 0.0                # one-pole filtered current
        self.e_hi = 0.0                  # EW{max(0,|I|-10)^2}
        self.e_rest = 0.0                # EW{1[|I|<0.5]}
        self.e_duty = 0.0                # EW{1[|I|>5]}
        self.e_i2 = 0.0                  # EW{I^2} for I_rms
        self.e_T = 25.0
        self.age_s = 0.0
        # variants
        self.v_islow = [0.0] * len(TAU_I_VARIANTS)
        self.v_sr = [0.0] * len(TAU_I_VARIANTS)
        self.v_ss = [0.0] * len(TAU_I_VARIANTS)
        self.l_ir = self.l_ii = 0.0        # long EW window, fast
        self.l_sr = self.l_ss = 0.0        # long EW window, slow
        self.g_ir = self.g_ii = 0.0        # sustained-load gate, fast
        self.g_sr = self.g_ss = 0.0        # sustained-load gate, slow
        self.load_s = 0.0
        self._last = None                # last (R_fast, R_slow) in mOhm

    # -- one sample ---------------------------------------------------------
    def update(self, dt, I, V, T, soc, soh):
        """Advance the nominal model and the statistics by dt seconds.

        Returns the nominal terminal voltage so a caller can log the residual.
        """
        if not (np.isfinite(dt) and dt > 0):
            return None
        dt = float(min(dt, 60.0))        # a pause is not a 10-hour time constant
        s = self.sc if I > 0 else self.sd
        th = s.theta(soc, soh, I, T)
        ocv, _ = s.ocv(soc, soh)
        M, _ = s.hyst_M(soc, soh)
        R0 = float(th["R0"][0]); R1 = float(th["R1"][0]); R2 = float(th["R2"][0])
        t1 = float(th["tau1"][0]); t2 = float(th["tau2"][0])

        v_hat = float(ocv[0]) + M * self.h + I * R0 + self.v1n + self.v2n
        r = float(V) - v_hat

        # statistics on the CURRENT sample, before the state advances
        self.i_slow = _ew(self.i_slow, I, dt, TAU_I)
        self.e_ir = _ew(self.e_ir, I * r, dt, TAU_EW)
        self.e_ii = _ew(self.e_ii, I * I, dt, TAU_EW)
        self.e_sr = _ew(self.e_sr, self.i_slow * r, dt, TAU_EW)
        self.e_ss = _ew(self.e_ss, self.i_slow ** 2, dt, TAU_EW)
        self.e_hi = _ew(self.e_hi, max(0.0, abs(I) - I_HI) ** 2, dt, TAU_EW)
        self.e_rest = _ew(self.e_rest, 1.0 if abs(I) < I_REST else 0.0, dt, TAU_EW)
        self.e_duty = _ew(self.e_duty, 1.0 if abs(I) > I_DUTY else 0.0, dt, TAU_DUTY)
        self.e_i2 = _ew(self.e_i2, I * I, dt, TAU_EW)
        self.e_T = _ew(self.e_T, float(T), dt, TAU_EW)
        self.age_s += dt

        # --- variants -----------------------------------------------------
        for q, tau in enumerate(TAU_I_VARIANTS):
            self.v_islow[q] = _ew(self.v_islow[q], I, dt, tau)
            self.v_sr[q] = _ew(self.v_sr[q], self.v_islow[q] * r, dt, TAU_EW)
            self.v_ss[q] = _ew(self.v_ss[q], self.v_islow[q] ** 2, dt, TAU_EW)
        self.l_ir = _ew(self.l_ir, I * r, dt, TAU_EW_LONG)
        self.l_ii = _ew(self.l_ii, I * I, dt, TAU_EW_LONG)
        self.l_sr = _ew(self.l_sr, self.i_slow * r, dt, TAU_EW_LONG)
        self.l_ss = _ew(self.l_ss, self.i_slow ** 2, dt, TAU_EW_LONG)
        # sustained load: the slow branch needs seconds of steady current before
        # its residual means anything, so the accumulator only advances after the
        # load has been held, and resets the moment it is not.
        self.load_s = self.load_s + dt if abs(I) > I_DUTY else 0.0
        if self.load_s >= LOAD_HOLD_S:
            self.g_ir = _ew(self.g_ir, I * r, dt, TAU_EW)
            self.g_ii = _ew(self.g_ii, I * I, dt, TAU_EW)
            self.g_sr = _ew(self.g_sr, self.i_slow * r, dt, TAU_EW)
            self.g_ss = _ew(self.g_ss, self.i_slow ** 2, dt, TAU_EW)

        # nominal model states advance last
        a1, a2 = np.exp(-dt / t1), np.exp(-dt / t2)
        self.v1n = self.v1n * a1 + R1 * (1 - a1) * I
        self.v2n = self.v2n * a2 + R2 * (1 - a2) * I
        if self.gamma:
            ah = np.exp(-abs(self.gamma * I * dt / 3600.0 / self.q))
            self.h = ah * self.h + (1 - ah) * np.sign(I)

        self._last = (R0 + R1) * 1000.0, R2 * 1000.0
        return v_hat, r

    # -- readout ------------------------------------------------------------
    def vector(self, soc, soh):
        """The 12 features. Units: mOhm for the residual slopes, A for currents."""
        rf, rs = self._last if self._last else (np.nan, np.nan)
        # EW{I r}/EW{I I} is a slope in ohms; the sign convention makes a cell
        # that is MORE resistive than nominal give a POSITIVE dR (discharge has
        # I<0 and r<0 when the real cell sags further than the model).
        d_fast = self.e_ir / (self.e_ii + EPS) * 1000.0
        d_slow = self.e_sr / (self.e_ss + EPS) * 1000.0
        return np.array([
            d_fast, d_slow,
            np.log10(self.e_ii + EPS),
            self.e_hi,
            self.e_rest,
            self.e_duty,
            soc, soh, self.e_T,
            np.sqrt(max(self.e_i2, 0.0)),
            rf, rs,
        ], dtype=np.float32)

    def vector_extra(self):
        """The diagnostic variants, in NAMES_EXTRA order. mOhm throughout."""
        out = [self.v_sr[q] / (self.v_ss[q] + EPS) * 1000.0
               for q in range(len(TAU_I_VARIANTS))]
        out += [self.l_ir / (self.l_ii + EPS) * 1000.0,
                self.l_sr / (self.l_ss + EPS) * 1000.0,
                self.g_ir / (self.g_ii + EPS) * 1000.0,
                self.g_sr / (self.g_ss + EPS) * 1000.0]
        return np.array(out, dtype=np.float32)

    def excitation(self):
        """The gate quantity: how much genuinely high-current history is in the
        window. Deliberately not innovation RMS - a 60 s innovation window on a
        drive cycle whose median current is a fraction of an amp is
        anti-correlated with the 30 A extrapolation risk it would claim to
        guard."""
        return float(self.e_hi)

    def serialize(self):
        return np.array([self.v1n, self.v2n, self.h, self.e_ir, self.e_ii,
                         self.e_sr, self.e_ss, self.i_slow, self.e_hi,
                         self.e_rest, self.e_duty, self.e_i2, self.e_T,
                         self.age_s], dtype=np.float32)

    def restore(self, a):
        (self.v1n, self.v2n, self.h, self.e_ir, self.e_ii, self.e_sr,
         self.e_ss, self.i_slow, self.e_hi, self.e_rest, self.e_duty,
         self.e_i2, self.e_T, self.age_s) = [float(x) for x in a]


def run_series(surf_dis, surf_chg, t, I, V, T, SOC, SOH, stride=1):
    """Feature vectors along a recorded series. Returns (features, residuals)."""
    tf = TrimFeatures(surf_dis, surf_chg)
    feats, res, idx = [], [], []
    for k in range(1, len(t)):
        dt = float(t[k] - t[k - 1])
        out = tf.update(dt, float(I[k]), float(V[k]), float(T[k]),
                        float(SOC[k]), float(SOH[k]))
        if out is None:
            continue
        res.append(out[1])
        if k % stride == 0:
            feats.append(tf.vector(float(SOC[k]), float(SOH[k])))
            idx.append(k)
    return (np.array(feats) if feats else np.empty((0, N_FEATURES), np.float32),
            np.array(res), np.array(idx))
