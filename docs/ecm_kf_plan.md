# ECM + Kalman filter SOC / SOH / SOP — design

Written 2026-08-15. The data facts live in `findings.md`, in particular
§4.3 (ECM decomposition).

**Restating the goal.** Use ECM parameters identified per SOH to estimate
SOC (Kalman), SOH and SOP together. The LSTM voltage model plus binary
search (`soh_extension_design.md`) stays as a comparison baseline; this
document is the main path.

---

## 0. This dataset was built for exactly this

The paper the UYPYDJ readme cites:

> J. Duque, P. J. Kollmeyer, M. Naguib, A. Emadi, "Battery **Dual Extended
> Kalman Filter State of Charge and Health Estimation** Strategy for Traction
> Applications," IEEE ITEC 2022.

The people who made the data designed the campaign for DEKF SOC/SOH
estimation. That is why there is an HPPC every 30 cycles, an OCV every 60,
and 0.5C/1C/2C capacity tests at every characterisation.

---

## 1. Structure

```
   measurement (V, I, T)
        │
        ├─────────────► ECM parameter model  θ(SOC, SOH) = {R0, R1, τ1, R2, τ2}
        │                                    OCV(SOC, SOH),  Q(SOH)
        │                                        │
        ▼                                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Dual EKF                                            │
   │    state x = [SOC, V1, V2]        ← fast dynamics     │
   │    param φ = [Q, R0 multiplier]   ← slow degradation  │
   └─────────────────────────────────────────────────────┘
        │                                        │
        ▼                                        ▼
      SOC estimate                            SOH estimate
        └──────────────┬─────────────────────────┘
                       ▼
        SOP(τ) : solve the ECM for the I* that drives V(τ) to V_lim
                 — propagated from the **current** V1, V2, so recent
                   driving history is carried in
```

### 1.1 State equations (2RC)

```
SOC_{k+1} = SOC_k − η · I_k · Δt / Q(SOH)
V1_{k+1}  = V1_k · e^(−Δt/τ1) + R1 · (1 − e^(−Δt/τ1)) · I_k
V2_{k+1}  = V2_k · e^(−Δt/τ2) + R2 · (1 − e^(−Δt/τ2)) · I_k
```

Measurement equation:

```
V_k = OCV(SOC_k, SOH) − V1_k − V2_k − R0 · I_k
```

The sign convention is the project's: **negative means discharge**.

### 1.2 Why SOP is solved through the ECM rather than looked up

The criticism prior work levels at lookup tables is that they cannot
reflect recent dynamic operating conditions. The ECM does not have that
problem — V1 and V2 hold the present state, so at the same SOC and SOH the
SOP depends on **what happened just before**. Starting from rest and
starting mid-discharge come out different on their own.

```
V(τ) = OCV(SOC) − V1(τ) − V2(τ) − I·R0 = V_lim
  with  V1(τ) = V1(0)·e^(−τ/τ1) + R1(1−e^(−τ/τ1))·I     (V2 likewise)
→ linear in I*, so there is a closed form
```

---

## 2. Already in place

| | Contents | Status |
|---|---|---|
| ECM parameters | `uypydj_ecm.csv`, 27,891 rows, 6 cells × SOH 0.68–1.00 × SOC 0.06–1.00 × 4 current levels × charge and discharge | done |
| OCV | `uypydj_ocv.csv`, 133 curves by SOH, 100-point SOC grid | done |
| Capacity Q(SOH) | `CAP` column in the cache (measured 0.5C discharge) | done |
| Drive cycles | 55.08 M samples cached, SOH labelled, rated SOC axis | done |
| SOP reference | `sop_reference.csv` (static ECM baseline) | done |
| LSTM baseline | 29.92 mV (27.81 mV over 24 conditions) | done |

## 3. To be built

| Step | Contents |
|---|---|
| **E1** | continuous θ(SOC, SOH) surface — interpolate or fit the discrete grid |
| **E2** | validate the ECM by open-loop drive-cycle simulation (voltage RMSE) |
| **E3** | EKF for SOC |
| **E4** | SOH estimation (Q, R0 multiplier) — dual EKF or RLS |
| **E5** | ECM-based SOP, compared against the reference and the LSTM |
| **T1–T4** | thermal model (§3.5) — couples into the E2 validation |

**E2 is the gate.** If the ECM cannot reproduce drive-cycle voltage, no
Kalman filter on top of it means anything. And that number is **directly
comparable** to the LSTM's 29.92 mV — the prior work also compares
LSTM/ECM/EM on one axis.

---

## 3.5 Thermal model — electro-thermal coupling

ECM parameters depend strongly on temperature, so without a thermal model
they are wrong wherever temperature moves. This campaign is fast charging,
so self-heating is large.

### 3.5.1 Structure

```
heat        Q_gen = I²·R_total(SOC,SOH,T)  +  I·T·(dOCV/dT)
                    └─ Joule (dominant)       └─ entropic term
capacity    C_th · dT_cell/dt = Q_gen − (T_cell − T_amb) / R_th
```

Feedback into the ECM: θ(SOC, SOH, T), Arrhenius form

```
R(T) = R_ref · exp( Ea/R_g · (1/T − 1/T_ref) )
```

### 3.5.2 What this data can and cannot identify

**Can — C_th, R_th.** UYPYDJ holds the chamber at 25 °C, but **cell
temperature swings 22–44 °C** (`findings.md` §1.1). Heating during fast
charge and cooling through drive cycles and rests is recorded as it
happened. Q_gen follows from current and resistance, so C_th and R_th can
be regressed out of the measured T trajectory. The absence of forced
temperature changes helps rather than hurts: the response is **pure
self-heating**.

**Measured — separability roughly holds at the resistance level and breaks
at the SOP level.**

This section originally said "Ea is assumed independent of aging, and that
cannot be checked." RPCWBY Test#1 and #2 give two temperatures across the
whole aging range, so it **could** be checked. The answer has two layers,
and reading them together misleads.

**Layer 1 — the resistance ratio moves gently.** Taking only the points
where both temperatures are at the 30 A current limit (the comparison has
to be at equal current) and inverting the 10 s resistance out of SOP:

| SOH | 1.000 | 0.935 | 0.886 | 0.832 | 0.777 |
|---|---:|---:|---:|---:|---:|
| R(25 °C) | 13.06 | 15.12 | 20.04 | 24.00 | 30.17 mΩ |
| R(10 °C) | 17.44 | 20.80 | 29.49 | 34.44 | 41.12 mΩ |
| **R10/R25** | 1.335 | 1.386 | **1.483** | 1.435 | 1.357 |

The ratio rises 11 % from 1.33 to 1.48 and comes back down. Test#2 gives
the same curve independently (1.306 → 1.476 → 1.452). **Separability
roughly holds, and an SOH correction at the 11 % level covers it.** Local
Ea runs 12.5 → 18.4 → 14.3 kJ/mol.

**Layer 2 — the SOP ratio collapses. But not because of resistance.**

| cycle | 1 | 1299 | 1428 | 1541 | 1769 | 1994 |
|---|---:|---:|---:|---:|---:|---:|
| 25 °C limit | current | current | current | current | current | current |
| 10 °C limit | current | current | current | **voltage** | voltage | voltage |
| SOP10/SOP25 | 0.961 | 0.888 | 0.880 | **0.821** | 0.710 | 0.532 |

At SOC 0.5. From cycle 1541 the cold side can no longer hold the 30 A
current rating and hits the voltage floor first; the ratio falls away from
that moment. 25 °C stays current-limited throughout.

**This is the argument for the ECM path.** A gentle resistance change meets
a voltage floor and is amplified non-linearly, and a model that solves the
voltage limit explicitly **reproduces the transition for free**. A lookup
table or a scalar SOH correction cannot express it.

> **Correction history.** This section once said "the SOP ratio collapses
> from 0.96 to 0.53, so separability is rejected." That was an overstatement.
> What collapses is SOP, and the cause is not that the temperature
> dependence of resistance changes with aging but that the limiting regime
> changes. The mistake was comparing the two temperatures at unequal current.

---

## 4. Constraints to pin down in advance

**Temperature.** UYPYDJ is a single 25 °C condition (`findings.md` §1.1).
The θ(SOC, SOH) obtained on this path therefore has **no temperature
dependence**. Mendeley HPPC gives six temperatures but only for fresh
cells. There are two options and either must be stated:

1. restrict SOC/SOH/SOP to 25 °C and declare temperature extension out of
   scope
2. assume **separability**, θ(SOC, SOH, T) = θ(SOC, SOH) · g(T), take g(T)
   from the fresh Mendeley data, and check only that the assumption holds
   within fresh cells — extrapolation into the aged range stays an
   assumption

**Low SOC × low SOH.** HPPC does not go below SOC 0.29 at SOH 0.70
(`findings.md` §4.4). That region of the θ surface is extrapolation, and it
is exactly where SOP is tightest. It must be marked in every result.

**Cell-to-cell spread.** At the same SOH, R1 differs by up to 5.58×
(`findings.md` §4.3.1). A θ averaged over six cells fits none of them.
Either use **per-cell θ**, or let the EKF adapt θ online — the latter is
what a real BMS does, and it is exactly the job of the parameter side of a
dual EKF.

---

## EKF improvement — the hysteresis term (2026-08-22)

### Why this was touched

After measuring SOP in amperes and propagating state-estimate error
through it, the loss caused by **SOC error exceeded the gain from the
hybrid correction** (7.26 → 4.94 A): a systematic 2 % SOC alone takes
4.94 → 14.23 A. Fixing SOC comes before correcting resistance.

Read with `sop_hybrid_spec.md` §11.3: R_eff bends 26 % across SOC
0.30–0.40 (−13.5 % and −14.9 % per 0.05) and is flat within ±5 % above it.
So SOC error at that knee passes straight into SOP.

### Diagnosis — measured, not guessed

The old filter's measurement model was `y = OCV + I·R0 + V1 + V2`, missing
the Plett hysteresis term `M(SOC,SOH)·h` that the project's own open-loop
ECM has — the term that cut open-loop error from 50.62 to 37.13 mV.

**The residual correlates with the hysteresis state:**

| SOH | voltage residual | residual vs h |
|---|---:|---:|
| 1.000 | 7.1 mV | +0.18 |
| 0.923 | 17.0 | +0.57 |
| 0.833 | 31.4 | **+0.80** |
| 0.691 | 87.5 | **+0.81** |

It grows with age. The Kalman gain has to put that residual somewhere, and
it puts it into SOC.

**And the SOC error is a bias, not a spread** — in every SOC band the
median absolute error equals |bias| to three digits, and the sign is
consistently negative. Discharging with h < 0, the true voltage sits below
what a model without h predicts, and the filter reads that as "less charge
than I thought." The worst band is SOC 0.3–0.4 (−0.031), which is the
R_eff knee.

### The fix — h is propagated, not estimated

h is **fully determined** by current history (Plett). So it is not a state:
it is propagated deterministically and added to the measurement prediction
only.

    y = OCV(SOC,SOH) + M(SOC,SOH)·h + I·R0 + V1 + V2
    H[SOC] = dOCV/dSOC + (dM/dSOC)·h

**The path that estimates h as a fourth state was also implemented and
compared, and it is worse.**

| Setting | overall RMSE | worst run | voltage residual | SOC 0–0.4 | 0.4–0.6 | 0.6–1.05 |
|---|---:|---:|---:|---:|---:|---:|
| none (old) | 0.0344 | 0.0753 | 47.1 mV | 0.0293 | 0.0259 | 0.0112 |
| **h deterministic** | **0.0261** | **0.0594** | **33.3 mV** | **0.0196** | **0.0172** | **0.0100** |
| h as a state | 0.0871 | 0.3144 | 91.8 mV | 0.0198 | 0.0180 | 0.0140 |

Three cells (CC, BOOST_REST, BOOST_NEGPULSE_1S) × 5 runs, started with the
initial SOC deliberately off by +0.20, scored only after convergence
(beyond 3000 s).

**Deterministic h improves every axis** — overall −24 %, worst run −21 %,
voltage residual −29 %, and −33 % in the target SOC 0–0.4 band. High SOC
does not get worse.

**Estimating h as a state gives a similar low-to-mid SOC gain but hurts
high SOC and diverges on one run** (SOC error 0.055 → 0.314, residual
336 mV). Given the freedom, it absorbs errors that are not hysteresis —
OCV table error, for instance. Both paths are kept; the default is
deterministic (`--gamma 20`, `--estimate-h` to make it a state).

### Not yet checked

- **How much of this translates into SOP.** → Done afterwards. Feeding the
  real EKF SOC into the SOP inversion gives 4.03 → 4.04 A against the
  oracle, effectively no change (`sop_end_to_end.csv`), and the residual
  regression is 3.38 A (`sop_hybrid_spec.md` §15.5). The 14 A collapse that
  §11.3's synthetic error propagation predicted does not happen — the
  synthetic systematic offset behaves differently from real filter error.
- γ = 20 was carried over from `ecm_simulate.py` and
  `sop_trim_features.py`. It was not re-tuned inside the EKF.
- Only three cells were examined. All six were not run.

---

## SOC error in aged cells — three rejections and two improvements (2026-08-23)

Even with the hysteresis term, SOC error remains in aged cells. On a
trajectory, a fresh cell is at median |error| 0.7 %p after an hour; at
SOH 0.706 it is 5.3 %p.

### What the cause is not

**(1) Not a bias — the sign flips run to run.** Written up from a single
trajectory as "skewed one way," but across runs CC run74 is +0.0019,
run99 is −0.0507, BOOST_REST run54 is +0.0519, run73 is +0.0036. A fixed
model error cannot do that.

**(2) Not OCV and not resistance.** The holdout cell was given its **own**
OCV with pooled resistance, and the reverse (both are undeployable oracles,
but they say which term is at fault). The results do not agree —
BOOST_REST run54 improves sharply on its own OCV, 0.0524 → 0.0088, while
run73 gets worse, 0.0271 → 0.0482.

**(3) Not insufficient convergence.** Lowering R_volt to 0.5/0.25/0.1× so
the filter trusts the measurement more makes **all twelve combinations
worse.** BOOST_REST run54 converges at 11,694 s but its final-20 % RMSE is
still 0.0409 — the place it converges to is wrong.

Together these leave one explanation: **in aged cells the residual of the
pooled measurement model is structural rather than resistance or OCV**, and
that has the same root as the "50 mV floor, the limit of 2RC plus a pooled
table" that §12.5 reached for SOP.

### Improvement 1 — put the trim multipliers into the measurement model (5–7 %)

The SOP arm already computes k_f and k_s while the filter was using the
uncorrected table. Put the same correction into the measurement model (k
comes from the last characterisation before the run — using the k that run
produced would be circular).

| Cell run | SOH | k=1 | with k | gain |
|---|---|---:|---:|---:|
| CC 79 | 0.799 | 0.0463 | 0.0442 | +4.5 % |
| CC 99 | 0.696 | 0.0584 | 0.0550 | +5.8 % |
| BOOST_REST 58 | 0.740 | 0.0473 | 0.0440 | +7.0 % |
| BOOST_REST 73 | 0.691 | 0.0271 | 0.0253 | +6.6 % |
| CC 0 | 1.000 | 0.0089 | 0.0089 | 0 % |

**It only works in the aged range.** Fresh cells have k near 1, so nothing
changes. That k_f reaches 1.23 (CC 99) or 0.72 (BOOST_REST 58) and still
only buys 5–7 % reconfirms the conclusion of (2) and (3): the error is not
a matter of resistance magnitude.

**And with the correction in place, R_volt still cannot be lowered.**

### Improvement 2 — update only at low current (−26 % in aged cells)

The measurement model's error enters **through I·R**. At rest,
y = OCV + M·h has no resistance term at all. So update only when |I| is
under a threshold and predict otherwise.

| Cell run | SOH | k, R base | R ×4 | **\|I\|<1 A only** |
|---|---|---:|---:|---:|
| CC 0 | 1.000 | 0.0089 | **0.0078** | 0.0099 |
| CC 39 | 0.902 | 0.0245 | **0.0208** | 0.0246 |
| CC 59 | 0.862 | 0.0342 | **0.0285** | 0.0309 |
| CC 79 | 0.799 | 0.0442 | 0.0395 | **0.0363** |
| **CC 99** | **0.696** | 0.0550 | 0.0483 | **0.0406** |
| BOOST_REST 43 | 0.785 | 0.0279 | 0.0304 | **0.0217** |
| BOOST_REST 58 | 0.740 | 0.0440 | 0.0468 | **0.0390** |

**Above SOH 0.85, R ×4 wins; below SOH 0.80, the gate wins.** The boundary
is the same in both cells. The gate loses on fresh cells because their R
error is small enough that updating under load carries information, and
gating only reduces the sample count.

### Cumulative

CC's worst run (SOH 0.696):

| Step | RMSE |
|---|---:|
| no hysteresis | 0.0753 |
| + deterministic hysteresis | 0.0584 |
| + trim multiplier k | 0.0550 |
| **+ \|I\|<1 A gate** | **0.0406** |

**From 5.8 %p to 4.1 %p.** Honestly, though: the SOC error in aged cells
is **not solved.** It is still 6× the fresh cell's 0.7 %p.

### What will bite in deployment

The gate can be scheduled on SOH, and R_volt is already scheduled there.
Vehicles rest often at lights and when parked, so update opportunities
should not be scarce. But **a long high-load drive straight after key-on**
gets almost no updates and falls back on coulomb counting, and that case
cannot be checked with this data — there is no guarantee that the rest
fraction of these drive cycles matches a real vehicle.

---

## SOC, final — nine attempts and three lines of answer (2026-08-23)

### Result

| Step | overall %p | worst run | SOH<0.80 |
|---|---:|---:|---:|
| original EKF | 3.80 | 9.22 | 5.47 |
| + deterministic hysteresis + trim multiplier k | 2.96 | 7.30 | 4.25 |
| + R_volt ×4 + \|I\|<1 A gate | 2.12 | 5.10 | 2.93 |
| **+ 30 s low-current hold** | **1.59** | **3.76** | **2.19** |

**Overall −58 %, worst −59 %, aged range −60 %.** Per cell: BOOST
4.10 → 1.07 (−74 %), CC 4.60 → 1.15 (−75 %), BNP_1S 5.18 → 1.65 (−68 %).
6 cells, 48 runs.

### Deployment form — no new state, no new learned parameter

    y = OCV + M·h + I·(k_f·R0) + v1 + v2      h propagated deterministically
    update: only after |I| ≤ 1 A has held for 30 s
    R_volt: SOH schedule × 4

k_f and k_s are the values the SOP arm's trim already computes, taken from
the most recent characterisation.

### How the last and largest step came about

At the current-gate stage the remaining error was measured **against true
SOC** — to see model error rather than estimation error. On the samples the
gate lets through, V_meas − (OCV + M·h) was **−115 to −142 mV** across SOC
0.4–0.7 and −10 to −18 mV at the ends.

Pooled OCV was the suspect, but the difference against the holdout cell's
**own** OCV was only +7 to +34 mV, so pooling was not the cause. What
remained was a missing RC term in the measurement, and that turned out to
be the gate's own defect: most `|I| < 1 A` samples are not real rest but a
brief low-current moment while driving, and with tau2 ≈ 8 s the RC voltage
from the preceding load is still there. Discharge makes it negative, which
is why the residual was consistently negative.

**30 s is 3.75 × tau2**, the point where the RC has decayed 97.6 %. Holding
longer improves the overall figure slightly (1.54 at 600 s) but **makes the
worst run worse again** (4.17) — update opportunities fall to 20 %, so
reliance on coulomb counting grows.

### What was rejected — and the pattern in it

| Direction | Attempt | Outcome |
|---|---|---|
| fix the model | substitute own OCV / R | sign varies run to run |
| | trim multiplier k | only 5–7 % |
| | relax hysteresis at rest | monotonically worse (2.12 → 2.24) |
| | cap M (100–0 mV) | monotonically worse (2.12 → 2.60) |
| | tune gamma (5–160) | 0.02 %p, insensitive |
| | offset state b (at rest) | collapse (fresh 42 %p) |
| | offset state b (under load) | worse collapse (12–34 %p) |
| **choose when to trust the model** | raise R_volt | **improvement** |
| | low-current gate | **improvement** |
| | 30 s rest hold | **largest improvement** |

**All seven attempts to fix or augment the model failed, and all three
attempts to choose when to trust it succeeded.** In aged cells the residual
of 2RC plus a pooled table is not expressible in any parameterisation —
the same place §12.5 reached for SOP with its "50 mV floor."

> **[Retracted — see `sop_hybrid_spec.md` §30.2]** The benchmark behind
> these numbers is circular: the label is `SOC = 1 + Ah/3.0` and the filter
> prediction is `soc + I·dt/3600/3.0`, the same equation, with the filter
> started at the exact initial SOC. Every one of the three successes
> reduces reliance on voltage, and that benchmark rewards that
> unconditionally — its optimum is "never use voltage at all" (pure coulomb
> counting scores 0.12 %p against the adopted filter's 1.51 %p). The
> pattern above is an artefact of the benchmark, not a finding about
> models. The de-circularised numbers are in §30.4.

### M is not hysteresis, but it must not be cut

`hyst_mV` in `uypydj_ocv.csv` is a median of 184 mV and a maximum of
666 mV at SOH 0.66–0.75. Real hysteresis in an NCA 21700 is 10–30 mV, and
this dataset's own rest voltage differs from OCV by at most 28.5 mV. The
test is C/20 so IR accounts for 3 mV; the rest is **the charge and
discharge curves sitting on different SOC axes**, which the steep low-SOC
slope amplifies to hundreds of millivolts.

Capping it nonetheless makes things monotonically worse. A drive cycle has
h ≈ −1, so M·h ≈ −M, and since `OCV_V` is the midpoint of charge and
discharge, **that term acts as a correction moving OCV onto the discharge
branch**. Only the name is wrong; the function is right.
(`ecm_surface.py:151` already uses `/2000` for the half-width, so it is not
a coefficient bug either.)

### What remains

The aged-cell 2.19 %p stands at 2.2× the fresh cell's 1.0 %p. But §15.5
showed that when this SOC is actually fed into SOP the residual regression
holds at 3.38 A rather than collapsing, so **SOC is not the bottleneck of
the SOP chain.**
