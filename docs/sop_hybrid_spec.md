# ΔV-Trim — design specification for the hybrid SOP arm

Written 2026-08-17. The design basis lives in `findings.md` and
`ecm_kf_plan.md`; this document covers only **what to build** and **why that
shape**.

This specification condenses the output of a multi-agent design review
(4 independent proposals × adversarial review in 2 directions × synthesis).
**Measurements the agents produced were not taken as given but reproduced** —
the reproductions appear in §2 beside the original claims.

---

## 0. Goal

Do not replace an existing BMS with AI. Leave SOC to the Kalman filter and
protection, balancing and SOE as they are, and put AI into **SOH and SOP
only**. Then ask:

> On an existing-class automotive BMS MCU with no AI accelerator
> (S32K344 class: 160 MHz Cortex-M7, 4 MB flash, 512 KB RAM), with all the
> existing BMS tasks still running, does adding SOH and SOP AI meet the
> memory and real-time deadlines?

The sub-question below it is this document's subject:

> Does the SOP have to be computed by a neural network directly, or is it
> enough for AI to update only the ECM parameters and solve SOP the
> conventional way?

The two arms are compared on one axis.

| Arm | Composition | Flash (int8) |
|---|---|---|
| **Full AI** | LSTM voltage model + binary search | 66 KB (h64) – 1,033 KB (h256) |
| **Hybrid (this document)** | 12 scalars → small NN → 2RC multipliers → closed form | about 26 KB (table 17.5 + model 2.1 + code) |

---

## 1. Structure

```
   drive-cycle residual r = V − V̂(pooled 2RC)
                │
                ▼
   12 exponentially-weighted scalars (O(1), no ring buffer)
                │
                ▼
   MLP 12-16-16-2  (514 parameters)   ← always reported beside linear 12→2 (26)
                │
                ▼
   k_f (shared R0·R1 multiplier),  k_s (R2 multiplier)
                │
                ▼
   2RC closed form → SOP → clamped by a table-based envelope
```

### 1.1 Why only two multipliers

**R0 and R1 are tied together.** The median τ1 is 0.244 s, so even at the
shortest SOP horizon (τ = 2 s), e1 = exp(−2/0.244) = 2.7e−4. R0 and R1 enter
`den` with **the same weight** and cannot be separated. This deletes
**structurally** — rather than by clipping — the failure in `ecm_refine.py`
where m1 went negative.

**τ, OCV, M and R0 alone are not outputs.** τ is non-monotone in temperature
and its correction was already abandoned (`ecm_temp_factor.py`); OCV and M do
not appear in the gradient under the loss below; and R0 alone is
unidentifiable at 1 Hz while being within 1.06× across cells anyway.

### 1.2 Why the loss is pulse ΔV rather than absolute voltage

The leave-one-out error of the pooled OCV is **45–72 mV** at SOH 0.76–0.80,
while the resistance correction moves about 40 mV at 29 A. Training on
absolute voltage lets **error in the OCV surface flow into the resistance
path.**

Labels use measured HPPC pulse ΔV. Every fitted pulse follows at least 600 s
of rest, so v1 = v2 = 0 and ΔV is exact rather than approximate.

```
ΔV̂(τ) = I_p · [ k_f·(R0n + R1n·(1−e1)) + k_s·R2n·(1−e2) ]
```

OCVn, Mn and h are **absent from the equation.** That is the entire safety
argument.

**`sop_reference.csv` never enters the loss.** It is ECM-derived, so using it
would fit assumptions to themselves. Evaluation only.

---

## 2. The measurements the design rests on — with reproductions

| Item | Agent's claim | My reproduction | Verdict |
|---|---|---|---|
| τ1 median / p95 / fraction < 1 s | 0.244 s / 0.730 / 99.1 % | 0.244 / 0.730 / 99.1 % | **match** |
| τ2 median / p95 | 7.95 s / 24.6 s | 7.95 / 24.59 | **match** |
| voltage-limited fraction (τ = 2 s / 10 s) | 9.5 % / 25.2 % | 9.5 % / 25.2 % | **match** |
| corr(k_fast, k_slow) | −0.802 | −0.687 | same direction, different magnitude |
| den error RMS (τ = 2/10/30 s) | 14.1 / 10.4 / 5.9 % | 15.8 / 12.4 / 7.7 % | same direction, mine larger |
| `sop()` defect (b), optimistic bias | 18.5 %, max +7.35 % | 12.8 %, max +11.0 % | **defect real** |

The difference in the last two appears to be how the SOH range was binned
(reproduction n = 95 against the original 36 points), and **the direction of
the conclusion is the same.** A larger den error in my reproduction means
more room for correction, which does not work against the design.

**Two design judgements follow:**

- k_f and k_s move **in opposite directions**, so "a single offset (Δs)
  sliding along the aging axis" is the wrong coordinate system. Aging moves
  R1 and R2 the same way; cell-to-cell variation moves them oppositely.
- The den error shrinks as τ grows (15.8 → 7.7 %), so **τ = 2 s is the
  horizon where correction is worth the most.** 30 s is worth the least.
  That is the opposite of intuition.

---

## 3. Inputs — 12 scalars, no ring buffer

Every exponential weight uses `α = 1 − exp(−dt/600)` with the **actual dt**,
which is why the cache needs a time axis (§6).

| # | Channel | Basis |
|---|---|---|
| 1 | `dR_fast = EW{I·r}/EW{I·I}` | the core channel; I² weighting selects high-current samples automatically |
| 2 | `dR_slow = EW{Ĩ·r}/EW{Ĩ·Ĩ}` (Ĩ low-passed at τ = 8 s) | separates the slow branch |
| 3 | `log10 EW{I·I}` | excitation energy — both an input and a gate |
| 4 | `I_hi = EW{max(0,\|I\|−10)²}` | high-rate excitation; the quantity the gate keys on |
| 5 | fraction of the last 600 s with \|I\| < 0.5 A | regime |
| 6 | fraction of the last 300 s with \|I\| > 5 A | regime |
| 7–9 | SOC, SOH, T | state |
| 10 | I_rms | the current level at which the feature was identified |
| 11–12 | R_fast_nom, R_slow_nom | so the network can form dR/R internally |

Normalisation uses (µ, σ) taken **from the training cells only**, clipped at
±4σ, frozen as 12 float32 pairs.

**Deleted:** the hysteresis multiplier k_h and the OCV offset (their gradient
is zero under the ΔV loss, so there is nothing to estimate); the 200×3 raw
ring buffer; and the 24-dimensional Mahalanobis OOD detector (measured to be
useless already at 1.5× the lab range, so it **removes the arm precisely in
the target region**).

---

## 4. Outputs and deployment guards

```
k_f = exp(0.470·tanh(u_f)) ∈ [0.625, 1.600]     shared by R0, R1
k_s = exp(0.588·tanh(u_s)) ∈ [0.555, 1.800]     R2
```

The bounds are the **measured LOCO requirement** (0.70–1.28 and 0.74–1.54 at
rank 3) plus margin. Setting them from the spread of the drive-cycle ridge
fit puts 24 of 30 points on the boundary.

Guard chain:

```
g = clip((I_hi − lo)/(hi − lo), 0, 1)
k ← 1 + g·(k_raw − 1)                    gate multiplying the output
k ← k + (k_new − k)·dt/300               slew (step changes in power limit are a driveability defect)
SOP_out = clip(SOP_trim, SOP_pess, SOP_opt)   the envelope comes from the table alone
```

The gate multiplies **the output rather than an input channel** because every
training window that teaches the correction is a high-excitation window, so
the loss would not compensate for shrinkage. On the output the loss cannot
route around it.

The envelope comes **from the table alone** because a guard that reads the
network's own output is not an independent channel.

---

## 5. The ablation ladder

Identical folds, seeds and loss at every rung, leave-one-cell-out. The
primary metric is held-out pulse ΔV RMSE (τ ∈ {2,10} s, rank 2/3); the
secondary is `limited_by` and the signed den % error per SOH band.

| Rung | What it isolates | Result that **rejects** the claim |
|---|---|---|
| A0 | k=1 pooled table (baseline) | — (den RMS 15.8 % at 2 s) |
| A1 | per-cell constant k_f (oracle) | A1 ≈ A4 means store one constant in NVM and stop; no network needed |
| A2 | oracle (k_f, k_s) per SOH band | if A2 cannot beat A0, **the output itself is wrong** — stop |
| A3 | linear 12→2, 26 parameters | A3 ≈ A4 means **ship A3** and report "the AI is 26 numbers" |
| A4 | MLP, 514 parameters (proposal) | inside seed noise of A0 means fold the arm and ship the 17.5 KB table |
| **A5** | remove residual channels 1–6 | **the primary falsifier.** A5 ≈ A4 means the correction is a function of state and age and belongs in the table |
| A6 | inject **another cell's** features at the same (SOC, SOH, T) | A6 ≈ A4 means the LOCO result is an artefact |
| A7 | tie k_s = k_f | if both survive, reduce to one output |
| A8 | re-add k_h and ΔV_oc | if either **helps**, there is leakage in the ΔV loss |
| A9 | inject ±30 mV OCV offset into the holdout and retrain | if k_f·k_s is not **0.000**, the offset is leaking into the resistance path |
| A10 | identify in \|I\| ≤ 3 A windows, evaluate on rank 3 pulses | worse than A0 means rate transfer fails |
| **A10b** | **evaluate on the loaded \|I\| > 20 A slice** | tests §7's largest risk directly (new) |
| A11 | all deployment guards on, cold start, no NVM | if gate and slew drop it to A1, ship A1 |
| A12 | oracle SOH × {A0, A4}, 2×2 | if the gain vanishes in the CNN-SOH row, it cannot be deployed as a chain |
| A13 | hybrid against the SOP LSTM (h64, h256) | the research question itself |

---

## 6. Preconditions — already handled

| Item | Status |
|---|---|
| `sop_from_ekf.py` defects (a) and (b) | **fixed.** All 67 voltage-limited cases land at exactly 2.5 V |
| cache time axis `t` | **added** (float64, `analysis/cache_t`) |
| fixed-point non-convergence | 346 of 400 — needs replacing with bisection (exposed via the `converged` flag) |

While fixing it **I introduced a new inconsistency**: θ was re-taken at the
final current while the current came from the solve at the previous point,
and 11 of 200 disagreed. Re-fixed to solve and report at the same point.

---

## 7. Remaining risks, ranked

### 1. The identification regime differs from the label regime

Features come from drive cycles that never rest; labels come from pulses
after 600 s or more of rest. As `ecm_refine.py` records, **this regime gap**
is what defeated all four alternative fits, and fitting the same multiplier
to drive-cycle residuals gave exactly zero gain, 35.3 mV against 35.3 mV.

> **The design review concluded "the only mitigation is new measurement."
> That is wrong.**
>
> Collecting new data is impossible, so the existing data was recounted to
> see whether it really is absent. Drive cycles contain **136,960 samples
> with \|I\| > 20 A** (0.25 % of the total), distributed across the whole SOH
> range and growing with age (51,354 of them in SOH 0.65–0.80).
>
> | SOH | >15 A | >20 A | >25 A |
> |---|---:|---:|---:|
> | 0.95–1.00 | 68,113 | 18,243 | 781 |
> | 0.85–0.90 | 77,559 | 24,373 | 2,418 |
> | 0.65–0.80 | 126,804 | 51,354 | 10,449 |
>
> So rung **A10b** is added: pick only the high-current instants under load
> and look at the voltage prediction error. There is no SOP label, but it
> **measures the physical quantity SOP depends on, under load.** That is
> stronger than A10 — A10 still evaluates on post-rest pulses, whereas A10b
> evaluates under load.
>
> This does not remove the risk. It moves it from "cannot be validated" to
> "validated, and here is the result."

### 2. One of the six cells is unreadable by the features

CC_CELL2's `dR_fast` stays within −1.3 to +2.7 mΩ across its whole life while
the required k_fast falls to 0.83–0.88. The reported r = +0.885 is made by
the other five. With n = 6, one silent fold is 17 % of the evidence.
**Report per cell and never pool.** If two fail, the honest conclusion is A1
(per-cell constant plus NVM).

### 3. The gain band is narrow, and its interior is a measurement gap

The correction's den leverage is 15.8 % at τ = 2 s, but the rows where SOP is
sensitive to den (voltage-limited) are only **9.5 %** at τ = 2 s, and their
median SOC is 0.31 — just above `SOC_FLOOR_OK = 0.29`. So the product of
(correction matters) × (SOP is sensitive) peaks in the **low SOC × low SOH ×
voltage-limited** wedge, which is exactly where aged HPPC has no measurement.
It overlaps the region the rejection rule says not to answer in.

---

## 7.5 Results — 2026-08-17

The implementation is complete and A0/A3/A4/A5 were run leave-one-cell-out
over six cells. The metric is the held-out cell's **measured HPPC pulse ΔV
RMSE** (τ ∈ {2,10} s, rank 2/3), λ_p = 1e−3, mean of three seeds.

| Holdout | A0 (uncorrected) | A3 (linear, 26) | A4 (MLP, 514) | A5 (residual channels removed) |
|---|---:|---:|---:|---:|
| BOOST | 61.7 | 46.0 | **43.9** | 55.2 |
| BOOST_NEGPULSE | 88.5 | **54.9** | 56.7 | 70.5 |
| BOOST_NEGPULSE_1S | 75.7 | **53.4** | 55.3 | 67.5 |
| BOOST_REST | 148.9 | **74.9** | 83.0 | 103.9 |
| CC | 74.4 | 57.9 | **57.8** | 72.3 |
| CC_CELL2 | 66.7 | 55.7 | **52.7** | 57.9 |
| **mean** | 86.0 | **57.1 (+33.5 %)** | 58.3 (+32.2 %) | 71.2 (+17.1 %) |

### Conclusion 1 — 26 parameters beat 514

The linear reader beats the MLP. It leads on 3 of 6 cells and leads by a wide
margin on the hardest fold (BOOST_REST, where A0 is 148.9 mV): 74.9 against
83.0. §5's pre-registered rule fires:

> A3 ≈ A4 means ship A3 and report "the AI in the hybrid arm is 26 numbers."

**26 parameters is 0.1 KB in float32.** Four orders of magnitude from the
Full AI arm's 66–1,033 KB.

The reason non-linearity does not pay is structural. k_f − 1 ≈ dR_fast /
R_fast_nom holds almost by definition, so what there is to learn is a linear
map to begin with. The other 488 parameters only add room to overfit — on
data with 387–533 labels per cell.

### Conclusion 2 — the primary falsifier passes

A5 zeroes the six residual channels and keeps only state and age (SOC, SOH,
T, I_rms, R_nom). The rejection condition was "A5 ≈ A4 means the correction
is only a function of state and age and should be absorbed into the table."

The result is **+17.1 % against +32.2 %, half.** The residual of the recent
response really does carry information that SOH and SOC do not.

It agrees with an independent measurement:

| | |
|---|---:|
| corr(SOH, required k_f) | **−0.012** |
| corr(dR_fast, required k_f) | **+0.940** |

(The agent design review reported +0.078 and +0.885; my reproduction was
stronger.)

That A5 still gives +17.1 % is because the pooled table itself carries
systematic error against SOH, and that share can be absorbed by updating the
table.

### A comparison not yet made

**The two arms use different metrics.** Full AI is drive-cycle voltage RMSE
(2 of 6 folds complete: M2 at 22.6 and 24.9 mV); the hybrid is HPPC pulse ΔV
RMSE (57.1 mV). Measuring them on one axis requires rung A13, which is
possible only after the 6-fold finishes. **The two numbers must not be placed
side by side right now.**

### Remaining rungs

A1 (per-cell constant), A2 (oracle upper bound), A6 (foreign-cell feature
injection), A7 (tying k_s), A8–A9 (loss integrity), A10/A10b (rate and load
transfer), A11 (deployment guards), A12 (SOH propagation), A13 (hybrid
against LSTM).

---

## 7.6 Rung A13 — comparing the two arms on one axis

`analysis/eval_a13.py`, `analysis/a13_psweep.py`. **Conclusion: over a 6-fold
leave-one-cell-out, the 26-parameter hybrid beats the 1.08 M-parameter LSTM
by 24 %.** Two wrong comparisons were made on the way there, and why they
were wrong is part of the result.

### 7.6.1 The setup holds

The reference LSTM was placed on exactly the pulses the hybrid was evaluated
on.

- Pulse indices were reproduced from the cache and checked against
  `uypydj_hppc_resistance.csv` — error 0.24–7.9 µV, the float32 storage
  limit. (The first attempt left direction out of the key, so discharge
  labels were paired with charge samples and it showed up as a 1.7 V error.)
- Both arms were **rescored on the identical surviving set.** A0/A3/A4
  reproduce §7.5 exactly (85.97 / 57.11 / 58.26 mV) — it is deterministic.
- The context window ends 200 samples before the target. HPPC is 10 Hz, so
  τ = 10 s is 100 samples (measured median 100, max 101, over 28,855 pulses),
  which stops 99 samples before the pulse starts. A logging rate faster than
  20 Hz breaks this design.

### 7.6.2 The first wrong comparison — different currencies

Given the recorded P directly, M1 gives 32.6 mV against the hybrid A3's
44.8 mV, which looks like a decisive LSTM win. **That table must not be used
to claim a ranking.** The LSTM's P channel is V·I as recorded, while the
hybrid's input is current only. They receive the same event in their own
currencies, and the currencies carry different amounts of information — give
the hybrid P and it can produce zero error via V = P/I.

### 7.6.3 The second wrong comparison — removing information moves you out of distribution

Under a query that solves P self-consistently (`isolve`: V ← f(P = V·I), 40
iterations, converged with a median move of 0.00 mV), M1 collapses 5×,
32.6 → 162.6 mV, while M2 barely moves, 70.8 → 78.3 mV. It is tempting to
read that as "M1 was reading V out of P," but **`isolve1` rejects that
reading.** Making only the target sample's P self-consistent gives M1
30.7 mV (0.94×, an improvement). The model does not read its own target.
What `isolve` destroyed is the shape of a trajectory whose ~100 pulse samples
were flattened to a constant V, and no real cell can produce that input. The
162.6 mV mixes missing information with out-of-distribution sensitivity and
is **an upper bound, not an estimate.**

### 7.6.4 The fix — match the input currency and retrain (A13c)

Rather than removing information at evaluation time, the model was **trained
with I instead of P as the input channel** from the start
(`train_soh.py --feats SOC,T,I`). Now both arms receive current and neither
receives voltage. Every other hyperparameter matches §7.6's 6-fold.

First, the cost of switching currency. Drive-cycle voltage RMSE:

| Holdout | M1 (P) | M1 (I) | M2 (P) | M2 (I) |
|---|---:|---:|---:|---:|
| BOOST | 23.32 | 25.90 | 22.41 | 19.46 |
| BOOST_NEGPULSE | 36.21 | 38.53 | 20.98 | 19.16 |
| BOOST_NEGPULSE_1S | 27.61 | 30.35 | 21.90 | 22.28 |
| BOOST_REST | 52.45 | 51.62 | 24.43 | 25.90 |
| CC | 33.92 | 34.93 | 24.92 | 24.14 |
| CC_CELL2 | 25.37 | 24.86 | 22.59 | 19.55 |
| **mean** | **33.15** | **22.87** | **34.36** | **21.75** |
| standard deviation | 9.75 | 1.38 | 9.07 | 2.58 |

**M2 improves when P is taken away** (22.87 → 21.75 mV). P = V·I was not
needed to predict voltage. M1 gets slightly worse (33.15 → 34.36), which
matches the observation in §7.6.3 that M1 leans harder on the P channel. So
this comparison did not win by crippling the LSTM.

### 7.6.5 A13 6-fold — measured pulse dV RMSE (mV)

| Method | BOOST | BOOST_NEGPULSE | BOOST_NEGPULSE_1S | BOOST_REST | CC | CC_CELL2 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| **hybrid linear (26)** | 32.7 | 38.8 | 46.9 | 62.6 | 41.5 | 46.4 | **44.8** |
| hybrid MLP (514) | 32.7 | 34.0 | 45.8 | 71.8 | 42.5 | 45.5 | **45.4** |
| Full AI M2, I input (1.08 M) | 48.7 | 51.7 | 51.3 | 99.6 | 60.4 | 40.3 | **58.7** |
| Full AI M1, I input (1.08 M) | 64.9 | 74.6 | 68.5 | 134.6 | 76.3 | 45.0 | **77.3** |
| uncorrected pooled ECM | 60.9 | 88.5 | 76.0 | 148.9 | 73.1 | 66.7 | **85.7** |

**The hybrid wins on five of six cells.** The one loss is CC_CELL2, and
excluding the worst cell (BOOST_REST) keeps the direction: hybrid 41.3
against M2's 50.5. The 22 % advantage seen in the 2-cell pilot holds at 24 %
over six.

**The way they diverge is consistent.** The hybrid's worst is 62.6 mV, M2's
is 99.6 mV. With a physical model underneath, an unseen cell is never worse
than 1.4× the mean, while pure AI stretches to 2.5×. A BMS has to guarantee
the worst case rather than the average, so that contrast is the core of the
deployment argument.

### 7.6.6 Current sensitivity — the sign is perfect and the magnitude is still blunt

Scale the last sample's current by s and look at the response slope. RC
branches cannot respond to a 0.1 s perturbation (measured median τ1 0.244 s),
so the physical expectation is **I·R0**.

| Holdout | M1 sign negative | M1/expected | M2 sign negative | M2/expected |
|---|---:|---:|---:|---:|
| BOOST | 100 % | 0.68 | 100 % | 0.68 |
| BOOST_NEGPULSE | 100 % | 0.71 | 100 % | 0.65 |
| BOOST_NEGPULSE_1S | 100 % | 0.69 | 100 % | 0.62 |
| BOOST_REST | 100 % | 0.65 | 100 % | 0.52 |
| CC | 100 % | 0.65 | 100 % | 0.64 |
| CC_CELL2 | 100 % | 0.64 | 100 % | 0.62 |
| **mean** | **100 %** | **0.67** | **100 %** | **0.62** |

**All 12 combinations have the right sign on 100 % of pulses.** It comes from
an I input where an echo is structurally impossible, so the reading needs no
qualification.

But at 0.62–0.67 × the physical expectation it is **33–38 % blunt.** Run a
binary search on it and it overstates the current needed to reach the voltage
limit, and therefore **answers optimistically about available power.** This
reproduced with P input and I input, in M1 and M2, and across all six cells —
the most robustly confirmed defect in this project, and the first thing to
check when validating SOP in watts.

### 7.6.7 What this section does not claim

- The metric is **pulse dV**, not SOP. (The ampere validation was done later
  in §11 — discharge 4.94 A, charge 2.05 A. The two items below were written
  before that work.)
- The hybrid's nominal is an ECM pooled over five training cells. With only
  six cells, one per protocol, how the advantage scales with cell count is
  unknown.
- The cause of the bluntness (0.62–0.67) was not established. That the
  training data's current distribution barely covers the SOP region is the
  leading hypothesis, unconfirmed.

---

## 7.7 Spike investigation — enumerating all 41 events, and why the guard was rejected

The trajectory plots (`analysis/fig_a13_traj_value.png`, `..._error.png`)
show the hybrid jumping intermittently. Before adopting it, what those jumps
are was established.

**The definition is on the multipliers, not on error.** k_f and k_s are what
the model actually emits, and a cell's resistance cannot move 15 % between
two characterisations 37 cycles apart. The per-cell median of adjacent
|Δlog k_f| is 0.013–0.029, and events above a 0.06 threshold number **41**
across all six cells.

### 7.7.1 The excitation gate is powerless

Correlation between `|Δlog k|` and the excitation percentile is **r = +0.029**.
The distribution is flat too — 7 events in the bottom 20 %, 18 in the middle,
16 in the top 20 %. Looking only at CC_CELL2's last point (excitation 98 %)
reads as "it jumps when excitation is excessive," but that does not hold over
all 41, and BOOST's cycle 150 and CC's 862 and 975 jump at 2 % excitation.
`TrimFeatures.excitation()` is there to filter low-information windows; it
does not predict these events.

### 7.7.2 Half of them are oscillation rather than spikes

20 up and 21 down, balanced, with adjacent pairs recurring — up then straight
back down (BOOST 112/150, BNP 937/975, BNP_1S 1168/1206, CC 1387/1462,
CC_CELL2 1056/1093 and others). The estimator is wobbling about a value
rather than drifting one way.

**Cycles 937 and 975 jump simultaneously in three cells (BOOST, BNP, CC).**
That is not per-cell coincidence but something common to the test schedule at
that time. The cause was not established.

### 7.7.3 The direction is mostly safe but differs by cell

26 conservative (overstating the drop) against 15 optimistic (understating).
But **BOOST_REST is 6/6 conservative** while **BOOST_NEGPULSE_1S is 5/7
optimistic**, including −105 and −93 mV. The optimistic direction calls
available power higher than it is, which is the dangerous side for SOP.

### 7.7.4 Three at the last characterisation

BNP (1660, +78 mV), BOOST_REST (1350, +93), CC_CELL2 (1806, +105). **All
three are upward jumps and all three are conservative.** They could be the
signal of resistance genuinely surging at end of life, or estimator noise.
There is no next measurement, so this data cannot separate them, and three in
the same direction is a narrow coincidence — suppressing them with a guard
would throw the signal away if it was real.

### 7.7.5 The guard experiment and its rejection

A two-strike rejection (hold on one jump, accept everything if the next
update goes the same way) was swept over thresholds and hold policies. Full
pulse set, six cells.

| Setting | mean | worst cell |
|---|---:|---:|
| **no guard** | 45.4 | 62.6 |
| thr 0.06, hold previous | **44.4** | **60.5** |
| thr 0.15, hold conservative | 44.8 | 62.6 |
| thr 0.20, hold previous | 44.9 | 62.6 |
| median filter (3) | 44.2 | 64.3 |

**The best is a 2.2 % improvement.** Catching all 41 buys that much.

Three things emerged.

1. **Holding conservatively goes the opposite way from expectation.** CC
   improves sharply, 45.2 → 41.6, while BOOST_REST worsens, 62.6 → 64.7. All
   of that cell's spikes are conservative, so holding conservative
   over-overestimates what was already overestimated. A textbook case of a
   safety rule costing accuracy.
2. **Raising the threshold buys nothing.** 0.15–0.20 catches only the six
   extremes and the mean is worse than at 0.06. The 3–6× oscillations are
   individually small but numerous.
3. **No combination gets BOOST_REST below 60.5 mV.** That cell's problem is
   not spikes but the sustained bias of §7.6.5 (+68.6 mV after cycle 800),
   and a guard does not touch bias.

The median filter improves the mean most but **worsens the worst cell** —
BOOST_REST ages fast, so its k really is moving, and a causal filter lags.
Improving the mean while worsening the worst is the wrong trade for a BMS.

**Conclusion: adopt without a guard.** If one is added, thr 0.06 with
hold-previous is the only setting that improves the mean and the worst
together, and the gain is 2.2 %. Either way A13's ranking is unchanged
(44.4 against M2's 58.7).

---

## 8. MCU budget (S32K344 class)

| Component | Flash | RAM |
|---|---:|---:|
| ECM table (4 rate steps, including the in-hull bitmask) | 17.5 KB | DTCM mirror recommended |
| Trim MLP, 514 float32 | 2.1 KB | 184 B activations |
| Feature front end (12 EW + nominal 2RC + h) | — | 152 B |
| SOH CNN int8 (shared) | 10.7 KB | about 2.0 KB |
| Code (.text estimate, to be replaced by the map file) | ~6 KB | — |
| **total** | **about 36 KB** | **about 3 KB + stack** |

Compute: about 5 µs for the 1 Hz feature update, about 68 µs for 12 SOP
solves at 10 Hz → **0.07 % CPU**.

**float32 rather than int8** — with a head of only two outputs, int8 puts
5.2–11.4 mV/LSB into k_f, which is larger than the entire accuracy budget.
Not worth doing to save 1.5 KB out of 4 MB.

**Caution:** the flash above is data only. `.text` for libm, the solver, CAN
packing and so on has to be confirmed from the map file, and the
cache-related preemption delay from 72 table lookups per tick evicting a
higher-priority task's D-cache is a separate item to validate. This is a
point about **the accuracy of the claim**, not about feasibility.

---

## 9. File plan

| File | Contents |
|---|---|
| `analysis/ecm_pool.py` | pooled surface with the holdout removed. Pooling in **response space** rather than parameter space (it is non-linear in τ2, so averaging (R2, τ2) pairs is wrong). Build gate: k_f = k_s = 1.00 ± 0.03 at SOH ≥ 0.97 |
| `analysis/sop_trim_features.py` | the 12 EW scalars, O(1), explicit dt. `.update/.vector/.excitation/.serialize` |
| `analysis/sop_trim_dataset.py` | pairs labels (HPPC ΔV, rank 2/3, rest ≥ 5τ2) with features (the **preceding** drive cycle) |
| `analysis/sop_trim.py` | `TrimMLP` (514) / `TrimLinear` (26), closed-form decoder, LOCO |
| `analysis/sop_guard.py` | **[never existed — the excitation gate is in `sop_trim_features.py`]** excitation gate, slew, table-based envelope, rejection rule |
| `analysis/sop_hybrid.py` | **[never existed — the inversion is `eval_sop_amps.py` / `sop_core.c`]** 11-step bisection SOP solve (replacing the fixed point), e1/e2 precomputed |
| `analysis/eval_sop_trim.py` | **[never existed — the ladder is `run_safety.py`]** drives the ablation ladder |
| `analysis/mcu_budget.py` | **[never existed — the budget is `run_mcu_table.py`]** **generates** the budget table (does not assert it) |

> **[Note — 37]** That table is the *plan*, not the repository. Four of the
> eight landed under those names: `ecm_pool.py`, `sop_trim_features.py`,
> `sop_trim_dataset.py`, `sop_trim.py`. Four never existed —
> `sop_guard.py` (the excitation gate lives in `sop_trim_features.py` and
> `sop_trim_dataset.py`), `sop_hybrid.py` (the inversion is
> `analysis/eval_sop_amps.py` and `mcu/fw_sop/Src/sop_core.c`),
> `eval_sop_trim.py` (the ladder is `repro/run_safety.py` → `ladder.csv`) and
> `mcu_budget.py` (the budget is `repro/run_mcu_table.py` → `mcu.csv`).
> `SETUP.md`, cited at §27 for its stale baud rate, is the design review's
> document and is not in this repository either.

Existing files to change: `ecm_surface.py` (merged fast branch, export the
in-hull bitmask) and `windows.py` (interpret ctx_len in seconds — it
currently mixes HPPC's 200 samples and the drive cycle's 200 seconds as if
they were the same time axis).

---

## 10. Adoption decision (2026-08-22)

**The SOP arm's baseline is the hybrid linear (A3).**

| | |
|---|---|
| Model | linear readout, 12 features → 2 outputs, **26 parameters** |
| Outputs | resistance multipliers k_f (R0+R1) and k_s (R2), bounded in log space |
| Inputs | 12 O(1) exponentially-weighted statistics, no ring buffer, 14 states (64 B NVM) |
| Nominal | leave-one-cell-out pooled 2RC ECM (pooled in response space) |
| Score | 6-fold measured pulse dV RMSE **44.8 mV** (uncorrected ECM 85.7, Full AI M2 58.7) |
| Worst cell | **62.6 mV** = 1.4× the mean (Full AI M2 is 99.6 = 1.7×) |
| Guard | none (§7.7.5) |

### Why this one

- **The worst case is short.** With a physical model underneath, an unseen
  cell never exceeds 1.4× the mean. Pure AI stretches to 1.7×, and in the
  optimistic (dangerous) direction.
- **The 514-parameter MLP does not beat the 26** (45.4 against 44.8). What is
  doing the work is parameterisation rather than capacity, which is a
  stronger result for the deployment argument.
- **The MCU budget is not the issue.** 26 parameters plus 14 states is a
  rounding error on any BMS MCU. The cost is in the pooled ECM table lookups
  and the 2RC propagation.

### Limits of the evidence this decision rests on

- **The metric is pulse dV, not SOP.** (§11 measured it in amperes:
  discharge 4.94 A; §16 charge 2.05 A.) That the current sensitivity is
  0.62–0.67 × the physical expectation reproduced across all six cells, and
  it means the SOP binary search **is optimistic about available power.**
  This is the largest open item in the adopted baseline.
- **Six cells, one per protocol.** Repeats of the same protocol (CC against
  CC_CELL2) diverge 15 % over life, so this data cannot distinguish anything
  smaller.
- **BOOST_REST's sustained bias was not fixed.** It is the pooled ECM losing
  ground on an out-of-distribution cell, and nothing can be done about it
  with this data except adding cells. The direction of the bias was confirmed
  to be conservative, at least.

---

## 11. SOP in amperes — measured labels and inversion (2026-08-22)

Every number in this project had been in millivolts. The arms are voltage
models and SOP is the current at which voltage reaches the floor, so 44.8 mV
is not an answer to "how many amperes." The conversion factor is not a
constant but dV/dI — resistance itself — so a model that gets resistance
wrong gets it wrong twice.

### 11.1 Measurement-based labels — `analysis/sop_label_measured.csv`

The existing `sop_reference.csv` cannot be a label. It is derived from the
very ECM the hybrid corrects, which is circular; 84.8 % of it extrapolates
beyond the maximum measured current (29.0 A); and 82.7 % is clipped at the
current ceiling rather than by voltage.

HPPC itself is used instead. At every (cell, cycle, SOC group), **four
discharge rates** were measured and each V(τ) recorded — four points of the
V-I characteristic at one operating point, model-free. Solving that to
V(τ) = 2.5 V gives I*. 7,406 rows.

**The label's extrapolation error was measured.** Fitting only the lower
current steps and predicting the higher step that was actually applied:

| Fit | Extrapolation factor | Voltage error (median) | in I* |
|---|---:|---:|---:|
| lowest 3 → highest step | 1.44× | −17.7 mV | 1.17 A |
| lowest 2 → highest step | 2.87× | −40.8 mV | 2.51 A |

The sign is consistently negative — R1 and R2 fall at high current
(Butler–Volmer), so a linear extrapolation reads voltage low and **reads I\*
small.** Even at 3× extrapolation it is 2.5 A (5–11 %), so the whole range is
usable; selecting rows by label quality would leave only low SOC and would
mean **selecting on the label rather than on the model.** So all rows are
evaluated and stratified into bands.

### 11.2 Result — hybrid 4.94 A, uncorrected ECM 7.26 A

`analysis/eval_sop_amps.py`. Fixed-point solution of
V_pre + I·R_eff(I) = V_min. 5,995 rows (1,129 excluded as outside the pooled
table's hull).

| Cell | n | ECM | Hybrid | Gain |
|---|---:|---:|---:|---:|
| BOOST | 1,097 | 5.32 A | 3.95 A | +25.8 % |
| BOOST_NEGPULSE | 936 | 7.33 | 4.71 | +35.8 % |
| BOOST_NEGPULSE_1S | 1,076 | 6.04 | 3.76 | +37.8 % |
| BOOST_REST | 752 | 11.75 | 6.47 | +45.0 % |
| CC | 1,099 | 6.36 | 4.96 | +21.9 % |
| CC_CELL2 | 1,035 | 6.71 | 5.81 | +13.5 % |
| **total** | **5,995** | **7.26 A** | **4.94 A** | **+31.9 %** |

**44.8 mV translates to 4.94 A.** It wins on all six cells, and the 31.9 %
gain is larger than the 24 % in the voltage domain.

### 11.3 The bias changes sign with current magnitude

Current is negative, so direction is read as magnitude — |predicted| >
|measured| means allowing more current, i.e. optimistic.

| Extrapolation factor | median \|I*\| | ECM | Hybrid |
|---|---:|---:|---:|
| ≤ 1.0 | 22.7 A | **+6.66 A optimistic** | **+3.84 A optimistic** |
| 1.0–1.5 | 34.0 | +5.27 optimistic | +2.82 optimistic |
| 1.5–2.5 | 57.6 | +1.13 optimistic | −0.35 conservative |
| > 2.5 | 95.6 | −1.73 conservative | −1.88 conservative |

**Optimistic at small I\*, conservative at large I\*.** The dangerous side
happens to be late life, where the power headroom is already gone —
allowing 6.66 A more at an I* of 22.7 A is a 29 % overpromise. This is
§7.6.6's "current sensitivity is 0.62–0.67 × the physical expectation"
showing itself in the ampere domain. A model with a shallow slope
overestimates small I* and underestimates large I*.

**The tail improves more than the RMSE.**

| | ECM | Hybrid |
|---|---:|---:|
| errors in the optimistic direction | 45.7 % | 42.1 % |
| **optimistic by 5 A or more** | **20.5 %** | **8.2 %** |
| optimistic by 10 A or more | 4.1 % | 2.6 % |
| optimistic by 20 A or more | 0.5 % | 0.1 % |

### 11.4 The reference LSTM cannot run an SOP binary search

`analysis/eval_sop_amps_ai.py`. With no closed form, current is found by
bisection — the paper's SOP method itself. The result is that M1 pinned to
the boundary on 57.3 % of rows and M2 on 96.0 %. Not a code problem: **at
s = 1 both models match the measurement within 9 mV.**

CC cell, 120 pulses, reference current −33.2 A:

| s | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 V | 3.320 | 3.115 | 2.931 | 2.861 | 2.809 | 2.715 | 2.641 |
| M1 dV/ds | — | −0.41 | −0.37 | **−0.07** | −0.05 | −0.05 | −0.04 |
| M2 V | 3.320 | **3.698** | **4.111** | **4.589** | 4.627 | 3.920 | 3.320 |

**Beyond s = 2, M1's slope bends 6× and flattens.** Even at 8× (266 A) it
sits at 2.641 V without reaching the floor. **M2 reverses sign** and gives
4.589 V at s = 3 — a physically impossible value above the cell's ceiling.

The 0.62–0.67 sensitivity measured in §7.6.6 was **the local slope near
s = 1.** SOP is not a local question but asks about s = 2–4, and while the
maximum training current is 30 A, SOP asks about 60–100 A. Neural networks do
not extrapolate; they flatten or go the wrong way. M2 being worse is
apparently because the context encoder responds more strongly to
out-of-distribution input, and the very structure that won 6/6 on drive
cycles is poison here.

### 11.5 What this section answers of the original question

To "can AI-based SOP go on a BMS without a dedicated accelerator," what this
data says is that **the problem is not accuracy but queryability.**

| | Pulse dV | SOP in amperes |
|---|---:|---|
| hybrid linear (26) | 44.8 mV | **4.94 A** |
| uncorrected pooled ECM | 85.7 mV | 7.26 A |
| Full AI M1 (1.08 M) | 77.3 mV | **cannot invert** (57 % saturated) |
| Full AI M2 (1.08 M) | 58.7 mV | **cannot invert** (96 % saturated, sign reversal) |

The ranking in which M2 beat M1 in the voltage domain loses meaning in
amperes. Neither produces an answer.

### 11.6 Remaining limits

- **Only τ = 10 s is validated.** τ = 2 s has zero interpolated rows — in two
  seconds even 29 A does not reach 2.5 V.
- **1,129 rows (15.9 %) fell outside the pooled table's hull.**
- The label itself has a 1.17–2.51 A bias toward reading I* small, so some of
  the "conservative" verdicts in the large-I* range may be label bias, and
  the optimistic range is **in reality more optimistic still.**
- Whether the LSTM's inversion failure is a property of the architecture or
  of this training distribution was not separated. There is no training data
  containing loads above 30 A.

---

## 12. Error budget — what the 79 mV at low SOH is made of

In `fig_hybrid_paper_layout.png` (b) the hybrid error grows monotonically
from 12 mV at SOH 1.00 to 88 mV at 0.68. **SOH is already an input feature.**
So it is not a lack of conditioning, and what remains has to be decomposed.

### 12.1 Scale explains less than half

|dV| itself doubles, 376 → 742 mV, while absolute error grows 7×, 11 →
79 mV. **Relative error also grows 3.8×, 2.8 % → 10.6 %.**

### 12.2 Oracle multipliers — what remains with perfect estimation

The optimal k_f and k_s are fitted per characterisation by least squares
after the fact. No estimator can do better.

| SOH band | mean \|dV\| | ECM | A3 | oracle k | A3 relative | oracle relative |
|---|---:|---:|---:|---:|---:|---:|
| 1.00-.95 | 376 mV | 6 | 11 | **4** | 2.8 % | 1.1 % |
| .95-.90 | 408 | 10 | 12 | **5** | 2.8 % | 1.2 % |
| .90-.85 | 471 | 32 | 26 | **11** | 5.4 % | 2.3 % |
| .85-.80 | 565 | 67 | 46 | **23** | 8.2 % | 4.1 % |
| .80-.75 | 645 | 121 | 61 | **38** | 9.5 % | 6.0 % |
| .75-.68 | 742 | 170 | 79 | **58** | 10.6 % | 7.8 % |

**Only 21 mV of the 79 comes from getting the multiplier wrong; 58 mV
remains even knowing k perfectly.** Giving each pulse its own k drives it to
0 mV (two degrees of freedom for two horizons), so 2RC suffices within a
single pulse, and the limit is **covering a whole characterisation with one
k.**

### 12.3 Giving the multiplier a shape barely helps

Set k_f = a0 + a1·x and re-solve the oracle for various x. Everything is
linear in the parameters, so it is one least squares per characterisation,
and a characterisation has 20 pulses × 2 horizons = 40 equations.

| SOH band | 2 k's | + SOC linear (4) | + current linear (4) | + both (6) | + SOC quadratic (6) |
|---|---:|---:|---:|---:|---:|
| 1.00-.95 | 4 mV | 4 | 4 | 3 | 3 |
| .90-.85 | 11 | 8 | 10 | 7 | 7 |
| .85-.80 | 23 | 17 | 20 | 14 | 16 |
| .80-.75 | 38 | 32 | 34 | 28 | 29 |
| **.75-.68** | **58** | **53** | **55** | **50** | **49** |

Tripling the parameters cuts only 16 % in the lowest band. **It is not a
problem of the multiplier's shape.** (Before writing this section a
SOC-dependent multiplier was expected to lower the ceiling substantially.
That was wrong.)

### 12.4 tau2 really does move, and consistently in one direction

k multiplies R only; tau1 and tau2 stay at their pooled values. tau2 can be
recovered from the stored values — the archive holds
ns(tau) = R2(1 − exp(−tau/tau2)) at two horizons, so their ratio is a single
equation in tau2 (reconstruction error 0.005 µΩ). The fast branch is not
handled this way: the measured tau1 of 0.244 s makes nf2 and nf10 agree to
four digits and inseparable, which is the same reason the trim has only one
fast multiplier.

| SOH band | 2 k's (tau fixed) | + tau2 free | gain | **tau2 factor** |
|---|---:|---:|---:|---:|
| 1.00-.95 | 4 mV | 4 | +6.7 % | 1.22× |
| .95-.90 | 5 | 4 | +8.5 % | 1.12× |
| .90-.85 | 11 | 8 | +30.7 % | 1.22× |
| .85-.80 | 23 | 18 | +22.6 % | **0.86×** |
| .80-.75 | 38 | 29 | +24.2 % | **0.55×** |
| .75-.68 | 58 | 50 | +14.6 % | **0.49×** |

**Aging halves tau2** — the pooled table says 7.15 s while an aged cell wants
about 3.5 s. The sign flips at SOH 0.85 and falls monotonically, so it is not
noise. Even so the ceiling in the lowest band goes 58 → 50 mV, the same level
as the 6-parameter multiplier (49 mV).

### 12.5 The budget and what follows from it

| Component | Size at low SOH |
|---|---:|
| multiplier estimation error (A3 → oracle) | **21 mV** |
| tau2 error | 8 mV |
| insufficient multiplier expressiveness | 8 mV |
| the floor that no method removes | **~50 mV** |

**The floor dominates, and it is the limit of the structure — 2RC plus a
pooled table.** More parameters do not make it go away.

Of the three movable pieces, **the largest is the 21 mV of estimation
error**, and the two expressiveness items together are 16 mV while taking the
parameter count from 26 to 36, which weakens the deployment argument. So
**keeping 26 and improving the estimate pays more.** The multiplier
oscillation shown in §7.7 — half of the 41 events being adjacent round trips
— is the leading source of that 21 mV.

---

## 13. Feature redesign attempts and their rejection (2026-08-22)

§12.5 said the multiplier estimation error, 21 mV, is the largest movable
piece. Two things were tried to find its cause, and **both were rejected.**
The diagnosis that came out matters more than the attempts.

### 13.1 First: the 21 mV is not read noise

Every label has 12 feature blocks and evaluation uses **the last one.**
Averaging all 12 should cut noise by sqrt(12). It got **worse** instead,
45.4 → 47.9 mV (median 46.5). The most recent block wins, so the multiplier
is not static noise: it **really moves, and the latest value matters.**

### 13.2 Do the features contain the answer — the two multipliers differ completely

The oracle k was regressed linearly in-sample. No regularisation, no holdout
— the ceiling of what these 12 features can do.

| | in-sample R² | LOCO R² | strongest single correlation |
|---|---:|---:|---|
| **log k_f** | **0.860** | 0.781 | dR_fast **+0.89** |
| **log k_s** | **0.481** | **0.221** | dR_fast +0.65 |

**k_f is in there and k_s is not.** And `dR_slow`, built for the slow branch,
does not even make the strongest terms for k_s — dR_fast is stronger.

### 13.3 Attempt 1: the filter time constant — rejected

`dR_slow`'s TAU_I is 8 s, matched to the measured median tau2 of 7.95 s.
§12.4 established that an aged cell's tau2 halves, so the fixed value was
suspected. TAU_I ∈ {2, 4, 8, 16, 32} s was computed simultaneously in one
pass, replacing only `dR_slow` (keeping 12 features).

**Sweeping 16× does not move k_s's in-sample R² of 0.481 in the third
decimal.** The filter constant is not the cause.

### 13.4 Attempt 2: read only during sustained load — rejected

Drive-cycle current reverses on a per-second scale, so the slow branch never
reaches steady state. A separate accumulator was added that updates only
while `|I| > 5 A` has been sustained for 3 seconds or more.

The distribution narrowed markedly (5–95 % from −0.5–28.4 to −0.2–11.0).
**But that was less information rather than more stability** — in-sample
rises slightly, 0.481 → 0.489, while **LOCO falls, 0.221 → 0.196.**

### 13.5 Fixing the channel count is what saved the verdict

Adding all nine channels raises k_s in-sample to 0.506 while **LOCO collapses
to 0.081.** With only 274 characterisations, growing the features to 21 does
that. Had the primary verdict not been fixed in advance as "replace only the
`dR_slow` channel, keep 12 features," the in-sample rise would have been read
as an improvement.

### 13.6 What came out instead — the two multipliers have opposite spread structure

| | within-cell sd | between-cell | **between-cell share** |
|---|---:|---:|---:|
| log k_f | 0.0887 | 0.1147 | **63 %** |
| log k_s | 0.1215 | 0.0634 | **21 %** |

**k_f's variation is mostly "which cell," and k_s's variation is mostly
"which moment in that cell's life."** That explains why feature redesign does
not help k_s — however much an instantaneous feature is refined, the thing to
be captured is not a between-cell difference but **one cell's trajectory in
time.**

For the same reason tying `k_s = rho · k_f` does not work either:
corr(log k_f, log k_s) = 0.583 and its LOCO R² is 0.067, worse than the
feature-based 0.221. The two multipliers follow different things.

### 13.7 Where this diagnosis points

- **k_f is a between-cell problem** → adding cells is the direct route. Even
  without new measurement, Mendeley and RPCWBY have more cells with HPPC.
- **k_s is a trajectory problem** → what is needed is **time structure**
  rather than an instantaneous feature. The multiplier flows smoothly with
  aging while the current estimator treats every characterisation as
  independent. Half of §7.7's 41 oscillations being adjacent round trips says
  the same thing.
- **Reading the load step after a rest** has not been tried. Attempt 2 chose
  "sustained load," but for the slow branch to mean anything the RC state has
  to start from zero, so the right condition is not sustained load but a
  **rest → load transition.** A vehicle does that at every key-on.

---

## 14. Adding cells — integrating RPCWBY (2026-08-22)

§13.7 proposed three routes; the two that stay inside the estimator (the
guard in §7.7 and the feature redesign in §13, plus time tracking) each
gained about 2 % on the mean and **made the worst cell worse.** The remaining
direct route is cell count. Without new measurement, `raw/RPCWBY` has more
aged Samsung 30T cells.

### 14.1 What that dataset is

Not a reference but **a dataset that measured SOP directly** (Chen, Emadi,
Kollmeyer, "Battery State of Power Measurement: A Generalized Methodology").

| | |
|---|---|
| Test#1 (RPC_CC) | one Samsung 30T, cycles 1 → 2013, 18 characterisations, aged by 1C CC discharge |
| Test#2 (RPC_US06) | one Samsung 30T, aged on the US06 profile, 15 characterisations |
| Voltage window | 2.55–4.15 V (UYPYDJ is 2.5–4.2) |
| Current limits | 30 A / −15 A |
| Logging | 1 Hz (UYPYDJ HPPC is 10 Hz within pulses) |

### 14.2 The first extraction was wrong, and it looked like a cell difference

Taking every `|I| > 1 A` stretch as a pulse gave a resistance at 3 A of
0.39–0.72 × UYPYDJ's. The same cell model cannot do that. The cause was
**the 1C discharge steps that move SOC** getting mixed in — 74 of them
exceeded 60 s in one file. During a long discharge V_pre is not a reference
point and the RC has already developed, so dV reads small.

Filtering to length 5–20 s with a preceding rest ≥ 20 s took 18,152 → 9,398
rows and the problem vanished. **What looked like a cell difference was a
difference in the definition of a pulse.**

### 14.3 Compatibility — 0.5 % at 10 s once currents are matched

Fresh pulses cluster at 30 A in RPCWBY (the search pins to the current
ceiling) and at 34 A in UYPYDJ, so comparing directly reads a 4 A difference
as a cell difference. UYPYDJ steps through four levels, so **30 A was
interpolated** per cell and SOC to match operating points.

| tau | RPCWBY / median of UYPYDJ's 6 cells |
|---|---:|
| 2 s | **+2.5 %** |
| **10 s** | **+0.5 %** |

0.5 % at the 10 s horizon where SOP lives. The two RPCWBY cells agree within
0.5 % of each other, so the +2.5 % at 2 s reads as a **systematic offset
between labs** rather than cell variation.

### 14.4 SOH anchors

Step_Index is reused within a file (one of 164 spans 919 minutes), so
grouping by it gives 3.15 Ah for a 3.0 Ah cell. They were found directly from
the current waveform — integrate the stretch that discharges continuously at
about 1C from the top of the window to the floor.

| Cell | Anchors | Capacity | SOH |
|---|---:|---|---|
| RPC_CC | 18 | 2.910 → 2.127 Ah | 1.000 → **0.731** |
| RPC_US06 | 15 | 2.980 → 2.395 Ah | 1.000 → 0.804 |

All run the full 4.15 → 2.500 V, monotonically decreasing, zero missing.

### 14.5 A defect caught while borrowing tau2 — and an independent check of §12.4

Two horizons alone cannot separate the branches, so tau2 is needed. The first
attempt inserted UYPYDJ's median of 8.111 s as a constant — **having just
confirmed in §12.4 that aging halves tau2 and used a constant anyway.**
RPCWBY is 1 Hz and a 10 s pulse gives 10 points, so tau2 can be fitted (tau1
at 0.244 s cannot be, but at horizons of 2 s and above the fast branch enters
only as one R0+R1 lump, so nothing is lost).

| SOH | 1.00–0.95 | 0.95–0.90 | 0.90–0.85 | 0.85–0.80 | 0.80–0.72 |
|---|---:|---:|---:|---:|---:|
| median tau2 | **8.53 s** | 8.28 | 7.34 | 5.78 | **4.03 s** |

**0.47×.** Almost identical to the 0.49× that §12.4 obtained by inverting
UYPYDJ's two stored horizons. **A different dataset, a different cell and a
different method** (direct fit against two-horizon inversion) give the same
value, so tau2's decrease with aging is not an artefact.

When the grid was first opened to 40 s, the 5–95 % of fits piled up at the
ceiling — a 10 s pulse cannot distinguish tau2 = 20 s from 40 s. Narrowing
the grid to 16 s and rejecting boundary-pinned fits normalised the
distribution (2.4–12.2 s).

### 14.6 Current bins are the common language

UYPYDJ steps four rates labelled 0–3, but RPCWBY's SOP search is continuous
over 3–30 A and has no such label. `rate_rank` was redefined as a **current
bin index** (edges 2/7/16/26/40 A, so that 30 A and 34 A fall in one bin).
`ECMSurface` already uses rank only as a tick on the current ladder and
interpolates with `rank_I`, so the code below is unchanged.

**Changing the key changes the baseline too.** So three numbers are reported
separately: (1) the old rank key with 6 cells = 44.8 mV, (2) the current key
with 6 cells = the effect of the key change, (3) the current key with 8 cells
= the effect of adding cells. (1)→(2) and (2)→(3) must not be reported mixed.

### 14.7 The build gate fails, and that is the gate doing its job

In the 8-cell pool, five of six fail (k_s ≈ 0.95). §14.3's systematic offset
is the cause — d2 is +2.5 % while d10 is +0.5 %, so (d10 − d2) shrinks, and
the two-horizon reduction amplifies that difference to 5 % in the branch
separation. That exceeds the 3 % tolerance.

The gate asks "does it match a fresh cell without correction?" The hybrid is
a thing that corrects, and the multipliers exist precisely to absorb such
offsets. So **the verdict comes from the holdout RMSE, not from the gate.**
If it gets worse, aligning RPCWBY to the five training cells with a scalar
first remains available.

### 14.8 What this experiment does and does not measure

RPCWBY cells enter **the pooled table only**, not the trim training. The
trim's features come from
`cache_t/uypydj_*_Fifteen_Drive_Cycles.npz` and RPCWBY has no drive data of
that form. So what is measured is **"does a better nominal help the
hybrid"**, not "does it improve with more training cells." Evaluation is
still leave-one-cell-out over UYPYDJ's six.

### 14.9 Result — adding cells is null on the mean and effective on the worst cell

| Configuration | ECM (A0) | hybrid mean | **worst cell** |
|---|---:|---:|---:|
| (1) rank key, 6 cells | 86.0 mV | 57.1 | 74.8 |
| (2) current key, 6 cells | 74.0 | **55.8** | 68.4 |
| (3) current key, 8 cells | 75.2 | 56.1 | **64.1** |

**(1)→(2), the key change**: current bins improve the uncorrected ECM
substantially (86.0 → 74.0, −14 %). Binning by current suits a pooled
nominal better than four rate labels. The hybrid moves a little
(57.1 → 55.8) and the worst cell −8.6 %.

**(2)→(3), adding cells**: the mean is **unchanged** at 55.8 → 56.1 (noise
level) while **the worst cell improves 68.4 → 64.1, −6.3 %.** That worst cell
is BOOST_REST — **the very cell where the guard (§7.7), the feature redesign
(§13) and Kalman tracking all failed.**

| Cell | 6 cells → 8 cells |
|---|---|
| **BOOST_REST** | 68.4 → **64.1** (−6.3 %) |
| BOOST_NEGPULSE | 52.8 → 50.4 (−4.5 %) |
| CC_CELL2 | 49.9 → 48.5 (−2.8 %) |
| CC | 61.7 → 61.7 (0 %) |
| BOOST | 46.3 → 50.4 (+8.9 %) |
| **BOOST_NEGPULSE_1S** | 55.8 → **61.4** (+10 %) |

**Three cells improve and two get worse, cancelling in the mean.** Exactly as
§13.6 diagnosed — k_f is a between-cell problem, so **the cells that were out
of distribution are precisely the ones that improve**, while cells that
already fitted well get worse as the lab offset mixes in.

The 8-cell pool is **broader and less precise.** A BMS has to guarantee the
worst case, so that trade is favourable, and above all **this is the first
time the worst cell has improved** — the three previous attempts all went the
other way, gaining a little on the mean and losing on the worst.

The likely cause of the two that got worse is §14.7's lab offset. Aligning
RPCWBY to the training cells could leave the worst-case improvement without
the cancellation — next section.

---

## 15. Regressing I* directly — and why pure data-driven fails (2026-08-23)

§11.4 showed the reference LSTM cannot run an SOP binary search. That failure
comes from **extrapolation** — training current never exceeds 30 A while SOP
asks about 60–100 A. So what happens in a form that needs no extrapolation,
with **I\* itself as the target**?

The features are exactly what the hybrid sees: the trim's 12 O(1) statistics
plus headroom voltage (V_pre − 2.5 V). The pulse itself is not seen — when a
vehicle asks "how much can I draw," it has not drawn yet. 6-cell
leave-one-cell-out, 5,995 rows.

| Form | Model | Mean | Worst cell |
|---|---|---:|---:|
| **D0  I\* direct** | linear | 11.27 A | 12.72 |
| **D0  I\* direct** | tiny MLP (529) | **17.39** | 18.05 |
| D1  I\*/I_ecm | linear | 5.05 | 6.63 |
| D1  I\*/I_ecm | tiny | 5.28 | 10.33 |
| **D2  I\*−I_ecm** | **tiny (529)** | **4.39** | **6.48** |
| D2  I\*−I_ecm | linear | 4.82 | 6.58 |
| ECM (uncorrected) | — | 7.25 | 11.75 |
| hybrid inversion (26) | — | 4.94 | 6.47 |

### 15.1 Pure data-driven is worse than an uncorrected ECM

D0 gives 11.27 A (linear) and **17.39 A (MLP)**, short of the ECM's 7.25 A —
even with the extrapolation problem removed.

The reason lies in how the target is composed. I\* spans 9.1–156.3 A, a
factor of 17, and most of that variation is **physics** set by SOC and V_pre.
Direct regression has to learn that from scratch, and 274 characterisations
are not enough.

**And the MLP is 54 % worse than linear.** Given capacity, it overfits six
cells.

### 15.2 With physics as the reference it works immediately — and then capacity helps

Learning **only the difference** from the ECM's answer (D2) gives 4.39 A,
beating the hybrid inversion (4.94) by 11 %. Same features, same split, same
rows.

**In D2 the MLP beats linear** (4.39 against 4.82), the opposite of D0. When
there is only a residual to learn, capacity helps; when the whole thing has
to be learned, it is poison. That contrast is the sharpest demonstration in
this project of what a physical baseline does.

The worst cell is 6.48 against 6.47, a tie with the hybrid. Both stall near
6.5 A on BOOST_REST.

### 15.3 This is not "doing SOP with AI"

D2 takes the ECM's I\* as an input. It does not exist without the physical
model. The accurate statement is **"529 parameters on top of an ECM"**, and
that is the best form found so far. Organised along the architecture axis:

| Approach | Target | Physical baseline | Mean | Outcome |
|---|---|---|---:|---|
| Full AI (1.08 M) | voltage → inversion | none | — | **cannot invert** |
| direct regression D0 (529) | I\* | none | 11.3–17.4 A | worse than the ECM |
| hybrid (26) | resistance multipliers | ECM | 4.94 A | closed form |
| **residual regression D2 (529)** | I\* − I_ecm | ECM | **4.39 A** | best |

The two forms without a physical baseline fail and the two with one succeed.
**What separates them is the presence of a baseline, not the parameter
count.**

### 15.4 Stated: this is the oracle state

SOC and SOH in this table are the file's true values. The end-to-end version
with an EKF attached is measured in §15.5.

### 15.5 End to end — attaching the EKF's SOC does not break it

SOC comes from the filter rather than the file, and **the ECM baseline is
re-solved at the estimated SOC too.** Leaving the baseline as the answer at
true SOC would have the model learn residuals on top of physics the BMS does
not have, adding accuracy to the chain that is not there.

| Form | Oracle (5,995 rows) | End to end (2,927 rows) |
|---|---:|---:|
| ECM (uncorrected) | 7.25 A | 5.46 |
| **D2 tiny (529)** | **4.39** | **3.38** |
| D2 linear | 4.82 | 3.72 |
| D0 linear | 11.27 | 5.41 |
| **D0 tiny** | 17.39 | **27.85** |

**The end-to-end column is better because of the row set, not the EKF** — the
ECM itself improves 7.25 → 5.46. The rank-3 pulse subset the filter covers is
an easier region. The two columns must not be compared across; each is read
only within itself.

Within the same rows: ECM 5.46 → **D2 tiny 3.38 A (38 % better)**, worst cell
9.64 → **5.26 A (45 %)**.

**SOC estimation error does not break D2.** §11.3's synthetic error
propagation predicted 4.94 → 14.23 A at a systematic 2 %, but attaching the
real filter produces no such collapse — real EKF error (median |·| 0.0139)
behaves differently from a constant offset fixed across a cell. §11's
recheck already showed the same.

**And D0 tiny gets worse end to end, 17.39 → 27.85 A.** Once SOC becomes an
estimate, the model that tried to learn physics from scratch wobbles. D0
linear, by contrast, holds at 5.41 A. §15.2's observation — **without a
physical baseline, capacity is poison** — reproduces more strongly end to
end.

### 15.6 The adopted chain (2026-08-23)

| Component | Method | Parameters | Performance |
|---|---|---:|---|
| SOC | EKF, state [SOC, V1, V2] + deterministic hysteresis | 0 | median \|error\| 0.0139 |
| SOH | dQ/dV CNN | 10,945 | RMSE 0.0128 |
| **SOP** | **pooled ECM + tiny MLP residual** | **529** | **3.38 A** (ECM 5.46 on the same rows) |

The "small AI" in SOP is 529 parameters sitting on a physical baseline. Pure
AI does not work at any size — 1.08 M cannot be inverted, and 529 is worse
than the ECM.

---

## 16. SOP in the charge direction (2026-08-23)

Claiming to have done SOP requires both directions, and §11–15 were all
discharge. The regenerative braking limit is half of a BMS, and being
optimistic there means overcharge, so the safety implications are if anything
larger.

### 16.1 Charge labels are far better conditioned than discharge

Same method — fit a line through HPPC's four current steps and solve to
V_max = 4.2 V. SOC > 0.92 and V_tau ≥ 4.195 V are excluded: 10.2 % of charge
rows sit at exactly 4.2000 V, which is the cycler truncating the pulse rather
than the cell's response, and is not a resistance measurement.

| Extrapolation factor | Charge | Discharge |
|---|---:|---:|
| **≤ 1.0 (interpolated)** | **34.1 %** | 8.0 % |
| 1.0–1.5 | 35.2 % | 14.1 % |
| 1.5–2.5 | 30.2 % | 32.7 % |
| **> 2.5** | **0.4 %** | **45.1 %** |
| difference between the two fits (median) | **0.2 A** | 1.4 A |

**Charge SOP is far more solidly validated by this data.** There is a
physical reason — the charge ceiling (4.2 V) is close to the cell while the
discharge floor (2.5 V) is far, so a measurable current reaches the ceiling
and not the floor.

### 16.2 The discharge trim must not be used on charge

The trim is trained only on **discharge** pulses with
`rate_rank in ("2","3")`. Multiplying charge resistance by those multipliers
**loses to the uncorrected ECM on two of six cells** — CC_CELL2 1.37 →
2.21 A (−62 %), BOOST_NEGPULSE 2.63 → 3.21 (−22 %). In contrast to winning
6/6 on discharge.

Training a separate trim on charge pulses gives 51.03 → 33.66 mV in the
voltage domain, **+34.0 %, 6/6** — the same level as discharge's +33.6 %.
What differs from discharge (0.90–1.10) is that **k_s stays pinned to 1 at
0.976–1.001**: on charge the slow branch barely needs correcting and only the
fast multiplier works.

In amperes it is 2.62 → **2.41 A**, a gain of +25.7 % → **+31.5 %**. The two
losing cells only halve their loss without changing sign (−62 → −31 %,
−22 → −9 %). Both are **cells where the ECM is already good**, and adding
multiplier estimation noise where there is nothing to correct is the same
thing §7.6.5 saw above SOH 0.95.

### 16.3 Tracing the remaining +1.02 A optimistic bias — three candidates rejected, one confirmed

| Candidate | Test | Result |
|---|---|---|
| the label's linear extrapolation | predict the highest rate from the lower ones | **opposite direction** — on charge R falls, so a straight line reads voltage +26.6 to +77.5 mV high, which makes the label I* too large and the prediction look conservative |
| pooled charge resistance | compare 4 steps × 4 SOH bands against measurement | agrees within 3.5–4.8 %; at low SOH and high rate the pooled value is actually 5 % larger (conservative) |
| ladder clamping | fraction of predicted I* outside the ladder | 62.8 % are outside, but the relative bias there is **smaller** (3.1 % against 10.8 %) and conservative in direction |
| **reference-point mismatch** | intercept against V_pre | **confirmed** |

The clue was that **the bias is fixed at about +1 A in absolute terms** —
+0.92 inside the ladder (I* 8.5 A) and +1.13 outside (37.0 A), independent of
the size of I*. With R_eff at 12–25 mΩ, +1 A corresponds to 20 mV.

The label starts from the **intercept** of the line fitted to four current
steps; the inversion starts from **the recorded V_pre of the first rank.**
The four pulses are applied in sequence and the cell drifts, so the two
differ by a median **+14.1 mV** (V_pre varies by 13.8 mV within a group).
Starting from the intercept instead:

| Inversion start voltage | n | RMSE | Bias |
|---|---:|---:|---:|
| recorded V_pre | 3,707 | 2.57 A | +1.06 A |
| **fitted intercept** | 3,707 | **2.05 A** | **+0.10 A** |

**The bias disappears.** This was not a model defect but the label and the
inversion starting from different reference points.

(This check took two attempts. The first hard-coded tau at 10 s while the
evaluation table mixes τ = 2 s and 10 s, so half of it solved 2 s labels
against 10 s resistance. It gave 5.84 A where the stored value for the same
rows was 2.41 A, and the discrepancy was traced to the harness before being
fixed.)

### 16.4 Discharge has no such defect — it has a different one

| | Discharge | Charge |
|---|---:|---:|
| intercept − first V_pre | −12.0 mV | +14.1 mV |
| **V_pre variation within a group** | **2.4 mV** | **13.8 mV** |
| substituting the intercept | 4.94 → 5.12 A (worse) | 2.57 → 2.05 A (better) |

**Between discharge pulses the cell barely drifts** (2.4 mV). On charge the
pulse raises SOC and the following rest does not fully return it, so it
drifts 6×. So V_pre is the right reference on discharge, and the intercept
(−12.0 mV) is just the intercept of a line where 45 % is extrapolated 2.5× or
more, which makes it worse as a starting point.

The discharge side's −0.76 A conservative bias is explained in both sign and
magnitude by the **extrapolation bias** §11.1 measured (1.17–2.51 A, in the
direction of reading I* small).

### 16.5 Summary

| | ECM | Hybrid | after correcting the label reference |
|---|---:|---:|---:|
| **charge** | 3.52 A | 2.41 A | **2.05 A** (bias +0.10) |
| discharge | 7.25 A | 4.94 A | — (no such defect) |

**Charge SOP is more than twice as accurate as discharge.** Its label
conditions are far better too.

The reference correction is **a diagnosis, not a deployable fix** — the
intercept is an after-the-fact value that knows all four current steps, while
a vehicle knows only its present terminal voltage. The practical implication
is on the label side: a measurement-based charge I* label absorbs the drift
across four pulses and is **about 1 A small.**

---

## 17. Two scorings, kept distinct (2026-08-23)

The same hybrid carries **two numbers** in this document, because the
scorings differ. Neither is wrong, but placed side by side they read as a
regression.

| Scoring | What it counts | With the rank key, 6 cells |
|---|---|---:|
| `sop_trim.py` itself | **every feature block** in that pool's dataset (12 per label) | **57.1 mV** |
| **A13** | the **intersection pulses** the LSTM could also be placed on, **last block only** | **44.8 mV** |

§7.5 and §14.9 use the former; §7.6.5 and §10 use the latter.

### 17.1 All four pools under the A13 scoring

| Pool | BOOST | BNP | BNP_1S | B_REST | CC | CC_2 | mean | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank key, 6 cells | 32.7 | 38.8 | 46.9 | 62.6 | 41.5 | 46.4 | 44.8 | 62.6 |
| **current key, 6 cells** | 30.7 | 38.9 | 47.7 | 60.9 | 40.6 | 38.0 | **42.8** | 60.9 |
| current key, 8 cells | 38.2 | 35.5 | 55.0 | 57.5 | 44.3 | 38.1 | 44.8 | **57.5** |
| aligned, 8 cells | 37.9 | 35.2 | 55.0 | 57.4 | 43.9 | 38.3 | 44.6 | 57.4 |

**The current key beats the rank key on both mean and worst** (42.8/60.9
against 44.8/62.6). The key change is a free improvement.

**Cell count is a trade between mean and worst** — 6 cells give a mean of
42.8, 8 cells a worst of 57.5. Same direction as §14.9's self-scoring, and
alignment (§14.7) remains pointless (44.8 → 44.6).

### 17.2 Which pool §10's adoption decision should be read against

The 44.8 mV §10 cited is the A13 scoring of the **rank key, 6 cells**,
because that was the only configuration at the time; §14 later introduced the
current key and 8 cells. Updated:

| | mean | worst |
|---|---:|---:|
| at the time of §10 (rank, 6 cells) | 44.8 mV | 62.6 |
| **taking the mean** (current, 6 cells) | **42.8** | 60.9 |
| **taking the worst** (current, 8 cells) | 44.8 | **57.5** |

By §10's own criterion — a BMS guarantees the worst case, not the average —
it is **the current key with 8 cells.** Though what that choice buys is the
worst cell 62.6 → 57.5 (−8 %) with the mean unchanged.

---

## 18. The temperature axis (2026-08-23)

Every measurement in this project was at 25 °C. The reference paper's figure
sweeps −20 to 40 °C, and there its ECM degrades 3×, from 25 mV at 40 °C to
77.5 mV at −20 °C. **Temperature is the ECM's weakest axis**, so the claim
that a corrected ECM beats a neural network is incomplete without that axis.

### 18.1 What is available

RPCWBY Test#3 measures one Samsung 30T at −20, −10, 0, 10, 25 and 40 °C with
2 s, 10 s and 30 s pulses — 18 combinations
(`analysis/rpcwby_temp_pulses.py`, 1,656 pulses from 24 files). It is **a
different cell from the six aging cells**, so every evaluation on it is a
cross-dataset generalisation test.

The measured cell temperature is consistently above the setpoint — −17.9 °C
at a −20 °C setting, −8.1 at −10. That is self-heating, and since
`ecm_temp_factor` uses `T_cell_C`, evaluation uses the measured value too.

### 18.2 Resistance holds across six temperatures and three horizons

Multiply the pooled table (UYPYDJ's 6 cells at 25 °C) by a temperature factor
g (measured separately from Mendeley) and predict a third cell. Data never
used in training.

| Setpoint T | tau=2 s | tau=10 s | tau=30 s |
|---|---:|---:|---:|
| −20 °C | 0.81 | 0.94 | 1.11 |
| −10 | 0.89 | 0.90 | 0.93 |
| 0 | 0.97 | 1.09 | 1.02 |
| 10 | 0.86 | 0.93 | 0.87 |
| 25 | 0.89 | 0.90 | 0.82 |
| 40 | 0.91 | 0.90 | 0.79 |

(predicted/measured ratio. The resistance itself varies **9×**, 9.6 to
88.2 mΩ.)

**Overall median 0.908, median relative error 13.7 %.** Across a 9× range the
ratio stays within 0.79–1.11.

But it **underestimates by 9 % everywhere**, and that direction is optimistic
for SOP. At tau = 30 s a temperature dependence appears (0.79 at 40 °C, 1.11
at −20 °C) — 30 s extrapolates a table reduced to 2 and 10 s, so the
direction is expected.

**This validates τ = 2 s and 30 s at the resistance level for the first
time.**

### 18.3 But per-temperature SOP labels cannot be made

Test#3 is an SOP **search** dataset. A search converges, so it leaves one
current per SOC rather than a fan — of 324 groups (T, tau, SOC on a 0.05
grid), only 62 have three or more pulses with a current span of 3 A or more,
and even those cluster at −20 and −10 °C.

Extending a line through those 62 to 2.55 V gives **extrapolation factors of
8.7–93.8×** and physically impossible I* values of 91–1411 A. Unusable as
labels.

The cause is physical. **The warmer it is, the less 30 A reaches 2.55 V** —
SOP at 25 °C and mid SOC is 60–100 A against a 30 A cycler ceiling. Only in
the cold does resistance grow 9× enough to reach it, which is why the
solvable groups cluster there. This is the same wall as UYPYDJ's discharge
labels being 45 % extrapolated.

### 18.4 The authors' SOP extraction — two temperatures, but they extrapolate too

`rpcwby_sop_summary.csv` holds Test#1/#2's SOP: 938 valid values, 2 cells ×
10 and 25 °C × cycles 1–1994 × SOC 0.005–1.000.

The physics is clean. At SOC 0.5 from cycle 1 to 1428, 25 °C goes
−100.5 → −87.4 W and 10 °C goes −96.5 → −76.9 W — **degradation is steeper in
the cold** (−20 % against −13 %).

**Not current-limited**: the upper values run continuously over
111.5–112.8 W and only 2–9 % cluster near 30 A. They are set by voltage.

But the median of |SOP| / 2.55 V is **34.3 A**, above their own 30 A ceiling.
Together with the paper's title, "A Generalized Methodology," this reads as
their extrapolating past the current they applied. It is **the same kind of
limit** as this project's labels, so using it as a reference is an
extrapolation-against-extrapolation comparison.

### 18.5 What the temperature axis actually gave and did not

| | Status |
|---|---|
| temperature generalisation of the resistance model | **validated** — 6 temperatures × 3 horizons, external cell, 13.7 % |
| tau = 2 s and 30 s | **validated at the resistance level** |
| per-temperature **SOP in amperes** | **impossible** — the 30 A ceiling does not reach the floor on the warm side |
| temperature generalisation of the hybrid | **impossible** — the trim's features come from UYPYDJ driving and Test#3 has no driving |

The last is structural. Testing the hybrid on the temperature axis needs
**driving and cold pulses on the same cell**, and none of the three datasets
has that combination.

### 18.6 Scored against the authors' measured SOP (10 / 25 °C)

`analysis/eval_sop_rpcwby.py`. What is scored is **the uncorrected pooled ECM
plus the temperature factor** — the trained trim is indexed by UYPYDJ
characterisation cycles and is meaningless on another lab's cell, so
kf = ks = 1. Nothing is fitted to this data.

**How usable rows are separated.** The authors' protocol applies a 30 A
ceiling to a 2.55 V floor. So every row is one of two things.

    current-limited   I = 30 A,      V_end > 2.55   ->  |SOP| = 30 × V_end
    voltage-limited   V_end = 2.55,  I* < 30 A      ->  |SOP| = 2.55 × I*

The two meet at exactly 2.55 × 30 = **76.5 W**. Above that the cycler set the
answer, so the voltage model cannot be tested and those rows are discarded.
The split is self-consistent — current-limited rows have V_end 2.551–3.759 V,
all inside [2.55, 4.15], and voltage-limited rows have I* 1.8–29.9 A, all
below 30 A. No row overlaps.

Of 938 rows, **342 are voltage-limited** (36 %); the other 64 % are unusable.

**The SOC axis is the rated one.** Line 117 of the readme:
`SOC = 1 - (Discharge_Capacity - Charge_Capacity)/cell_rated_capacity`. Same
convention as this project's axis. It was assumed to be on an aged-capacity
basis and converted by SOC × SOH, but the readme rejects that.

**Result** (6 pool variants × 344 rows = 2064, of which 319 in hull;
τ = 10 s, v_pre = OCV − M):

| | n | RMSE | Bias | median ratio |
|---|---:|---:|---:|---:|
| all | 319 | 6.44 A | −4.18 A | 1.210 |
| 10 °C | 300 | 6.59 | −4.61 | 1.240 |
| 25 °C | 19 | 3.14 | +2.49 | 0.930 |
| SOH 0.88–0.95 | 78 | 4.21 | −3.01 | 1.152 |
| SOH 0.80–0.88 | 144 | 7.46 | −5.92 | 1.314 |

Current is negative, so **a ratio of 1.21 means seeing 21 % more usable
current** — the dangerous side for a BMS. And that overestimate **deepens
with age** (1.152 → 1.314).

**The source of the error is resistance.** Matching the measured I* requires
R to be **1.29×** larger (predicted 36.9 mΩ, required 48.1 mΩ). The voltage
headroom v_pre − 2.55 is a wide median 0.969 V, so OCV error does not explain
it, and turning the hysteresis term M on and off moves the ratio only between
1.21 and 1.29. Dividing the R ratio by SOH gives 0.869 (SOH ≥ 0.88) → 0.726
(0.80–0.88) → 0.757 (< 0.80).

**Two external tests point the same way.** Test#3's resistance validation
gave predicted/measured = 0.908; here it is 0.778. Both say the pooled ECM
reads resistance **low**, and it deepens toward the corners (low SOC, aged,
high current). Test#3 was a gentle 0.33C condition and this is up to 30 A, so
the difference in magnitude matches the direction too.

**The required correction is within the trim's range.** k_f =
exp(0.470 tanh u) has a ceiling of 1.60× and the required value is 1.29×. So
the hybrid's correction mechanism is **shaped to fit** this error. That
cannot be confirmed here, though — the trim's features come from driving
windows and this evaluation has no such pipeline. **Test#2 is US06 driving,
so it is possible in principle** (18.7).

**The limits, stated.** Only 15 % pass the hull, and 25 °C has just 19 rows
(when warm, 30 A does not reach the floor, so voltage-limited rows survive
only at low SOC, which is outside the pool's SOC floor of 0.053).
**So this is a test at 10 °C, not a test of temperature contrast.**

### 18.7 Remaining

Test#2 is the **only** file that holds US06 driving and measured SOP on one
cell. Standing up the trim feature pipeline on it would score the hybrid on
an external cell, against measured SOP, at 10 and 25 °C. That is the
strongest validation possible in this project, and it has not been done.

---

## 19. SOP error is not symmetric (2026-08-23)

RMSE penalises optimism (reading current higher than it is) and conservatism
equally. In a BMS they are nothing alike — **optimism breaches undervoltage
protection, conservatism only leaves output unused.** Through §15 this
project had measured SOP only by RMSE. Split by sign, a different picture
appears.

### 19.1 The headline number was diluted

Over all 5,995 rows the hybrid's optimism rate is 42.1 % (ECM 45.7 %). But
that set is filled with rows whose labels cannot be trusted. Labels come from
stretching HPPC's 4-rate fan to the voltage floor, and
`extrap = |I*| / max|I_meas|` is that extrapolation factor.

| extrap | n | hybrid optimism | mean overshoot | median \|I*\| |
|---|---:|---:|---:|---:|
| ≤ 1.0 (interpolated) | 143 | **97.9 %** | +3.93 A | 22.7 A |
| 1.0–1.5 | 508 | **78.7 %** | +4.15 A | 34.0 A |
| 1.5–2.5 | 2,134 | 46.3 % | −0.27 A | 57.5 A |
| 2.5–5 | 3,210 | 31.0 % | −2.16 A | 95.6 A |

The median |I*| of the high-extrapolation rows is 95.6 A — **32C** on a 3 Ah
cell. Those labels are inflated, and measured against them any model looks
conservative.

**Not an SOH selection effect.** In every SOH band the trustworthy labels are
far more optimistic — 80.6 % against 40.3 % above SOH 0.90, 68.7 % against
31.2 % over 0.80–0.90, and 88.3 % against 57.6 % below 0.80.

**So the honest number is not 42.1 % but 82.9 % on the 651 trustworthy
rows** (ECM 84.8 %).

### 19.2 The label's chord explains 9 %p of that

`sop_label.py` produces two fits. `lin4` is least squares on all four rates;
`lin2hi` uses only the top two. R falls with current (0.7× from 2.6 A to
29.6 A), so a line through four points is pulled up by the low-rate points,
overestimates resistance in the high-current range, and therefore reads |I*|
**low.**

Switching the label to `lin2hi` takes optimism from 82.9 to **73.9 %**
(symmetric trim) and 52.5 to 44.1 % (pinball q = 0.9). So about 9 %p of the
apparent optimism is the label. **The core finding survives both labels.**

`lin2hi` is physically better but also moves the label in the direction that
flatters the model, so **the safety margin is calibrated on the conservative
`lin4` while `lin2hi` is reported alongside as the physical best estimate.**
`eval_sop_amps.py --label-fit` produces both.

### 19.3 The required resistance multiplier — where τ = 2 s breaks

v_pre and v_min are shared by label and model, so the difference comes
entirely from R_eff(I*). Over the 651 trustworthy rows, the factor R would
need to match the measured I*:

| | median | 90th percentile |
|---|---:|---:|
| hybrid, all | 1.109 | — |
| ECM, all | 1.198 | — |
| SOH 0.90–1.01 | 1.055 | 1.085 |
| SOH 0.80–0.90 | 1.052 | 1.176 |
| SOH < 0.80 | 1.143 | 1.484 |
| **tau = 2 s** | **1.323** | **1.843** |
| tau = 10 s | 1.076 | 1.257 |

τ = 10 s SOP is nearly right and **τ = 2 s is 32 % short.** The 90th
percentile of 1.843 exceeds the trim's k_f ceiling of exp(0.470) = 1.60 — a
correction the present structure cannot reach.

**τ = 2 s is always extrapolated.** All 143 rows with extrap ≤ 1.0 are
τ = 10 s. So τ = 2 s SOP has never been validated by interpolation in this
dataset.

**But 10–19 % is real even in pure interpolation.** At extrap ≤ 1.0
(τ = 10 s, 143 rows) the hybrid still needs 1.187× (lin4) or 1.101×
(lin2hi). An underprediction unrelated to extrapolation remains.

**External comparisons point the same way.** The required factor against the
RPCWBY authors' SOP is 1.29 (§18.6), and Test#3's resistance validation gives
predicted/measured 0.908 (§18.2). Three independent tests all say the pooled
ECM reads resistance low.

### 19.4 Yet on measured pulses it is unbiased

Comparing the holdout's pooled nominal (k = 1) against measured dV gives a
ratio of 0.98–1.08 — at both horizons, both rates (20.2 A, 29.2 A) and every
SOH. **The model is right where it was measured.**

Not a contradiction. SOP asks about **beyond** the fan, and there the label
extrapolates with a chord while the model extrapolates with a tangent, fixing
R at the end of the rate ladder. R falls with current, so the tangent sits
above the chord and the model's I* is larger. §19.3's factor is **a mixture
of resistance-model error and the difference between two extrapolation
conventions**, and the last paragraphs of §19.2 and §19.3 are that
decomposition.

### 19.5 How to fix it — a per-horizon margin helps most

`sop_safety.py`. |I_hat| ← lambda·|I_hat|, with lambda found by bisection on
the other cells so that the exceedance rate hits the target (LOCO). Choose
the conditioning axis.

Target 5 %, hybrid, trustworthy labels:

| Conditioning axis | actual exceedance | worst overshoot | usability | lambda |
|---|---:|---:|---:|---|
| global | 5.5 % | 12.61 A | 0.691 | 0.612 |
| **tau** | **5.5 %** | **6.95 A** | **0.794** | t2=0.497, t10=0.754 |
| tau + SOH | 6.6 % ✗ | 14.29 A | 0.812 | 6 cells |

**Splitting by tau alone raises usability 0.691 → 0.794 and halves the worst
overshoot.** A single global lambda was making τ = 10 s pay for τ = 2 s's
failure.

**Adding SOH as a second axis overfits.** The cells thin out, the calibration
does not transfer to a new cell, it overshoots the target (6.6 %), and the
worst overshoot gets worse rather than better.

**The derate transfers between cells** — target 5 % gives an actual 5.4–5.9 %,
target 1 % gives 1.0–1.7 %.

### 19.6 Pinball (quantile) loss — moving the margin into training

The trim's loss was Huber, i.e. symmetric. `sop_trim.py --quantile q` was
added:

    e = |dV| − |dV_hat| ,   L = mean( max(q e, (q−1) e) )

Sign normalisation (s = sign(I)) lets the same code serve both directions.
For q > 0.5, |dV_hat| becomes the q-quantile of |dV|.

**Why this is not an arbitrary safety factor.** The SOP inversion
I* = (V_min − V_pre)/R_eff is monotonically decreasing in R. A monotone
transform preserves quantiles, so

    P(|I*_hat| > |I*_true|) = P(R_hat < R_true) = 1 − q

That is, q in resistance space directly sets the exceedance rate in current
space — **in principle.**

### 19.7 How the history snapshot is chosen is the second knob

The dataset pairs one pulse with **12 different drive-history windows**
(their `m_exc` differ). The physical resistance is one number, but the model
sees 12 histories, so 12 k's come out. A deployed BMS will see one of them,
and which one is unknown.

`trim_k` keeps only the **last** row per (cycle, SOC, rank). Those are all at
the end of a characterisation, where k has converged to one value — the raw
data for CC cell cycle 40 holds 12 different k_f values (0.934–1.030) while
the SOP evaluation uses only 1.0296. That is why adding SOC to the key
changes **not a single row.**

For safety, taking an **upper quantile** over the history uncertainty is
right. Reading resistance large makes SOP conservative.
`eval_sop_amps.py --trim-agg` selects it. No retraining needed.

### 19.8 Both knobs, end to end (raw, no derate, 651 trustworthy rows)

| Loss | theoretical exceedance | optimism (last) | optimism (max) | worst (max) | A-RMSE (last) | k_f | mV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Huber | — | 82.9 % | 61.1 % | 28.5 A | 6.63 A | 0.988 | 57.1 |
| q=0.50 | 50 % | 81.6 % | 57.8 % | 27.2 A | 6.65 A | 0.991 | 56.7 |
| q=0.70 | 30 % | 73.6 % | 47.0 % | 26.0 A | 5.77 A | 1.024 | 59.5 |
| q=0.80 | 20 % | 65.9 % | 39.6 % | 25.8 A | 5.38 A | 1.041 | 65.3 |
| q=0.90 | 10 % | 52.5 % | 28.9 % | 25.5 A | **5.08 A** | 1.065 | 78.0 |
| q=0.95 | 5 % | 41.5 % | 23.2 % | 25.3 A | 5.13 A | 1.102 | 97.8 |
| q=0.99 | 1 % | 24.4 % | 15.7 % | 23.9 A | 5.59 A | 1.188 | 132.1 |

Three things read out.

**(1) q works monotonically but is heavily attenuated.** A theoretical 10 %
comes out as an actual 52.5 % (last) / 28.9 % (max). The attenuation is
smooth and consistent. The reason is §19.4 — the loss lives on measured pulse
dV while SOP asks about beyond the fan. Error dimensions absent from the dV
loss (OCV, rate interpolation, chord against tangent) are added at the
inversion step. **q is a knob, not a guarantee.**

**(2) The two knobs multiply.** At any q, max aggregation drops optimism a
further 20–24 %p. The k ceiling (1.60) is not reached even at q = 0.99
(k_f 1.188).

**(3) mV and amperes point to different q.** mV RMSE is minimised at q = 0.50
(56.7 m) and degrades monotonically, while **ampere RMSE is minimised at
q = 0.90 (5.08 A)**. The response-space metric the whole trim is tuned on is
not the selection criterion for the SOP task.

**q = 0.50 passes as a control.** The pinball median is L1, hence symmetric,
and should give essentially the same result as the existing Huber — mV
+34.0 % against +33.6 %, optimism 81.6 % against 82.9 %. **So q = 0.9's gain
comes from the asymmetry itself rather than from replacing Huber with
pinball.**

### 19.9 Recommended configuration

Target 5 %, per-horizon margin, 651 trustworthy rows, cell-holdout
calibration:

| Configuration | actual exceedance | worst overshoot | usability | lambda |
|---|---:|---:|---:|---|
| current (Huber + last) | 5.5 % | 6.95 A | 0.794 | t2=0.497 t10=0.754 |
| **Huber + max aggregation** | **5.1 %** | **5.62 A** | 0.826 | t2=0.543 t10=0.840 |
| q0.80 + max | 5.4 % | 7.89 A | 0.843 | t2=0.602 t10=0.921 |
| q0.90 + max | 5.8 % | 8.65 A | **0.849** | t2=0.617 t10=0.958 |
| q0.95 + max | 6.9 % ✗ | 9.05 A | 0.846 | — |
| ECM (reference) | 6.5 % ✗ | 4.87 A | 0.761 | t2=0.449 t10=0.665 |

**Raising q improves the middle (usability) and worsens the tail.** q = 0.90
gives 2.8 % more usability at the cost of 54 % more worst overshoot
(5.62 → 8.65 A). A bad trade for a BMS.

The reason is structural. A high q raises k generally, which brings the
lambda that hits the exceedance target closer to 1. But the tail comes from
**rows where the model is badly wrong**, and the model's own quantile knows
nothing about those rows. lambda is multiplicative and presses the tail down
too. **The derate is a better tail suppressor than a learned quantile.**

**The tighter the target, the stronger the recommendation:**

| Target | Huber+max | q0.80+max | q0.90+max |
|---|---|---|---|
| 10 % | 10.9 % / 7.03 A / 0.867 | 10.6 % / 9.31 / 0.874 | 11.1 % / 9.64 / 0.871 |
| 5 % | 5.1 % / **5.62** / 0.826 | 5.4 % / 7.89 / 0.843 | 5.8 % / 8.65 / **0.849** |
| 2 % | 2.5 % / **3.50** / 0.774 | 3.4 % / 5.21 / 0.781 | 3.2 % / 7.03 / **0.791** |
| 1 % | 1.7 % / **2.50** / **0.745** | 1.7 % / 3.83 / 0.716 | 1.8 % / 5.68 / 0.713 |

(exceedance / worst / usability.) **At 1 % Huber+max wins on both tail and
usability.**

**Recommendation: Huber trim + max history aggregation + per-horizon
derate.** Against the current setting it improves usability (0.794 → 0.826)
**and** worst overshoot (6.95 → 5.62 A) **at the same time**, hits the target
most accurately, and **needs no retraining.** τ = 10 s needs only a 16 % cut.

`docs/fig_sop_safety.png` is the whole frontier. **Which axis to buy is a
design decision, but the margin has to be per horizon** — that axis wins for
both objectives.

### 19.11 So does it not exceed? — it does

Answered in three layers.

**(1) By design it exceeds.** The target was set at 5 %, so 5.1 % exceeds and
the worst is 5.62 A. Reduced, not removed.

**(2) It can be driven to zero. It is expensive.** Taking the largest lambda
under which not one row on the other cells exceeds, and applying it to a new
cell, leaves 0.5 % (3 rows) with a worst of 0.89 A. Adding another 10 %
(×0.90) takes exceedance to zero over the 651 rows — **usability falls from
0.826 to 0.620.** That means discarding 38 % of the predicted current, not
17 %.

| Margin | exceeding rows | worst | usability |
|---|---:|---:|---:|
| zero-exceedance calibration as is | 3 / 651 | 0.89 A | 0.688 |
| ×0.90 | **0** | 0 | 0.620 |
| ×0.80 | 0 | 0 | 0.551 |

**(3) And this calibration does not transfer to another lab's cell.**
Applying the same margin to the voltage-limited rows of RPCWBY Test#1/#2
(external cell, 10 s, 53 rows):

| Applied margin | exceedance | worst | usability |
|---|---:|---:|---:|
| none (§18.6) | 92.5 % | 15.05 A | 1.294 |
| ECM per-horizon **5 % target** (t10 = 0.665) | **13.2 %** | 2.05 A | 0.861 |
| 0.665 × 0.90 | 1.9 % | 0.21 A | 0.775 |

**A margin aimed at 5 % produces 13.2 % on an external cell.** Among the six
UYPYDJ cells the target holds (aim 5 % → 5.1 %), and changing lab more than
doubles it. So **the calibration transfers between cells but not between
datasets.**

**That is as far as the claim goes.** "Exceedance can be held near target"
holds only within one dataset; on a new cell or in a new lab the margin has
to be recalibrated, and for deployment that recalibration procedure has to be
part of the design.

**The remaining blind spots**, stated:

- Only **651** of 5,995 rows are scoreable. The rest have labels too inflated
  to judge on.
- **τ = 2 s has never been validated by interpolation in this dataset**
  (§19.3), and it is exactly where the required R multiplier is largest.
- Internal evaluation is all 25 °C. The external comparison is 10 °C with
  only 3 rows at 25 °C.
- The charge direction is not covered anywhere in §19.

### 19.10 Not yet done

- There is no data in this dataset to validate τ = 2 s by interpolation
  (§19.3). RPCWBY Test#3 measures τ = 2 s at six temperatures but SOP labels
  cannot be made from it (§18.3).
- A symmetry check on the charge direction. §19 is all discharge.
- §19.4's decomposition — fully separating resistance-model error from the
  chord/tangent convention needs data measuring τ = 2 s inside the fan.

---

## 20. Scoring the hybrid entirely from outside (2026-08-24)

§18.6 applied only the uncorrected pooled ECM to an external cell, because
the trim is indexed by UYPYDJ characterisation cycles and could not be moved.
This section removes that index: **the features are recomputed from the
RPCWBY cells' own aging drives.**

Different cycler, different lab, different cell, different aging protocol.
Only the trim weights come from UYPYDJ. It is the most external test the
hybrid receives in this project.

### 20.1 The pipeline (`rpcwby_us06_trim.py`)

The pairing rule matches UYPYDJ — each characterisation takes the last 12
blocks (600 s) of **the preceding drive** and never takes anything from the
characterisation itself.

One extra gate is needed only here. Test#2's drive files end entirely in a
CC-CV charge right before the characterisation, so using the last 2 h as-is
captures only charge blocks outside the training distribution. Only blocks
with duty (EW{1[|I| > 5 A]}) of 0.05 or more are counted — the 5th percentile
of duty across UYPYDJ blocks is 0.028.

**Two traps, measured.**

*An SOC anchor must not depend on a current threshold.* Test#2 changes the
CC-CV termination current from 0.15 A to about 1 A partway through, and the
later drives end in discharge rather than full charge. Finding full charge by
an |I| threshold finds no point after cycle 1333, and SOC leaked out to
1.00–1.62. Taking the maximum accumulated charge in each full-charge stretch
(V ≥ 4.19) as an anchor and **re-fixing it periodically** makes the whole
range physical at 0.14–1.00 (excursions 1.4–3.4 %) and also erases the drift
from coulombic efficiency at every cycle.

*The cell temperature channel cannot be chosen by magnitude.* Test#2 cycles
the chamber between 10 and 25 °C, so the chamber channel's standard deviation
exceeds the cell's (5.71 against 5.54). The cell responds to load within
seconds while the chamber moves over hours, so what separates them is not
magnitude but **the standard deviation of the differences.**

### 20.2 Result — error and tail halve on both external cells

Raw, no margin. The median over six pool variants, collected to unique rows.
(The hybrid shows the values of two independently trained symmetric trims
side by side.)

| Test#2 — US06 driving, n=14, 10 °C | exceedance | worst | usability | RMSE |
|---|---:|---:|---:|---:|
| pooled ECM | 100 % | 10.05 A | 1.152 | 4.32 A |
| hybrid | 71.4 / 64.3 % | 5.26 / 3.84 A | 1.067 / 1.026 | **2.26 / 1.76 A** |

| Test#1 — constant 18 A, n=31 | exceedance | worst | usability | RMSE |
|---|---:|---:|---:|---:|
| pooled ECM | 80.6 % | 13.97 A | 1.275 | 6.92 A |
| hybrid | 77.4 / 61.3 % | 7.23 / 4.70 A | 1.097 / 1.013 | **3.97 / 3.57 A** |

**The trim finds the required correction on its own.** Median k_f is
1.143–1.166 on Test#2 and 1.188–1.282 on Test#1. The multiplier §18.6
**measured** as required on this data is 1.29, and 38 of those 53 rows are
Test#1. The trim never saw that label — only the cell's own aging drives.

Seed-to-seed variation is about 15 %. Direction and magnitude reproduce; the
exact value differs per training run.

### 20.3 It works under constant-current excitation too — the hypothesis was wrong

Test#1's aging is a **constant 18 A discharge**, not a drive cycle. The
trim's core feature is EW{I r}/EW{I I}, the slope of the residual regressed
on current, so with a single current value the regression becomes a line
through one point and transfer to 30 A is pure extrapolation — it was
expected to break.

**It does not.** On Test#1 the RMSE goes 6.92 → 3.97/3.57 A and the worst
overshoot 13.97 → 7.23/4.70 A, improving in the same proportion as Test#2.
The kind of excitation is not a constraint within the reach of this test.

### 20.4 But the margin (lambda) must not be transplanted

Applying the lambda calibrated on UYPYDJ in §19.9 does not hold the 5 %
target — 7.1 % on Test#2 and **16.1 %** on Test#1.

That is a failure of transplanting lambda, not of the hybrid. Being more
accurate internally, the hybrid received a gentler lambda (0.840) and
therefore runs at **a less conservative point** on the external cell
(usability 0.922). The ECM received 0.665 and runs at 0.848. Two different
points were compared.

**Matched on usability, the hybrid is equal or better on both cells:**

| Matched at usability 0.85 | ECM exceedance | hybrid exceedance |
|---|---|---|
| Test#2 | 2/14 = 14.3 % (lam 0.738) | **0/14 = 0.0 %** (lam 0.796) |
| Test#1 | 2/31 = 6.5 % (lam 0.667) | **1/31 = 3.2 %** (lam 0.775) |

Panel (c) of `docs/fig_sop_external.png` is the whole range — at every
usability the hybrid curve sits below the ECM's.

**Deployment conclusion: the model transfers and the margin does not.** The
trim halves the error on a cell from a new lab and a new protocol without
retraining. lambda must be recalibrated on the target fleet, and that
recalibration procedure has to be part of the design (§19.11).

### 20.5 Limits

- **n is small.** 14 rows on Test#2, 31 on Test#1. The 95 % upper bounds on
  exceedance are around 19.3 % and 9.2–18.9 %, so the arms are not well
  separated by exceedance alone. What separates them is usability.
- **Effectively a single point at 10 °C.** Test#1 has only 3 rows at 25 °C.
  There is still no temperature contrast.
- **All τ = 10 s.** Test#1/#2 pulses are 10 s long, so τ = 2 s is not
  validated here either (§19.3).
- Only some of the 146 (Test#2) / 173 (Test#1) voltage-limited rows pass the
  hull. The low-SOC end is outside the pooled table's (SOC, SOH) convex hull.

---

## 21. Eliminating overprediction — the zero-exceedance configuration and its price (2026-08-24)

What §19 and §20 left was "the optimistic bias was reduced." From a safety
management standpoint that is not enough — reading SOP higher than it is
breaches undervoltage protection, so the requirement becomes **eliminating**
it rather than reducing it. This section sets that as the target and prices
it.

### 21.1 Setting the calibration target to zero rather than 5 %

> **[Updated — 34.1]** lambda here is the median of six
> leave-one-cell-out fits applied to every cell, so the evaluated cell
> helped set the lambda it is scored under. Under strict per-cell
> held-out calibration the exceedance is 1 or 2 per setting, not zero.
> The usable current barely moves; the claim does.

lambda is set by bisection, leaving the holdout cell out, so that **not one
row exceeds**
on the rest (the same LOCO as §19.5, only with the target at 0).
651 internal trustworthy rows, symmetric trim + max aggregation:

| Training target | exceeding | rate | 95 % upper | worst overshoot | usability | lambda |
|---|---:|---:|---:|---:|---:|---|
| 5 % | 33 | 5.1 % | 6.72 % | 5.62 A | 0.826 | t10=0.840 t2=0.543 |
| 2 % | 16 | 2.5 % | 3.71 % | 3.50 A | 0.774 | t10=0.782 t2=0.518 |
| 1 % | 11 | 1.7 % | 2.78 % | 2.50 A | 0.745 | t10=0.749 t2=0.512 |
| **0** | **3** | **0.46 %** | **1.19 %** | **0.89 A** | **0.688** | **t10=0.693 t2=0.489** |

**A complete zero is not reached on a new cell** — three rows remain. But the
worst overshoot falls from 5.62 to **0.89 A**. Against an |I*| of 20–30 A
that is 3–4 %, inside the headroom undervoltage protection already has.

**The price is usability 0.826 → 0.688.** A further 14 %p of pack peak
current is given up.

### 21.2 tau is the right axis here too — adding SOH blows up the tail

| Conditioning | cells | exceedance | worst overshoot | usability |
|---|---:|---:|---:|---:|
| global | 1 | 0.31 % | 0.89 A | 0.513 |
| **tau** | 2 | 0.46 % | **0.89 A** | **0.688** |
| tau + SOH (2 bands) | 4 | 0.92 % | **7.84 A** | 0.709 |
| tau + SOH (3 bands) | 6 | 1.69 % | 8.17 A | 0.729 |
| SOH (2 bands) | 2 | 0.77 % | 22.52 A | 0.520 |

Adding SOH gains 2 %p of usability while **the worst overshoot blows up from
0.89 to 7.84 A.** The cells thin out, lambda goes loose where the training
cells happened to be fine, and a new cell is not. The overfitting §19.5
observed at a 5 % target is larger at a target of 0.

### 21.3 The trim's internal disagreement is not an uncertainty measure

The relative spread of k_f across the 12 blocks was tried as a margin axis.
Discarding less output where the model is confident would be worth a lot.

**It does not work.** corr(spread, relative overshoot) = −0.087 and
corr(spread, |relative overshoot|) = +0.170, and by quartile a larger spread
is if anything **less** optimistic (exceedance 75.5 % → 38.2 %). The worst
overshoot is in Q3, not Q4. So the spread does not point at the dangerous
rows. Using it as a margin axis would give a rule with reversed direction,
becoming more conservative where it is already conservative. Rejected.

### 21.4 The zero-exceedance lambda does transplant to external cells

The internally calibrated lambda(t10) was applied **as is** to §20's two
external cells. No recalibration.

| | exceeding | 95 % upper | worst | usability |
|---|---:|---:|---:|---:|
| Test#2 (US06) hybrid + 0.693 | **0/14** | 19.3 % | 0.00 A | **0.740** |
| Test#1 (constant current) hybrid + 0.693 | **0/31** | 9.2 % | 0.00 A | **0.761** |
| Test#2 ECM + 0.520 | 0/14 | 19.3 % | 0.00 A | 0.599 |
| Test#1 ECM + 0.520 | 0/31 | 9.2 % | 0.00 A | 0.663 |

§20.4 said the 5 %-target lambda does not transplant (16.1 % on Test#1).
**The zero-target lambda does** — with enough margin the between-lab
difference fits inside it.

An ensemble taking the **minimum** over six pool variants was not needed
(usability 0.699 / 0.711, in fact lower). The median already gives zero
exceedance.

### 21.5 What the trim is worth in this configuration

At the same zero-exceedance standard:

| | internal usability | Test#2 | Test#1 |
|---|---:|---:|---:|
| ECM | 0.594 | 0.599 | 0.663 |
| **hybrid** | **0.688** | **0.740** | **0.761** |

**26 parameters buy back 10–14 %p of pack peak current at zero
overprediction.** This is the strongest deployment argument so far — not an
RMSE improvement, but a difference in usable output at the same safety
grade.

### 21.6 What it does not guarantee

- **"Zero" is an observation, not a guarantee.** The 95 % upper bounds are
  1.19 % internally (over 651 held-out rows) and 9.2–19.3 % externally (over
  rows of one external cell, conditional on the tested grid — neither figure
  is a cell-level or population-level risk). A regression model cannot
  guarantee an upper bound. The real guarantee comes from the undervoltage
  cutoff as a protection layer, and the SOP estimator's job is to make that
  cutoff fire rarely.
- **External validation is all τ = 10 s.** τ = 2 s carries lambda 0.489, a
  51 % cut, and has never been validated by interpolation in this dataset
  (§19.3).
- **Effectively one point at 10 °C.** There is still no temperature contrast.
- Low SOC lies outside the pooled table's convex hull and is not scored.

---

## 22. The temperature axis opens — the Test#3 sheet (2026-08-24)

§18.3 concluded that SOP labels cannot be made from Test#3's **raw pulses.**
That was right, but it missed that the authors had written the converged
values of that search into a sheet. `rpcwby_sop_summary.py` ran only
`--sheets Test#1,Test#2`, and Test#3 has a different layout and was not read.

`rpcwby_sop_test3.py` reads that layout: three column bands (A = 2 s,
F = 10 s, K = 30 s), and within each band, blocks starting with a
"SOC | <temperature>°C" header repeating downward.

### 22.1 What is actually in it — not what was expected

| Band | rows | Temperatures |
|---|---:|---|
| tau = 2 s | 14 | 25 °C only |
| tau = 10 s | 84 | −20, −10, 0, 10, 25, 40 |
| tau = 30 s | 10 | 25 °C only |

**The six temperature points exist only at tau = 10 s.** τ = 2 s and 30 s
have 25 °C only. (The raw files have all six temperatures at all three
horizons — the authors simply did not transfer them into the summary.)

54 rows are voltage-limited (|SOP| ≤ 76.5 W). **At −20 and −10 °C all 14
rows are voltage-limited and cover the whole SOC 0.02–1.00 range** — the
first time in this project that voltage-limited SOP covers the entire SOC
axis. On the warm side the 30 A ceiling does not reach 2.55 V and only a few
low-SOC rows survive (exactly §18.3's physics).

### 22.2 SOH is the 25 °C capacity, not the block capacity — and this choice swings the conclusion 4×

The sheet records a capacity per temperature block (40 °C 2.6333 / 25 °C
2.6 / 0 °C 2.4193 / −20 °C 2.1352 Ah). Used as SOH, −20 °C becomes 0.712.

**That is not aging.** File dates give the test order — 7/11 0 °C (2.4193),
7/12 25 °C (2.6), 7/13 40 °C (2.6333), 7/14 −20 °C (2.1352). The capacity
**rises and falls with time.** Aging cannot raise capacity. This is the
difference in dischargeable capacity with temperature.

**And putting the block capacity into SOH counts temperature twice** —
g_temp already carries the temperature effect. SOH is defined at a reference
temperature, so the 25 °C value 2.6/3.0 = **0.867** is used for every block.

The weight of that choice, in numbers. At −20 °C the predicted/measured
factor is

    SOH 0.867 (adopted)        1.121   -> the model is optimistic
    SOH 0.712 (block capacity) 0.275   -> the model is 4× conservative

**The sign reverses**, because the pooled table's resistance rises steeply at
low SOH. The basis for the choice is the two paragraphs above, and if that
basis is wrong so is §22.3's conclusion.

### 22.3 The margin depends on temperature — and the size of it is moderate

Pooled ECM (uncorrected), tau = 10 s, median over six pool variants, 27
unique rows:

| T setpoint | n | SOC range | median factor | max factor | zero-exceedance lambda |
|---|---:|---|---:|---:|---:|
| −20 °C | 10 | 0.20–1.00 | 1.119 | 1.604 | **0.623** |
| −10 °C | 10 | 0.20–1.00 | 1.366 | 1.522 | **0.657** |
| 0 °C | 4 | 0.20–0.50 | 1.176 | 1.234 | 0.810 |
| 10 °C | 2 | 0.20–0.30 | 1.188 | 1.238 | 0.808 |

Monotone. A linear fit gives lambda ≈ 0.790 + 0.0093·T, extrapolating to
1.024 at 25 °C.

**The ECM is optimistic at every low temperature** — the same direction as
§19 and §20. Three datasets, two labs and six temperatures point at the same
sign.

### 22.4 §21's lambda is insufficient in the cold. A single 0.623 covers it.

§21 calibrated lambda(tau = 10 s) = 0.693 on 25 °C data. Applied in the cold:

| | exceeding | worst |
|---|---:|---:|
| −20 °C | 2/10 | 0.67 A |
| −10 °C | **4/10** | 1.00 A |
| 0, 10 °C | 0 | 0 |

**Sufficient above 0 °C and broken below −10 °C.**

Tightening lambda to 0.623 gives zero exceedance across all four datasets:

| | exceeding | 95 % upper | usability |
|---|---:|---:|---:|
| internal UYPYDJ, 25 °C | 0/504 | 0.6 % | 0.627 |
| external Test#2 (US06 driving) | 0/14 | 19.3 % | 0.665 |
| external Test#1 (constant current) | 0/31 | 9.2 % | 0.684 |
| **Test#3, −20 to +10 °C** | **0/26** | 10.9 % | 0.762 |

**The price is internal usability 0.697 → 0.627, i.e. 7 %p.** Seven
percentage points of pack peak current buys coverage down to −20 °C. Simpler
than a per-temperature lambda table, and it does not thin the table's cells
(§21.2's overfitting).

### 22.5 τ = 2 s does not close — an equipment limit, not a shortage of data

The sheet's τ = 2 s band is 25 °C only, 3 of its 14 rows are
voltage-limited, and all three are at SOC 0.02–0.10, outside the pooled
table's convex hull and therefore unscored.

**The raw files were dug into.** Test#3's 24 CSVs have τ = 2 s at all six
temperatures, and the July files are genuine search files (247–361 short
pulses). The existing extraction (`rpcwby_temp_pulses.csv`) was discarding
almost all of them — because of a `MIN_REST_S = 20 s` gate. A search fires
pulses back to back, so the preceding rest is a median 2 s. What survived was
only the separate September/October runs (about 50 pulses, never reaching the
floor), producing the false picture of "one converged pulse."

Releasing the gate and looking again, **almost no SOC group brackets the
floor (2.55 V):**

| τ = 2 s file | short pulses | min V_end | SOC groups bracketing 2.55 |
|---|---:|---:|---|
| 7/20 10 °C | 361 | 2.498 | 1 (SOC ≈ 0.05, 0.5–4.3 A) |
| 7/21 40 °C | 270 | 2.530 | 1 (SOC ≈ 0.05, 0.5–5.8 A) |
| 7/19 25 °C | 247 | 2.577 | **0** |
| 7/20 −10 °C | 296 | 2.574 | **0** |
| 7/20 −20 °C | 297 | 2.570 | **0** (SOC 0.4–1.0 has V_end cut at exactly 3.200) |

**The reason is physical and comes out in numbers.** The unlimited I* the
pooled ECM predicts (25 °C):

| SOC | SOH | tau = 2 s | tau = 10 s |
|---|---:|---:|---:|
| 0.20 | 0.95 | **76.3 A** | 51.6 A |
| 0.50 | 0.95 | **108.7 A** | 86.5 A |
| 0.80 | 0.95 | **127.2 A** | 101.9 A |
| 0.30 | 0.85 | 56.5 A | 35.5 A |
| 0.50 | 0.85 | 77.6 A | 59.7 A |

A short horizon has low resistance, so reaching the floor takes **more**
current. A 30 A cycler cannot get there except at low SOC or low temperature,
where resistance grows 4–5×.

**So τ = 2 s SOP cannot be measured with this equipment.** It is not a matter
of digging out more data. Every τ = 2 s label in this project will remain
extrapolated, and §21's lambda(τ = 2 s) = 0.489 stays unvalidated and
conservative. That is the right handling — there is no basis for reducing
margin on a horizon that cannot be validated.

(The same wall partly applies at τ = 10 s — 86.5 A at SOH 0.95, SOC 0.5.
τ = 10 s is scoreable because aging, cold and low SOC bring I* below 30 A,
which is why §22.1's voltage-limited rows cluster there.)

### 22.6 Correcting an earlier verdict

§18.3 said "measured SOP labels cannot be made from Test#3." Half right —
**the authors' sheet had them** (§22.1), and they opened the temperature axis
(§22.3, §22.4). The part about the raw pulses is right, and the reason is now
precise: not that the search fails to converge, but that **30 A does not
reach the floor.**

`analysis/rpcwby_sop_test3.csv`, `analysis/sop_test3_eval.csv`,
`analysis/eval_sop_test3.py`.

---

## 23. Hysteresis effects — testing the trim's premise head on (2026-08-24)

The trim stands on the assumption that recent history changes effective
resistance and that 12 O(1) statistics can read it. UYPYDJ cannot test that
assumption — changing the history changes the cell and the cycle too.

**Test#8 holds everything else fixed and varies only history**: the same
cell, the same 0 °C, the same SOC, with only the C-rate of the immediately
preceding pulse at 0 / C/3 / 1C / 2C / 3C / 4C. The authors named the figure
`fig_historyEffect.png`.

### 23.1 The effect is real, large, and sits in the voltage-limited region

The sheet was parsed with `rpcwby_sop_test8.py` (72 rows).
Amplitude (max − min) / mean:

| SOC | 0C | 1C | 2C | 4C | amplitude | voltage-limited |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | −90.2 | −89.7 | −89.7 | −88.7 | 1.6 % | 0/6 |
| 0.30 | −84.4 | −82.7 | −81.2 | −79.2 | 6.3 % | 0/6 |
| **0.20** | **−75.5** | −65.2 | −60.9 | **−58.1** | **26.9 %** | **6/6** |
| 0.15 | −48.1 | −43.2 | −42.3 | −43.3 | 13.1 % | 6/6 |
| 0.10 | −23.6 | −18.6 | −17.4 | — | 31.8 % | 5/5 |

At SOC ≥ 0.3 the 30 A ceiling sets the answer, so the voltage model is not
tested. **The 17 voltage-limited rows sit exactly where the hysteresis effect
is large.**

Extraction check: the median temperature immediately before measurement is
0.9 / 1.1 / 2.0 / 3.1 / 3.9 / 4.4 °C per file, matching the bars in the
authors' figure.

### 23.2 The effect is in V_pre, not in resistance

Taking only converged pulses (V_end = 2.55 V) and matching SOC:

| SOC ≈ 0.20 | 0C | C/3 | 1C | 2C | 3C | 4C |
|---|---:|---:|---:|---:|---:|---:|
| **V_pre [V]** | **3.456** | 3.413 | 3.365 | 3.299 | 3.239 | **3.197** |
| R = dV/I [mΩ] | 38.8 | 39.8 | 39.5 | 37.7 | 35.0 | **32.6** |
| I* [A] | 23.4 | 21.7 | 20.6 | 19.9 | 19.7 | 19.8 |
| T [°C] | 1.1 | 1.3 | 1.9 | 3.1 | 3.9 | 4.4 |

**The headroom (V_pre − 2.55) falls 29 %, 0.906 → 0.647 V. Resistance
actually falls 16 %** — self-heating. So the hysteresis effect is a **state
of the voltage baseline** left by diffusion polarisation, not a change in
resistance.

### 23.3 The trim cannot express this — structurally

The trim's output is one pair of resistance multipliers (k_f, k_s). The
design paragraph in `sop_trim_dataset.py` states it: the loss lives on
dV = V(t0+tau) − V(t0−), so "OCV, hysteresis and the RC initial states are
absent by construction." That exclusion is the device that stopped 45–72 mV
of OCV error leaking into the resistance multipliers, and here that very
exclusion is what keeps the effect out of reach.

Measurement confirms it. Over SOC 0.18–0.30, as the C-rate rises,

    required factor (R measured / R nominal)   0.642 -> 0.570   (slope −0.0136 /C)
    k_f                                        0.687 -> 0.732   (slope **+0.0099** /C)

**The trim moves the opposite way.** The association learned from driving
("recent high current = more resistive") reverses in the face of
self-heating at 0 °C.

### 23.4 Yet the hybrid gets it right — if V_pre is the measured value

The SOP inversion is V_pre + I·R_eff = V_min, and a vehicle's BMS
**measures** V_pre. Then §23.2's effect simply enters as an input.

dI*/d(C-rate), the slope of the hysteresis effect:

| SOC | measured | predicted (measured V_pre) | predicted (model V_pre, OCV−M) |
|---|---:|---:|---:|
| 0.20 | −0.767 | **−0.668** | **+0.583** |
| 0.25 | −0.924 | **−0.374** | **+0.700** |

**With the measured terminal voltage the sign and magnitude are right; with
the model OCV the sign reverses.** Fixing V_pre leaves only R varying, and R
falls with self-heating, so the predicted I* rises — exactly backwards.

**Design rule: V_pre in the SOP inversion must be the measured terminal
voltage and must not be replaced by the model OCV.** §11's end-to-end path
already uses the label's `V_pre_V` (measured). §20's and §22's external
evaluations used OCV−M because the labels do not carry V_pre — those cells
rest before characterisation so V_pre ≈ OCV holds, but where that
approximation breaks is now known.

### 23.5 What remains

- **The trim's premise is half right.** History changes SOP substantially
  (27 % at SOC 0.2), but the channel is a voltage state rather than
  resistance. A resistance multiplier cannot reach it.
- **The hybrid's SOP still comes out right** — because the physics side
  (voltage inversion) carries it through the measured V_pre. The physics does
  what the AI cannot.
- **The absolute level is not established.** At 0 °C the model reads I* at
  about half (12.9 A predicted against 23.4 A measured at SOC 0.20). That is
  the conservative side, but Test#8's SOH is not in the sheet and was assumed
  to be 0.90, and that assumption dominates the absolute level. **What is to
  be read here is the slope, not the level.**
- 0 °C is outside the temperature range the trim was trained on
  (25.96–30.66 °C). Whether §23.3's sign reversal is temperature
  extrapolation or the history association cannot be separated with this
  data.

`analysis/rpcwby_sop_test8.py`, `analysis/rpcwby_t8_trim.py`,
`analysis/rpcwby_sop_test8.csv`. (`analysis/cache/t8/` was a local working
cache; it is gitignored and no longer present, and the scripts rebuild it.)

---

## 24. Data design audit — what can and cannot be separated (2026-08-24)

The question §23 left open — whether the trim's sign reversal is temperature
extrapolation or the history association — **closes with the data already in
hand.** And something more important emerges on the way: there are many rows,
but **contrast is extremely unevenly distributed across axes.**

### 24.1 The sign reversal is not temperature

Splitting k_f by recent load over 64,308 UYPYDJ blocks at 25 °C:

| Feature | bottom 20 % k_f | top 20 % | difference |
|---|---:|---:|---:|
| I_rms | 0.9920 | 1.0032 | **+0.0113** |
| duty | 0.9956 | 1.0021 | +0.0065 |
| I_hi | 0.9977 | 1.0014 | +0.0037 |

**The same direction holds at 25 °C.** So the sign reversal seen in Test#8
(0 °C) is not an artefact of temperature extrapolation but an association the
trim carries everywhere. This closes the open item in §23.5.

### 24.2 The trim is a residual reader, not a history reader

Correlation of k_f with each feature over the same 64,308 blocks:

    dR_fast   +0.915      R_fast_nom  -0.405
    dR_slow   +0.889      T           +0.265
    I_rms     +0.075      duty        +0.033      I_hi  +0.065

**k_f is effectively a function of dR_fast alone.** dR_fast =
EW{I·r}/EW{I·I} is the slope of the residual measured *now*, not an integral
of history. The load-related features contribute a negligible +0.03–0.08.

That explains three things at once.

- **Why it works under constant-current excitation** (§20.3) — the residual
  slope is defined even at a single current.
- **Why it misses Test#8's hysteresis effect** (§23.3) — that effect lives in
  V_pre, and with the loss on a dV difference, V_pre is structurally removed.
- **Why learning invariance over 12 blocks worked** (§24.3) — the residual
  slope is nearly identical across the 12 windows.

**The trim's claim has to be narrowed from "it reads history" to "it reads
the resistance deviation of the pulse response."**

### 24.3 The data teaches the model *not* to respond to history

`sop_trim_dataset.pair()` attaches **the last 12 blocks of the preceding
drive** to one label. One target, 12 inputs. Checked: in **all 5,359
(100 %)** (cycle, SOC, rank) keys, the label Y across the 12 windows is
completely identical (median within-key standard deviation 3e-8 V).

That pairing is **invariance training**: "produce the same physical
resistance whatever history you saw." No amount of extra data teaches history
sensitivity through it.

More fundamentally, **every labelled pulse in UYPYDJ follows a long rest**
(`MIN_REST_MULT = 5 × tau2`). History contrast on the label side is zero to
begin with. This file's own design paragraph already records it — "the
features are measured in a regime where the cell never rests, the labels in
one where it always has."

### 24.4 Contrast audit by axis

| Axis | UYPYDJ (6 cells) | RPCWBY | paired with a label? |
|---|---|---|---|
| SOH | continuous 0.68–1.00 | Test#1/2 0.73–1.00 | **plentiful** |
| SOC | 0.05–1.00, 17 points | 0.005–1.00, 17 points | **plentiful** |
| current / rate | 2.6–29.6 A, 4 steps | up to 30 A | **plentiful** |
| temperature | **one point, 25 °C** | Test#3 −20–40 °C, **tau = 10 s only** | 26 rows |
| horizon tau | 2 / 10 s | 2 / 10 / 30 s, no crossing with temperature | partial (zero at tau = 2 s) |
| **history** | 12 windows → 1 label | Test#8 6 conditions → 6 labels | **18 rows, all of it** |
| direction | charge / discharge | charge / discharge | present |

**There are tens of thousands of rows, and the labels carrying history
contrast are 18, all from one cell at one temperature.** The labels carrying
temperature contrast are 26, all at one horizon. Most of the remaining rows
sweep SOH and SOC repeatedly — two axes that are already saturated.

**So the answer to "there is so much data, why can it not teach this" is
design, not volume.** These datasets were designed as aging tests, and an
aging test by definition moves SOH and holds everything else.

### 24.5 What should be measured then

Closing the history axis means extending Test#8 — no exotic equipment
needed.

    same cell, same temperature, SOC widened to 0.2–0.8,
    the pulse immediately before the measurement varied 0 / 1C / 2C / 4C,
    then a 10 s SOP search right after.

That is the only structure in which **the label varies with history.** And
§23.4 already says what would come of it — even measured that way, a
resistance multiplier will not reach it. The effect is in V_pre, and V_pre is
something a vehicle already measures. **So the conclusion on this axis is not
"make the model bigger" but "use the measured terminal voltage."**

The temperature axis needs SOP labels crossed with tau, and as §22.5 showed,
τ = 2 s is unreachable with a 30 A cycler. That axis does not close without
changing the equipment.

---

## 25. Safety in the charge direction (2026-08-24)

§19–24 are all discharge. A BMS does not run without a regeneration limit,
and overprediction on charge is the symmetric hazard, **overvoltage.** This
section applies the same yardstick to charge.

### 25.1 The label situation is better on charge

| | total | trustworthy (extrap ≤ 1.5) |
|---|---:|---:|
| discharge | 5,995 | 651 (11 %) |
| **charge** | 6,946 | **4,648 (69 %)** |

The reason is physical — the 4.2 V charge ceiling is close to OCV, so the
headroom is small, I* is small, and it lands inside the fan. The statistics
are 7× stronger than discharge.

The same holds externally. Of RPCWBY Test#1/#2's 889 `SOP_char` values,
**265 are voltage-limited** (SOC 0.10–1.00, 10 and 25 °C, two cells, 30
cycles) — 5× the discharge external set of 53 rows.

**Separating voltage- from current-limited works differently than on
discharge.** The charge current ceiling is 15 A and the voltage ceiling
4.15 V, so both cases give P < 62.25 W — one P threshold does not separate
them. Instead, for a row to be current-limited, V = P/15 has to be the
terminal voltage **during charge**, and below 45 W that would be under 3.0 V,
which is impossible. So P < 45 W is judged voltage-limited. Self-consistent —
those rows have I* = P/4.15 of 0.0–10.8 A, all below 15 A.

### 25.2 The max-aggregation recommendation reproduces on charge

4,648 trustworthy rows:

| | optimism | worst overshoot | usability | RMSE |
|---|---:|---:|---:|---:|
| ECM | 72.9 % | 31.45 A | 1.084 | 3.65 A |
| hybrid (current, last) | **81.6 %** | 32.73 A | 1.067 | 2.52 A |
| hybrid (max aggregation) | **54.4 %** | 23.07 A | 1.009 | **2.23 A** |

Exactly the shape of discharge — the trim improves RMSE while **worsening**
optimism (72.9 → 81.6 %), and §19.7's max aggregation reverses that without
retraining. RMSE improves along with it.

### 25.3 Charge fails in two different ways — so a ratio is the wrong yardstick

| \|I*\| range | n | exceedance | worst absolute overshoot | median relative overshoot |
|---|---:|---:|---:|---:|
| 0–3 A (high SOC) | 252 | **92.9 %** | 2.09 A | +36.7 % |
| 3–10 A | 1,236 | 66.0 % | 7.15 A | +9.0 % |
| 20–30 A | 1,302 | 44.9 % | 18.80 A | −0.9 % |
| above 30 A (low SOC) | 882 | 29.4 % | **23.07 A** | −2.7 % |

**High SOC has large relative and small absolute error. Low SOC is the
reverse.**

Allowing 2.0 A where the true limit near full charge is 1.3 A is a 54 %
exceedance by ratio, but the voltage hits 4.2 V slightly early and the CV
loop absorbs it. Allowing 55 A instead of 32 A at SOC 0.14 is an entirely
different event. **A ratio counts the two the same and amperes count them
differently.** So the safety criterion on charge has to be **absolute
overshoot** rather than exceedance rate.

### 25.4 Setting the tolerance in amperes halves the price

lambda is chosen on the training cells so that
`max(lambda·I_pred − I_true) ≤ tol`, then measured on the holdout
(tau-conditioned):

| tolerance | charge usability | charge actual worst | discharge usability | discharge actual worst |
|---|---:|---:|---:|---:|
| 0 A | 0.458 | 0.03 A | **0.688** | 0.89 A |
| **0.5 A** | **0.594** | **0.75 A** | 0.701 | 1.40 A |
| 1.0 A | 0.621 | **3.35 A** | 0.715 | 1.98 A |
| 2.0 A | 0.645 | 5.30 A | 0.744 | 3.14 A |

**Allowing 0.5 A on charge raises usability by 14 %p** (0.458 → 0.594). From
1.0 A the new cell's actual worst jumps to 3.35 A and the calibration does
not transfer — 0.5 A is the knee.

**On discharge the curve is flat, so tightening to zero is cheap.** The
optimum differs by direction.

lambda(tol = 0.5 A): tau = 10 s **0.598**, tau = 2 s **0.566**. The ECM at
the same standard is 0.507, so **the trim buys back 8.7 %p** (9.4 %p on
discharge).

### 25.5 Conditioning and margin form — what differs from discharge and what does not

**tau conditioning barely helps on charge.** At the zero-exceedance standard
usability is 0.458 (tau) against 0.447 (global), in contrast to discharge's
0.688 against 0.513 — discharge's τ = 2 s was specifically broken (§19.3)
while charge's two horizons are similar (lambda 0.566 against 0.598).

**tau + SOH overfits here too**: usability 0.568 with the worst overshoot
going 0.03 → 7.38 A.

**Three margin forms** were compared — I_bound = lambda·I_pred − delta:

| Form | exceeding | worst | usability |
|---|---:|---:|---:|
| **multiplicative lambda only** | 2/4648 | **0.03 A** | 0.458 |
| additive delta only | 6 | 12.74 A | 0.000 |
| both | 9 | **11.11 A** | 0.568 |

Additive alone drives delta to 18–23 A and kills the output entirely. Mixed
gives 11 %p more usability while the tail blows up — one more parameter and
the margin goes loose where the training cells happened to be fine, the same
overfitting as §21.2. **Multiplicative only is the answer.**

### 25.6 External validation — and charge has no temperature dependence

§20's feature files are used as they are (`TrimFeatures` selects direction by
the sign of current, so the same feature vector serves both). The charge trim
is `runs_trim_chg`.

No margin, median over six pool variants, unique rows:

| | exceeding | worst | usability |
|---|---:|---:|---:|
| Test#2 ECM | 34/94 | 2.72 A | 0.785 |
| **Test#2 hybrid** | **10/94** | **0.29 A** | 0.672 |
| Test#1 ECM | 42/107 | 4.47 A | 0.818 |
| **Test#1 hybrid** | **8/107** | **1.05 A** | 0.683 |

**Matched on usability, the hybrid's tail is consistently smaller:**

| Usability | ECM worst | hybrid worst |
|---|---:|---:|
| Test#2 0.65 | 0.64 A | **0.07 A** |
| Test#2 0.70 | 1.40 A | **0.68 A** |
| Test#1 0.65 | 1.34 A | **0.49 A** |
| Test#1 0.70 | 2.27 A | **1.35 A** |

The **count** of exceedances is similar or slightly higher for the hybrid.
So what the trim does on charge is not shift the distribution but **compress
the tail** — a different character from discharge (§19.5).

**No temperature dependence.** With the hybrid and lambda 0.598:

    Test#2: 10 °C 1/51 worst 0.01 A usability 0.396 | 25 °C 1/43 worst 0.00 A usability 0.406
    Test#1: 10 °C 1/60 worst 0.02 A usability 0.418 | 25 °C 0/47 worst 0.00 A usability 0.404

The two temperatures are effectively identical. **In contrast to §22.4, where
discharge had to tighten lambda 0.693 → 0.623 in the cold.** Charge does not
need it.

**The internal lambda transfers on the conservative side** — externally only
1–2 rows exceed and usability drops to 0.40, because the hybrid is already
conservative externally (usability 0.67). Same conclusion as §20.4: **the
model transfers and the margin has to be recalibrated on the target fleet.**

### 25.7 Adopted configuration (both directions) — values after the 2026-08-25 rebuild

These are the values after rebuilding the pipeline with the temperature
defects excluded (findings.md §7.3).

    history aggregation   max (running max over the preceding 2 h of driving)
    margin form           multiplicative only, per tau (additive and mixed blow up the tail — §25.5)
    V_pre                 measured terminal voltage; never substitute the model OCV (§23.4)

| | lambda(10 s) | lambda(2 s) | usable current | exceeding | worst |
|---|---|---|---|---|---|
| discharge (25 °C) | 0.679 | 0.462 | 70 % | 4/631 (0.6 %) | 1.19 A |
| **discharge (−20 to 25 °C)** | **0.623** | 0.462 | **~64 %** | — | — |
| charge (temperature independent) | 0.567 | 0.544 | 57 % | 3/4543 (0.07 %) | 0.58 A |

The cold value of 0.623 for discharge is what §22.4 required from Test#3's
measured SOP at −20 to +10 °C. The 0.679 calibrated at 25 °C alone breaks
below −10 °C. **Charge needs no temperature correction because §25.6 found
10 and 25 °C effectively identical.**

At the same safety level the ECM gives 59 % on discharge and 51 % on charge —
**the hybrid uses 11 %p more pack current on discharge and 6 %p more on
charge.** That is what 26 parameters are worth.

**Raw, before the margin**, a majority still exceeds (discharge 69.4 %,
charge 61.7 %; the ECM 84.3 / 73.0 %). It is unusable without a margin.

τ = 2 s cannot be validated by measurement with this equipment (§22.5, 30 A
does not reach the floor), so it is left conservative.

### 25.8 Recounting per cell — §16.2's verdict flips

§16.2 said the charge trim **loses to the ECM on two of six cells**
(CC_CELL2 −31 %, BOOST_NEGPULSE −9 %). That was on RMSE. Recounted on the
safety criterion, with lambda taken per tau on the other five cells
(tol = 0.5 A):

| Cell | n | RMSE ECM | RMSE hybrid | gain | usability ECM | usability hybrid | gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| BOOST | 851 | 3.99 A | 1.40 A | +64.9 % | 0.553 | 0.608 | +10.0 % |
| **BOOST_NEGPULSE** | 727 | 2.88 A | 2.96 A | **−2.6 %** | 0.462 | 0.539 | **+16.7 %** |
| BOOST_NEGPULSE_1S | 858 | 4.63 A | 2.30 A | +50.4 % | 0.573 | 0.651 | +13.6 % |
| BOOST_REST | 580 | 3.05 A | 2.77 A | +9.1 % | 0.430 | 0.527 | +22.5 % |
| CC | 836 | 4.50 A | 2.04 A | +54.6 % | 0.567 | 0.611 | +7.7 % |
| **CC_CELL2** | 796 | 1.36 A | 1.80 A | **−32.7 %** | 0.464 | 0.572 | **+23.3 %** |
| mean | | | | +24.0 % | | | **+15.6 %** |

**RMSE 4/6, safety 6/6.** And **the two cells that lose on RMSE win the most
on safety** (+16.7 %, +23.3 %).

**Not an aggregation artefact.** Counted with §16.2's `last` aggregation the
RMSE is still 4/6 with the same losing cells (−6.1 %, −42.5 %) — max
aggregation only reduces the size of the loss. What changed is **the
criterion.** (Though on the safety criterion, aggregation does turn 4/6 into
6/6.)

**Why it flips — the margin is set by the tail.**

| Cell | ECM worst | hybrid worst | ECM p99 | hybrid p99 |
|---|---:|---:|---:|---:|
| BOOST | 14.48 A | 8.72 A | 11.29 A | 4.87 A |
| BOOST_NEGPULSE | **31.45 A** | 23.07 A | 3.92 A | 4.65 A |
| BOOST_NEGPULSE_1S | 15.51 A | 8.93 A | 12.78 A | 7.32 A |
| BOOST_REST | 4.72 A | 8.31 A | 3.11 A | 3.61 A |
| CC | 15.44 A | 10.34 A | 12.24 A | 7.58 A |
| CC_CELL2 | 5.94 A | 7.24 A | 4.71 A | 4.45 A |

The hybrid's worst is smaller in 4 of 6 and its p99 in 5 of 6. So on any
5-cell training set the hybrid receives **a looser lambda** — measured,
lambda_ECM 0.465–0.470 against lambda_hybrid 0.582–0.591, **25 % looser.**
That difference becomes the usability difference directly.

**CC_CELL2 shows it best.** On that cell the ECM's RMSE of 1.36 A is the best
of the six, while its usability at 0.464 is among the worst. **RMSE measures
the middle of the distribution and the margin measures the end.** What §16.2
called "a cell where the ECM is already good" meant good in the middle, and
for safety that is not grounds for a verdict.

**Robust to the tolerance**: **6/6** at tol = 0 / 0.5 / 1.0 A, and 5/6 at
2.0 A (one tied at −0 %). Mean gains +14.6 / +15.6 / +12.1 / +7.4 %.

### 25.9 Remaining

- §16.2's **two cells losing to the ECM on charge** (−9 %, −31 %) was on
  RMSE. On the safety criterion (§25.4) the trim buys back 8.7 %p, so that
  verdict flips — though it was not recounted per cell there.
- Charge τ = 2 s has labels, unlike discharge (2,129 trustworthy rows). But
  30 s does not.
- Why lambda is over-conservative on external charge (the hybrid is already
  conservative externally) was not decomposed per cell.

---

## 26. Where the charge margin's 40 % comes from (2026-08-24)

What §25 left is lambda(tau = 10 s) = 0.598, meaning only 59 % of the current
the cell can take is used and 41 % is discarded. Whether that can be reduced
was probed in four directions, and two things were found — one genuine model
defect and one piece of physics that cannot be fixed. Three attempts failed
for the same reason.

### 26.1 One measured point sets the safety factor

Of the 2,092 calibration points, **one** sets lambda:
BOOST_NEGPULSE_1S, SOC 0.95, SOH 0.691, where the true limit is 1.6 A and the
model says 3.6 A (2.18×).

Removing that one point barely moves lambda, 0.598 → 0.600 (similar points
queue behind it). Removing the top 1 % gives 0.625, and the top 5 % gives
**0.768.**
The distribution is nearly perfect at the centre (ratio 1.01) and exceeds 1.63
in the top 1 %. **The structure is that one tail is being blocked at the cost
of cutting everything.**

### 26.2 Why conditioning fails — variance decomposition

Splitting the log variance of the predicted/actual ratio:

    condition (tau x SOC x SOH, 32 bins) explains    0.122
    cell explains                                    0.283
    bin x cell                                       0.584

**Even after the bin is known, the cell explains a further 0.461** — four times
the share the condition explains. Per bin, cell means spread over 1.10–1.74×
(median 1.22), which is the same size as the within-bin standard deviation
(1.03–1.34).

**"The model is wrong by this much under this condition" does not hold.** At the
same SOC, SOH and horizon it depends on which cell, and which cell it is cannot
be known in advance.

All three measured failures come from here.

| Attempt | Result | Why |
|---|---|---|
| Split the safety factor by SOC (2–3 bins) x SOH (2 bins) | usable current +1~10 %p, **true worst 0.75 -> 16.61 A** | Not because the bins are thin (217–757 points per bin). The **worst** within a bin swings 1.5–1.6× between cells and breaks through on a new cell |
| Subtract the median bias on a (tau, SOC, SOH) grid and recalibrate | 59 -> 56~61 %, worst 0.75 -> 2.13~4.19 A | The median bias does not reproduce either. Per-bin cell-mean spread (1.12–1.60×) exceeds the bias itself (~2 %) |
| Per-cell online correction with dR_fast | 59 -> 56 %, worst 0.77 A | **Cell bias is estimated well** (error within 0.035 on 5 of 6, cell-to-cell correlation +0.962). But what sets lambda is not the cell's mean bias, it is the outlier **inside** the cell, so there is no gain |

**Two mistakes worth recording.** (a) The grounds offered for the median-bias
grid came from misreading a table — it was read as "every cell is 1.14–1.30×
at SOC 0.7–0.9", but the actual values were split 1.20/1.00/1.27/1.02/1.19/1.06,
and 1.27 was the max/min column. (b) An attempt was made to explain the
near-full region as "a region where small margin amplifies OCV error", but the
holdout error of the pooled OCV is in fact **smaller** at SOC>=0.9 (median
1.7 mV, 95 %ile 26.0 mV; overall 6.5 / 55.8 mV), and in any case the internal
evaluation's v_pre is the label's **measured** value, so OCV error does not
enter.

### 26.3 The genuine defect — RC parameters must not be interpolated on the rank axis

Because v_pre is measured, the margin is exact, and therefore
**predicted/actual ratio = R_actual/R_predicted.** The problem is entirely
resistance.

Raw values by rank on the charge side at SOC 0.95, SOH 0.72:

| rank | I | R2 | tau2 | D10 |
|---|---:|---:|---:|---:|
| 0 | 1.26 A | 23.3 mOhm | **5.71 s** | 49.0 mOhm |
| 1 | 4.94 A | 460.0 mOhm | **3000 s** | 40.2 mOhm |
| 2 | 13.89 A | 0.0 mOhm | 3000 s | 39.4 mOhm |

The fits at rank 1 and above **diverged** — once tau2 sits at the 3000 s bound,
1 − e^(−10/3000) = 0.0033 over a 10 s window, so R2 contributes nothing to D10
whether it is 460 or 0. R2 and tau2 cancel each other and the values are not
determined.

`ECMSurface.theta` interpolates the five parameters **individually** along the
rank axis and returns them. That yields, at 1.6 A, a combination
(R2 64 mOhm, tau2 285 s) that exists at no rank and has no grounds. It is
merely that the midpoint of two diverged fits looks like a value that did not
diverge.

The result is unphysical. The model's 10 s equivalent resistance goes

    1.0 A 49.0 -> 1.6 A **32.8** -> 5.0 A 40.2 mOhm

dipping and then rising. The measurement decreases monotonically,
52.2 -> 41.3 -> 29.6 mOhm (Butler–Volmer). Resistance rises with increasing
current at 21–31 of the 39 grid points.

`ECMSurface.d_tau` was added — build the equivalent resistance D(tau) per rank
first and interpolate **that**. Divergence is trapped inside its own rank. At
1.6 A, 32.8 -> 48.2 mOhm (measured 52.2); at SOC 0.90/SOH 0.75,
37.2 -> 56.0 mOhm. Regions that already fit are untouched (16.9 -> 16.8 at
SOC 0.50/SOH 0.90).

### 26.4 But it is not adopted

| Direction | Interp | Optimism | Safety factor | Usable current | True worst |
|---|---|---:|---|---:|---:|
| discharge | theta | 61.1 % | 0.693 / 0.489 | 69 % | 0.89 A |
| discharge | dtau | 61.2 % | 0.689 / 0.482 | 69 % | 1.06 A |
| charge | theta | 54.4 % | 0.598 / 0.566 | 59 % | **0.75 A** |
| charge | dtau | **42.5 %** | 0.608 / 0.590 | 59 % | **2.78 A** |

Charge optimism drops 12 %p, but the holdout cells' **true worst gets worse,
0.75 -> 2.78 A**, and usable current is unchanged. On discharge every item
degrades slightly.

**It improves the middle of the distribution and degrades the end that safety
uses.** This is a pattern that has repeated in this project (19.7, 21.2, 25.5),
and as long as the adoption criterion is safety, the answer is rejection. The
diagnosis is valid, so it is kept as `--interp dtau`. If work ever moves the
metric back to RMSE, it becomes an adoption candidate then.

### 26.5 The point that sets the worst — correction: it was a temperature-channel defect, not a hysteresis effect

**The first ruling was wrong.** Both the ruling and the correction are left
below.

On the discharge side the point that sets the worst absolute exceedance is
**BOOST_NEGPULSE cycle 487, SOC 0.14–0.23.** An exhaustive scan finds 13
(cell, direction, cycle) triples that spike 1.4× or more above neighbouring
cycles, and **487 alone spikes in all 20 combinations at once** (charge 2.02×,
discharge 2.97×). The rest are scattered noise covering 1–3 combinations.

The raw voltage records it too — at the same V_pre (3.48 V) and the same 27 A,
dV is 0.408 V at cycle 450, **0.708 at 487**, and 0.404 at 525.

**This was first read as the same kind of thing as the hysteresis effect of
Test#8 in §23.** That it exists only at SOC<0.3, appears in both directions and
at all rates at once, and recovers at the next characterisation all seemed to
fit.

**Wrong. Opening the temperature in the raw data shows the cell temperature over
that region is 3.8 C.**

| Cycle | Median temp over V<3.55 | over V>3.90 |
|---|---:|---:|
| 450 | 25.8 C | 26.0 |
| **487** | **3.8 C** (0.4–27.6) | 25.8 |
| 525 | 25.8 C | 26.0 |

Resistance being 2–3× the 25 C value at 0.4–4 C matches the temperature factor
of §22 exactly (4.5× at −20 C, about 2× at 0 C). Existing only at low SOC,
spiking in both directions and at all rates at once, and recovering at the next
characterisation are all explained by temperature.

**This defect was already known.** findings.md 7.1 recorded BOOST_NEGPULSE
cycles 488 and 505 as defective "in the first ~20 % only" — the drive cycles
immediately after characterisation 487, and the same event.

**But that filter was applied only to the drive cache (cache_t).** The HPPC
resistance table (`uypydj_hppc_resistance.csv`) has no temperature column at
all, so the same filter could not be applied. The resistance table and the SOP
inversion both assume 25 C, yet labels taken from a 4 C cell were going straight
in and setting the discharge safety factor.

**The impact is on discharge only.** Listing the deciding points per holdout:

| | Deciding point | Nature |
|---|---|---|
| discharge t10 | BOOST_NEGPULSE 487, SOC 0.23 (6 of 12 combinations) | **temperature defect** |
| discharge t2 | CC 1762, SOC 0.37, SOH 0.730 | normal |
| charge t10/t2 | BOOST_NEGPULSE_1S 1881, SOC 0.95/0.78, SOH 0.691 | normal |

Removing it at the evaluation stage loosens discharge lambda(t10) from 0.693 to
0.715 (the next point, CC 1425, then sets it, so the true worst goes
0.89 -> 1.15 A), and **charge does not change at all, staying at 0.598.**

**So the charge deciding point of 26.1 (near full, SOC 0.95, true 1.6 A vs
predicted 3.6 A) is normal data unrelated to temperature.** Why the charge
margin is tight remains open.

### 26.6 Conclusion — a substantial part of the 41 % cannot be reduced

The reason the charge margin is tight split in two.

- **Model defect** (26.3): real and fixable, but fixing it does not loosen the
  safety factor.
- **Data defect** (26.5): half the discharge deciding points came from a
  temperature-channel defect. Removing it loosens lambda from 0.693 to 0.715.
  **No effect on charge.**

**The charge 41 % is still unexplained.** The deciding point is normal data near
full charge (SOC 0.95, SOH 0.691, true 1.6 A vs predicted 3.6 A), and as
26.2–26.4 showed, it is not resolved by conditioning, by online correction, or
by fixing the interpolation.

**And one lead closed this time.** §23.4 offered the measured terminal voltage
as a channel, but at cycle 487 V_pre matches its neighbours (3.477 / 3.483 /
3.484) and the difference only appears once current is applied. The drive
features are normal too (dR_fast −1.47 against neighbours −1.60, +1.56, −2.38,
−1.41; k_f 0.987). **Neither V_pre nor the drive residual predicted that state**
— though that was a temperature defect, so for the hysteresis effect it remains
untested.

---

## 27. MCU measurement — measured on the board (2026-08-25)

The question this project started from was "does AI-based SOH/SOP run on an
ordinary BMS". Through §26, accuracy and safety were all answered, but
**computation was never measured once.** This section closes that.

    Board     NUCLEO-H563ZI, STM32H563ZIT6
    Core      Cortex-M33 250 MHz, FPv5-SP hardware FPU, ICACHE on
    Build     arm-none-eabi 14.3, -Os, float32
    Timing    DWT CYCCNT, interrupts off, call region only
    Comms     USART3 921600 baud (same protocol as the existing CEMA bench)

### 27.1 What was deployed — the table is stored as D(tau)

Python uses `LinearNDInterpolator` over scattered data. That cannot be used on
the MCU, so it must be resampled onto a regular grid, and **that gridding
itself creates error.**

Two storage formats were measured: storing the five RC parameters, versus
precomputing and storing D(2 s) and D(10 s) per rank. **The latter gives the
same error at less than half the size**, and above all makes the defect found
in §26.3 structurally impossible — RC is not interpolated, so no baseless
combination can arise between diverged tau2 fits.

The grid size was chosen by measurement. The source table is SOC 20 x SOH 17
points, so there is no reason to be finer than that. Relative error in the 10 s
equivalent resistance against Python scattered interpolation:

| Grid | Median | 95 % | Max | Size (both directions) |
|---|---:|---:|---:|---:|
| 24x12 | 0.65 % | 4.46 % | 21.6 % | 18 KB |
| **32x16** | **0.30 %** | **2.88 %** | 16.8 % | **32 KB** |
| 48x24 | 0.20 % | 1.75 % | 12.1 % | 72 KB |

32x16 is the knee. Above it the size doubles and the error falls by only 0.1 %p.

**Gridding barely touches SOP.** On the 657 rows of real trustworthy labels

    optimism    70.6 -> 70.9 %
    RMSE        5.38 -> 5.50 A
    worst       20.35 -> 19.53 A   (it actually fell)
    median diff 0.217 A

This is buried under the retraining spread measured in §7.3 (optimism 13.2 %p).

Assets total **36.4 KB** — ECM grid 32 KB + OCV/hysteresis 4 KB + trim 0.39 KB.

### 27.2 First, checking that the C implementation matches Python

The comparison was done on the host by **linking the same `sop_core.c` as the
firmware**. If that does not match, there is no way to know what the MCU
measurement measured.

| | Median relative error | Max |
|---|---:|---:|
| trim k (26 parameters) | 1.1e-07 | 3.8e-07 |
| R_eff | 2.7e-03 | 6.1e-02 |
| SOP I* | 2.4e-03 | 3.3e-02 |
| **SOH CNN** | **7.4e-10** | 1.8e-07 |

Trim and SOH are at the float32 precision limit — the formulas are exactly the
same. The 2.7e-03 on R_eff is not implementation error but the **intended
gridding**, and agrees with the 0.30 % of §27.1. SOH matches PyTorch down to
the RMSE (0.0123).

### 27.3 Measurements

n = 200, medians. Interrupts off, call region only.

| Stage | Median | p95 | Max | Iterations |
|---|---:|---:|---:|---:|
| One table lookup D(tau) | 2.91 us | 3.05 | 4.22 | — |
| Trim (12->2, 26 parameters) | 6.34 us | 6.50 | 6.58 | — |
| Fixed-point inversion | 43.06 us | 47.10 | 64.91 | **17** |
| **One SOP (trim + inversion)** | **49.28 us** | 53.29 | 71.19 | 17 |
| Trim feature update (per sample) | 12.42 us | 12.50 | 14.93 | — |
| SOC EKF one step (predict + update) | 7.46 us | 7.76 | 7.83 | — |
| SOC EKF (predict only) | 4.86 us | 4.96 | 5.00 | — |
| **SOH CNN (3 seeds)** | **17,919 us** | 17,919 | 17,919 | — |

[Superseded — 36.4] Re-measured on the board with real dQ/dV curves: **19,442.25 us** median. And the CNN is no longer what runs — ridge does the same job in **6.50 us**.

The 17 iterations match the Python prediction exactly (median 17, 95 % 18).

**One expectation was wrong.** It was expected that "what dominates is the table
lookup, not the 26 parameters", but the trim at 6.34 us is **2.2×** the table
lookup's 2.91 us. The reason 26 MACs cost more than a bilinear interpolation is
two `expf` and two `tanhf` calls — transcendentals are far more expensive than
MACs. Still, inversion accounts for 87 % of the total, so the big picture was
right.

**The same code moves 20 % depending on binary layout.** In the SOP-only build
the table lookup was 2.41 us; adding FEAT/EKF made it 2.91 us. This is ICACHE
and flash placement. That spread must be quoted alongside any absolute value.

### 27.4 Cycle budget

| What | Period | Per period |
|---|---|---:|
| SOC EKF | per sample | 7.5 us |
| Trim feature update | per sample | 12.4 us |
| SOP (discharge/charge x tau 2/10 s = 4 calls) | per decision | 197.1 us |
| **Total** | | **217.0 us** |

| Rate | Load | Headroom |
|---|---:|---:|
| 1 Hz | 0.02 % | 99.98 % |
| 10 Hz | 0.22 % | 99.78 % |
| 100 Hz | 2.17 % | 97.83 % |
| 200 Hz | 4.34 % | 95.66 % |

**SOC and SOP together are 4 % of the CPU even at 200 Hz.** For practical
requirements (1–10 Hz) they are effectively free.

**Only SOH is a constraint.** Taking 17.9 ms in one bite overruns a 100 Hz
period (10 ms). But it runs only once at the end of a charge, so at once per
hour the average load is 0.0005 %. There are three options — use a single seed
(5.97 ms, fits inside 10 Hz; the accuracy cost is unmeasured), split it across
several periods, or separate it into a low-priority task. **The quantity does
not need real-time behaviour, so any of them works.**

### 27.5 Memory

| Configuration | Flash | RAM | Stack high-water |
|---|---:|---:|---:|
| SOP only | 64.6 KB | 12.9 KB | 624 B |
| **SOP + SOH** | **197.2 KB** | **24.8 KB** | 624 B |
| (reference) GRU-128 network | 449 KB | 82 KB | — |

The whole thing is less than half of a single neural network. It fits an
S32K344-class part (4 MB flash, 512 KB RAM) with room to spare.

### 27.6 Alongside existing measurements on the same board

The CEMA bench in `MCU_DEPLOYMENT_REPORT_BUNDLE_20260804` was measured on the
same board.

| Method | Latency | Flash | RAM |
|---|---:|---:|---:|
| Coulomb counting | 5.9 us | 28 KB | 12.6 KB |
| 2RC EKF (FP64 software) | 378.6 us | 418 KB | 13.1 KB |
| **Hybrid one period (SOC+SOP)** | **217 us** | 64.6 KB | 12.9 KB |
| GRU-128 network | 191,789 us | 449 KB | 82 KB |

**That is 1/884 of the neural network.**

It is tempting to put the existing EKF's 378.6 us next to this run's 7.46 us and
say "51×", but **it must not be written that way.** The existing one is software
FP64, and this one is float32 and additionally folds the fast branch into an
instantaneous term to simplify the propagation. Precision and structure changed
together and their shares were not separated. Precision is presumed dominant,
but it was not measured.

### 27.7 The cost of adding AI

Pure physics (EKF + inversion x4) is 179.7 us, hybrid is 217.0 us — **1.21×.**

**The price of adding 26 parameters is 21 %, and in exchange the pack current
used at the same safety level of §25.7 is 11 %p higher on discharge and 6 %p
higher on charge.**

### 27.8 Three things stepped on along the way

- **The 115200 baud in `SETUP.md` is stale.** The actual firmware is 921600.
  When responses come back as 0 bytes, look here first.
- **Reading a float directly out of a packed struct hard-faults on Cortex-M33.**
  CFSR 0x01000000 (UsageFault UNALIGNED), HFSR FORCED. It must be received as
  bytes and `memcpy`'d into an aligned struct. The wire format is 73 B and
  therefore unaligned, so this cannot be avoided.
- **`soh_cnn.py` had never saved weights** — only predictions. MCU deployment
  needs the weights, so `--save-model` was added and it was retrained (RMSE
  0.0128, 10,945 parameters at that time). **The currently adopted numbers are
  RMSE 0.0135 / bias +0.0001** — the values after removing the 8
  temperature-defect curves (30.12).

### 27.9 int8 quantisation — a trade that buys flash, not speed

"Actual RMSE after int8 quantisation — unmeasured" in findings.md §10 had been
open for a long time. Closing it.

**The two targets are different in kind.** The SOH CNN weights (128 KB) are a
neural network; the ECM lookup table (36 KB) is a smooth function. They must not
be handled the same way.

**SOH CNN — per-channel symmetric int8, biases kept as float.**

| Scheme | RMSE (holdout CC) | Weights |
|---|---:|---:|
| fp32 | 0.0123 | 128.3 KB |
| int8 per-tensor | 0.0128 | 32.8 KB |
| **int8 per-channel** | **0.0125** | **32.8 KB** |
| int6 per-channel | 0.0111 | 24.8 KB |
| int4 per-channel | **0.0216** | 16.9 KB |

Rechecking over the full 6-fold gives **0.0128 -> 0.0129 (+1.7e-4)** — effectively
free. int4 collapses. (int6 coming out better on CC is noise over 52 samples.)

**ECM lookup table — per rank x horizon scaling.** The finer the axis split, the
narrower the scale:

| Scale | Median | 95 % | Max | Size |
|---|---:|---:|---:|---:|
| int8 global | 0.43 % | 1.51 % | 3.35 % | 8 KB |
| int8 per horizon | 0.34 % | 1.14 % | 2.46 % | 8 KB |
| **int8 rank x horizon** | **0.23 %** | **0.77 %** | 2.15 % | **8 KB** |
| int16 global | 0.00 % | 0.01 % | 0.01 % | 16 KB |

**The error of rank x horizon int8 is smaller than the gridding error (0.30 % /
2.88 % in §27.1)** — quantisation is buried under gridding. On the 657 rows of
real trustworthy labels, optimism 70.9 -> 71.4 %, worst overshoot
19.53 -> 19.51 A, median difference 0.005 A.

**Board measurements (n=200):**

| | float32 | int8 | Change |
|---|---:|---:|---:|
| **Flash** | 197.2 KB | **76.2 KB** | **−61 %** |
| RAM | 24.8 KB | 24.8 KB | — |
| One SOP | 49.28 us | 53.65 us | +8.9 % |
| Period total | 217.0 us | 235.2 us | +8.4 % |
| One SOH | 17,919 us | 19,442 us | +8.5 % |
| SOH RMSE (6-fold) | 0.0128 | 0.0129 | +1.7e-4 |

**int8 is slower.** It is because the Cortex-M33 has a hardware FPU — a float32
multiply is one cycle, whereas int8 requires sign extension + float conversion +
multiply, so the instruction count rises. Using CMSIS-NN's SIMD (`SMLAD` etc.)
would flip that, but it would require quantising the activations too and pulling
in a runtime, which conflicts with the reason a direct implementation was chosen
in §27.1 (so that what costs how much is not hidden inside a runtime).

**So on this board int8 is a trade that buys flash, not speed.**

**Recommendation: float32 by default, int8 as an option.** Even 197 KB is
comfortable on an S32K344-class part (4 MB flash) and the speed headroom is
ample. Use `--int8` only on flash-tight targets (`export_mcu_tables.py --int8`,
`export_soh_mcu.py --int8`). The C side branches on the `SOP_GRID_INT8` /
`SOH_INT8` macros. The int8 build also passed the host comparison (SOH 1.2e-07).

### 27.10 Cutting seeds — two is the knee

The SOH 17.9 ms is with all three seeds. How much is lost by cutting (full
6-fold, averaged over all combinations):

| Configuration | RMSE | vs baseline | Latency | Spread across combinations |
|---|---:|---:|---:|---:|
| 3 seeds | 0.0128 | — | 17,919 us | — |
| **2 seeds** | **0.0129** | +1.0e-4 | **11,946 us** | ±3.9e-4 |
| 1 seed | 0.0132 | +4.1e-4 | 5,973 us | ±8.4e-4 |

**Two seeds is the knee** — 1.5× faster at a cost of +1.0e-4.

One seed is 3× faster, but **which seed is drawn drives the result** (individual
seed RMSE 0.0123 / 0.0143 / 0.0129; spread across combinations ±8.4e-4). Picking
by holdout would consume the holdout on that choice, so it is not recommended
unless the 3× is genuinely needed.

### 27.11 Full integer path — speed does not flip on this core

The int8 of §27.9 was int8 **weights only**, with float activations. A float
conversion sat before the multiply, making it 8.5 % slower. The real question is
"does going integer all the way through the activations beat the hardware FPU".

`soh_simd.c` was written — per-layer symmetric int8 through the activations,
int32 accumulation, requantisation once at the end of each layer. It uses the
M33's DSP extension (`__SMLAD`, two int16 MACs per cycle) directly. Not pulling
in all of CMSIS-NN is for the same reason as §27.1 — writing the kernels
directly is what keeps what costs how much from being hidden inside a runtime.
Activation scales were calibrated on the **training cells with the holdout
removed** only.

**The first version was 32 % slower** (25,581 vs 19,443 us). Counting MACs per
layer made the reason clear:

| Layer | MACs | Share |
|---|---:|---:|
| conv1 | 5,120 | 5.4 % |
| **conv2** | **81,920** | **86.0 %** |
| dense1 | 8,192 | 8.6 % |
| dense2 | 32 | 0.0 % |

**SIMD had been applied only to dense1, which is 8.6 % of the total.** The
dominant conv2 got only integer multiplies, while each layer newly acquired a
requantisation (multiply + round + saturate).

Packing conv2's input-channel axis into int16 pairs and applying SMLAD recovered
26 %, **25,581 -> 18,885 us**. Even so it is **5 % slower than fp32
(17,919 us).**

| Path | Latency | 6-fold RMSE |
|---|---:|---:|
| **fp32** | **17,919 us** | **0.0128** |
| int8 weights only | 19,441 us | 0.0129 |
| full integer + SIMD | 18,885 us | 0.0136 |

**The reason is structural.** `SMLAD` is two int16 MACs per cycle, and the FPv5
hardware FPU is one float32 MAC per cycle — only 2× of theoretical headroom, and
the requantisation cost eats it. Cortex-M55's Helium (MVE, 8–16 int8 MACs per
cycle) would change that, but the M33 does not have it.

**Accuracy gets worse too** — full integer is 0.0136 over 6-fold, 8× the loss of
int8-weights-only (0.0129). (On the single CC holdout cell it came out 0.0100,
better than fp32, but over 6-fold that was noise. One more example of why one
cell must not decide.)

**Conclusion: full integerisation gains nothing on this core.** It is slower and
less accurate. The recommendation of §27.9 (fp32 by default; int8 weights only
when flash is tight) stands unchanged.

### 27.12 EKF — precision is 7.4×, structure is 1.2×

The reason §27.6 said "51× versus the existing 378.6 us" must not be written is
that precision (FP64 software -> float32 hardware FPU) and structure (full 2RC ->
fast branch folded into an instantaneous term) changed together. All four
combinations were built and split apart. Only the arithmetic type and the state
propagation were changed; the flow was kept identical.

| Variant | Latency | vs float32 |
|---|---:|---:|
| **folded 2RC + float32** | **7.77 us** | 1.00x |
| folded 2RC + double | 57.24 us | **7.37x** |
| full 2RC + float32 | 9.00 us | 1.16x |
| full 2RC + double | 73.05 us | 9.40x |

**Precision is 7.37×, structure is 1.16×.** 88 % of the difference is software
double and the structural simplification contributes only 16 %. The guess in
§27.6 ("precision is presumed dominant") was right and there is now a number.

**The remaining 5× still cannot be attributed.** That is the gap between the
existing CEMA's 378.6 us and the 73.05 us of full 2RC + double here. That EKF
does more work — it propagates the full 3x3 covariance, for one — so **it is not
the same algorithm, and that 5× cannot be split into precision and structure.**
Comparing would require running the same EKF at two precisions, which is outside
this bench's scope.

**Deployment implication**: if an automotive BMS's EKF is written in double and
the core has no double-precision FPU, **moving it to float32 without changing
the algorithm alone gives 7×.** The headroom of §27.4 (0.02 % at 1 Hz) sits on
top of that.

**The target MCU is actually in that condition.** The NXP S32K344 this project
takes as its reference is a mainstream BMS pack-master (BMU) class part —
Cortex-M7 160 MHz, 4 MB flash, 512 KB SRAM, ASIL-D, lockstep — and **its FPU is
single precision only (fpv5-sp-d16), so double is software-emulated.** Its FPU
configuration matches the H563 measured here (M33, FPv5-SP), so the 7.37×
applies directly to the target. (Cortex-M7 can also be configured with a
double-precision FPU, but the S32K3 family is not.)

For the same reason the integer-path ruling of §27.11 carries to the target
unchanged — Cortex-M7 is Armv7E-M, so it has the DSP extension only and no
Helium (MVE). Helium exists only on Armv8.1-M's Cortex-M55 / M85 / M52, which
are still rare at automotive grade.

### 27.13 What remains

- Re-measure the integer path on a core with Helium (M55/M85) — the ruling of
  §27.11 could flip. The M33's SMLAD is only two int16 MACs per cycle.
- A like-for-like comparison running the same EKF at two precisions (the
  remaining 5× above)

---

## 28. Pack level — the min operation does not protect (2026-08-25)

> **Everything in this section is a resampling simulation.** There is no pack,
> no module, no HIL bench. `analysis/sop_pack2.py` draws single-cell
> evaluation rows within condition groups and takes their min, so it can show
> that a cell-calibrated margin is *not* safe by construction on a series
> string — a negative result a simulation can carry — but it cannot show that
> any margin *is* safe on real hardware. Inter-cell error correlation beyond
> the condition bin, a shared current trajectory, thermal gradients and
> imbalance are all absent. The deliverable of this work is cell-level margin
> design; pack behaviour is a sensitivity study pointing at future work.

Sections 19–27 are all single-cell. In a series pack the current is common, so

    pack SOP = min over cells ( per-cell I* )

and putting a cell estimator on a pack passes through that min. **How min
transforms the error** is the system-level question.

### 28.1 Expectation — min ought to protect

For the pack to exceed, two things are needed **at once**: (a) the limiting cell
is overestimated, and (b) no other cell falls below that limit. The more cells,
the harder (b) becomes, so **if the errors are independent and symmetric** the
min operation itself makes the pack safer than a cell, because the minimum of
noisy values is pulled below the minimum of the true values.

### 28.2 Measurement — discharge goes the other way

N-cell packs were simulated 4,000 times by resampling cell holdout errors within
condition groups (SOH band x SOC band x tau). Pack cells have similar SOH, so
draws are within the same SOH band only.

| lambda | N=1 exceed | N=12 | N=96 | | N=1 exceed | N=12 | N=96 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **discharge** | | | | **charge** | | |
| 0.90 | 22.8 % | 33.6 % | 44.0 % | | 17.3 % | 18.2 % | 13.4 % |
| 0.80 | 8.9 % | 16.7 % | 22.2 % | | 5.2 % | 4.3 % | 3.6 % |
| 0.70 | 4.1 % | 11.3 % | **14.1 %** | | 1.6 % | 0.4 % | **0.0 %** |

**Charge gets safer with more cells, as expected. Discharge gets worse.**

### 28.3 Cause — the weaker the cell, the more optimistically it is estimated

Measuring the correlation between measured I* and the predicted/measured ratio
within a condition group:

| | corr(measured I*, ratio) | ratio in bottom 25 % of measured | top 25 % |
|---|---:|---:|---:|
| discharge | **−0.385** | 1.054 | 0.971 |
| charge | **−0.411** | 1.136 | 0.989 |

**The weaker the cell, the larger the ratio** — i.e. the more it is
overestimated. And what sets the pack is the weakest cell. **The min operation
picks "the weakest cell", and that cell happens to be "the most optimistically
estimated cell".** The protection of §28.1 is a story about errors being
unrelated to the true value; here the error is bound to the true value by a
**negative correlation**, so it does not hold.

Physically this is natural. At the same condition, a cell with small I* is a cell
with large resistance, and the pooled table gives an average resistance, so it
sees such cells as under-resistive — the same fact §26 stated as cell-to-cell
deviation not being explicable by condition, seen from another side.

**The reason charge survives** is that despite the same correlation (−0.411) the
tail of its error distribution is far thinner than discharge's (worst overshoot
23.07 A in §25.2 vs discharge 28.51 A, but the |I*| scale differs, so charge is
narrower in relative terms). The thinness of the tail beats the harm of the
correlation.

### 28.4 Implication — cell margin cannot be carried over as pack margin

**Using a cell-calibrated lambda directly on a pack raises the discharge
exceedance rate more than threefold** (4.1 -> 14.1 % at N=96, lambda 0.70). Every
number through §27 is cell-level, and pack-level calibration must be done
separately.

That said, **the adopted configuration does not expose the problem.** At the
lambda values set on a zero-exceedance criterion (discharge 0.679/0.462, charge
0.567/0.544), exceedances are zero for every N from 1 to 192, and usable current
actually **improves** — discharge 69 -> 78 %, charge holding at 57 %. When the
margin is large enough, the harm of the correlation fits inside it.

**[Updated — 31.2]** The lambda above is from the A3 era. After adoption changed
to A8 (discharge 0.683/0.470, charge 0.586/0.560) the simulation was rerun and
**zero exceedance holds.** Charge 10 s does give 4.4 % as N grows if counted at
zero tolerance, but the worst overshoot is 0.09 A, **inside charge's design
tolerance of 0.5 A**, so it is not an exceedance. The flat lambda table of §28.2
did not reproduce (31.4).

**So the conclusion is not "it works on a pack too". It is narrower: in this
simulation the shipped margin survives the min only because it was set on a
cell basis with room to spare.** That is a statement about the resampling, not
about a pack — no pack was built or measured, so nothing here demonstrates
pack-level safety. What it does establish is the negative: a lambda that looks
safe at cell level is broken through threefold once the min is applied, so any
attempt to reduce the margin would have to be revalidated on real pack
hardware, which this work does not have.

### 28.5 Limitations

- Pack cells were **resampled from our six cells**. A real pack has cells of the
  same process and same history, so its spread would be narrower. That is, this
  simulation is **pessimistic**. Conversely our six cells have different ageing
  protocols, so their deviation structure differs from a real pack's — the
  direction is known, the magnitude is not.
- The **shared component** of cell-to-cell error was not modelled explicitly.
  Drawing within a condition group shares the condition's portion (0.122 in
  §26.2) automatically, but shared factors that exist only on a pack, such as a
  temperature gradient, are not in this data.
- A real pack measures every cell voltage, so **the limiting cell is
  observable**. Using that removes the need to leave min to estimation — §23.4's
  "V_pre must be a measured value" becomes stronger at pack level. That path
  cannot be tested with this data.

---

## 29. Pre-paper check — filling three gaps (2026-08-25)

Before drafting, what a reviewer would ask was enumerated in advance. Three
places were empty, and one of them was **the assumption every SOP number so far
had been standing on.**

### 29.1 SOH error propagates into SOP — and the sign decides safety

SOP inversion takes SOH as an input (an axis of the resistance table). **Every
evaluation so far fed the label's true SOH.** A real system estimates it.

Injecting a systematic error:

| SOH error | Optimism | Worst overshoot | Median ratio |
|---|---:|---:|---:|
| −0.03 | 24.1 % | 16.65 A | 0.937 |
| −0.01 | 58.5 % | 18.08 A | 1.027 |
| **0 (true)** | **70.6 %** | **20.35 A** | 1.070 |
| +0.01 | 81.9 % | 21.82 A | 1.121 |
| **+0.02** | **88.5 %** | **30.99 A** | 1.189 |
| +0.05 | 99.0 % | 37.77 A | 1.406 |

**Reading SOH high makes resistance look small and SOP optimistic.** Reading it
just 2 %p high jumps the worst overshoot from 20.4 to 31.0 A.

~~**The adopted SOH arm's bias of +0.0010 is, of all things, on the dangerous
side.**~~ **[Retracted — 30.12]** 90 % of that bias came from 8
temperature-defect curves. With the defects removed the bias is **+0.0001**,
effectively zero, and there is no basis for saying the adopted arm is skewed in
the dangerous direction.

**The mechanism of the table above (the injection experiment) is still valid** —
reading SOH high does make SOP optimistic. What is not valid is the diagnosis
that "our arm is skewed that way".

(Left for reference: injecting the defect-included error distribution
RMSE 0.0128 / bias +0.0010 gave optimism 70.6 -> 72.4 % and worst
20.35 -> 22.72 A. It was not re-measured with the defect-excluded version
(RMSE 0.0135 / bias +0.0001).)

**The safety factor must be reset:**

| SOH input | lambda(10 s) | lambda(2 s) | Exceed | Worst | Usable current |
|---|---:|---:|---:|---:|---:|
| true SOH | 0.679 | 0.462 | 4/657 | 1.19 A | **70 %** |
| **estimated SOH** | **0.594** | **0.423** | 6/645 | 1.85 A | **63 %** |

**7 %p is the price of SOH estimation.** Every number through §27 sits on top of
true SOH, so the paper must use the **estimated-SOH version as the main result**
and quote the true-SOH version alongside as an upper bound.

This is why the SOH arm and the SOP arm must not be evaluated separately — only
chaining them gives the system's answer.

### 29.2 Sensor noise is not a problem

The voltage resolution of an automotive BMS AFE is on the order of mV. Injecting
noise into V_pre:

| V_pre noise | Optimism | Worst overshoot |
|---|---:|---:|
| none | 70.6 % | 20.35 A |
| 1 mV rms | 70.8 % | 20.25 A |
| 5 mV | 70.8 % | 20.97 A |
| 10 mV | 70.9 % | 20.68 A |

**Even 10 mV is buried.** It is because the margin (V_pre − V_min) has a median
of 1.0 V.

This reinforces §23.4's "V_pre must be the measured terminal voltage" — **sensor
noise (mV) is an order of magnitude smaller than model OCV error (45–72 mV).**
Using the measurement is clearly the better side.

### 29.3 Feature ablation — the residual channels do the work, state and age do not

Are all 12 features needed? §24 measured the correlation of k_f and dR_fast at
+0.915, so
"surely one would do" is a natural question.

Refits were made per holdout cell using subsets only (linear, so least squares —
for ranking comparison):

| Feature set | dV RMSE | vs full |
|---|---:|---:|
| no correction (k=1) | 87.2 mV | — |
| **all 12** | **58.2 mV** | — |
| dR_fast alone | 62.9 mV | +8.2 % |
| dR_fast + dR_slow | 62.5 mV | +7.5 % |
| residual channels only (0–5) | 61.6 mV | +5.8 % |
| **state/age only (6–11)** | **75.3 mV** | **+29.4 %** |
| dR_fast + SOC, SOH | 64.0 mV | +9.9 % |

**dR_fast alone gets 92 % of the full set.** Conversely, keeping only state/age
(SOC, SOH, T, I_rms, R_nom) is 29 % worse.

This quantitatively reconfirms the A5 falsifier of §7.6 — the rejection condition
was "if the correction is merely a function of state and age, it should be
absorbed into the table", and **it cannot be absorbed.** What creates the
correction is the residual channels, that is, **how far the measured voltage
departs from the nominal model.**

**Deployment implication**: it may not be necessary to compute all 12. With
dR_fast alone the EW states drop to two (e_ir, e_ii) and the feature-update cost
falls.

**[Measured — 33.1]** On the board, **13.25 -> 5.99 us (−55 %).** [Updated 2026-08-31: the stored table now reads 13.27 us for the A3 feature update, measured on the all-cell deployment header rather than a leave-one-cell-out fold. The A8 figure and the −55 % are unchanged.] The
safety-factor price was measured in 29.7 / 32.7 and there is **none** — on
charge A8 actually beats A3.

### 29.4 SOC was using true SOH too

After finding §29.1 on SOP, SOC was checked. **It was the same problem.**

`ekf_soc.py` uses SOH in two places — as an axis of the resistance/OCV tables
(`theta`, `ocv`, `hyst_M`) and as the **schedule of the measurement noise
R_volt** (110 mV at SOH 0.70, 15 mV at 1.00). The evaluation loop feeds the
cache's true value (`soh = np.nanmedian(SOH)`).

The second is particularly bad. **Reading SOH high makes R_volt small, which
raises the Kalman gain and overtrusts the model.** The SOH arm's bias is
positive, so the direction is dangerous.

The SOH arm's predictions were stitched in the same way as for SOP (the
preceding charge) and rerun (SOH 0.68–0.70 region, 6 runs per cell):

| Cell | SOH true / estimated | SOC RMSE true -> estimated |
|---|---|---|
| CC | 0.696 / 0.706 | 2.87 -> 2.46 %p |
| BOOST | 0.692 / 0.697 | 2.95 -> 3.11 |
| BOOST_NEGPULSE | 0.682 / 0.687 | 2.97 -> 3.29 |
| **BOOST_REST** | 0.691 / **0.745** | 3.01 -> **4.19** |
| CC_CELL2 | 0.687 / 0.679 | 3.81 -> 3.96 |
| BOOST_NEGPULSE_1S | 0.693 / 0.697 | 3.03 -> 3.10 |
| **overall** | | **3.11 -> 3.35 %p (+0.24)** |

**The loss is concentrated in one cell.** BOOST_REST reads SOH 5.4 %p high and
its SOC error grows 1.18 %p. On the rest the SOH error is within ±0.008, so the
impact is negligible.

(These numbers look only at the most-aged region, so they are a different
condition from the 1.59 %p over the full six-cell trajectories.)

**[Added 2026-08-26 — these numbers must be re-measured]** The SOC benchmark
used here is circular. The true label is `SOC = 1 + Ah/3.0` and the filter's
prediction is `soc + I dt/3600/3.0` — the same equation — and the filter was
started from the exact initial SOC. So turning the voltage correction off
entirely gives the best RMSE, 0.12 %p (§30.1).

This section's logic is "reading SOH high makes R_volt small, raises the Kalman
gain and overtrusts the model", but on that benchmark **every change that raises
the gain comes out bad unconditionally** (measured: tightening to R_volt=0.005
gives 1.51 -> 2.90 %p). So this table cannot separate how much of the +0.24 %p
price is due to SOH error and how much to "having used voltage more". It must be
re-measured on the perturbation benchmark of §30.

**All three arms were standing on true SOH.** The paper should bundle this into
one section — "evaluating state estimators separately does not give the system's
answer" is a methodological point in itself, and this work quantified its price
at 3 %p discharge, 4 %p charge, and 0.24 %p SOC.

### 29.5 Reproduction spread — correction

§7.3 wrote that "the trim's optimism moves 8–9 %p under a 2 % change in the
training data" and lumped that together with the seed-to-seed variation of §20
(15 %) as the same property. **That was wrong.**

Building four independent training runs with the same data and settings, varying
only the seed:

| Source of spread | n | Optimism range | Availability range | lambda range |
|---|---:|---:|---:|---:|
| **seed only** | 4 | **0.00 %p** | 0.02 %p | 0.001 |
| training data differs by 2 % | 2 | 8.28 %p | 0.98 %p | 0.015 |

**It reproduces perfectly with respect to seed.** It is a 26-parameter linear
model, so the optimum converges to a single solution — obvious, but it was
written down without checking. The median k_f differs only in the fifth digit:
1.032025 / 1.032042 / 1.032039.

The 8.28 %p is because **the training data changed**, and even that is not a
clean comparison because the evaluation row set changed with it (657 -> 631).

**So the headline numbers need no seed error bars.** What must be disclosed
instead is data sensitivity, and that is already recorded in §7.3 as "removing
the 6 temperature-defect cycles (2.1 %) moves it this much". The 15 % observed
in §20 was between **two training runs of the symmetric trim**, and those two
differed in data as well as code state — it is not a seed effect.

### 29.6 What remains before the draft

**Must do**
- Redo the §27–28 numbers on the estimated-SOH version. §29.1/29.4 produced the
  cell level (discharge 70 -> 67 %, charge 57 -> 53 %, SOC 3.11 -> 3.35 %p). The
  pack level (§28) is not done.
- The safety-factor price of the dR_fast-only reduction (A8) — in training

**Nice to have**
- External validation of the SOH arm
- A comparison group against a standard SOP technique from the literature

**Cannot do** (grounds recorded in 22.5 / 26 / 28.5)
- tau = 2 s validation, temperature x horizon crossing, more cells

### 29.7 Measured result of the dR_fast-only trim (A8)

What §29.3 had only estimated by ablation was actually trained and measured.
A8 is the trim that keeps only dR_fast of the 12 EW features and drops the other
11.

On the voltage basis (mean over the 6 holdout cells, against A0 = fixed k):

| Trim | A0 | Model | Improvement |
|---|---|---|---|
| all 12 features | 85.36 mV | 58.76 mV | +31.2% |
| dR_fast alone (A8) | 85.36 mV | 62.81 mV | +26.4% |

On voltage alone A8 is 6.9% worse. That broadly matches the +8.2% estimated by
ablation.

But converting to current flips the order:

| Trim | Optimism | RMSE | λ(10 s) | Usable current | Worst overshoot |
|---|---|---|---|---|---|
| all 12 features | 69.4% | 5.43 A | 0.679 | 69.8% | 1.19 A |
| dR_fast alone (A8) | 62.8% | 4.95 A | 0.683 | 68.9% | 0.97 A |

The two values that decide post-deployment performance — the safety factor λ and
the usable current — are effectively the same (λ 0.683 vs 0.679, current 68.9%
vs 69.8%). Meanwhile the raw prediction before multiplying by λ is better for A8
on every item: optimism 6.6 %p lower, RMSE 0.48 A lower, worst overshoot 0.22 A
lower.

Interpretation: the other 11 features do reduce the voltage residual further,
but that gain does not carry over into current. Turning voltage error into
current is a division (dV/dR), so there is no reason for the places that win on
voltage to coincide with the places that win on current. That means using
voltage RMSE as the trim's evaluation metric can select the wrong deployment
performance.

Deployment advantages of A8: the feature update drops from 12 EW states to 2,
and the trim input dimension goes 12 -> 1, so parameters go 26 -> 4.

**Adoption**: A8 is the final configuration.

Checked — the evaluation row sets of the two versions are exactly the same (631
trustworthy-label rows, `meas` agreeing), and the gain is not concentrated in one
cell. Per cell:

| Cell | n | Optimism 12 -> A8 | RMSE 12 -> A8 |
|---|---:|---|---|
| BOOST | 153 | 73.2 -> 66.0 % (−7.2) | 4.76 -> 4.24 A |
| BOOST_NEGPULSE | 43 | 25.6 -> 23.3 % (−2.3) | 4.80 -> 4.34 A |
| BOOST_NEGPULSE_1S | 169 | 87.0 -> 78.7 % (−8.3) | 5.53 -> 4.85 A |
| BOOST_REST | 25 | 8.0 -> 4.0 % (−4.0) | 4.78 -> 4.73 A |
| CC | 153 | 77.1 -> 69.9 % (−7.2) | 6.18 -> 5.89 A |
| CC_CELL2 | 88 | 54.5 -> 50.0 % (−4.5) | 5.42 -> 4.81 A |
| **overall** | **631** | **69.4 -> 62.8 % (−6.7)** | **5.43 -> 4.95 A** |

**A8 is better on both metrics in 6 of 6 cells.** The trim is trained leaving one
cell out anyway, so this table is itself a holdout result. Seed spread was
measured at 0.00 %p in §29.5, so this difference is not seed.

The full 12-feature configuration is kept in §16 as a comparison group.

## 30. The SOC benchmark was circular

### 30.1 The label and the prediction are the same equation

The true label is `SOC = 1 + Ah/3.0` (readme line 117), and the EKF's prediction
step at `ekf_soc.py:231` is

    soc_p = soc + I * dt / 3600.0 / Q_RATED_AH        # Q_RATED_AH = 3.0

Same current, same divisor, same equation. The evaluation loop starts the filter
exactly at `soc0 = soc[0]` (`f.x[0] = soc0` in `run()`). Then integrating the
current alone reproduces the label by definition, and the voltage correction can
only **add** error.

Measured (36 drive-cycle runs, 6 cells):

| Configuration | Overall RMSE | Worst | Aged <0.80 |
|---|---:|---:|---:|
| **voltage correction off (pure current integration)** | **0.12 %p** | **0.48** | **0.20** |
| adopted configuration (gate 1 A / 30 s) | 1.51 | 3.86 | 2.13 |
| strong voltage correction (R_volt = 5 mV) | 2.90 | 6.99 | 4.53 |

Not using voltage at all is 12× better, and the more voltage is used the worse it
gets. The remaining 0.12 %p is dt quantisation and the `soc_span` limit.

### 30.2 Which explains the past conclusions

`ecm_kf_plan.md` wrote that "seven attempts to fix the model all failed; three
attempts to decide when to trust it all succeeded". The three that succeeded —
raising R_volt, the low-current gate, holding for 30 s — are **all in the
direction of trusting voltage less**. Since the benchmark's optimum is "do not
trust voltage", anything in that direction wins. The seven that failed were all
attempts to fix the voltage model, and as long as voltage is used they cannot
win.

This session fell into the same trap. A rule that grows R with the residual
magnitude gave 1.51 -> 0.84 %p (−44.6 %) and even passed leave-one-cell-out
validation (all 6 folds picked the same constants k=20, w=0.003), but what that
rule does is raise R by a median factor of 505 and suppress the voltage
correction 97.8 % of the time. Raising k to 200 makes it better still. It was
measuring how close it had got to pure current integration.

**Cell-level cross-validation cannot catch this error.** The circularity is not
per cell; it is in the label definition.

### 30.3 What is affected and what is not

Not affected: SOP (labels measured from HPPC currents), SOH (labels are
capacity). The labels are unrelated to the filter's prediction equation.

Affected: everything that compared EKF variants on this benchmark. The R_volt
schedule, the low-current gate, the hold time, and §29.4's "estimated-SOH price
of +0.24 %p" belong here.

### 30.4 A benchmark that breaks the circularity

The reason to use a Kalman filter is that three things do not hold in a real
vehicle: the initial SOC is unknown, the current sensor has an offset, and it has
a gain error. The label is kept on the true current and only the filter is given
the distorted current and initial value.

Overall RMSE (%p, mean over 36 runs):

| Distortion | pure current integ. | EKF no gate | **EKF adopted (gate)** | EKF+spread k=20 | EKF+spread k=200 |
|---|---:|---:|---:|---:|---:|
| none | 0.12 | 3.11 | 1.51 | 0.84 | 0.77 |
| initial SOC +10 %p | 9.95 | 3.11 | 1.57 | 1.04 | 1.51 |
| initial SOC −10 %p | 10.03 | 3.10 | 1.57 | 0.94 | 1.04 |
| current offset +0.10 A | 10.32 | 2.41 | 2.86 | 8.15 | 8.64 |
| current offset −0.10 A | 10.39 | 3.96 | 3.77 | 7.56 | 8.64 |
| current gain +1 % | 0.37 | 3.14 | 1.55 | 0.98 | 0.94 |
| current gain −1 % | 0.29 | 3.08 | 1.51 | 0.81 | 0.74 |
| **mean over the 6 distortions** | 6.89 | 3.13 | **2.14** | 3.25 | 3.58 |

The ranking flips.

1. **The adopted configuration is already best** (2.14). The spread rule that
   looked like −44.6 % on the circular benchmark is in fact a **52 % loss**,
   2.14 -> 3.25. It is discarded.
2. **The one weakness is current-sensor offset.** It is 1.5 %p under the other
   conditions but 2.86 / 3.77 %p under offset. Looking at only the last quarter
   gives 2.18 / 3.46, so it is residual error that does not converge away.
3. Pure current integration diverges under offset, with a last-quarter RMSE of
   15.3 %p. That is exactly what the voltage correction does, and the circular
   benchmark could not see that work.

### 30.5 Current-offset state

The offset state `q_b` that was abandoned in the past was a constant added to the
predicted **voltage** (`y += xp[self.ib]`, limited to ±0.25 V). A current-offset
state had never been tried.

It was added this time: `I_true = I_meas - ib` is used in both the coulomb
integration and the voltage model, and the Jacobian gets
`dsoc/d(ib) = -dt/3600/Q` and `dy/d(ib) = -R0`. The SOC drift the offset creates
surfaces as OCV error, so it is observable.

(Measurement in progress — recorded in 30.6.)

### 30.6 R_volt and the gate were re-chosen — the values are the same, the reason is not

The R_volt schedule (110 mV at SOH 0.70, 15 mV at 1.00) and the gate (|I| <= 1 A
held for 30 s) were all chosen on the circular benchmark. They were re-chosen on
the perturbation benchmark: 4 R_volt multipliers x 5 gates x 7 perturbations,
selected leave-one-cell-out.

**Result: all six folds picked multiplier x1.00 and gate 1 A / 30 s. Change
0.0 %.**

The values not changing does not mean the circularity was harmless. Putting the
three tables side by side shows why those values survived.

**With no distortion** (what the circular benchmark saw, %p):

| Multiplier | no gate | 1A immediate | 1A 30s | 1A 120s | 3A 30s |
|---|---:|---:|---:|---:|---:|
| x0.25 | 3.84 | 3.49 | 2.14 | 1.77 | 2.82 |
| x1.00 | 3.11 | 2.68 | 1.51 | 1.33 | 1.90 |
| x2.00 | 2.81 | 2.33 | 1.26 | **1.13** | 1.56 |

The optimum is at the bottom right — raise R_volt and lengthen the hold, i.e.
use voltage less. Exactly as §30.1 predicted.

**With a current offset** (%p):

| Multiplier | no gate | 1A immediate | 1A 30s | 1A 120s | 3A 30s |
|---|---:|---:|---:|---:|---:|
| x0.25 | 3.84 | 3.50 | **2.58** | 3.27 | 2.90 |
| x1.00 | 3.18 | 2.92 | 3.32 | 3.68 | 2.68 |
| x2.00 | 3.13 | 3.17 | 4.29 | 4.55 | 3.58 |

**The direction is exactly opposite.** R_volt must be reduced — voltage used more
— to catch the offset. The circular benchmark's optimum (x2.00, 1A 120s) is 4.55
here, the worst in the whole table.

**Mean over the 7 perturbations** (%p):

| Multiplier | no gate | 1A immediate | 1A 30s | 1A 120s | 3A 30s |
|---|---:|---:|---:|---:|---:|
| x0.25 | 3.84 | 3.49 | 2.28 | 2.27 | 2.85 |
| x0.50 | 3.45 | 3.08 | 2.07 | 2.11 | 2.40 |
| x1.00 | 3.13 | 2.75 | **2.05** | 2.09 | 2.14 |
| x2.00 | 2.90 | 2.58 | 2.16 | 2.20 | 2.16 |

The reason the current values come first is that they are the **balance point of
two forces pushing in opposite directions**. The circular benchmark was seeing
only one of them, and happened to stop near the balance point.

That is the lesson this section leaves — a circular benchmark can produce the
right answer, but when it does, that answer is **right for no reason**. Explaining
why that value is the value requires breaking the circularity.

### 30.7 Measured result of the current-offset state

Measuring the state added in §30.5. Does the filter find the offset it was given:

| q_ib | injected 0.00 | +0.05 | +0.10 | +0.20 | −0.10 |
|---|---:|---:|---:|---:|---:|
| 1e-10 | +0.002 | +0.052 | +0.102 | +0.202 | −0.098 |

**Error 2 mA.** Zero runs diverged.

RMSE (%p, gate 1 A / 30 s):

| Injected offset | state off | state on (p0=0.25) | state on (p0=1e-4) |
|---|---:|---:|---:|
| 0.00 A | **1.51** | 2.44 | 1.94 |
| +0.05 A | **1.67** | 2.42 | 1.81 |
| +0.10 A | 2.86 | 2.42 | **1.81** |
| +0.20 A | 5.98 | 2.41 | **2.41** |
| −0.10 A | 3.77 | 2.44 | **2.70** |
| mean of 4 | 3.53 | 2.43 | **2.21** |

**The one decisive knob is p0_ib.** Tightening the initial uncertainty from 0.25
(standard deviation 0.5 A) to 1e-4 (0.01 A) more than halves the price paid when
there is no offset, 0.90 -> 0.43 %p, and when there is an offset it actually
improves. With large initial uncertainty the state absorbs the true SOC error
instead early on; tightening blocks that.

`ib_clip` has no effect (0.3 A and 2.0 A are the same). `q_ib` is the same over
1e-12 to 1e-10; 1e-8 gives marginally better RMSE but the offset estimate error
worsens threefold, 7 -> 22 mA.

**The break-even is about 0.07 A.** If the real current-sensor offset is smaller
than that, adding the state is a loss. Against a 30 A full scale, 0.07 A is
0.23 %, right in the middle of a typical automotive sensor's offset error
(0.1–0.5 % of full scale). So this decision depends on the sensor specification
and is not obvious either way.

**Note**: the table above was chosen using all 36 runs. Confirmation
leave-one-cell-out is in progress.

### 30.8 Final confirmation — SOC can barely be improved

The R_volt multiplier x gate x the current-offset state's p0_ib were put on one
grid and chosen leave-one-cell-out (24 configurations x 7 perturbations x 36
runs).

**All six folds picked `multiplier x1.00 / gate 1 A 30 s / p0_ib = 1e-5`.**

Measured on the unseen cell:

| Holdout | current adopted | with offset state | Change |
|---|---:|---:|---:|
| BOOST | 1.85 | 1.79 | −3.6 % |
| BOOST_NEGPULSE | 1.90 | 1.86 | −1.9 % |
| BOOST_NEGPULSE_1S | 2.39 | 2.30 | −3.9 % |
| BOOST_REST | 2.12 | 2.14 | +0.7 % |
| CC | 1.74 | 1.68 | −3.9 % |
| CC_CELL2 | 2.29 | 2.31 | +0.8 % |
| **total** | **2.05** | **2.01** | **−1.9 %** |

Breakdown by perturbation (%p):

| Configuration | none | initial SOC ±10 | offset +0.1 | offset −0.1 | gain ±1% | mean |
|---|---:|---:|---:|---:|---:|---:|
| state off | **1.51** | 1.57 | 2.86 | 3.77 | 1.53 | 2.05 |
| p0=1e-5 | 1.65 | 1.71 | **2.35** | **3.30** | 1.68 | **2.01** |
| p0=1e-4 | 1.94 | 1.99 | **1.81** | **2.70** | 1.96 | 2.05 |

It pays 0.14 %p when there is no offset and earns 0.5 %p when there is, so on
average they nearly cancel. Opening p0 further to 1e-4 earns more under offset
(1.81) but the everyday price grows (1.94) and the mean comes out the same.

**Recommendation: do not include it in the default configuration.** It adds one
state, raising EKF computation by about 30 %, for a gain of 1.9 %. That said,
the break-even is an offset of about 0.07 A (0.23 % of a 30 A full scale) and a
typical automotive current sensor's offset is 0.1–0.5 % of full scale, so **if
the target BMS's sensor specification is above that, including it is right.** It
is not obvious either way.

It is left in `ekf_soc.py` as `q_ib` / `p0_ib` / `ib_clip` (off by default).

### 30.9 The number this section leaves

~~The SOC number to use in the paper is not 1.51 %p but **2.05 %p.**~~
**[Corrected — 34.2]** 2.05 is the mean over all SEVEN rows, and the seventh is
the undisturbed case. The mean over the six disturbances this sentence
describes is **2.14 %p**, which §30.4's own table prints. Quote 2.14 and label
it, or quote 2.05 and say it includes the undisturbed row.

The former figure, 1.51 %p, feeds the filter the very current that made the
label and starts it at the exact initial value; the latter averages over
initial SOC error ±10 %p, current offset ±0.1 A, and gain error ±1 %.

### 30.10 What remains

- **Asymmetry**: offset −0.1 A (3.77 / 3.30) is always worse than +0.1 A
  (2.86 / 2.35). The drive cycle runs SOC 0.85 -> 0.35, i.e. in the discharge
  direction, so a negative offset adds to discharge, but why it is harder to
  observe was not examined. The `soc_span` lower bound is suspected.
- **Re-measuring 29.4**: the "estimated-SOH price of +0.24 %p" must be
  re-measured on the perturbation benchmark.
- **p0_ib grid ends**: 1e-5 is an interior optimum between off and 1e-4, but the
  space between was not examined more finely. The optimum is shallow, so there
  appears to be no real gain.

### 30.11 Re-measuring 29.4 — the mechanism claim was wrong and the price is half

§29.4 produced "using estimated SOH gives SOC 3.11 -> 3.35 %p (+0.24)" on the
circular benchmark and pointed at the **R_volt schedule** as the cause —
"reading SOH high makes R_volt small, raises the Kalman gain and overtrusts the
model."

The two paths were separated and re-measured on the perturbation benchmark. SOH
enters in two places: (a) the axis of the resistance/OCV tables, (b) the
measurement-noise R_volt schedule.

The SOH arm's predictions were stitched from the preceding charge (36 runs,
estimate error RMSE 0.0129, bias +0.0032 — consistent with the arm's overall
0.0128 / +0.0010).

| SOH input | mean over 7 perturbations | none | offset +0.1 | offset −0.1 | vs true |
|---|---:|---:|---:|---:|---:|
| both true | **2.05 %p** | 1.51 | 2.86 | 3.77 | — |
| **R_volt estimated only** | **2.05** | 1.51 | 2.87 | 3.77 | **+0.00** |
| table axis estimated only | 2.17 | 1.65 | 2.79 | 3.97 | +0.12 |
| both estimated | 2.17 | 1.65 | 2.82 | 3.96 | +0.12 |

**The R_volt path's price is 0.00.** The mechanism claim of §29.4 was wrong. It
all comes from the axis of the resistance/OCV tables.

Working out the magnitude makes it obvious. R_volt interpolates 110 -> 15 mV over
SOH 0.70 -> 1.00, a slope of about 0.32 V per SOH. An SOH error of 0.013 moves
R_volt by 4 mV. Over a 15–110 mV range, 4 mV is negligible. §29.4 did not measure
that magnitude and wrote "particularly bad" from the direction alone.

**The total price is also +0.12 %p, not +0.24.** Per cell:

| Cell | SOH error (mean) | true -> estimated |
|---|---:|---|
| BOOST | +0.0075 | 1.85 -> 1.94 (+0.08) |
| BOOST_NEGPULSE | +0.0041 | 1.90 -> 2.04 (+0.14) |
| BOOST_NEGPULSE_1S | +0.0067 | 2.39 -> 2.32 (−0.07) |
| **BOOST_REST** | **+0.0095** | 2.12 -> **2.64 (+0.52)** |
| CC | −0.0060 | 1.74 -> 1.86 (+0.12) |
| CC_CELL2 | −0.0024 | 2.29 -> 2.26 (−0.03) |

**§29.4's observation that the loss is concentrated in one cell holds** —
BOOST_REST is +0.52 and the mean of the other five is +0.05. Two cells actually
improve marginally, which is taken as noise level.

**Updated price of "all three arms were standing on true SOH"**: discharge
70 -> 67 %, charge 57 -> 53 %, **SOC 2.05 -> 2.17 %p.**

Note: when picking the 36 runs, `np.linspace` picked the same file twice on
BOOST_NEGPULSE_1S (cycle 1149 twice). That cell has 5 effective runs.

## 31. Pack level again, on estimated SOH (2026-08-26)

> Same standing caveat as §28: this is the resampling simulation, not pack
> hardware. "Re-measured" below means the simulation was re-run on estimated
> SOH, not that anything was measured on a pack.

Every number in §28 sat on true SOH (the point §29 raised). It is re-run
here.

### 31.1 The adopted evaluation configuration was pinned down

§28's pack script was not kept as a file, so it was rebuilt
(`analysis/sop_pack2.py`). In the process the settings that reproduce §16's
lambda were found and are recorded here — they were not in the documentation
until now:

| Direction | Trim directory | Aggregation | Tolerance | lambda(10s / 2s) |
|---|---|---|---|---|
| discharge | `runs_trim_v2` | `max` | 0.0 A | **0.679 / 0.462** |
| charge | `runs_trim_chg_v2` | `max` | **0.5 A** | **0.567 / 0.544** |

The 0.5 A tolerance is the value §25 set as charge's knee. Counting the charge
lambda at zero tolerance gives 0.438 / 0.421, so without this value §16's numbers
cannot be reproduced.

### 31.2 Estimated SOH tightens lambda and cuts current

lambda was re-set leave-one-cell-out for the true and estimated versions
separately. Packs were simulated 4,000 times by resampling N cells with
replacement within condition groups (SOH band x SOC band).

| Direction | tau | SOH input | lambda | N=192 exceed | N=192 worst | N=192 usable current |
|---|---:|---|---:|---:|---:|---:|
| discharge | 10 s | true | 0.679 | 0.0 % | 0.00 A | **82.9 %** |
| discharge | 10 s | **estimated** | **0.628** | 0.0 % | 0.00 A | **66.7 %** |
| discharge | 2 s | true | 0.462 | 0.0 % | 0.00 A | 83.5 % |
| discharge | 2 s | **estimated** | 0.456 | 0.0 % | 0.00 A | 81.7 % |
| charge | 10 s | true | 0.567 | 4.4 % | 0.10 A | 58.3 % |
| charge | 10 s | **estimated** | **0.501** | 0.0 % | 0.00 A | 53.3 % |
| charge | 2 s | true | 0.544 | 0.0 % | 0.00 A | 56.4 % |
| charge | 2 s | **estimated** | 0.512 | 0.0 % | 0.00 A | 54.5 % |

**Re-setting lambda on estimated SOH preserves §28.4's "zero exceedance".** The
4.4 % on the charge 10 s true version counts exceedance at zero tolerance, and
its worst is 0.10 A, inside charge's design tolerance of 0.5 A — i.e. not an
exceedance by the charge criterion.

**The price is usable current.** Discharge 10 s loses 16.2 %p, 82.9 -> 66.7 %.
tau = 2 s barely moves (83.5 -> 81.7 %).

### 31.3 Without re-setting the margin, discharge at tau = 2 s breaks through

Using the lambda set on true SOH directly on the estimated-SOH version.  N is
the string length in the resampling simulation, not cells on a bench:

| Direction | tau | lambda used | N=1 | N=12 | N=48 | N=96 | N=192 | N=192 worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| discharge | 10 s | 0.679 | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.00 A |
| **discharge** | **2 s** | 0.462 | 0.9 % | 7.8 % | 23.9 % | 34.5 % | **42.0 %** | 0.19 A |
| charge | 10 s | 0.567 | 0.5 % | 2.9 % | 1.7 % | 0.3 % | 0.0 % | 0.04 A |
| charge | 2 s | 0.544 | 1.1 % | 1.3 % | 0.0 % | 0.0 % | 0.0 % | 0.00 A |

It is stable across three seeds at 41.5 / 42.1 / 43.2 %.

**The magnitude is small.** The worst overshoot of 0.18 A is 0.59 % of that
condition's true median |I*| of 31.9 A. The exceedance *rate* jumping to 42 % is
because lambda sits by definition exactly on the zero-exceedance boundary, and at
a boundary a very small change swings the rate a lot. Quoting the rate alone
would exaggerate — the rate and the magnitude must be read together.

**Implication**: §28.4 said "any attempt to reduce the margin must be revalidated
on a pack", and **attaching an SOH estimator is one of those "attempts".** At
cell level the tau=2 s lambda barely moves, 0.462 -> 0.456, and looks harmless,
but passing through the pack's min turns that 1.3 % difference into an exceedance
rate of 0 -> 42 %. The two effects (estimated SOH, pack min) are each harmless
alone and only surface when they overlap.

### 31.4 §28.2 did not reproduce

§28.2's flat lambda table (discharge at lambda=0.70: N=1 4.1 %, N=96 14.1 %) does
not reproduce under the same definition. The rebuild gives N=1 9.2 %, N=96
28.6 %, about double. The N=1 value does agree with a direct cell-level
calculation on the same data (10.0 %), so the reimplementation is at least
internally consistent. Which lambda definition §28.2 used cannot be recovered
from the documentation (it likely used a per-condition calibrated lambda).

**§28.2's absolute numbers are not quoted.** The directional claim (discharge
gets worse with more cells, charge gets better) does hold for discharge in the
reimplementation (9.2 -> 21.6 -> 28.6 %). Charge is not monotone in the
reimplementation (5.5 -> 12.8 -> 10.7 %), so §28.2's charge claim is
unconfirmed.

§28.3's correlation reproduces on charge (original −0.411, rebuild −0.378).
Discharge differs: original −0.385 vs rebuild **−0.608**.

(These numbers come from the evaluation of the adopted trim A8 —
`results/tables/correlation.csv`. The first rebuild measured with the default
trim and gave −0.400 / −0.587. Changing the trim changes the correlation, so
which version it is must be written alongside.)

## 32. Literature comparison group — why online adaptation does not work as-is (2026-08-26)

The obvious question a reviewer will ask: "instead of learning 26 parameters
offline, why not just update the resistance with online RLS?" That is the
standard technique for adaptive SOP.

### 32.1 The trim's main feature is already RLS

The main feature in `sop_trim_features.py` is

    dR_fast = EW{I * r} / EW{I * I},    r = V - V_hat (drive voltage vs nominal model)

and that is an **exponentially weighted least squares** regression of the
residual on the current. That is, the trim is "online RLS plus a learned map
turning that estimate into a multiplier", and §29.3 shows dR_fast alone gives
92 % of the gain.

So the real question is **"is the learned map worth anything, or would using
dR_fast directly do?"** It was measured with `sop_baselines.py` /
`sop_shrink.py`.

### 32.2 The parameter ladder

Leave-one-cell-out, voltage RMSE (mean of per-cell RMSE — the same convention as
the trim tables):

| Method | Parameters | RMSE | vs A0 |
|---|---:|---:|---:|
| **A0** no correction (= classical SOP on the HPPC resistance table) | 0 | 85.36 mV | — |
| **direct plug-in** k = 1 + dR/R_nom | 0 | **134.71 mV** | **+57.8 %** |
| **direct plug-in** k = exp(dR/R_nom) | 0 | 145.78 mV | +70.8 % |
| **shrinkage coefficient** k = exp(a * dR/R_nom) | 2 | 69.10 mV | −19.0 % |
| **A8** dR_fast alone | 4 | 62.81 mV | −26.4 % |
| **A3** 12 features | 26 | 58.76 mV | −31.2 % |

**Using the RLS estimate directly is worse than doing nothing** (+57.8 %).

### 32.3 The reason — it is magnitude, not sign

The direct plug-in's k correlates 0.67–0.94 with the learned k (consistent with
the +0.89 the documentation quotes). The direction is right. What is wrong is
the magnitude:

| Cell | median dR_fast (mOhm) | R_fast_nom (mOhm) | median learned k_f | median direct k_f |
|---|---:|---:|---:|---:|
| BOOST | +2.92 | 12.71 | 1.027 | 1.214 |
| BOOST_NEGPULSE | −2.85 | 16.21 | 0.955 | 0.800 |
| CC | +3.06 | 12.11 | 1.032 | 1.245 |

The direct plug-in **overcorrects by 5–7×**. Fitting the shrinkage coefficient
recovers that value:

| Holdout | a_f | a_s |
|---|---:|---:|
| BOOST | 0.160 | 0.400 |
| BOOST_NEGPULSE | 0.180 | 0.300 |
| BOOST_NEGPULSE_1S | 0.180 | 0.280 |
| BOOST_REST | 0.200 | 0.160 |
| CC | 0.200 | 0.320 |
| CC_CELL2 | 0.240 | 0.220 |

**a_f is stable at 0.16–0.24 independently of cell.** It reads physically — drive
cycles mostly run below 5 A while SOP lives around 30 A. Because R depends
strongly on current, a residual slope measured at low current **transfers only
about 1/5 to high current.** That transfer ratio is the substance of what the
learned map does.

This gives §29.3's A5 falsifier the same conclusion from the other side — what
creates the correction is indeed the residual channels (hence 92 % from dR_fast
alone), but **the residual must not be trusted as-is** (hence learning is
needed).

### 32.4 Upper bound — if HPPC could be run in the vehicle

Running RLS on the **same cell's past HPPC measured dV** (causal, forgetting
factor swept):

| Forgetting factor | 1.0 | 0.9999 | 0.999 | 0.99 | 0.95 |
|---|---:|---:|---:|---:|---:|
| RMSE | 49.92 mV | 43.02 | 32.06 | 27.89 | **21.07** |

Better than the trim (58.76 mV). **But it is not a deployable comparison group**
— a real vehicle BMS does not run HPPC. A forgetting factor of 0.95 has an
effective memory of about 20 samples, so it is effectively "reading a value
measured a few minutes ago under similar conditions".

The use of this row is not ranking but quantifying the **value of periodic
characterisation**. **This gain, visible in voltage, disappears when moved to
current** — see §32.5.

### 32.5 In current again — the ladder spreads further

Since §29.7 showed that winning on voltage does not guarantee winning on current,
the same five versions were run through the SOP inversion and re-measured
(discharge, tau = 10 s, tolerance 0.0 A, lambda calibrated leave-one-cell-out):

| Method | Parameters | Optimism | RMSE | lambda | **Usable current** | Worst overshoot |
|---|---:|---:|---:|---:|---:|---:|
| A0 no correction | 0 | 79.8 % | 5.62 A | 0.520 | **59.3 %** | 0.00 A |
| direct plug-in | 0 | 4.3 % | 10.10 A | 0.461 | **32.6 %** | 0.00 A |
| shrinkage coefficient | 2 | 15.7 % | 4.72 A | 0.692 | **62.4 %** | 0.00 A |
| **A8** dR_fast alone | 4 | 54.4 % | 3.21 A | 0.683 | **69.1 %** | 0.00 A |
| **A3** 12 features | 26 | 61.9 % | 3.39 A | 0.679 | **70.3 %** | 0.00 A |

Read on the value that decides deployment (usable current):

- **The direct plug-in is a disaster** — 59.3 -> 32.6 %. Worse than doing
  nothing. That an RLS estimate must not be used directly as a resistance
  multiplier shows up even more strongly in current.
- **Two shrinkage coefficients buy only 3.1 %p over A0** (59.3 -> 62.4 %). One
  transfer ratio is not enough.
- **A8 buys a further 6.7 %p over the shrinkage coefficient** (62.4 -> 69.1 %).
  That is what the learned map is actually worth.
- A3 adds 1.2 %p over A8 (69.1 -> 70.3 %). Small for a difference of 26
  parameters versus 4 — the same direction as the A8 adoption in §29.7.

The gap on the current basis is larger than on the voltage basis (shrinkage
69.10 mV vs A8 62.81 mV, a 9 % difference). The order is the same but the
magnitude is not.

### 32.6 The full ladder in both directions

The charge trim format was exported too (`sop_baseline_fill.py`), and the
HPPC-RLS k was collected per cycle and moved into current.

Voltage RMSE (mean of per-cell RMSE):

| Method | Discharge | Charge |
|---|---:|---:|
| A0 | 85.36 mV | 49.90 mV |
| direct plug-in | 134.71 | 87.32 |
| shrinkage coefficient | 69.10 | 39.82 |
| HPPC-RLS (ff=1.0) | 49.89 | 33.59 |

Fitted shrinkage coefficients:

| Direction | a_f | a_s |
|---|---|---|
| discharge | 0.16–0.24 | 0.16–0.40 |
| charge | 0.26–0.32 | **0.00–0.02** |

**Charge's a_s is effectively zero** — the slow branch does not contribute to the
charge correction.

Usable current (tau = 10 s, discharge tolerance 0.0 A / charge tolerance 0.5 A):

| Method | Parameters | Discharge voltage | Discharge current | Charge voltage | Charge current |
|---|---:|---:|---:|---:|---:|
| A0 no correction | 0 | 85.36 mV | 59.3 % | 49.90 mV | 50.7 % |
| direct plug-in | 0 | 134.71 | 32.6 % (−26.8) | 87.32 | 39.6 % (−11.1) |
| shrinkage coefficient | 2 | 69.10 | 62.4 % (+3.1) | 39.82 | 57.4 % (+6.8) |
| **A8** dR_fast alone | 4 | 62.81 | 69.1 % (+9.8) | **36.73** | **59.6 % (+8.9)** |
| **A3** 12 features | 26 | **58.76** | **70.3 % (+10.9)** | 34.13 | 57.9 % (+7.2) |
| [upper bound] HPPC-RLS | 0 | 49.89 | 71.1 % (+11.8) | **33.59** | 55.2 % (+4.5) |

**Two new things surface.**

**[Updated — 34.4, 34.9]** The adoption below is corroborated: nested
selection that never sees the test cell picks A8 in 16 of 22 folds. The
**aggregation** is not — `q75` wins 10 folds against `max`'s 7, so `--trim-agg
max` is weakly determined, not established. And "4 parameters" should read
**four effective deployed coefficients**: the header ships a twelve-input
layer, 50 floats, of which four are load-bearing.

**(1) On charge, A8 (4 parameters) beats A3 (26)** — 59.6 % vs 57.9 %. On
discharge A3 leads by 1.2 %p (70.3 vs 69.1). Taking both directions together,
**A8 matches or beats A3 with one sixth the parameters.** The A8 adoption
decision of §29.7 is now supported in both directions.

The gap against the shrinkage coefficient (2 parameters) differs by direction —
discharge 6.7 %p (62.4 -> 69.1), charge 2.2 %p (57.4 -> 59.6). That matches the
a_s ~ 0 above: the charge correction is effectively close to a single scalar on
the fast branch, so two parameters already capture much of it, and learning is
worth less than on discharge.

**(2) The HPPC upper bound is not an upper bound.** On charge it is 55.2 %, below
even the shrinkage coefficient (57.4 %). That is so even though its voltage RMSE
of 33.59 mV is the best in the table. On discharge it beats A3 by only +0.8 %p.

That is, **even if a vehicle could afford periodic HPPC, deployment performance
would barely improve.** The voltage gain of §32.4 (58.76 -> 49.89 mV) disappears
when moved to current. The same phenomenon §29.7 showed on A8 appears here for
the third time — the operation that turns voltage error into current by dividing
by resistance does not preserve rank.

**These three cases make one methodological claim**: evaluating SOP by voltage
RMSE selects the wrong deployment performance. It is a direct counterexample to
the literature's practice of comparing SOP techniques by voltage-model accuracy.

### 32.7 A8 is adopted in both directions

Charge A8 was trained (`runs_trim_a8_chg`) to complete the table. Voltage
49.90 -> 36.73 mV (+26.4 %), the same margin as discharge's +26.4 %.

**The expectation was "between the shrinkage coefficient and A3", and it was
wrong — A8 overtook A3** (current 59.6 % vs 57.9 %). A3 leads on voltage (34.13
vs 36.73 mV) and A8 leads on current. The phenomenon of §32.6 appears here for
the fourth time.

Final configuration: **A8 in both directions.** Parameters 26 -> 4, EW states
12 -> 2.

### 30.12 Only the SOH dataset failed to filter defects

While building the reproduction package, `run.py` marked the `soh_data` stage as
"upstream is newer". Checking it showed a real defect.

**Temperature-channel defect exclusion happens only in the characterisation
layer** — `uypydj_ecm.py`, `uypydj_ocv.py` and `uypydj_hppc_resistance.py` use
`temp_defects`. So SOP is filtered on both the surface and the label side.

`soh_charge_dataset.py` reads the raw data directly and filtered nothing. Of the
290 SOH curves, **8 sat on defective cycles**:

    BOOST#1459, BOOST_NEGPULSE_1S#3, #1315, CC#1384, CC#1497,
    CC_CELL2#640, #753, #1878

Measured impact (cell holdout, mean over 3 seeds):

| | defects included (previous) | defects excluded |
|---|---:|---:|
| RMSE | 0.0128 | 0.0135 |
| **bias** | **+0.0010** | **+0.0001** |

**RMSE actually worsens (8 fewer samples) and the bias falls tenfold.**

This changes §29.1's account. §29.1 wrote "the adopted SOH arm's bias of +0.0010
is, of all things, on the dangerous side" and used it as grounds for a safety
argument, but **90 % of that bias came from the defective curves.** On clean data
it is +0.0001, effectively zero. The mechanism itself (§29.1's injection
experiment, where SOH bias pushes SOP optimistic) remains valid, but **the claim
that the adopted arm is skewed in that dangerous direction is retracted.**

Propagated downstream (estimated-SOH version, tau = 10 s):

| Direction | defects included | defects excluded |
|---|---:|---:|
| discharge usable current | 67.1 % | 67.2 % |
| charge usable current | 54.6 % | **56.6 %** |

Charge improves by 2.0 %p. The price against true SOH shrinks from 59.6 -> 54.6 %
(as previously written) to 59.6 -> 56.6 %.

**And there are 64 defects, not 18.** The 18 the documentation has quoted so far
(HPPC 6, OCV 2, drive 10) are the subset the characterisation layer filters. The
full set the audit found is:

| Kind | Defects | Where it is filtered |
|---|---:|---|
| HPPC | 6 | characterisation layer |
| OCV | 2 | characterisation layer |
| drive | 10 | characterisation layer |
| **halfC** | **20** | **nowhere (SOH only, fixed this time)** |
| **other** | **20** | **nowhere** |
| **schedule** | **6** | **nowhere** |
| CAP | 0 | — |

Whether the halfC / other / schedule defects flow down other paths has not yet
been examined.

**Fixed**: `soh_charge_dataset.py` now excludes all seven kinds. It was rebuilt
from the raw data and confirmed byte-identical to the masked version (282
curves).

## 33. A8 put on the board and re-measured (2026-08-26)

> **[Updated — 34.9]** The timings in this section stand — they were
> re-measured on corrected firmware and moved by less than 0.1 µs. What
> did not stand is the model: `fw_sop/Inc/sop_tables.h` was a separate
> file, not a symlink, and held A3 weights, so the firmware measured
> here compiled the superseded model. The A8 header did not compile at
> all until §34.9 fixed it.

§29.7 / §32.7 changed the adopted configuration to A8, but §27's MCU numbers were
measured with A3. The A8 path (`sop_feat_update_a8`, command 0x6D) was added to
the firmware and re-measured. NUCLEO-H563ZI, 250 MHz, n = 500, interrupts
disabled, DWT cycle counter.

### 33.1 The feature update is 55 % faster

| Stage | A3 (12 features) | A8 (dR_fast alone) | Change |
|---|---:|---:|---:|
| Trim feature update | 13.25 us (13.27 remeasured) | **5.99 us** | **−55 %** |
| p95 | 13.46 | 6.17 | |
| Max | 14.92 | 6.42 | |

What A8 deletes is eight EW updates and their auxiliary state. What remains is
the nominal propagation (v2n, h), two table lookups (`nominal_rf_rs`,
`ocv_lookup`), and two EWs (e_ir, e_ii). **What remains is the expensive part, so
it does not fall below half** — of the 5.99 us, the two table lookups take around
3 us (REFF alone is 3.11 us).

EW states drop from 12 to 2. The struct shrinks from 15 floats to 5, so the NVM
needed to survive a key cycle goes from 64 B to 24 B.

### 33.2 Cycle budget

| What | Period | A3 | A8 |
|---|---|---:|---:|
| SOC EKF | per sample | 7.5 us | 7.1 us |
| Trim feature update | per sample | 12.4 us | **6.0 us** |
| SOP (discharge/charge x tau 2/10 s = 4 calls) | per decision | 197.1 us | 201.7 us |
| **Total** | | **217.0 us** | **214.8 us** |

**The period total falls by only 1 %.** It is because inversion accounts for
94 %. Halving the feature update is invisible in the total.

**But looking only at what runs every sample it is different** — SOC EKF +
feature update falls 34 %, from 19.9 us to 13.1 us. SOP inversion runs only on a
decision while the feature update runs on every sample, so in a system running at
100 Hz this is the real load: 1.99 ms/s down to 1.31 ms/s.

Within the same build the table lookup moved 2.91 -> 3.11 us (the binary-layout
dependence recorded in §27.3). **A8 and A3 were measured side by side in the same
binary, so the comparison above is unaffected by it.**

### 33.3 Two things fixed in the reproduction package

**(1) There were two copies of `sop_core.c/h`.** `mcu/` and `fw_sop/Inc|Src/`
were independent files. The file header says "host and MCU use the same code",
and the interpretation of the MCU measurements hangs on that, yet **nothing
enforced it.** Checking showed they had happened to match so far. They were
changed to symlinks to make it structural (`soh_core` and `soh_simd` too).

**(2) A new command was silently ignored.** `CEMA_Protocol_Loop`'s dispatch
filters commands through an explicit list, and 0x6D was missing. The firmware
dropped the command and the host **read the previous response**, plausibly
returning 0 cycles — it came back as a wrong value, not an error. Fixed by adding
it to the list.

The second is a protocol design problem. Without returning an error on an unknown
command, every new measurement added falls into the same trap.

### 33.4 Drive defects — verified but application deferred

While §30.12 fixed the missing defect filtering in the SOH dataset, the same
question was put to the SOP path. **The 10 drive defects are filtered nowhere** —
defect exclusion exists only in the characterisation layer (`uypydj_ecm` /
`_ocv` / `_hppc_resistance`), and `build_uypydj_cache` and `sop_trim_dataset` do
not use it.

**How far it reaches**

| Path | Reach |
|---|---|
| SOC benchmark, 36 runs | **0** — the chosen files contain no defects |
| Trim dataset | **18** of 7,098 labels (one cell, BOOST_NEGPULSE) |

The trim side was actually rebuilt and checked (BOOST_NEGPULSE, the two
defective drive files excluded):

- The paired row count is 10,296, **unchanged.** Those 18 labels attach to an
  earlier drive run instead.
- **216 rows (2.1 %) have changed features.** dR_fast moves by a median of
  1.70 mOhm and a maximum of 7.58 mOhm — about 10 % of this cell's nominal
  R_fast (roughly 16 mOhm), a large change for those rows.
- Over the whole dataset (63,108 rows) it is **0.34 %.**

**It was applied and the canonical version was kept.** Without touching the
canonical cache, a defect-excluded version was assembled in separate directories
(`cache/trim_v3`, `cache/trim_chg_v3`), A8 was retrained in both directions, and
the evaluation was run.

Voltage (mean over cell holdouts):

| Direction | canonical (defects included) | defects excluded (v3) |
|---|---:|---:|
| discharge | 62.81 mV | 62.78 mV |
| charge | 36.73 mV | 36.70 mV |

Deployment metrics (lambda calibrated leave-one-cell-out):

| Direction | tau | lambda canon/v3 | Optimism canon/v3 | Usable current canon/v3 |
|---|---:|---|---|---|
| discharge | 10 s | 0.683 / **0.683** | 54.4 / **54.4 %** | 69.1 / **69.1 %** |
| discharge | 2 s | 0.470 / **0.470** | 92.1 / **92.1 %** | 55.8 / **55.8 %** |
| charge | 10 s | 0.586 / **0.586** | 57.1 / **57.1 %** | 59.6 / **59.6 %** |
| charge | 2 s | 0.560 / **0.560** | 59.9 / **59.9 %** | 57.5 / **57.5 %** |

**All eight values agree to the decimal.**

The predictions themselves change on 83.8 % of the discharge evaluation rows, but
**the median change is 6 mA** — 0.02 % of a 30 A |I*|. Every cell changing is due
to the cell-holdout structure: BOOST_NEGPULSE's corrected data enters the
**training set** of the other five cells' models.

The reason it does not reach was also confirmed — BOOST_NEGPULSE has 43
trustworthy labels at tau=10 s and **zero at tau=2 s.** The 26 contaminated
labels barely touch the evaluation.

**Conclusion: keep the canonical version.** Excluding the defects is the correct
treatment but it does not change the deployment numbers, so there is no reason to
disturb the current cache that 38 verified numbers stand on.
`sop_trim_dataset.py` contains the exclusion code, so **rebuilding the dataset
next time picks it up automatically.** The v3 version is left in `cache/trim_v3`
and `cache/trim_chg_v3` (five cells are symlinks to the canonical version; only
BOOST_NEGPULSE is a real file).

**One thing fixed**: `sop_trim_dataset.py` was discarding `feat_cycle` (which
drive run a feature came from) instead of keeping it in the metadata. That forced
this comparison to go the long way round by reproducing the pairing. It is now
stored as `m_feat_cycle` — the next time the same question comes up it can be
answered without a rebuild.

**Deferred-state record**: `analysis/cache/trim_backup/` held two versions
(`_predrivefix` = the same as the then-canonical version, `_drivefixed` = the
defect-excluded version). It was gitignored and **no longer exists** — see the
resolution below. `sop_trim_dataset.py` already has the exclusion code, so
rebuilding produces the excluded version automatically.

> **[Resolved — 37.1]** The deferral ended: the defect-excluded set is the
> canonical one, and `analysis/cache/trim_backup/` no longer exists — it was
> never in git, and rebuilding the cache now produces the adopted version
> directly. Eight verified numbers moved when it was adopted; the rebuild
> itself came back byte-identical, which is how we learned the cache had
> never been the stale link.

### 33.5 What the 64 defects really are — a logging fault, not a cold test

After §30.12 confirmed there are 64 defects rather than 18, the question was
whether the other 46 reach the SOP path. In the process **a more fundamental
question** arose: is what the audit ruled a "defect" really a fault, or an
intentional temperature test?

The audit's rule is simple — the fraction of samples whose recorded temperature
is outside 15–45 C. That rule cannot distinguish three things.

Splitting the 64 by the median of the recorded temperature:

| Ruling | Count | Example |
|---|---:|---|
| physically impossible (T_med < −40 C) | 9 | BOOST_NEGPULSE_1S#2 (−195 C), #3 (−200 C) |
| **near 0 C** | 28 | BOOST#1458–1463, CC#1495–1501 |
| 26–75 C | 6 | BOOST_NEGPULSE#487 (26 C, only part of the region outside) |
| no T_med value | 21 | |

**The 28 near 0 C were the problem.** Across the whole dataset (3,713 files with
a T_med), only 28 are near 0 C, and they form **complete blocks**:

| Cell | Cycles | File kinds |
|---|---|---|
| BOOST | 1458–1463 | HPPC, drive, halfC, other, schedule (8 files) |
| CC | 1495–1501 | HPPC, **OCV**, drive, halfC, other, schedule (9 files) |

Every kind of file from one characterisation round points at 0 C together. **It
looks like an intentional low-temperature test.** If so, valid data has been
discarded as "defective", and §26's premise that "UYPYDJ is 25 C only" would be
wrong too.

**Physics settled it.** At 0 C a lithium-ion cell's resistance is 3–5× the 25 C
value. The exclusion filter was turned off, those HPPCs' resistances were
extracted, and they were compared with neighbouring cycles (discharge,
tau = 10 s, per-cycle medians):

| Cell | 1425 | **cycle in question** | 1537 | Ratio |
|---|---:|---:|---:|---:|
| BOOST | 33.68 | **1462: 34.88** | 39.69 | **0.98** |
| CC | 32.86 | **1500: 37.06** | 40.10 | 1.00 |

**They sit exactly on the ageing trend line.** Not 3–5× but 0.98–1.00×. The cell
was not at 0 C — **the temperature channel merely recorded 0 C.**

Forming blocks is explained too: if one temperature sensor fails during a round,
every file in that round records 0 C together.

**Conclusion**
- The audit's ruling is correct. The exclusion stands.
- §26's "UYPYDJ is 25 C only" stands as well.
- However, **it would be better to change the audit's basis from temperature
  range alone to "temperature range + does the resistance depart from the trend
  line".** The present rule would have discarded a genuine low-temperature test
  as defective too. There was none in this dataset, but applying the same code to
  a dataset with a temperature axis, such as RPCWBY, would discard everything.

**Why this section stays**: "a 0 C block was found" was a plausible finding and
came close to being written into the documentation as such. One physics check
stopped it. The point is that a temperature record must not be judged on its own.

### 33.6 Deployment build — keeping A8 only

§33.1 put A3 and A8 in the **same binary** and measured them side by side. A real
deployment build keeps A8 only. `make EXTRA_CFLAGS=-DSOP_A8_ONLY` removes the
12-feature path.

| | comparison (both) | deployment (A8 only) | Difference |
|---|---:|---:|---:|
| Flash (text) | 143,964 B | 142,060 B | **−1,904 B** |
| RAM (bss) | 28,276 B | 28,276 B | 0 |

[Updated 2026-08-31 — 35.4] The comparison build was 143,932 B when this was
written, giving −1,872 B. Rebuilding it against the all-cell deployment header
moved it to 143,964 B; the deployment build is unchanged at 142,060 B. Both
rows are now in `build_size.csv` and checked by `verify.py`, which they were
not before.

What was removed was confirmed by ELF symbols — `sop_feat_update` is absent from
the deployment build (only `sop_feat_update_a8` remains).

**The saving is small, 1.3 %.** The 143 KB is dominated by the SOH tables (about
100 KB) and the ECM grid (32 KB), so 1.9 KB of code is buried. RAM does not fall
because `sop_feat_t` is shared by both paths.

**Where it matters is NVM.** Of the struct's 16 floats, A8 uses 6 (v1n, v2n, h,
e_ir, e_ii, age_s). The amount that must be stored to survive a key cycle falls
from **64 B to 24 B**. What constrains a real vehicle is not RAM but NVM writes,
so this is the deployment argument.

**Note — FEAT_A8 is actually slower in the deployment build**: 5.99 us
(comparison) vs 6.96 us (deployment), +16 %. This is the binary-layout dependence
recorded in §27.3 (the same cause as the table lookup moving 2.41 -> 2.91 us:
ICACHE and flash placement). **The A3 vs A8 comparison of §33.1 was done within
one binary, so it is unaffected.** Any absolute value must be quoted with the
build it came from.

### 33.7 Two things fixed in the build

**(1) The Makefile had no external-definition hook.** It used `CFLAGS :=`, so
passing `make CFLAGS=...` on the command line wipes out the toolchain flags
entirely and breaks the build. `CFLAGS += $(EXTRA_CFLAGS)` was added.

**(2) The method for checking a removed command was wrong.** To see whether
FEAT (0x65) was absent from the deployment build, that command was sent and 32 B
came back. It looked as though it had not been removed, but no — when the
firmware ignores a command byte, **the following 73 B body is reinterpreted as
commands** and a valid byte among them produces a response. Exactly the trap
§33.3 pointed out, again. Checking via ELF symbols is the correct method.

With that, §33.3's point is confirmed twice: **a protocol that does not return an
error on an unknown command silently ruins measurements.** It is left as a
firmware fix to make.

---

## 34. External audit — what did not survive it (2026-08-27)

The repository was cloned fresh and audited against a submission checklist:
strict held-out calibration, nested selection, fair baselines, external
validation, an end-to-end chain, and deployment evidence. Sections 34.1–34.9
record what the audit measured. Six published claims did not survive, and
they are marked at their original sites rather than deleted — the reasoning
that produced them is still worth reading, as in §26.5.

The audit's own artifacts are in `.paper_state/` (`paper_map.yaml` maps every
claim to its status, `evidence_ledger.yaml` carries the measurements) and
`manifests/`.

### 34.1 "Zero exceedance" was an artefact of pooling the safety factor

`run_safety.py` fits six leave-one-cell-out lambdas and applies their
**median** to every cell. Cell *i* is therefore scored under a lambda that
five of the six contributing folds were fitted on data containing. That is
not a held-out calibration, and the exceedance count it produces is
optimistic by construction.

Giving each held-out cell its own lambda, fitted with that cell removed
entirely (`repro/run_safety_strict.py`):

| direction | tau | lambda range | exceed | rows | 95 % upper | usable | worst cell |
|---|---:|---|---:|---:|---:|---:|---|
| discharge | 10 s | 0.683–0.708 | **1** | 491 | 0.96 % | 69.6 % | 59.9 % BOOST_REST |
| discharge | 2 s | 0.470–0.502 | **2** | 140 | 4.43 % | 58.3 % | 54.3 % BOOST |
| charge | 10 s | 0.586–0.591 | **1** | 2461 | 0.19 % | 59.5 % | 53.5 % BOOST_REST |
| charge | 2 s | 0.560–0.569 | **2** | 2082 | 0.30 % | 57.5 % | 49.7 % BOOST_REST |

The exceedances come from exactly the cells whose lambda_i is larger than the
others: removing a cell loosens what constrains lambda, and the looser lambda
then breaks on the cell it was not allowed to see. Pooling to the median hid
this by pulling those lambdas back down.

The magnitudes stay small — worst 0.97 A against a median |I\*| near 30 A —
and the usable current barely moves (69.8 → 69.6 % on discharge 10 s). What
changes is the claim, not the performance.

**Zero observed exceedance is a measurement. It is not zero risk.** With no
events in *n* rows the 95 % upper bound on the true rate is about 3/*n*, and
that bound is the number to quote.

### 34.2 The SOC headline averaged over a set it did not name

§30.9 says the number to use is 2.05 %p and describes it as the average over
initial-SOC error, current offset and gain error. **2.05 is the mean over all
seven rows of `soc_perturb.csv`, and the seventh is the undisturbed case.**
The mean over the six disturbance rows is 2.14 %p — which §30.4's own table
already prints.

| quantity | value |
|---|---:|
| undisturbed alone | 1.51 %p |
| mean over the six disturbances | **2.14 %p** |
| mean over all seven rows | 2.05 %p |
| worst single condition (offset −0.10 A) | 3.77 %p |

Nothing pinned 2.05 to a table, so `verify.py` never saw the discrepancy.
`repro/run_soc_headline.py` now writes all four aggregations side by side.
Quote one and label it.

### 34.3 Ridge beats the SOH CNN

The CNN was reported with no baseline beside it. Under identical nested
cell-held-out splits, identical 64-bin features and identical target
(`repro/run_soh_baselines.py`):

| method | pooled RMSE | worst cell |
|---|---:|---:|
| **ridge** | **0.0094** | **0.0130** |
| PLS | 0.0096 | 0.0126 |
| SVR (RBF) | 0.0099 | 0.0146 |
| gradient boosting | 0.0101 | 0.0179 |
| **1D CNN (shipped at the time, 10,945 parameters)** | **0.0135** | **0.0293** |

[Superseded — 36] The CNN is a comparison group now, not the arm: ridge
was adopted, and nested selection places the CNN last on all six folds.
| mean baseline | 0.0878 | 0.0988 |

Ridge is 30 % better pooled and 2.3× better on the worst cell, with 65
coefficients against 10,945 parameters — and it removes the 17.9 ms SOH
inference from the board entirely. Confirmed with a fixed alpha and no inner
selection (0.0093–0.0095), so it is not a selection artefact.

Two related corrections. `soh_cnn.py` is a plain two-layer 1D CNN; there is
no physics or residual term in it, and it must not be called physics-aware.
And the representation is not load-bearing: dQ/dV, time-per-voltage-bin, and
both concatenated give 0.0094 / 0.0096 / 0.0095. What *is* load-bearing is
the full 3.55–4.05 V window — the low-voltage 48 of 64 bins give 0.0132 and
the high-voltage 48 give 0.0169, against 0.0094 for the whole
(`repro/run_soh_ablations.py`).

### 34.4 A8 survives nested selection; the aggregation choice does not

§29.7 adopts A8 over A3 by comparing usable current on the leave-one-cell-out
evaluation — the rows the paper then reports. `--trim-agg max` was picked from
five options the same way. The evaluation was both the selection set and the
reported test set.

Redone properly (`repro/run_nested_selection.py`): outer leave-one-cell-out,
inner leave-one-out again over the five training cells using models trained on
the remaining four — 120 leave-two-out fits. The grid is scored on the inner
splits only and the winner applied to the untouched outer cell.

| choice | what the inner splits picked |
|---|---|
| rung | **A8 in 16 of 22 folds**, A3 in 6 |
| aggregation | q75 in 10, **max in 7**, median 3, last 1, q90 1 |

**A8's adoption is corroborated** by selection that never saw the test cell.
**The aggregation is not**: q75 wins more often than the shipped `max`. Present
the aggregation as a weakly determined choice, not an established one.

Scored on the untouched outer cell the nested protocol gives 68.42 % discharge
and 59.83 % charge at tau = 10 s, against the published 69.61 and 59.49 — and
in two of the four settings the nested protocol scores *higher*. Selecting on
the evaluation was worth at most about 1.2 %p. That is a good result, and it
could only be stated after running the nested protocol.

**Tolerance cannot be selected, and that is a finding.** The objective is
defined relative to it: raising the tolerance loosens both the lambda fit and
what counts as an exceedance, so usable current rises about 1.6 %p per 0.25 A
with the exceedance count unmoved. A search returns the largest value offered
every time. Tolerance is a design constraint — how much overshoot the pack is
allowed — and stays declared.

### 34.5 The SOP targets are pulse-derived, not measured

A 30 A cycler cannot reach the discharge current this cell can take, so I\* is
a projection of a fit through four HPPC rates down to the voltage floor.
Stratifying by `extrap = |I*| / max|I_measured|` (`repro/run_label_quality.py`):

| direction | ≤ 1 interpolated | 1–1.5 | > 1.5 | median extrap |
|---|---:|---:|---:|---:|
| discharge | 8.0 % | 14.1 % | **78.0 %** | 2.67 |
| charge | 34.1 % | 35.1 % | 30.8 % | 1.15 |

Calling these "directly measured SOP labels" overstates them. **Pulse-derived
current-limit reference** is what they are.

The conclusion survives the labels, which is the point of checking: A8 beats
A0 at every extrapolation ceiling in both directions, including `extrap ≤ 1`
where every label is interpolated (discharge 10 s 74.7 vs 70.6 %; charge 10 s
62.5 vs 54.4 %).

### 34.6 Larger models do not buy anything the trim does not already have

A reviewer will ask whether a sequence model, given the same causal window,
does better. On the same twelve 600 s drive blocks, the same output head, the
same loss, optimiser, schedule and seeds, and the same inversion
(`repro/run_sop_seq_baselines.py`):

| | voltage RMSE, disch / chg | usable current, strict lambda, tau = 10 s |
|---|---|---|
| GRU (4,482 par.) | **44.8 / 25.8 mV** | 69.5 [67.1–71.2] / 59.9 [58.0–61.6] % |
| LSTM (5,954 par.) | 45.4 / 26.6 mV | 69.1 [67.0–70.5] / 60.0 [57.7–62.1] % |
| A3 (26 coeff.) | 58.8 / 34.1 mV | 70.0 [65.5–72.7] / 58.0 [54.6–61.0] % |
| **A8 (4 eff. coeff.)** | 62.8 / 36.7 mV | 69.6 [65.5–72.1] / 59.5 [56.3–62.4] % |
| FFRLS adaptive ECM | 106.1 / 65.0 mV | 43.3 / 49.1 % |

The sequence models are 28–30 % better on voltage and **statistically
indistinguishable on usable current** — all four intervals overlap in both
directions — at 1,100–1,500× the parameters. This is §32.6's
voltage-does-not-predict-current claim, now shown against models three orders
of magnitude larger rather than only against simpler ones.

A forgetting-factor RLS adaptive ECM plugged straight into the resistance is
worse than no correction at all (43.3 vs 59.3 % on discharge), reproducing the
direct-plug-in result of §32.5.

### 34.7 External validation of the frozen A8

RPCWBY Test#2 is the only sheet in either archive carrying a drive cycle
(US06) and SOP measurements on the **same cell**, which is what a trim indexed
on preceding drive history needs. 375 rows, 10 and 25 °C, SOH 0.98 → 0.80, all
six frozen folds, nothing refitted, scored through the constant-power search of
Chen et al. 2026 under Test#2's own limits (`repro/run_external_a8.py`):

| direction | A0 RMSE | A8 RMSE | A0 bias | A8 bias | A0 optimism | A8 optimism |
|---|---:|---:|---:|---:|---:|---:|
| charge | 8.36 W | **7.21 W** | +3.44 | +2.6 | 43.1 % | 44.0 % |
| discharge | **2.25 W** | 3.65 W | +0.05 | **−2.0** | 51.2 % | **20.7 %** |

The trim transfers, and the two directions transfer differently. On charge it
improves accuracy and bias. On discharge it costs 1.4 W of RMSE and buys a
sign flip — the model turns conservative and over-prediction falls from 51 %
of rows to 21 %. Worse on RMSE, better on the criterion this project adopts
by. **Report both.** The fold spread is tight (3.44–3.85 discharge).

Limits: 38 % of model calls fall outside the pooled hull; Test#2 is 10 and
25 °C only; one external cell.

Separately, and **not to be merged with the above**, the physics-only model
was scored on RPCWBY Test#3 across six temperatures under the same search
(`repro/run_chen2026_baseline.py`): 1.7–4.2 W RMSE at 0–40 °C but 17.5 W at
−10 °C and 36.2 W at −20 °C, with a **+29 W optimistic bias against a 30.8 W
measured mean** — it claims nearly double the power the cell can deliver. The
temperature factor is borrowed from the Mendeley sweep on a different, un-aged
cell, and no aged low-temperature data exists anywhere in this project.
**Do not claim aged low-temperature generalization.**

### 34.8 The causal chain, and SOC is the term that matters

Every SOP number above feeds the inversion the label's own SOC. A deployed BMS
has an estimate. All four corners, each under strict per-cell held-out lambda,
with SOC entering as the adopted EKF's terminal error on the drive run nearest
before each characterisation (`repro/run_end_to_end.py`):

| discharge, tau = 10 s | exceed / n | lambda | usable |
|---|---:|---:|---:|
| oracle SOH + oracle SOC | 1 / 491 | 0.683 | 69.61 % |
| estimated SOH + oracle SOC | 1 / 514 | 0.659 | 68.45 % |
| **oracle SOH + estimated SOC** | **20 / 455** | 0.589 | 66.91 % |
| **estimated SOH + estimated SOC** | **26 / 488** | 0.573 | 65.54 % |

On charge, oracle SOH with estimated SOC collapses lambda from 0.586 to 0.404
and usable current from 59.5 to 41.4 %.

Estimated SOH costs about 1 %p. **Estimated SOC takes discharge exceedance
from 1 in 491 to 20 in 455**, and those exceedances are *after* lambda is
refitted per held-out cell on the same estimated-SOC data. It is not a bias
the safety factor can price out.

> **[Corrected — 35.1]** The four rows above are not scored on the same
> pulses; *n* moves because the label-trustworthiness filter is computed from
> whichever SOC is in force. On the 385 pulses all four corners keep, the
> fully estimated corner shows **4 exceedances against 3** for the oracle
> corner, not 26 against 1. The direction survives and the magnitude does
> not. What replaces it is a sharper mechanism: the rows only the estimated
> corner admits carry a 24.3 % exceedance rate. See §35.1 and §35.2.

The both-estimated corner is the only one a vehicle can execute. Every row
above it still receives a ground truth the vehicle does not have.

One row is not monotone: on charge the both-estimated corner (56.3 %) beats
oracle-SOH-with-estimated-SOC (41.4 %). Errors displacing the operating point
in opposing directions is a plausible reading and was not tested. Nothing
should be built on that row.

Not modelled, and therefore not claimed: sensor dropout, fallback logic,
temperature-sensor error, and drift between the pulse and the end of the
preceding drive.

### 34.9 The board was not running the adopted model

Three defects, all in the export path, all found by trying to rebuild it.

`mcu/fw_sop/Inc/sop_tables.h` was an independent **file**, not a symlink to
`mcu/sop_tables.h`. It held A3 weights (`trim_w_dis[0] = 0.2283459`) while the
exporter wrote A8 (`0.2146806`). **The firmware behind §27 and §33's timings
compiled the superseded model.**

The A8 header did not compile at all. It emitted 27 literals of the form `0f`
— an integer constant with a float suffix — because `"%.7g"` renders an exact
zero as `0`, and A8 has exact zeros in its eleven masked feature columns. Both
gcc and arm-none-eabi-gcc reject it. **The adopted configuration's header had
never been built.**

And `repro/stages.py` invoked `export_mcu_tables.py --rung A8`, an argument the
exporter did not have; its own defaults pointed at `runs_trim_v2`, which is A3.

Fixed, rebuilt, flashed and re-measured. The flashed image now contains the
deployment weights and not the A3 ones, verified by searching the objcopy'd
binary for the IEEE-754 pattern. Timing is unchanged — full cycle 214.8 µs
median, 307.1 µs worst case — which the array shapes predict, so §27 and §33's
numbers stand. `repro/run_parity.py` agrees the C with Python to 9.2 × 10⁻⁶
over 5,000 random states, automatically, which nothing previously did.

The shipped model is now fitted on **all six cells** by the recipe the LOCO
folds used, against an all-cell ECM pool that did not previously exist —
`export_mcu_tables.py --holdout` had been exporting one arbitrary
leave-one-out fold as the deployment model. That fit carries no held-out score
by construction; every validated number remains the LOCO folds.

**On the parameter count.** A8 has two non-negligible weights and two biases —
four effective coefficients, and the masked columns are at most 5.6 × 10⁻⁴¹.
But the header ships a twelve-input layer: 24 weights, 2 biases, 12 mu, 12 sd,
50 floats. Write **"four effective deployed coefficients"**, not "four
parameters".

### 34.10 What the audit could not do

- **Frequency-dependent fractional-order SOP** (Lai et al. 2024) needs EIS.
  Neither UYPYDJ nor RPCWBY contains any; there is nothing to identify the
  model from. A time-domain substitute would not be their method.
- **Pack and HIL.** `sop_pack2.py` resamples single-cell evaluation rows within
  condition groups. There is no pack hardware. Call it **pack-level simulation
  sensitivity**, never pack validation, and state what it omits: inter-cell
  error correlation beyond the condition bin, a shared current trajectory,
  thermal gradient, and imbalance.
- **UYPYDJ's licence** is stated nowhere in the dataset readme. Confirm with
  the depositor before redistributing anything derived from it.

  > **[Corrected — 37.8]** It is stated, in section 2 METADATA of that
  > readme: *"Licenses/restrictions: CC BY 4.0"*, alongside the depositors'
  > certification that the data is free of licensing and intellectual
  > property issues. The file's SHA-256 matches `raw_data.yaml`, so the text
  > is in the file as downloaded. All three datasets are CC BY 4.0. This
  > entry stood for four days because nobody read line 47.

---

## 35. Second audit round — what the first round got wrong

The first audit (§34) was reviewed and returned NO-GO. Six of its findings
were confirmed, two were wrong in a way that mattered, and two defects it
introduced went undetected until the review: continuous integration was red
on every commit, and the MCU evidence contradicted itself. This section
records what changed. Nothing in §34 was deleted; the claims that fell are
marked in place, as §26.5 requires.

### 35.1 The four end-to-end corners were not scored on the same rows

§34.8 compared 1/491, 1/514, 20/455 and 26/488. The denominators move because
the filters that decide which pulses are evaluated — the trustworthy-label
test (`extrap ≤ 1.5`) and the surface hull — are computed against **whichever
SOC is in force**. Shifting SOC therefore moves rows into and out of the
evaluated set, and the four corners were compared across four different sets.

`repro/run_end_to_end.py` now also scores every corner on the intersection.
The row key is cell, cycle, horizon, pre-pulse voltage and measured current —
SOC-free by construction, so it names the same physical pulse under every
corner. The script asserts the key is unique per corner and that the four
intersected sets have identical *n*; a paired comparison that silently
compares different counts is the defect being fixed, so it fails loudly.

**discharge, τ = 10 s, n = 385 in every row**

| | exceed | 95 % upper | worst overshoot | usable |
|---|---:|---:|---:|---:|
| oracle SOH + oracle SOC | 3 | 2.00 % | 1.043 A | 73.05 % |
| estimated SOH + oracle SOC | 1 | 1.23 % | 0.157 A | 73.82 % |
| oracle SOH + estimated SOC | 1 | 1.23 % | 0.918 A | 69.31 % |
| estimated SOH + estimated SOC | 4 | 2.36 % | 1.171 A | 69.15 % |

**charge, τ = 10 s, n = 2 421 in every row**

| | exceed | 95 % upper | worst overshoot | usable |
|---|---:|---:|---:|---:|
| oracle SOH + oracle SOC | 1 | 0.20 % | 0.518 A | 59.49 % |
| estimated SOH + oracle SOC | 3 | 0.32 % | 1.442 A | 57.62 % |
| oracle SOH + estimated SOC | 1 | 0.20 % | 1.537 A | 41.37 % |
| estimated SOH + estimated SOC | 6 | 0.49 % | 3.135 A | 56.27 % |

On identical pulses the fully estimated corner shows **4 exceedances against
3** on discharge and **6 against 1** on charge, and costs about **4 %p of
usable current**. It is not the 26-against-1 collapse §34.8 reported. That
figure was an artifact of the comparison, and it is withdrawn.
Artifact: `analysis/results/tables/end_to_end_paired.csv`.

### 35.2 What replaces it: a wrong SOC moves which rows are scored

> **[Narrowed — 37.2]** This section was headed "a wrong SOC corrupts the
> filter, not the prediction", which reads as a measured failure rate of an
> onboard admission filter. No such filter was implemented or tested. What
> was measured is the *offline evaluation inclusion rule*. The numbers below
> stand; the interpretation is narrowed in §37.2.

The paired view answers only half the question, because in a vehicle there is
no oracle SOC to compute the label-trustworthiness test with. So the rows each
corner keeps *outside* the intersection have to be scored too, not averaged
away (`end_to_end_drift.csv`):

| discharge, τ = 10 s | kept | outside | scored | exceed | rate |
|---|---:|---:|---:|---:|---:|
| oracle SOH + oracle SOC | 491 | 106 | 106 | 1 | 0.94 % |
| estimated SOH + oracle SOC | 514 | 129 | 129 | 1 | 0.78 % |
| oracle SOH + estimated SOC | 455 | 70 | 20 | 0 | 0.00 % |
| **estimated SOH + estimated SOC** | 488 | 103 | **103** | **25** | **24.27 %** |

Twenty-five of the fully estimated corner's twenty-six exceedances are in the
103 rows the oracle corner never evaluates — a **24.3 % exceedance rate**
against 0.94 % for the oracle corner's own extra rows. On charge the effect is
small (2 421 of 2 461 rows are common) and the discharge case is where it
lives.

So the mechanism is not that a wrong SOC makes the prediction worse on a given
pulse; it is that **a wrong SOC corrupts the test that decides whether the
label can be trusted at all**, and admits pulses that should have been
rejected. The safety consequence in §34.8 survives; its explanation does not.
This is worse than a per-pulse bias, not better: a bias can be priced into λ,
and a corrupted admission rule cannot.

### 35.3 λ fitted to bound the worst row is fragile

In the paired charge table the oracle-SOH + estimated-SOC corner holds
exceedance at 1 — but only by dropping λ from 0.5860 to **0.4035**, costing
18 %p of usable current, while the *fully* estimated corner recovers to
0.5259. Utility is non-monotonic in how much state error is present.

The cause is structural: λ is fitted to bound the worst single training row,
so one row moves the factor and therefore the whole utility number. §34
reported this row as an oddity. It is not an oddity; it is a property of
max-based calibration, and any deployment using this λ inherits it. A
quantile-based or distributionally-robust λ would not have it, and was not
tested.

### 35.4 The board, the header and the manifest now agree

`.paper_state/evidence_ledger.yaml` said the flashed image held the all-cell
weight 0.2105654. `manifests/mcu_evidence.yaml` said it held the
leave-one-cell-out CC fold 0.2146806, and recorded a header hash that no file
in the tree matched. The ledger was right; the manifest had not been updated
when the header was replaced. Two documents disagreeing about what is on the
board is worse than either being wrong, because nothing in the repository
objected.

What was done, in order, all of it on hardware:

1. `repro/stages.py` `mcu_export` was still exporting with the **default
   leave-one-cell-out directories**. Re-running the graph would have silently
   replaced the shipped header with a fold. It now passes `--deployment` and
   the all-cell run directories.
2. That corrected command regenerates the committed header **byte for byte**
   (SHA-256 `2a4378de…`).
3. The firmware was rebuilt from current sources and the image searched for
   IEEE-754 patterns: `0.2105654` (all-cell) occurs once; `0.2146806` (CC
   fold) and `0.2283459` (A3) do not occur.
4. The board was re-flashed and re-benchmarked at n = 500. FULL median
   **50.52 µs**, p95 56.42, max 73.32 — identical to the stored `mcu.csv`,
   which was therefore already measured on the deployment model.
5. `manifests/mcu_evidence.yaml` was rewritten to the measured values, and two
   tests now fail if it drifts again: one checks the recorded hash against the
   header on disk, one checks that the manifest and the ledger name the same
   flashed weight.

Comparison build text moved 143 932 → **143 964 B** (the all-cell constant
pool); the deployment build is unchanged at 142 060 B. Both rows are now in
`build_size.csv` and checked by `verify.py`, which they were not before.
Stack high-water is 560 B, not the 624 B recorded in August — the
unknown-command NACK path added during the first audit changed the worst-case
call chain. No published table quotes it.

### 35.5 External validation: where the hull reaches, and where λ fails

§34.7 reported RMSE for the frozen A8 folds on RPCWBY Test#2 and called it a
transfer result. Two things were missing: which operating points the pooled
hull actually covers, and whether the *safety factor* transfers, which is the
only property the paper claims.

**Coverage** (`external_a8_coverage.csv`). The hull is a band, not the
envelope:

| SOC | ≤ 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40–0.95 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| in hull | 0 % | 7.7 % | 26.9 % | 42.3 % | 88.5 % | 100 % | 0 % |

Overall 54.2 % of discharge calls and 70.3 % of charge calls are in hull. The
model declines to answer at both ends of the SOC range — including the low-SOC
region where discharge power limits actually bind. **Nothing about the transfer
of this model below SOC 0.30 has been shown.**

**Safety** (`external_a8_safety.csv`). λ is a ratio, so it carries from
current to power without a unit argument. `lambda_needed` is the largest
factor that would leave zero exceedance on the external data; `margin` is
`needed / frozen`.

| | frozen λ | needed | margin | exceed | worst overshoot |
|---|---:|---:|---:|---:|---:|
| discharge (6 folds) | 0.6832 | 0.886–0.975 | **1.30–1.43** | **0** | 0 W |
| charge (6 folds) | 0.5860 | 0.3969 | **0.677** | 9–11 | 2.1–3.0 W |

On discharge the frozen factor is genuinely conservative on a dataset it was
never fitted to, with 30–43 % of margin left over and zero exceedance in all
six folds. **On charge it is not conservative enough**: it would need to fall
to 0.397, and as shipped it overshoots by up to 3.0 W on 9–11 of 248 in-hull
rows in every fold.

The claim that survives is therefore narrow, and this is how it should be
stated: *six frozen UYPYDJ leave-one-cell-out A8 models were transferred
without refitting to the in-hull portion of RPCWBY Test#2; on **discharge**
the frozen safety factor remained conservative with 1.30–1.43× margin and zero
exceedance, over the 54 % of calls the pooled hull covers, which excludes
SOC below 0.30 and SOC = 1.0. On **charge** the frozen factor did not
transfer.* Anything broader than that sentence is not supported.

### 35.6 Continuous integration was red on every commit

The `check` job had been failing since the workflow was added, at the unit and
integration test step, and this was not noticed before pushing. Three
failures, none of them flaky, all of them the same mistake — tests that
assumed the author's machine:

* Two MCU tests asserted on guard messages from `export_mcu_tables.py`, which
  died at `import torch` before reaching argparse. The guards decide things
  knowable from file names alone, so torch and the model classes are now
  imported lazily and the `--rung` check is hoisted out of the export path
  into argument validation. Both guards now fire, and are tested, on a machine
  with no torch and no raw data.
* `test_stage_graph_inputs_resolve` treated a missing `raw/` as failure. A
  clean clone has no `raw/` — the datasets are third-party downloads. External
  inputs now resolve against `manifests/raw_data.yaml`, and a second test
  requires every input no stage produces to be either a declared raw root with
  file hashes or committed to the repository. That test immediately found
  `results/cold_check`, which is the committed case and is now documented as
  such.

The full CI sequence — environment smoke test, compile, 53-value verification,
both table producers, 40 tests, C/Python parity, three figures — was then run
locally against a Python with `torch` masked, which is what the runner has.
All steps pass.

### 35.7 The SOH arm

Ridge regression on the shipped input, with its penalty chosen by grouped
inner selection on the five training cells only, beats the 1D CNN that was
deployed at the time, on the same leave-one-cell-out splits: pooled RMSE **0.0094 against 0.0135**, and
worst cell **0.0130 against 0.0293** — a factor of 2.25 on the cell that
matters. [Updated — 37.5: the CNN was retrained through a deterministic pool
and its worst cell is now 0.0291, a factor of 2.24. The earlier figure came
from a fit that could not be reproduced at all; see §37.5.] The CNN's seeds are fixed, not tuned, so the comparison does not
favour ridge by search budget.

The CNN is therefore **not** presented as a result. It is what was deployed
and timed, and it is reported for that reason only. The SOH arm is a secondary
result of this paper; the primary claim is the SOP path. Replacing the
deployed model with ridge would be an improvement in accuracy *and* in MCU
cost — it is a dot product — but it would change the shipped artifact, the
MCU timing evidence and the estimated-SOH end-to-end corners, so it is left as
stated work rather than done quietly.

> **[Superseded — 36]** It was done. Demoting was the wrong call: nested
> selection shows the CNN placing **last on all six folds** before any
> held-out cell is touched, so it is not a worse model that happened to ship,
> it is a model no honest procedure would have picked. Ridge is deployed,
> measured on the board at **6.50 µs against 19 442 µs**, and halves the
> firmware. The SOP inversion got 5.3 % slower and that turned out to be an
> instruction-cache placement effect, not a cost of ridge — §36.4.

### 35.8 What the contribution is

Not superiority. On usable current the sequence baselines are within noise of
the trim at 1 100× the parameters (§34.6), and ridge beats the CNN on SOH.

> **[Corrected — 37.3]** This section then claimed **deployment-efficient
> equivalence**. That word is not available. An equivalence or noninferiority
> claim needs a margin fixed before the data is seen and a formal test, and
> neither exists here — choosing a margin now would be the same defect as
> choosing a model on the test set. §37.3 replaces it with what the
> measurements support, and `method_comparison.csv` makes it checkable: A8
> places 3rd, 3rd, 2nd and 5th of six across the four direction × horizon
> conditions, and only FFRLS separates from it by bootstrap interval. The
> defensible sentence is *competitive safety-adjusted usable current against
> the A3, LSTM, GRU, FFRLS and shrinkage baselines tested here, using four
> effective deployed coefficients* — and "four effective deployed
> coefficients", never "a four-parameter model", because the header ships 50
> floats.

The production claim in the first draft is withdrawn. Oracle-state validation
overstates system safety: it scores a row set the vehicle could not have
selected.

> **[Narrowed — 37.2]** The original wording here said the gap is "in the
> admission rule rather than in the prediction", which reads as a measured
> failure rate of an onboard filter. No onboard filter was implemented or
> tested. What was measured is that the *offline evaluation inclusion rule*
> is SOC-dependent: estimated SOC changes which pulse-derived labels satisfy
> it, and the subset only the estimated corner admits shows elevated
> exceedance. §37.2 states it at that scope.

### 35.9 What still cannot be done

- **A seventh cell.** The all-cell fit that ships is not separately validated
  and cannot be; the leave-one-cell-out numbers are the honest estimate of
  what it does on a cell it has not seen.
- **Charge-direction external safety.** The frozen λ fails there. Fixing it
  means refitting λ on data that includes RPCWBY, which stops it being a
  transfer result. Reporting the failure is the only honest option available
  without a third dataset.
- **Low-SOC external transfer.** Would need pooled surfaces built from cells
  characterised below SOC 0.30 on the external chemistry. The data exists in
  RPCWBY; the surfaces do not.
- **A quantile or distributionally-robust λ.** §35.3 shows the max-based
  factor is fragile. Nothing was substituted, because changing the safety
  definition after seeing the test set is the failure this audit exists to
  prevent.

---

## 36. The SOH arm was replaced

§35.7 left this as a stated decision rather than a done one: ridge beat the
CNN that was deployed at the time, on the same splits, and the paper demoted
it instead of replacing it. That was the wrong call, for a reason §35.7 did not notice —
the CNN is not merely worse, it is a model that **no honest selection procedure
would have picked**.

### 36.1 The CNN loses before it ever sees a held-out cell

Choosing ridge because it beat the CNN on the leave-one-cell-out table is
selection on the evaluation set: the comparison in `soh_baselines.csv` already
saw every held-out cell. So the family was re-chosen the way it would have to
be chosen in practice. For each outer held-out cell, every candidate — mean,
ridge, PLS, SVR, gradient boosting and the CNN itself — is scored by
leave-one-cell-out over the **five training cells only**; the winner is refit
on those five and only then meets the held-out cell (`repro/run_soh_nested.py`).

| outer holdout | chosen | inner ridge | inner CNN | outer RMSE |
|---|---|---:|---:|---:|
| BOOST | gradient boosting | 0.0105 | 0.0142 | 0.0091 |
| BOOST_NEGPULSE | ridge | 0.0103 | 0.0184 | 0.0112 |
| BOOST_NEGPULSE_1S | ridge | 0.0101 | 0.0149 | 0.0060 |
| BOOST_REST | ridge | 0.0088 | 0.0102 | 0.0130 |
| CC | ridge | 0.0092 | 0.0135 | 0.0106 |
| CC_CELL2 | gradient boosting | 0.0122 | 0.0144 | 0.0056 |

**The CNN is last on all six folds**, with no fold close. The procedure's own
pooled error is 0.0095 — indistinguishable from always-ridge at 0.0094 — so the
cost of selecting rather than assuming is about one part in a hundred.

Ridge is what ships, on four folds out of six and on a criterion decided before
the comparison: a 65-coefficient linear map fits the part, a gradient-boosted
ensemble does not. That is a deployability constraint, not a test score.

| | pooled RMSE | worst cell | coefficients |
|---|---:|---:|---:|
| **ridge (adopted)** | **0.0094** | **0.0130** (BOOST_REST) | **65** |
| 1D CNN (baseline) | 0.0135 | 0.0291 (BOOST_REST) | 32,835 |

The worst cell matters more than the pooled figure: the CNN was 2.2× its own
pooled error on BOOST_REST, ridge is 1.4×. The arm that used to fail worst on
the hardest cell no longer does.

### 36.2 What runs on the board

    soh = b + sum_i w_i * (x_i - mu_i) / sd_i

64 multiply-accumulates. `analysis/export_soh_ridge.py` writes the same
`soh_mu`/`soh_sd` arrays the CNN header defined, so `soh_core.c` standardises
identically and only the inference body changes; the CNN body is kept under
`#else` so the comparison build still exists and the saving is measured rather
than asserted. `SOP_CMD_SOH_Q`, the integer kernel opcode, is **refused** by
the ridge build rather than answered with the float timing.

Like the SOP header, the exported fit is the all-cell one. `--holdout` anything
else needs `--allow-fold` and is a benchmark artifact.

### 36.3 Downstream

Every estimated-SOH result was recomputed. The SOP arm is unchanged — SOH
enters only through `results/soh_pred.npz`:

| | CNN | ridge |
|---|---:|---:|
| estimated-SOH usable current, discharge 10 s | 67.2 % | **69.0 %** |
| end-to-end, both estimated, discharge 10 s | 26 / 488 | **20 / 485** |
| paired, both estimated (common rows) | 4 / 385 | **3 / 388** |

The §35.2 drift finding is unchanged and slightly stronger: the rows only the
fully estimated corner admits carry **27.8 %** exceedance (27 of 97) against
0.97 % for the oracle corner's extra rows. A better SOH model does not repair
a filter that a wrong SOC has corrupted, which is what §35.2 predicted.

### 36.4 The board: half the firmware, and an honest surprise

| | CNN | ridge | change |
|---|---:|---:|---:|
| SOH inference | 19 442.25 µs | **6.50 µs** | **2 991× faster** |
| flash (text) | 144 012 B | **72 700 B** | **−49.5 %** |
| RAM (bss) | 28 276 B | **13 172 B** | −53.4 % |
| deployment build (A8 only) | 142 060 B | **70 796 B** | −50.2 % |

The 19.4 ms is itself new: the SOH arm had never been timed, and the ledger
carried "17.9 ms, unmeasured" as an estimate. `mcu/bench_soh.py` now sends real
dQ/dV curves from the cache and checks the board's answer against the same fit
in NumPy — **282 of 282 within 2 × 10⁻⁴**. SOH runs once per charge, not in the
control cycle, so this buys headroom rather than cycle budget.

**And the SOP inversion got slower.** FULL median 50.50 → **53.20 µs**, +5.3 %,
from removing 71 kB of code that the SOP path never calls. Reporting that as a
cost of ridge would be wrong and hiding it would be dishonest, so it was
isolated. Same source, same board, minutes apart:

| | CNN image | ridge image | ridge − CNN |
|---|---:|---:|---:|
| ICACHE on (deployment) | 50.50 µs | 53.20 µs | **+5.3 %** |
| ICACHE off (`-DSOP_NO_ICACHE`) | 107.89 µs | 106.38 µs | **−1.4 %** |

With the instruction cache disabled the ordering **reverses** and the smaller
image is faster, as it should be. All twelve hot functions moved by 0x1C–0x20
bytes when the CNN left, and eight of them changed their 32-byte alignment.
The penalty is instruction-cache placement, not work.

Two things follow, and the second is the one worth carrying out of this paper.
The published SOP timing is now 53.20 µs, because that is what the deployment
firmware does. And **a microcontroller timing claim at this granularity is a
property of the image, not of the algorithm**: an unrelated model swap moved an
untouched hot path by 5 %, in the wrong direction, reproducibly. Anyone
comparing embedded inference costs across papers is comparing link maps as much
as arithmetic. `analysis/results/tables/mcu_icache.csv` carries the experiment.

### 36.5 What this does to the paper's claim

It strengthens the part that was already the contribution and removes the part
that was embarrassing. The SOH arm is no longer a 10,945-parameter network
reported because it was deployed; it is 65 coefficients that beat the network
on every cell and cost 6.5 µs. The SOP arm's claim — four effective
coefficients matching models three orders of magnitude larger — now has a
matching SOH arm rather than a contradicting one.

What it does **not** change: everything in §35. The estimated-SOC finding, the
corrupted-filter mechanism, the charge-direction external transfer failure and
the withdrawn production claim all stand exactly as written, and the numbers
were recomputed rather than carried over.

### 36.6 What was not done

- **Gradient boosting**, which the inner selection prefers on two of six folds,
  was not deployed. The reason is size, not accuracy, and it was decided before
  the outer scores were read — but it does mean the deployed family is not the
  argmax of the procedure on every fold, and saying otherwise would overstate
  it.
- **Recovering the 5.3 %.** `-falign-functions=32` changes nothing (the image
  is byte-identical). Linker-level placement of the SOP hot path would probably
  recover it and was not attempted; tuning the layout until the number improves
  is also exactly the kind of search that needs a pre-registered stopping rule,
  and there is none.
- **The CNN's int8 kernel** has no ridge counterpart, so `soh_simd.c` compiles
  to nothing and the SIMD comparison in §27 stands only for the superseded
  model.

---

## 37. Third review round — reproducibility, and the words the results support

The second round (§35–36) was reviewed and returned a conditional go: the
science was salvageable, the package was not submittable. Six findings, and
the first one is the one that mattered.

### 37.1 Raw-to-result reproducibility, resolved

`manifests/raw_rebuild.yaml` had recorded that a full rebuild moved 235
numeric cells across ten tables, and left the decision open with
`AUTHOR DECISION REQUIRED`. It has been taken: **the defect-excluded rebuild
is now the canonical result.**

Getting there cost two wrong diagnoses, and both are worth recording because
each was reached by inference where a measurement was available.

*First wrong diagnosis.* `voltage.csv` still read 67.61 where the manifest
said a rebuild gives 69.3, so the cache was assumed stale. It was not. Rebuilt
from scratch in both directions, all six cells, `cache/trim` came back
**byte-identical** — same row counts, `X` and `Y` allclose. Seventy minutes of
wall clock to learn that the thing I should have compared first was the cache
against the defect list.

*Second wrong diagnosis.* The committed models were then assumed to predate
the fix. Their training labels `Y` are identical to the retrained ones in all
six cells, so they were fitted on the same data; only the weights differed, by
0.02–0.06 % in prediction.

*What it actually was.* `sop_baseline_fill.py`'s `fit_alpha` is an argmin over
`np.linspace(0, 1, 51)`, step 0.02 — a **step function of its input**. A
0.05 % shift in the trim predictions moves that argmin one step, and that
lands as a 2.5 % change in a published voltage RMSE. The manifest had already
named this ("a grid quantum, not a measurement drift") without connecting it
to the reproducibility claim.

Retraining reproduced the documented rebuilt values exactly — `voltage.csv`
shrink CC 67.61 → 69.3 and `ladder.csv` rmse_A 4.72 → 4.77, both as recorded —
and moved **eight verified numbers**:

| check | was | now |
|---|---:|---:|
| `volt.disc.A8` | 62.81 | **62.78** |
| `volt.char.A8` | 36.73 | **36.70** |
| `volt.disc.A3` | 58.76 | **58.71** |
| `volt.char.A3` | 34.13 | **34.09** |
| `volt.disc.direct` | 134.71 | **134.57** |
| `volt.disc.shrink` | 69.10 | **69.23** |
| `ladder.disc.shrink` | 62.4 | **62.0** |
| `sop.chg.10s.usable.est_soh` | 54.1 | **54.2** |

The A8 charge figure, 36.70 mV, is exactly what §33.4 predicted the
defect-excluded data would give. `ladder.disc.shrink` read 62.4 twice and 62.0
twice during this audit as the grid step flipped under it; that is the
amplifier, not a measurement.

One number that moved mid-audit was not a measurement at all: a 55.54 for the
estimated-SOH charge figure came from a `safety_strict` table that had not
been re-run after `run_evals`. Three ordering mistakes of that shape happened
in this round, all from hand-assembling partial re-runs, which is why the
regeneration is now one script in dependency order.

### 37.2 Estimated state, under a lambda the vehicle actually has

§35.1 scored each corner under a lambda refitted inside that corner, which
answers "how well could this be calibrated" and hides the state error in the
recalibration. A deployed system does not recalibrate. `run_end_to_end.py` now
also carries the oracle corner's per-cell lambda across unchanged:

**discharge, τ = 10 s, 388 rows, λ frozen at the oracle corner's calibration**

| | exceed | usable | worst overshoot |
|---|---:|---:|---:|
| oracle SOH + oracle SOC | 3 | 73.02 % | 1.050 A |
| estimated SOH + oracle SOC | 2 | 72.20 % | 0.233 A |
| oracle SOH + estimated SOC | 3 | 72.59 % | 0.676 A |
| **estimated SOH + estimated SOC** | **1** | **72.01 %** | 0.641 A |

Under the shipped safety factor, on identical pulses, **estimated state does
not increase exceedance** — one against the oracle corner's three — and costs
about 1 %p of usable current. The apparent penalty in §35 was row-set drift
plus per-corner recalibration.

**And the drift finding has to be stated at its actual scope.** §35.2 called
it "a wrong SOC corrupts the filter". No onboard filter was implemented or
tested. What was measured is that the **offline evaluation inclusion rule** —
the trustworthy-label test, an extrapolation distance computed against SOC —
is SOC-dependent: the 97 rows only the fully estimated corner admits carry 27
exceedances, 27.8 %, against 0.97 % for the oracle corner's own extra rows.
That is a property of the evaluation protocol. It still means oracle-state
validation overstates system safety, because it scores a row set the vehicle
could not have selected, and it is still the reason the production claim is
withdrawn. It is not a measured failure rate of an admission filter.

### 37.3 Not equivalence — competitive, and not first

§35.8 claimed "deployment-efficient equivalence". That word needs a margin
fixed before the data is seen and a formal noninferiority test; choosing one
now would be the same defect as choosing a model on the test set. It is
withdrawn, and `method_comparison.csv` replaces it with what the data
supports — every method with its cell-cluster bootstrap interval and its rank
in each of the four direction × horizon conditions:

| method | ranks across the four conditions |
|---|---|
| lstm | 1, 1, 4, 4 |
| gru | 2, 2, 3, 2 |
| a3 | 4, 4, 1, 3 |
| **a8 (adopted)** | **3, 3, 2, 5** |
| shrink | 5, 5, 5, 1 |
| ffrls | 6, 6, 6, 6 |

**A8 is never first.** Three of twenty intervals separate from it, and all
three are FFRLS. So "outperformed" is false and "equivalent" is unavailable.
The defensible sentence is *competitive safety-adjusted usable current against
the A3, LSTM, GRU, FFRLS and shrinkage baselines tested here, using four
effective deployed coefficients* — and "four effective deployed coefficients",
never "a four-parameter model", because the header ships 50 floats.

`qc.py` now fails if "equivalence", "outperformed", "four-parameter model",
the onboard-filter framing or "production" reappear.

### 37.4 The validity envelope, measured

I recorded in the ledger that nothing in the three DOIs provides sub-0.30 SOC
or off-25 °C data. **That was false and I should have checked before writing
it.** RPCWBY Test#3 sweeps −20 to 40 °C on the same cell at SOC 0.02–0.95, and
it was already reduced in the repository.

Test#3 has no paired drive cycle, so the A8 trim cannot be computed on it —
only the nominal 2RC layer the trim sits on. That layer is what decides
whether a prediction is safe, so its envelope is worth measuring:

| T set | in hull | λ needed | margin | exceed | 95 % upper | worst overshoot |
|---:|---:|---:|---:|---:|---:|---:|
| 40 °C | 8 / 14 | 0.9887 | 1.447 | 0 | 31.2 % | 0 W |
| 25 °C | 7 / 14 | 0.9672 | 1.416 | 0 | 34.8 % | 0 W |
| 0 °C | 8 / 14 | 0.9534 | 1.396 | 0 | 31.2 % | 0 W |
| −10 °C | 8 / 14 | 0.6001 | **0.878** | 1 | 47.1 % | 6.37 W |
| −20 °C | 8 / 14 | 0.3319 | **0.486** | 5 | 88.9 % | 26.90 W |
| **0–40 pooled** | **31** | | | **0** | **9.2 %** | |

Below 0 °C the shipped λ = 0.6832 is plainly wrong, and the sample is large
enough to say so: −20 °C needs 0.332 and overshoots by 26.9 W on five of eight
points.

Above it, the honest statement is weaker than "conservative". Each temperature
contributes **7–8 in-hull points**, so an observed zero carries a one-sided
95 % upper bound of about **31 %** — pooled across 0–40 °C, 31 points and
**9.2 %**. Zero observed is not zero risk at this sample size, which is the
same caution §32.6 applies to the main result and it applies here with far
less data.

And the scope is narrower than "temperature range" suggests: **Test#3 has no
paired drive cycle, so the A8 trim cannot be computed on it at all.** What is
bounded above is the nominal 2RC layer the trim sits on. The hybrid's
temperature validity is not established by this experiment and must not be
stated as if it were.

The pooled hull **never reaches below SOC 0.30 at any temperature**, which
also explains the 0 % low-SOC coverage in the Test#2 result rather than
leaving it an unexplained hole. On Test#2 the frozen λ holds on discharge (margin 1.30–1.43, zero
exceedance) and fails on charge (needs 0.3969 against the shipped 0.5860,
9–11 exceedances of 248 in-hull rows per fold).

**RPCWBY is one physical cell.** The six "folds" are six UYPYDJ-trained models
on that same data — six models, not six cells — so their exceedance counts are
correlated and "zero exceedance in all six folds" is not six independent
confirmations. "External multi-cell validation" is forbidden in the ledger.

### 37.5 The pipeline is now bit-reproducible

The trainers called `torch.manual_seed` and nothing else. cuDNN still chose
kernels by benchmarking, CUDA reductions still accumulated in nondeterministic
order, and cuBLAS had no fixed workspace. `analysis/determinism.py` sets all
of it, and **raises rather than warning** when it cannot: a run that cannot be
reproduced should say so.

Verified by running the same command twice: predictions bit-identical for all
six cells, and the checkpoints hash the same.

The strictness earned its keep immediately. `use_deterministic_algorithms(True)`
refused to train the SOH CNN —

    RuntimeError: adaptive_avg_pool2d_backward_cuda does not have a
    deterministic implementation

— which means **every CNN fit ever made through that layer was irreproducible
regardless of seed, and nobody knew**. On a length-32 activation
`AdaptiveAvgPool1d(8)` is exactly `AvgPool1d(4)` (checked numerically, max
|Δ| = 0) and `AvgPool1d` has a deterministic backward. `soh_cnn.pool_to_8()`
takes the equivalent fixed-kernel pool whenever the length divides, and names
the non-reproducible case in its docstring. Same function, computed
differently. The CNN then reproduced bit-for-bit across two runs, and its
worst cell moved 0.0293 → 0.0291 — the old figure came from a fit that could
not be reproduced at all.

The rest of the chain was checked rather than assumed: `eval_sop_amps.py`
reads the stored `pred_*.npz` in NumPy and does no GPU inference, and every
table producer that uses randomness seeds it. Training was the only unseeded
link.

The claim is not "training is deterministic" but "the pipeline gives the same
numbers", so it was tested that way. The whole chain was re-run from scratch —
four trims retrained, baselines refitted, every evaluation rescored, every
safety table and the method comparison rebuilt — and the 54 result tables
diffed cell by cell against the previous run:

> **0 of 54 tables changed, 0 numeric cells moved.**

`manifests/model_provenance.yaml` now records the SHA-256 of the training data
behind each committed model, because mtimes cannot catch a model outliving its
data in a fresh clone.

### 37.6 The gates that were not gates

Three things reported success without being able to fail, and all three were
found by using them rather than reading them:

* The **lint step** had ended in `|| true` since the day it was added. `E9,F`
  now blocks — 72 real defects fixed on the way — and a test fails if every
  ruff invocation becomes non-blocking again.
* **`run_soh_deploy_tables.py --check`** computed the comparison, printed
  MISMATCH and returned 0. It returns the result now, and a test corrupts each
  table and requires `--check` to fail.
* **`optional`** was a stage key nothing read, so the superseded CNN showed as
  `absent` beside genuinely broken stages. A second test now fails on any
  stage key no code in `repro/` reads, with the allowlist derived from the
  source rather than written down.

Twenty-one verified numbers had no producer stage; eleven stages were added
and the count is zero. The two all-cell fits that become the header on the
board had **no stage at all** — the command existed only in a shell history —
and a test now walks everything that reaches the board and requires each to be
some stage's output.

The figure was the worst of it: `results_fig_soh_traj.png` plotted ridge while
its title said "partial-charge CNN, 10,945 parameters". Producers now record
what made a prediction file, the figure reads it and refuses to render without
it, and a test forbids a model name or parameter count appearing in the figure
source at all.

### 37.7 What is still not done

- **A second external cell.** RPCWBY is one. Temperature and low-SOC coverage
  could be extended and were; cell-to-cell external generalisation cannot be,
  and no wording may imply it.
- **Charge-direction external safety.** The frozen λ fails there. Refitting on
  data that includes RPCWBY would stop it being a transfer result, so it is
  reported as a failure.
- **Pack and HIL.** No hardware. `sop_pack2.py` is a Monte Carlo sensitivity
  study and `qc.py` fails if it is ever called a pack validation again.
- **A formal equivalence test.** Deleted rather than faked; a margin chosen
  after seeing the comparison is not a margin.
- **The grid amplifier.** `fit_alpha` is still an argmin over a discrete grid.
  Determinism removes the input noise so the step is no longer reachable by
  chance, but a 2.5 % move there can still mean a 0.05 % change upstream, and
  the function now says so.

### 37.8 The state files had not been re-canonicalised

A fourth review found that §37's work had been *appended* to the state
manifests without the superseded entries being marked. Every one of these was
a document saying something the code had stopped doing:

* `paper_map.yaml` still offered `24.3 %`, `385 rows`, `50.52 µs` and
  "deployment-efficient equivalence" as **replacements** — the corrections
  from round two, left standing after round three moved past them. Marked
  `[SUPERSEDED]` with the current values and three new claims added.
* `freeze_log.yaml` still froze "SOH CNN predictions, 6 LOCO folds × 3 seeds".
  The frozen artifact is ridge, and the entry now carries SHA-256 for it, the
  all-cell fit and the kept CNN reference.
* `evidence_ledger.yaml` still carried `open_choice: AUTHOR DECISION
  REQUIRED` for the raw rebuild, which §37.1 had resolved.
* `generalization_scope.yaml` said the frozen A8 "has not been run" outside
  UYPYDJ. It has: one external cell, discharge transfers, charge does not,
  54.2 % coverage.

A stale manifest is worse than a missing one, because it reads as provenance.

Two mechanical defects came with it. `expected.json` named a stage
`external_a8` that does not exist (the stage is `external`), and a test now
checks every stage name — which immediately found seventeen more: the
`safety_strict` stage did not declare or produce the per-method tables that
named it, and two MCU tables named `mcu_measure` instead of
`soh_deploy_tables`. The stage now runs all five method scorings and declares
all 21 outputs, which also closes the ordering trap that produced a stale
table earlier in this audit.

**And the licence question was never open.** The manifest said UYPYDJ's licence
was "stated nowhere in the dataset readme". It is stated, in section 2 of that
readme — *"Licenses/restrictions: CC BY 4.0"* — with the depositors'
certification that the data is free of licensing and intellectual property
issues. The file's SHA-256 matches the manifest, so the text was there all
along. All three datasets are CC BY 4.0. The entry stood for four days because
nobody read line 47, and it was blocking release.

### 37.9 Zero observed is not zero risk, and the sample here is small

§37.4's temperature table reported zero exceedance from 0 to 40 °C without
saying how little that rests on. Each temperature contributes **7–8 in-hull
points**, so an observed zero has a one-sided 95 % upper bound near **31 %**;
pooled across 0–40 °C, 31 points and **9.2 %**. Those bounds are now columns in
`external_temp_envelope.csv` rather than something a reader has to compute.

The scope was also stated too broadly. Test#3 has **no paired drive cycle**, so
the A8 trim cannot be evaluated on it at all — what the sweep bounds is the
nominal 2RC layer beneath the trim. "The validity range this work can claim is
0–40 °C" was wrong twice over: wrong about which model, and wrong about what
7–8 points can support. Both the spec and README now say *the physics layer
shows no exceedance from 0 to 40 °C above SOC 0.30, bounded at 9.2 %*, and say
explicitly that the hybrid's temperature range is not established.

### 37.10 A sweep, instead of waiting for the next reviewer

Four review rounds each found defects of the same few shapes, which is a
signal about the process rather than the code: everything so far had been
fixed reactively, one report at a time. So the classes were swept
systematically instead, and each sweep that found something was turned into a
test so it stays swept.

| class | swept by | found |
|---|---|---|
| gates that cannot fail | AST: does `main()` ever return non-zero? | clean (the one `\|\| true` is the advisory lint, deliberately) |
| tests with no assertion | AST over every `test_*` | none |
| manifest hashes vs files | recompute all 116 | all match |
| stage outputs that do not exist | walk the graph | none |
| tables no stage produces | set difference | none |
| `expected.json` referencing absent tables | direct | none |
| **hand-maintained numbers** | **manifest values against the tables** | **`timing_us` was two firmwares stale** |
| **document paths that do not resolve** | **regex over backticked repo paths** | **six** |

The two finds are worth stating plainly.

`manifests/mcu_evidence.yaml` carried the per-cycle cost as six typed-in
fields — 13.09, 201.74, 214.83, 293.26, 307.12 µs — and they were still the
**pre-ridge firmware** after the board had been reflashed and re-measured
twice. Nobody retyped them because nobody had to. A number a human copies
after every measurement is a number that will eventually be wrong, so
`run_mcu_table.py` now derives the composite into `mcu_cycle.csv` and
`verify.py` checks it. The real figures are **227.79 µs median and 339.84 µs
worst case** — 2.278 % and 3.398 % of a 100 Hz budget — against the 214.83 and
307.12 the manifest and `paper_map` had been asserting.

Six backticked paths in this document pointed at nothing: four are the design
review's proposed files that never landed, two were local caches that are
gitignored and gone. Mentioning them is fine; mentioning them without saying
so is not, and a reader following the reference had no way to tell. Each now
carries the explanation in its own block, and a test requires that — reusing
`qc.py`'s block rule, so "this text is a record, not a claim" has one
mechanism in this repository rather than two.

### 37.11 The SOH arm is ridge; the CNN is a baseline

Adopting ridge (§36) changed which model the paper reports, and the labels
had not followed. `soh_baselines.csv` still called the CNN "shipped",
`soh_model_cost.csv` called it "superseded" — audit language, not the name of
a comparison group — and `soh_nested.csv` carried `inner_1D CNN (shipped)`.

Worse, `paper_map.yaml` still listed **`soh.rmse` as VERIFIED with the CNN's
0.0135**. The entry a reader would take as "the SOH result this paper stands
on" was the model that had been replaced. `mcu.cycle_time` was VERIFIED at
214.8 µs and 142.1 KB for the same reason.

Both are `SUPERSEDED` now with the current values attached, and the tables
read the way the paper argues:

| | pooled RMSE | worst cell | coefficients |
|---|---:|---:|---:|
| **ridge (adopted)** | **0.0094** | **0.0130** (BOOST_REST) | **65** |
| gradient boosting | 0.0101 | 0.0179 | — |
| PLS | 0.0096 | 0.0126 | — |
| SVR (RBF) | 0.0099 | 0.0146 | — |
| 1D CNN (baseline) | 0.0135 | 0.0291 | 32,835 |
| mean baseline | 0.0878 | 0.0988 | — |

The CNN now sits with PLS, SVR and gradient boosting, which is where the
evidence puts it: nested selection scores every family on the five training
cells before touching the held-out one, and the CNN places **last on all six
folds and is chosen zero times**.

This one was not found by a sweep or a reviewer's file list — it was found by
being asked whether the paper really had that many claims, and counting.
Eleven survive, not the thirteen I reported, because two of the "verified"
ones were describing a model the paper no longer uses.

### 37.12 Prior load, at the temperature with the least room

§37.4 left 0 °C as the cold edge of where the frozen λ still holds, with a
margin of 1.396 against 1.447 at 40 °C. RPCWBY Test#8 sweeps the one axis
Test#3 does not — the rate the cell was discharged at **before** the pulse:
0, C/3, 1C, 2C, 3C and 4C, at 0 °C, over thirteen SOC points. If recent load
history moves what the physics layer owes, it should move it here.

| prior rate | in hull | λ needed | margin | exceed | 95 % upper |
|---|---:|---:|---:|---:|---:|
| 0C | 8 / 12 | 1.1421 | 1.672 | 0 | 31.2 % |
| C/3 | 8 / 13 | 1.1375 | 1.665 | 0 | 31.2 % |
| 1C | 8 / 12 | 1.1352 | 1.662 | 0 | 31.2 % |
| 2C | 8 / 12 | 1.1352 | 1.662 | 0 | 31.2 % |
| 3C | 8 / 12 | 1.1338 | 1.659 | 0 | 31.2 % |
| 4C | 8 / 11 | 1.1309 | 1.655 | 0 | 31.2 % |
| **pooled** | **48** | | | **0** | **6.1 %** |

**Prior load barely moves it.** λ_needed spans 1.1309–1.1421 across a
twelve-fold change in prior rate — a spread of 1 %, and every value is well
above the shipped 0.6832. Zero exceedance in all 48 in-hull points, bounded
at 6.1 % — an exact-binomial bound over those rows, which are one physical
cell measured across SOC and prior rate. It is conditional on this grid and
this cell, and it is not a cell-level or population-level risk; a single
external cell cannot support one at any sample size.

The table above is scored against the CC surface. **That was a choice, and it
was the most favourable of six** — see §37.14, which repeats the whole test
against every internal surface. The margin to quote from this experiment is
1.351, not the 1.672 above.

Two things to take from that, and one not to.

The frozen factor has **more** room here than the Test#3 sweep suggested at
the same temperature (margin 1.66 against 1.396), which is worth stating
because it is the opposite of a caveat: Test#8's protocol reaches a different
part of the operating space and the model is further inside its envelope
there.

The hull is again the binding constraint, not the physics: 8 of 12–13 points
per rate, SOC 0.30–0.95 every time. Below 0.30 there is still nothing.

What this does **not** show is that the trim is insensitive to prior load.
Test#8 has no paired drive cycle either, so the A8 trim cannot be computed on
it — this is the nominal 2RC layer, and the trim exists precisely because that
layer needs correcting from drive history. A reader could easily take "prior
rate does not matter" as undermining the trim's premise; it does not, because
the trim was never evaluated here.

### 37.13 The SOC arm had a mean and nothing else

The SOP arm publishes a per-cell held-out λ, the worst cell and a cell-cluster
bootstrap interval. The SOH arm publishes a per-cell RMSE and the worst cell.
The SOC arm published `2.140 %p` — one pooled mean over six disturbances — and
a reader had no way to tell whether that is what every cell does or the average
of one good cell and one bad one. The per-run errors existed; they were
averaged away on the way to the table.

`repro/run_soc_percell.py` takes the pooled number apart. It re-simulates
nothing: `soc_perturb.npz` already holds every run's RMSE, and `soc_runs.pkl`
carries the cell label of each run, so the decomposition cannot disagree with
the headline. The row order in that array is config-major, and reading it
wrong would silently publish another filter's numbers under the adopted
filter's name, so the script asserts its own indexing against three values in
`soc_headline.csv` before it writes anything.

| Cell | undistorted | mean of 6 | worst of 6 | worst disturbance |
|---|---:|---:|---:|---|
| CC | 1.201 | **1.834** | 3.406 | current offset −0.10 A |
| BOOST | 1.292 | 1.948 | 3.458 | current offset −0.10 A |
| BOOST_NEGPULSE | 1.269 | 2.001 | 3.742 | current offset −0.10 A |
| BOOST_REST | 1.565 | 2.213 | 4.243 | current offset −0.10 A |
| CC_CELL2 | 1.818 | 2.372 | 4.321 | current offset −0.10 A |
| BOOST_NEGPULSE_1S | 1.910 | **2.474** | 3.599 | current offset +0.10 A |

Pooled 2.140 %p, worst cell 2.474, cell-cluster 95 % interval [1.962, 2.326]
over six clusters. The spread is 1.35×, which is tight — the headline is not
carried by one cell — and a current offset of ±0.10 A is the worst disturbance
for every cell but one. That last point is the useful one for a deployment:
the sensitivity that matters is current-sensor bias, not initial-SOC error.

**Two things this is not.** Every filter reads its own cell's
characterisation surface, so this is a per-cell *calibrated* deployment and
the spread is over operating conditions within a cell — it is not a
leave-one-cell-out transfer like the SOP and SOH arms, and nothing here says
the filter works on a cell it has never seen. And the six cells each carry a
different aging protocol, so a per-cell spread is a cell-and-protocol spread.

That second point is worth stating on its own, because it reaches past the SOC
arm. **The dataset runs one physical cell per protocol.** Holding out
BOOST_REST holds out a protocol and a cell at once, and no result in this
repository can say which of the two moved. CC and CC_CELL2 — same protocol,
second cell — are the only clean read on cell-to-cell variation in the entire
set, and they span 1.834 to 2.372 %p, a 1.29× ratio against the 1.35× spread
over all six. Nearly the whole apparent protocol effect is reproducible from
two cells running the *same* protocol. `generalization_scope.yaml` therefore
holds `protocol` at PARTIAL as well as `cell`, and the claim limit is "six
protocol-cell combinations", never "generalises across charge protocols".

### 37.14 Which surface Test#8 is scored against changes the answer

§37.12 ran against `--holdout CC` because that is the script's default. Test#8
is external, so nothing is held out for it: all six internal surfaces are
models built on internal cells and applied to a cell none of them saw, and
every one is equally entitled to be used. Reporting one was a selection, and
it happened to land on the most favourable.

`run_external_crate.py --all-surfaces` runs all six.

| surface | λ needed (min–max over rates) | margin | exceed |
|---|---:|---:|---:|
| CC_CELL2 | 0.9232–0.9325 | **1.351** | 0 / 48 |
| BOOST | 0.9891–0.9922 | 1.448 | 0 / 48 |
| BOOST_REST | 1.0706–1.0746 | 1.567 | 0 / 48 |
| BOOST_NEGPULSE | 1.0961–1.1091 | 1.604 | 0 / 48 |
| BOOST_NEGPULSE_1S | 1.1241–1.1354 | 1.645 | 0 / 48 |
| CC | 1.1309–1.1421 | **1.655** | 0 / 48 |

**The conclusion survives; the number does not.** Every surface clears the
shipped 0.6832 with room, and every surface gives zero exceedance in all 48
in-hull points, so "the frozen factor holds at 0 °C across prior load" does
not depend on which surface is picked. But the margin spans 1.351 to 1.655, a
1.22× range — against a ~1 % spread across prior C-rate *within* any one
surface. **Prior load is not what moves this number. The choice of internal
surface is**, by twenty times as much, and §37.12's headline had that
variation hidden inside a default argument.

So the honest reading of Test#8 is narrower than it first looked: it says the
nominal 2RC layer, built on any of the six internal cells, is conservative on
this external cell at 0 °C regardless of prior discharge rate, with at least
35 % of margin left over. It does not say the margin is 66 %.

**Do not pool the six columns.** They are the same 48 measurements scored six
times; 288 would read as six times the evidence and there is none of it. The
summary row in `external_crate_surfaces.csv` reports 48 for that reason.

### 37.15 The temperature envelope had the same defect, one script over

§37.14 found that Test#8's margin depended on which internal surface it was
scored against, and that the published number came from the default argument.
`run_chen2026_baseline.py` — which produces `external_temp_envelope.csv`, and
with it five of the published checks and every temperature claim in the
README — takes the same `--holdout`, defaults it the same way, and its own
help text already said why that is arbitrary: *"RPCWBY is an external cell, so
no UYPYDJ cell is 'held out' here; every surface is equally external."*

| surface | 0–40 °C worst margin | 0–40 °C exceed | −10 °C margin | −20 °C exceed |
|---|---:|---:|---:|---:|
| CC_CELL2 | **1.156** | 0 / 32 | 0.716 | 5 / 8 |
| BOOST_REST | 1.170 | 0 / 32 | 0.701 | 3 / 8 |
| BOOST_NEGPULSE | 1.213 | 0 / 32 | 0.728 | 5 / 8 |
| BOOST | 1.248 | 0 / 32 | 0.861 | 4 / 8 |
| BOOST_NEGPULSE_1S | 1.339 | 0 / 31 | 0.869 | 4 / 8 |
| CC | **1.394** | 0 / 31 | 0.878 | 5 / 8 |

**Both conclusions are surface-independent, and that is the useful result.**
Zero exceedance from 0 to 40 °C on every one of the six. Exceedance at −20 °C
on every one of the six. A margin below 1 at −10 °C on every one of the six.
Nothing about "the frozen λ holds down to 0 °C and fails below it" depends on
which surface was picked.

**The margin is surface-dependent, and the published one was again the most
favourable.** 1.394 was CC; the worst of the six is 1.156. Quote 1.156.

That is now two tables, in two scripts, where a default argument chose the
best of six folds and nothing in the repository would have said so.
`tests/test_no_hidden_fold_selection.py` fails if a third appears: any script
that defaults a fold selector and whose stage publishes a checked table must
also register a sweep. It found `run_external_a8.py` the first time it ran.

### 37.16 The SOC filter was chosen on a SOH it will not have

`soc_perturb_bench.py` hands every configuration its cell's **true** SOH. The
adopted filter — gate at 1 A, 30 s rest hold — was selected on that benchmark,
and §29.4 later re-scored it under the ridge estimate. Re-scoring is not
re-selecting: `soc_est_soh.py:56` hard-codes `i_gate=1.0, rest_hold_s=30.0`,
so it can say what the chosen filter costs and not whether it is still the one
that would be chosen. Picking a model under a condition the deployment does
not have is the same defect as scoring an external test against one surface
out of six (§37.14).

`repro/run_soc_soh_selection.py` re-runs the whole comparison under four SOH
inputs: the true value, the ridge estimate, and a deliberate ±0.02 bias — the
bias separates "sensitive to SOH being wrong" from "sensitive to *this*
estimator", whose error varies by cell and correlates with cell condition.

| config | oracle | estimated | bias +0.02 | bias −0.02 |
|---|---:|---:|---:|---:|
| **EKF adopted (gate)** | **2.140** | **2.206** | **2.405** | **2.231** |
| EKF no gate | 3.133 | 3.147 | 3.935 | 2.539 |
| EKF gate + spread k=20 | 3.248 | 3.332 | 3.356 | 3.472 |
| EKF gate + spread k=200 | 3.584 | 3.619 | 3.644 | 3.664 |
| pure current integration | 6.892 | 6.892 | 6.892 | 6.892 |

**The winner is stable.** The adopted configuration places first under every
SOH input, so the choice was not an artefact of the oracle benchmark, and the
estimated-SOH column is the one to quote: **2.206 %p, not 2.140**.

The full ranking is *not* stable, and saying otherwise would overstate this.
Under a +0.02 bias three placements below first change: no-gate falls 2 → 4
while both spread variants rise. Nothing in the paper rests on those, but the
sentence "the ranking is stable" would have been wrong.

**Overestimating SOH costs three times what underestimating it does** — +0.264
%p against +0.091. The asymmetry has a mechanism: a high SOH reads as a
healthier cell, `R_volt` drops (110 mV at 0.70, 15 mV at 1.00), and the filter
trusts a model that is wrong by more than it thinks. Under-estimating errs
toward distrust, which is the safe direction. For a deployment that is a
reason to bias the SOH estimator low, and this repository does not do that —
the ridge estimator's bias is −0.0003, essentially centred.

### 37.17 So does the SOP method ranking, and four baselines had never been scored

The same question asked of the SOP arm found a larger version of it.
`run_evals.py` carried a per-method flag for whether to also run the estimated
SOH, and it was set for A8 and A3 only. The four baselines in the six-method
comparison — LSTM, GRU, FFRLS, shrink — **had never been evaluated under
estimated SOH at all**, and `run_method_comparison.py` filtered
`soh_arm != 'oracle'`. The paper's central comparative claim was therefore an
oracle-SOH statement, and no other statement was available.

Eight evaluations and five safety tables later, it is. Re-running the oracle
arm first reproduced every published table byte-for-byte, so the two arms
differ only in the SOH.

| condition | 1st on oracle | 1st on estimated | A8 |
|---|---|---|---|
| charge τ = 10 s | lstm | gru | 3rd → **4th** |
| charge τ = 2 s | lstm | gru | 3rd |
| discharge τ = 10 s | a3 | **a8** | 2nd → **1st** |
| discharge τ = 2 s | shrink | shrink | 5th |

**A8's rank vector moves from [3, 3, 2, 5] to [4, 3, 1, 5]**, and the leader
changes in three of the four conditions. Twelve of twenty-four rows change
placement. What does *not* change is the separation count: 3 of 20 intervals,
all FFRLS, in both arms. The weak claim the paper is entitled to — that the
data does not separate A8 from the other four — survives the switch intact.

Degradation is ordered, and the order is the interesting part:

| method | mean Δ, oracle → estimated |
|---|---:|
| a3 | −0.82 %p |
| **a8** | −0.69 |
| lstm | −0.63 |
| gru | −0.11 |
| ffrls, shrink | +0.06 |

The more a method leans on the SOH-indexed surface, the more estimated SOH
costs it; shrink and FFRLS are unmoved because they adapt from the measured
signal instead. A8 loses less than A3, which is why it takes first place at
discharge τ = 10 s under the honest condition while placing second under the
oracle one.

**The published ranking sentence has to be rewritten**, and the replacement is
not worse for the paper: on the deployment condition A8 leads the longest
discharge horizon outright, and drops one place on charge.

### 37.18 The third surface selection, and the rule that found it

§37.14 found Test#8 scored against one internal surface out of six, and
§37.15 found the temperature envelope with the same defect one script over.
Two instances of a pattern is a pattern, so it went into a test rather than a
memo: `tests/test_no_hidden_fold_selection.py` parses every script in
`repro/` for an argument that selects a fold, cell, surface or arm **by
default**, and fails if that script's stage publishes a checked table without
also registering a sweep. Either the selection is swept, or it is listed as
deliberate with a reason.

It found a third the first time it ran: `run_external_a8.py --surface`,
default `CC`, feeding `external_a8_coverage.csv` and `external_a8_safety.csv`
— which carry three published checks and the whole external-validation
claim.

| surface | discharge margin | disc exceed | charge margin | chg exceed | in-hull % |
|---|---:|---:|---:|---:|---:|
| BOOST_REST | **1.125** | 0 | 0.706 | 38 | 55.2 |
| BOOST_NEGPULSE | 1.186 | 0 | 0.610 | 47 | 54.2 |
| BOOST_NEGPULSE_1S | 1.210 | 0 | 0.677 | 67 | 53.9 |
| CC_CELL2 | 1.246 | 0 | **0.610** | 58 | 55.0 |
| CC (published) | 1.297 | 0 | 0.677 | 57 | 54.2 |
| BOOST | **1.302** | 0 | 0.677 | 63 | 55.0 |

**Both conclusions are surface-independent.** Discharge: zero exceedance on
every one of the six. Charge: a margin below 1 on every one of the six, with
38 to 67 exceedances. The external result — discharge transfers, charge does
not — does not depend on which internal surface carries it, and neither does
the hull coverage, which sits between 53.9 % and 55.2 %.

**The margins were again reported from near the top of the range.** Published
discharge 1.297 against a worst of 1.125; published charge 0.677 against a
worst of 0.610. Quote the worst.

That is three tables, in three scripts, where a default argument picked one
fold of six and nothing in the repository would have said so. In all three
the qualitative conclusion survived and only the number moved, which is worth
stating plainly: the defect did not change what the paper concludes. It would
have changed what a reviewer could believe about how the numbers were chosen.

### 37.19 "Usable current" was a percentage of the wrong thing

`usable = median(λ · pred / meas)`, so 100 % means the system permits exactly
what the cell can deliver. The adopted trim scores about **69 % on discharge
and 55 % on charge**, which reads as a weak result. It is not one, and the
denominator is why.

λ is the largest factor with no exceedance on the training cells, so it is set
by **the single worst row of the most demanding cell** — BOOST_NEGPULSE_1S in
most conditions. Every other cell is then scored against its own true
capability, which *no single-λ policy can reach by construction*. The metric
was measuring the price of a fleet-wide safety margin and reporting it as
though it were model error.

Two reference points were missing. `run_usable_reference.py` adds both.

> **[Corrected — 37.20, 2026-09-01]** This subsection first read "holding one
> of these six cells out costs nothing" and that is wrong as a summary. The
> comparison is not iso-risk: where the deployed λ exceeds the fleet λ it
> permits more current and pays for it in exceedance. Corrected below, with
> the exceedance of both policies in the table.

**Against the all-cells ceiling — an identity for five cells, and a risk
trade for the sixth.**

At matched risk `vs_fleet_pct` is **100.0 for every method in every
condition** — an identity, not a measurement. What the comparison actually
surfaces is where the exceedances are:

| method | disc τ=10 | disc τ=2 | chg τ=10 | chg τ=2 | 합계 | fleet λ 로는 |
|---|---:|---:|---:|---:|---:|---:|
| **a8** | 1 | 1 | 2 | 3 | **7** | 0 |
| a3 | 1 | 1 | 2 | 3 | **7** | 0 |
| lstm | 1 | 2 | 1 | 1 | **5** | 0 |
| gru | 1 | 2 | 1 | 2 | **6** | 0 |
| shrink | 2 | 1 | 2 | 3 | **8** | 0 |
| ffrls | 3 | 2 | 11 | 9 | **25** | 0 |

Every one of those sits on the fold that holds the binding cell out, and
**under the all-cells factor there would be none at all** — for any method.
The exceedance in this repository's safety tables is not a property of the
estimator. It is the price of calibrating λ without having seen the cell
that sets it.

§37.20 sets out the retraction in full — which folds are an identity, which
one is the risk trade, and why the two policies are not comparable on usable
current alone. The table above is the part that generalises past A8: **every
method has exceedances under the deployed factor and none under the fleet
factor**, so this is a property of the calibration procedure rather than of
any estimator.

**Against a per-cell oracle λ — field calibration is worth 5 to 19 %p.**

| method | discharge τ=10 | discharge τ=2 | charge τ=10 | charge τ=2 |
|---|---:|---:|---:|---:|
| a8 | **95.2** | 94.6 | 81.8 | 85.3 |
| a3 | 94.5 | 94.9 | 82.6 | 85.9 |
| lstm | 96.0 | 95.0 | 86.2 | 88.9 |
| gru | 95.0 | 95.0 | 88.2 | 89.3 |
| shrink | 91.8 | 95.4 | 83.6 | 86.1 |
| ffrls | 75.0 | 84.9 | 80.4 | 82.3 |

Discharge is nearly saturated: a λ fitted on the evaluated cell itself would
buy the adopted trim under 5 %p at
τ = 10 s. Charge is not — 18 %p sits
there, because the per-cell λ spread is much wider on charge
(0.527–0.839) than on discharge (0.683–0.919), over a
lower base.

So the honest decomposition of the published 68.89 %:

```
68.89 %  =  100.4 %  of the best a single λ could ever do
         ×   95.2 %  of what a per-cell calibrated λ would do
         ×  (the rest is the intrinsic cost of one conservative factor)
```

None of the gap is generalisation error and almost none is prediction error —
the median `pred/meas` is 0.946 and 95 % of rows sit within 8.6 % of truth. It
is the price of covering the worst row of the worst cell with one number, and
the way to reduce it is a per-cell field calibration or a weaker safety
criterion, not a better model.

**Both new columns are oracle bounds.** They use the evaluated cell's own
labels. They may be reported and must never be used to select λ, a tolerance
or a model — that is selection on the test set, which is the defect this
audit exists to remove. The published `usable_pct` stays the headline; these
two say what it is a percentage of.

### 37.20 Retraction: "holding a cell out costs nothing"

§37.19 first summarised the `vs_fleet` column as *holding one of these six
cells out costs nothing*. That is wrong, and the sixth review named the reason
exactly: **the two policies being compared do not carry the same risk.**

The column divides usable current under the leave-one-cell-out λ by usable
current under the all-cells λ. Where those two factors are equal the ratio is
100.0 and nothing has been measured. Where they differ, the deployed factor is
the **looser** one, so it permits more current — and a ratio above 100 is that
permission, not a free gain.

| held out | λ deployed | λ fleet | vs fleet | exceed, deployed | exceed, fleet |
|---|---:|---:|---:|---:|---:|
| BOOST | 0.6831 | 0.6831 | 100.0 | 0 | 0 |
| BOOST_NEGPULSE | 0.6831 | 0.6831 | 100.0 | 0 | 0 |
| BOOST_REST | 0.6831 | 0.6831 | 100.0 | 0 | 0 |
| CC | 0.6831 | 0.6831 | 100.0 | 0 | 0 |
| CC_CELL2 | 0.6831 | 0.6831 | 100.0 | 0 | 0 |
| **BOOST_NEGPULSE_1S** | **0.6930** | 0.6831 | **101.4** | **1** | **0** |

Discharge τ = 10 s. The pattern repeats in all four conditions, and the four
binding-cell rows carry **every one of the seven exceedances** in the A8
estimated-SOH arm — 1, 1, 2, 3 — against **zero** under the fleet factor on
the same rows. Worst overshoot 3.03 A on charge τ = 10 s.

So the statement inverts. The usable current is unchanged only because the
exceedance is not: the ratio above 100 and the seven exceedances are the same
event seen twice. Cap the deployed λ at the fleet factor and the ratio is
100.0 everywhere, which is the identity again — there is no iso-risk gain to
report.

**What survives.** For five of six held-out cells the training cells alone fix
the same safety factor the whole set would have chosen, exactly, in every
condition and for every method. And the failure mode is specific and visible:
it is the cell that sets the constraint, and holding *that* one out leaves the
factor too loose. Both are worth reporting; neither is "costs nothing".

`usable_reference.csv` now carries `exceed_deployed` and `exceed_fleet` so the
comparison cannot be read without its risk. `qc.py` fails on the retracted
phrasing.

### 37.21 How wrong is adding per-stage timings?

§37.10 relabelled 339.84 µs a derived cycle-budget estimate rather than a
WCET, because no integrated loop was ever timed. Honest, and it leaves the
size of the error unmeasured. The firmware answers part of it: `SOP_CMD_FULL`
runs the trim and the solve inside one DWT window, while `SOP_CMD_TRIM` and
`SOP_CMD_SOLVE` time the same two pieces separately over the same 500
operating points. That is 80 of the 340 µs, paired on (SOC, SOH, τ).

| quantity | integrated | summed | Δ % |
|---|---:|---:|---:|
| median over 500 points | 53.206 | 53.320 | +0.214 |
| maximum over 500 points | 81.092 | 81.020 | **−0.089** |
| sum of the two stage maxima | 81.092 | 81.352 | +0.321 |
| worst single point | 51.164 | 51.564 | +0.782 |

Summing is the larger figure at 88.4 % of operating points, and the quantity
`mcu_cycle.csv` actually uses — the sum of stage maxima — sits 0.32 % above the
integrated maximum. **It is not conservative pointwise.** At the single worst
integrated point the paired sum is 0.089 % *below* it, because the two maxima
fall at different operating points. Small here, and not a property to rely on.

This bounds the summation error where it can be measured. It does not extend
to the EKF or the feature stage, which are only ever timed alone; an
integrated loop over all four needs firmware that cannot be built on the audit
host (no `arm-none-eabi` toolchain); and cache and pipeline interactions over
a longer chain are exactly what a two-stage pair cannot show. **339.84 µs is
still not a WCET.**

### 37.22 The cycle timed as a cycle, and what a pack costs to compute

> **Nothing here is pack validation.** There is no second cell, no pack and no
> HIL bench; all N cells are handed the same current, voltage and temperature.
> This measures what the embedded implementation COSTS at pack scale — a
> compute cost on one MCU — not whether the estimate is right on a pack. Pack
> behaviour stays blocked (`evidence_ledger.blocked_work`).

§37.10 relabelled 339.84 µs a derived cycle-budget estimate rather than a
WCET, and §37.21 could only bound the summation error over the 80 µs the
firmware already paired, because **no command ran the four stages of a control
cycle inside one DWT window**. That was recorded as blocked on hardware. It
was not: the NUCLEO-H563ZI is on the bench and the toolchain ships with
STM32CubeIDE. What was missing was twenty lines of firmware.

`SOP_CMD_CYCLE` runs feature update → SOC EKF → trim → inversion in one
window. `SOP_CMD_PACK` repeats that for N cells, each with **its own state**
(124 B, not shared), and reduces by min — what a series pack master does.
Both sit behind `-DSOP_BENCH_PACK` and are absent from the default build,
because the pack arrays are 24 KB and would otherwise inflate the deployment
footprint `build_size.csv` reports. The default build is byte-identical:
text 72700, bss 13172.

**Summing overstates the cycle, by little.** Paired on the same 200 operating
points `sop_mcu_bench.csv` used:

| | integrated | summed | Δ |
|---|---:|---:|---:|
| median | 67.116 µs | 68.226 µs | **+1.65 %** |
| maximum | 93.484 µs | 96.108 µs | **+2.81 %** |

Summing is the larger figure at 59.5 % of points. So the per-stage addition is
conservative here at the median and at the maximum — the opposite sign from
§37.21's two-stage pairing, where the sum of stage maxima sat 0.32 % above the
integrated maximum but 0.089 % *below* it at the single worst point. Neither
error is large. **What matters is that the size is now measured rather than
assumed**, and 339.84 µs stays a four-solve budget estimate, not a WCET: this
measures one solve, and cache and pipeline behaviour over a longer chain is
still not what a 200-point sweep shows.

**A pack costs exactly N times a cell, and 192 cells do not fit 100 Hz.**

| N | median | maximum | per cell | % of a 10 ms period |
|---:|---:|---:|---:|---:|
| 1 | 67.2 µs | 93.7 µs | 67.17 | 0.7 |
| 12 | 798.6 µs | 1.12 ms | 66.55 | 8.0 |
| 48 | 3.19 ms | 4.47 ms | 66.51 | 31.9 |
| 96 | 6.38 ms | 8.94 ms | 66.50 | 63.8 |
| **192** | **12.77 ms** | **17.87 ms** | 66.50 | **127.7** |

Per-cell cost is flat to four significant figures from 1 to 192 cells — it
*falls* slightly, 67.17 → 66.50 µs — so 24 KB of per-cell state costs nothing
in cache behaviour at this size and the scaling is linear with no surprise.

The consequence is concrete. On this part a 192-cell pack runs the estimator
at **78 Hz median and 56 Hz worst case**, not 100 Hz. At 96 cells it fits with
36 % of the period left; at 192 it does not fit at all. A deployment at that
size needs a 10 Hz loop, a faster part, or the cells split across modules —
and that is a design constraint the paper can state, because it was measured.

**Say it once more where the numbers are, not only at the top.** This is a
compute cost at pack scale on one MCU. It is not pack validation, it must
never be quoted as any, and pack behaviour remains blocked.
