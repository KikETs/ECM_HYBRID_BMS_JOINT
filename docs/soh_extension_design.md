# SOH-dependent SOP — extension design

Written 2026-08-15. The premises live in `docs/findings.md`; this document
only covers how those facts force the design.

---

## 0. One-sentence summary

The prior work learns (SOC, T, P) → V and finds SOP by binary search. The
extension is **not to add SOH as an input** but to draw a state vector from
the cell's recent measured response and condition the voltage model on it.
The reason is §1.

---

## 1. Why the naive extension fails

The natural extension is to widen the input to (SOC, T, P, **SOH**). The
data we already have refutes it.

`findings.md` §4: at the same SOH 0.75 the 10 s discharge resistance of six
cells spreads over 26.6–64.1 mΩ, a factor of **2.41**. Of that, **1.58×
sits between two cells running the same protocol.**

If two cells have the same SOH and resistances differing by 1.6×, a model
conditioned on SOH learns their average and is wrong on both. SOP is
defined as the headroom to a voltage limit, so resistance error becomes SOP
error directly.

**The ECM decomposition says why** (`findings.md` §4.3.1). The ohmic
resistance R0 of the six cells agrees within 1.05× everywhere; almost all
the spread comes from the charge-transfer resistance R1 — 5.58× at SOH
0.75. Capacity fade and interfacial degradation follow different
mechanisms, and SOH measures only the first.

**And that pins down what z has to capture.** R1's time constant is short,
0.15–1.05 s. It appears in the voltage response as soon as load is applied,
so it is observable from a few minutes of measured (V, I, T) before the
pulse. The design does not ask the context encoder for something it cannot
in principle reach.

**This variant is still built and run.** Showing a predicted failure is
evidence, and above SOH 0.92 it should actually work well — the six cells
agree within 2–4 % there. "How far can it be used" is part of the result.

---

## 2. Model — conditioning through a context encoder

```
        measured history (V, I, T)             candidate pulse trajectory
        last W_ctx seconds, before the pulse   (SOC, T, P), 200 s
              │                                      │
        ┌─────▼─────┐                          ┌─────▼─────┐
        │  context  │ ──► z (8–16 dims) ─────► │  voltage  │ ──► V(t+τ)
        │ encoder,  │                          │   LSTM,   │
        │ 1-layer   │                          │  2×256    │
        │   GRU     │                          └───────────┘
        └───────────┘
```

- **z is computed only from measurements before the pulse, and is frozen
  during the binary search.**
- The search therefore remains **one forward pass per candidate**. That is
  the binding constraint.

### 2.1 Why past voltage is not fed to the voltage LSTM directly

The prior work's choice of (SOC, T, P) alone has an elegant reason: for a
hypothetical constant-power pulse all three are **known in advance**. Build
the trajectory for each candidate power, run one forward pass, done.

Putting past voltage in the input means the voltage during the pulse has to
be predicted, rolling autoregressively for τ seconds. Error accumulates, and
a place opens for teacher-forcing leakage. (The same trap was already hit
while reproducing the DCT paper.)

The context encoder uses the information in past voltage without breaking
that structure, because z is frozen at the pulse's start.

### 2.2 Variants to compare

| | Conditioning input | What it tests |
|---|---|---|
| **M0** | none (prior work as published) | baseline |
| **M1** | SOH scalar | the failure §1 predicts |
| **M2** | learned z | does it absorb cell-to-cell spread |
| **M3** | z + SOH | are the two complementary |

---

## 3. Data pipeline — the 146 TB problem

All of UYPYDJ is about **61 million samples**. Materialising windows the
way the current code does gives

```
61e6 × 200 × 3 × 4 B = 146 TB
```

Impossible. The current approach worked at 978 k windows (2.35 GB) because
the data was small.

### 3.1 Solution — keep the series, make windows a view

```
61e6 × 3 × 4 B = 732 MB      ← fits on the card as-is
```

Put the contiguous per-file arrays on the GPU, build a **copy-free view**
with `Tensor.unfold(0, 200, 1)`, and gather by batch index alone. A separate
global index of (file id, offset) keeps windows from crossing file
boundaries.

This is not a performance optimisation but a **feasibility** matter.

**Implemented — `analysis/windows.py::WindowSet`.** Measured:

| | Value |
|---|---|
| identical to the old `build_windows` | raw X·Y exact, max difference after scaling 0.000e+00 |
| UYPYDJ CC zip (5.78 M samples) | series 92 MB, 5.77 M windows (13.8 GB if materialised) |
| batch gather (5000 windows) | **0.80 ms** — about 1 % of the 82 ms LSTM step |
| group holdout | index split only, no series copy |

There was one trap in the scaler. The stored series contains samples no
window ever touches — the first W voltages of each file are never a target.
Fitting the scaler on the raw series changes the range relative to the old
path, and the measured difference in the scaled target was 1.7e-2. That is
not rounding: it is a **silent recalibration of the reported RMSE**. Fixed
to fit only on positions that actually occur (`covered_mask`).

### 3.2 Stride — adjacent windows overlap 99.5 %

At 1 Hz, adjacent windows share 199 of 200 samples. There is no reason to
use them all.

| Stride | Windows | Epoch time (est.) | 1000 ep |
|---:|---:|---:|---:|
| 1 | 61.0 M | 17 min | 12 days |
| 10 | 6.1 M | 104 s | 29 h |
| **30** | **2.0 M** | **34 s** | **9.5 h** |

(linearly extrapolated from the measured 16.7 s/epoch at 978 k windows)

**Start at stride 30.** Jittering the offset each epoch means the whole set
is eventually seen across epochs, so nothing is lost.

### 3.3 What goes into training

| Source | Role | Temperature | SOH |
|---|---|---|---|
| UYPYDJ drive cycles | main data on the aging axis | 25 °C | 1.00→0.69 |
| UYPYDJ HPPC | the high-current region (±34 A) | 25 °C | 1.00→0.69 |
| Mendeley drive cycles | the temperature axis | 6 values | fresh |
| RPCWBY Test#3 (2s/30s) | high current × temperature | 6 values | fresh |

Drive cycles do **not** visit the SOP current limit. The power distribution
measured in the aging campaign (whole cache, subsampled 37×):

| Power [W] | Drive cycle | HPPC |
|---|---:|---:|
| −140 to −100 | **0.000 %** | 1.355 % |
| −100 to −70 | 0.154 % | 3.566 % |
| +60 to +130 | 0.050 % | 3.789 % |
| peak discharge power | **−86.9 W** | **−130.3 W** |

Drive-cycle peak discharge power stops at 86.9 W. There are **zero** samples
with |P| > 100 W. Not rare — **absent**. SOP is by definition a quantity in
that region, so without HPPC there is no basis at all for learning SOP on
the aging axis. Same logic as the prior work including Test#3 for fresh
cells, and the deficit here is worse.

Test#3's **10 s runs are not used for training.** That is the prior work's
validation protocol and it is kept.

---

## 4. Held-out design — cut by cell

**One protocol is one cell.** Because of the cell-to-cell spread shown in
`findings.md` §4.1, validating on other cycles of the same cell means the
spread has already been seen and generalisation is overstated.

So **leave-one-cell-out**: train on five cells, validate on the whole sixth.
6-fold.

The question this answers is exactly the right one — *given an unseen cell
that has aged, can its SOP be predicted.* If M1 (SOH scalar) breaks here and
M2 (z) holds, §1's claim is demonstrated.

Temperature generalisation is handled separately, as before, by a
temperature holdout on Mendeley. There is no data that validates both axes
at once (§5).

---

## 5. What can and cannot be claimed

| | Verifiable | Basis |
|---|---|---|
| SOP across all SOH at 25 °C, SOC ≳ 0.33 | **yes** | UYPYDJ 6 cells, leave-one-cell-out |
| SOP at six temperatures, fresh | **yes** | Mendeley + Test#3 |
| **SOP of an aged cell at low SOC** | **no** | §5.1 |
| **SOP of an aged cell at low temperature** | **no** | no such file exists |

The fourth is a structural limit of the data (`findings.md` §1.1). The model
will emit a number, but that number is unvalidated extrapolation and is
marked as such in the results.

### 5.1 Low SOC × low SOH is simply not measured

The lowest SOC HPPC reaches, by SOH band (discharge 10 s, rated axis):

| SOH band | n | lowest SOC | 5th percentile |
|---|---:|---:|---:|
| 0.95–1.00 | 2080 | 0.056 | 0.090 |
| 0.90–0.95 | 2808 | 0.091 | 0.136 |
| 0.85–0.90 | 2808 | 0.149 | 0.183 |
| 0.80–0.85 | 2392 | 0.196 | 0.229 |
| 0.75–0.80 | 2236 | 0.228 | 0.275 |
| 0.65–0.75 | 2184 | **0.288** | 0.325 |

The HPPC protocol steps SOC down in *percent of capacity*, so on the rated
axis the same number of steps covers a progressively narrower band as the
cell ages. Drive cycles also bottom out at 0.087 and rise with age.

**SOP is tightest exactly at low SOC, and late in life there is no
measurement there at all.** All six cells behave the same way, so changing
cells does not fill it.

SOP predictions for SOC < 0.29 at SOH < 0.80 are therefore unverifiable
extrapolation. Those cells in the result tables are left blank or flagged as
extrapolated — there is precedent in this project for a "pass" verdict being
issued on one extrapolated grid cell and then reversed.

There is one mitigation: state explicitly the assumption that the
temperature dependence from Mendeley and the SOH dependence from UYPYDJ
**separate multiplicatively**, then at least check that the assumption holds
within the fresh data (is the ratio of R(SOH=1) across temperatures
constant?). Confirming it still leaves extrapolation into the aged range an
assumption, but a checked assumption is not the same as an unfounded one.

---

## 6. SOP validation reference

The problem: **there is no measured SOP for an aged cell.** The Test#3 SOP
runs are fresh only.

The fix: build a reference from HPPC. The already-extracted
`uypydj_hppc_resistance.csv` holds (SOH, SOC, τ, rate_rank) → R, so by the
standard ECM route

```
I*(τ) = (OCV(SOC) − V_min) / R(τ, SOC, SOH)
SOP(τ) = V_min × I*(τ)            (clipped by current and power ratings)
```

OCV comes per SOH from UYPYDJ's OCV_0.05C test.
**Extracted — `analysis/uypydj_ocv.py` → `uypydj_ocv.csv` (133 curves,
11,686 rows).**

The test runs 0.05C (±0.150 A) full → discharge → charge over 41 hours at
60 s intervals. The two legs are averaged on a common SOC grid to give a
pseudo-OCV — at 0.05C the IR drop is small and has **opposite sign** on the
two legs, so the average cancels it to first order.

**Why fresh OCV cannot be reused, quantified:**

| Fixed SOC | Correlation r | OCV, SOH 1.00 → 0.70 |
|---|---:|---:|
| 0.5 | 0.899 | **−149.8 mV** |
| 0.8 | 0.908 | **−157.1 mV** |
| 0.2 | 0.855 | −444.8 mV (see caveat) |

150 mV lands directly in the reference. For scale, the voltage model being
reproduced has error in the 40 mV range, so using fresh OCV plants a
**systematic error larger than the model error** in the reference.

**Two limits are stated explicitly:**

1. The −444.8 mV at SOC 0.2 is not a pure aging effect. An aged cell already
   reaches the voltage floor near there — the bottom of the OCV curve is
   0.020 above SOH 0.95 but 0.190 at SOH 0.70–0.80 (the same phenomenon as
   the gap in §5.1). The roughly −150 mV at SOC 0.5 and 0.8 is closer to the
   pure aging effect.
2. The pseudo-OCV average **weakens at low SOC.** Median hysteresis is
   108.9 mV over SOC 0–0.1, twice the 44–56 mV of other bands. The curve is
   steep there, so SOC alignment error is amplified into voltage difference.
   The reference in that band is marked less reliable accordingly.

This reference **depends on ECM assumptions and is therefore not ground
truth.** So two things are watched together:

1. **Direct metric (assumption-free)**: voltage prediction error on the
   held-out cell's HPPC pulses. That is the physical quantity SOP depends
   on, compared directly against measurement.
2. **Indirect metric (ECM-based)**: the difference between the SOP from the
   formula above and the model's binary-search SOP.

The first is the primary metric; the second is for interpretation.

### 6.1 Reference produced — `analysis/sop_reference.py` → `sop_reference.csv` (6,594 rows)

Using the textbook form `SOP = V_min · I*` directly is wrong in two places.

**First, there is not one resistance.** It differs by current level
(`findings` §3.2), so the solve is iterated self-consistently — use the R of
the measured rate nearest the current obtained, solve again, to a fixed
point.

**Second, which limit binds has to be decided first.** At mid SOC this cell
hits the **current rating (35 A) before voltage** (82.7 % of all rows).
There the terminal voltage sits well above the floor, so `V_min · I`
understates SOP. Computing the actual terminal voltage at the limiting
current handles both cases and reduces to the original formula when voltage
is the binding limit.

**Result (10 s, SOC 0.5, W):**

| SOH | 0.95 | 0.90 | 0.85 | 0.80 | 0.75 |
|---|---:|---:|---:|---:|---:|
| six-cell mean | 115.0 | 109.8 | 105.9 | 99.9 | 82.2 |
| **max/min across cells** | **1.00×** | 1.01× | 1.03× | 1.09× | **1.41×** |

The 2.41× seen in resistance eases to 1.41× in SOP (voltage headroom
buffers it). Even so, **knowing SOH pins SOP down only to within 41 %.**
§1's claim reappears in the final quantity of interest.

**Limits — flagged per row:**

In aged cells the cycler clamps the upper pulse currents while the rating is
35 A, so 84.8 % of rows involve rate extrapolation. To offset that, each row
also carries **`SOP_measured_floor_W`** — computed from the largest pulse
actually applied, so it contains no rate extrapolation at all (66.5–77.8 W
at SOH 0.75).

**Re-measuring with the ECM, though, shows the extrapolation matters
little.** The median highest measured current is 29.6 A, close to 35 A, and
fitting R(I) as a power law and extrapolating to 35 A moves R1 by +1.8 % and
R2 by −8.4 % (mixed signs, so not even a one-way bias). R0's current
dependence is 1.02×, effectively none. The "84.8 % extrapolated" flag counts
occurrences, not magnitude, and the magnitude is within 10 %.

---

## 7. Stages and compute budget

| Stage | Contents | Estimate |
|---|---|---|
| **S1** | window-view pipeline (§3.1), validated by reproducing existing results | half a day |
| **S2** | connect the UYPYDJ loader to the training path, pilot on one 25 °C cell | data layer **done**, training link remaining |
| **S3** | compare M0/M1 on one fold (one cell held out) | 9.5 h each |
| **S4** | implement the context encoder, M2/M3 on the same fold | 9.5 h each |
| **S5** | full 6-fold for the winning variant only | 6 × 9.5 h |
| **S6** | produce the SOP reference and evaluate the two metrics of §6 | half a day |

Running S5 for every variant is 4 × 6 × 9.5 h = 228 hours, which is not
possible. **Screen on one fold first and run the full set only for the
winner.** Choosing a variant from a single fold is itself a selection bias,
so only 6-fold results go into the final report.

S1 is the precondition — without it UYPYDJ cannot enter training at all.

### 7.1 S2 data layer, measured (`build_uypydj_cache.py`, `windows.load_uypydj_cells`)

| | Value |
|---|---|
| cache | one `.npz` per cell, 575 runs / **55.08 M samples** / 1.6 GB |
| windows (stride 30) | **1,830,526** — matches the 1.86 M design estimate |
| GPU residency | series 881 MB, gather 0.60 ms |
| per-cell balance | 231 k – 330 k windows (BOOST_REST lowest) |
| leave-one-cell-out | split verified with `by_group()` (1,504,790 / 325,736) |

Building the cache turned up eight files with a defective temperature
channel, handled by rule — `findings.md` §7.1. The pre-fix cache was
discarded and rebuilt.

---

## 7.6 Results — leave-one-cell-out, 6-fold complete (2026-08-18)

M0/M1/M2 were implemented as in §2.2 and run over all six cells. The metric
is drive-cycle voltage RMSE on the held-out cell; hidden 256×2, train on
five cells and validate on one, stride 60, HPPC fraction 0.5, milestones at
20 % of total epochs (printed at startup to avoid the trap in §5.2).

| Holdout | M1 (SOH scalar) | M2 (context z) | Gain |
|---|---:|---:|---:|
| BOOST | 23.32 | 22.41 | +3.9 % |
| CC_CELL2 | 25.37 | 22.59 | +11.0 % |
| BOOST_NEGPULSE_1S | 27.61 | 21.90 | +20.7 % |
| CC | 33.92 | 24.92 | +26.5 % |
| BOOST_NEGPULSE | 36.21 | 20.98 | +42.1 % |
| **BOOST_REST** | **52.45** | **24.43** | **+53.4 %** |
| **mean** | | | **+26.3 %** |

**M2 leads on all six cells.** No fold flips sign.

(For reference M0 — no conditioning input — was 53.44 mV on a single fold.
That age information is necessary at all is established first; M1 versus M2
is decided on top of it.)

### The decisive pattern: the worse M1 is, the more M2 recovers

M1 23.3 → 3.9 % gain, M1 52.5 → 53.4 % gain. And **M2 converges to
20.98–24.92 mV regardless of cell** while M1 spreads 23.32–52.45, a factor
of 2.2.

That is the claim itself. The less well a cell is captured by an SOH scalar,
the more information its recent response carries, and moving the
conditioning there removes the cell-to-cell performance spread.

BOOST_REST is the cell that has stood out five times in this project: the
only bad one in the SOH CNN (2.76 %p against 0.63–1.23 for the rest), the
minimum of the resistance spread (R1 4.79 mΩ), the only failure of the
hybrid arm's pooled gate (k_s 1.038), and the worst hybrid A0 at 148.9 mV.
It is worst in M1 too at 52.45 mV, and **M2 lifts it to 24.43, the level of
the other cells.**

### Against the pre-registered criterion

The criterion written down before running M2 was: "under 22 mV on a single
fold (13 % or better against M1) supports the claim; 22–27 mV is
undecidable and requires 6-fold." The first fold landed at 22.59, in the
undecidable band, so the 6-fold was run. **A mean of +26.3 % with 6/6
consistency** clears the bar. The bar was not moved to fit the result.

### A comparison not yet made

The hybrid arm (`sop_hybrid_spec.md` §7.5) is scored on **measured HPPC
pulse ΔV**, while this table is **drive-cycle voltage**. The two numbers
must not be placed side by side. A comparison on one axis exists only at
rung A13.

---

## 8. To check before starting

- [ ] whether the calibration-schedule reproduction run finishes and how
      close it lands to 21.54 mV (in progress). If it does not reproduce,
      the extension's baseline is unstable.
- [ ] rerun the baseline (drive cycles only) on the calibration schedule
      too — the present 45.61 vs 43.93 are both values from an error state.
- [x] **Checked — it does not cover it.** See §5.1. The lowest HPPC SOC is
      0.056 at SOH 1.00 but only 0.288 at SOH 0.70. Low SOC × low SOH is a
      measurement gap; that combination is marked unverifiable and work
      proceeds.

---

## 7.7 The size axis — how far it can shrink, and that the answer differs by cell

`/tmp/run_size.sh`. Below hidden 256 (§7.6), 128 and 64 were run on two
cells. CC_CELL2 is a middling fold; BOOST_REST is the fold where M1 was
worst in §7.6. Everything else matches the 6-fold hyperparameters.

| Hidden | Parameters (M1/M2) | CC_CELL2 M1 | M2 | BOOST_REST M1 | M2 |
|---:|---:|---:|---:|---:|---:|
| 256 | 1,057,793 / 1,078,729 | 25.37 | **22.59** | 52.45 | **24.43** |
| 128 | 266,753 / 284,105 | 26.49 | 26.91 | 51.22 | **28.20** |
| 64 | 67,841 / 83,401 | **27.02** | 30.12 | 57.27 | **31.39** |

### The value of M2 is set by the cell, not by the size

Looking at CC_CELL2 alone gives the conclusion "the context encoder costs
you once the model is small" — an 11 % win at 256 becomes a 1.6 % loss at
128 and an 11 % loss at 64. **BOOST_REST does not behave that way at all.**
Even at hidden 64 it wins 45 %, 57.27 → 31.39.

What separates them is M1's score. On cells an SOH scalar captures well
(M1 at 25–27 mV) the capacity the context encoder consumes is a net loss in
a small model; on cells it captures badly (M1 at 51–57 mV) it wins large
regardless of size. The observation from §7.6 — the worse M1 is, the larger
M2's gain — survives the size reduction.

### What this means for deployment

Judged on the worst cell, **M2 at hidden 64 (83,401 parameters) beats M1 at
hidden 256 (1,057,793)** — 31.39 against 52.45 mV. A 13× smaller model
improves the worst case by 40 %. A BMS has to guarantee the worst case
rather than the average, so that is the right comparison.

### Budget for an S32K344-class part (160 MHz Cortex-M7, 4 MB flash, 512 KB RAM, no accelerator)

The recurrent part is assumed to run streaming. This project already
confirmed that carrying state forward and replaying 200 samples each time
give 28.80 against 28.77 mV, effectively identical, so the 200-sample window
is a training-time construction rather than a runtime buffer.

| Hidden 64 | M1 | M2 |
|---|---:|---:|
| int8 weights | 66.3 KB (1.62 % of flash) | 81.4 KB (1.99 %) |
| state RAM (fp32) | 1.0 KB (0.195 %) | 1.2 KB (0.244 %) |
| 1 Hz streaming | 66,624 MAC/s (0.042 %) | 68,416 MAC/s (0.043 %) |
| one context-encoder pass | — | 2,573,312 MAC ≈ 16 ms |

**The recurrent part is not a budget problem.** Weights are 2 % of flash,
state is 0.25 % of RAM, compute is 0.04 % of the core. Even hidden 256 fits
at int8 1,053 KB, 25 % of flash — what binds is RAM, not storage, and even
that is 4 KB when streaming.

**The whole cost sits in one place, the context encoder.** One GRU pass over
200 samples is 2.57 M MAC, about 16 ms at one MAC per cycle. It cannot run
every step. Fortunately it need not — z summarises the cell's recent
response and does not change on a per-second scale, so it can be refreshed
every few tens of seconds with a cached z in between. That is the real
design decision when deploying M2, and how far the refresh interval can be
stretched before performance degrades has not been measured.

### Not yet measured

- Actual RMSE after int8 quantisation. The storage figures above are int8
  but the accuracy is from fp32 training. Quantisation loss was not
  measured.
- Degradation as the z refresh interval lengthens.
- The remaining four folds at hidden 64. The two cells disagreed in
  direction, so generalising the size-axis conclusion needs the full 6-fold.
