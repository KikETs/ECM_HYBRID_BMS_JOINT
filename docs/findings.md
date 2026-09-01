# Samsung INR21700-30T — data audit and reproduction status

> **[Historical — read spec §34–37 for current results]** This file is the
> design-era findings log. Its tables record what each configuration measured
> at the time, and most of those configurations have since been superseded:
> the SOH arm is ridge, not the CNN (§36); the trims were retrained
> deterministically (§37.5); the per-cycle cost and the deployment build both
> moved (§37.10). `.paper_state/paper_map.yaml` is the authority on which
> claims still hold, and `analysis/results/tables/` on what the numbers are.
> Numbers here are dated, not wrong.

> **[Audited 2026-08-27]** Six claims in this file and the spec did not
> survive an external audit: the zero-exceedance framing (§34.1), the SOC
> headline 2.05 %p (§34.2), the SOH CNN against ridge (§34.3), the
> aggregation choice (§34.4), the SOP labels called measured (§34.5), and
> the model that was actually on the board (§34.9). Each is marked where
> it was claimed. `.paper_state/paper_map.yaml` maps every claim to its
> status.

Written 2026-08-15. Subject: `~/바탕화면/DL/Samsung30T`.
The goal is to reproduce the prior work's SOP prediction model and extend
it to SOH-dependent SOP.

This document **separates what has been confirmed from what has not**.
Every number carries the script that produced it.

---

## 1. Data sources and what each provides

| Source | Path | Provides | Temperature | Aging |
|---|---|---|---|---|
| Mendeley `9xyvy2njj3` | `raw/Mendeley` (438 MB) | 72 drive cycles, HPPC, OCV, CC discharge | **6 values** (−20 to 40 °C) | none (fresh) |
| RPCWBY | `raw/RPCWBY` (1.7 GB) | Test#3 SOP pulses (2/10/30 s) | 6 values | none |
| UYPYDJ | `raw/UYPYDJ` (22 GB) | 6 fast-charge aging protocols, 279 HPPC runs | **25 °C only** | **SOH 1.00→0.69** |

All three come from the same McMaster lab, the same cell, and the same sign
convention (negative means discharge). Integrity of the nine zips confirmed
(5,531 files, 33.6 GB unpacked).

### 1.1 Temperature × SOH coverage — correcting the first version's claim

> **Correction.** This section originally asserted that "SOH and temperature
> live in different sources, so the low-temperature behaviour of an aged
> cell cannot be validated." **Wrong.** RPCWBY was judged from Test#3 alone;
> Test#1 and #2 of the same dataset give two temperatures across the whole
> aging range.

| Source | Temperature | SOH | Current |
|---|---|---|---|
| Mendeley HPPC | 6 values (−20 to 40 °C) | fresh | 4 levels |
| RPCWBY Test#3 | 6 values | fresh | SOP high current, 2/10/30 s |
| **RPCWBY Test#1 and #2** | **10 and 25 °C** | **whole range** | SOP high current |
| RPCWBY Test#8 | 0 °C | fresh | prior C-rate 0–4C |
| UYPYDJ | 25 °C | whole range | 4 levels |

The aged files of Test#1 (CC discharge profile) and Test#2 (US06 profile)
**swing between 9.9 and 25.6 °C chamber temperature within a single file**
(8.4 % of all samples below 15 °C), across cycles 22 to 2013.

```
JC_aging_chanel1_CC_Cycle22    chamber  9.9–26.9 °C   cell 11.0–38.8 °C
JC_aging_chanel1_CC_Cycle2013  chamber  9.9–25.2 °C   cell 10.9–48.1 °C
```

The cell's upper temperature rising from 38.8 to 48.1 °C is deepening
self-heating as resistance grows.

**So a temperature × SOH intersection exists, for two temperatures.** The
separability assumption for θ(SOC, SOH, T) can be *validated rather than
assumed*. Not all six temperatures appear in the aged range, so an aged cell
at −20 °C is still extrapolation, but the first version's "impossible" was
wrong.

> **Lesson:** a dataset was judged from one part of it (Test#3). One test out
> of eight in the archive was opened, and a limit on its capability declared.

---

## 2. The SOC axis — one convention for this project

```
SOC(t) = 1.0 + Ah(t) / 3.0        (rated 3.0 Ah, fixed)
```

Not divided by measured capacity (the capacity achieved at a given
temperature and age). Dividing folds temperature and SOH into the SOC axis,
where they cannot be separated again.

Verified:

| Source | Published SOC column | Verdict |
|---|---|---|
| RPCWBY Test#3 | inverted Q = 3.0000 Ah, R² = 1.000000 | rated basis — **usable as is** |
| UYPYDJ | inverted Q = that file's CAP (2.9717 / 2.7386 / 2.5976), R² = 1.000000 | aged-capacity basis — **conversion required** |

### 2.1 UYPYDJ's SOC column cancels SOH

The aging protocol swings between 10 and 80 % *of the current capacity*. So
on the published axis the full-charge point sits fixed at 79–80 % regardless
of age.

Regression of the full-charge point over 54 drive-cycle files
(SOH 1.000 → 0.874):

| SOC axis | change over SOH 1.00 → 0.87 | correlation r |
|---|---:|---:|
| published column | **−0.06 %p** | 0.033 |
| reconstructed on rated basis | **−10.03 %p** | 0.987 |

That is not obscuring; it is **cancellation**. With SOH-dependent SOP as the
goal, using this column as published makes the study impossible.

The conversion is exact and simple. End of discharge is set by the voltage
floor, so it is the same physical state at any age, and the two axes differ
only by the capacity ratio:

```
SOC_rated = (SOC_published / 100) × CAP / 3.0
```

Implemented in `analysis/data30t.py::load_uypydj_mat`, which also returns
the published value as `SOC_aged` so the conversion stays auditable.

Note: `Ah` cannot be used directly in UYPYDJ. As the readme states, the
counter resets at each charge and test, making it file-relative. (HPPC files
are the exception — they start at 0 from full charge and can be used
directly.)

---

## 3. HPPC resistance — the raw material for SOH-dependent SOP

`analysis/uypydj_hppc_resistance.py` → `analysis/uypydj_hppc_resistance.csv`
(57,680 rows, 279 HPPC runs, 6 protocols)

### 3.1 Pulse structure (checked, not assumed)

104 ten-second pulses per file = 13 SOC steps × (4 discharge + 4 charge).
Fresh discharge amplitudes −2.98 / −11.89 / −23.77 / −34.17 A (≈ 1C / 4C /
8C / 11.5C). Logging is 1 Hz, so resistance can be read at τ = 2 s and 10 s.

```
R(τ) = (V(t₀+τ) − V(t₀⁻)) / I_pulse
```

**Do not label C-rate by measured current.** At low SOC the upper currents
hit the voltage floor and the cycler clamps them, so what should be four
levels smears into 92 levels on the rated basis and 70 on the aged basis.
The within-protocol ordinal (`rate_rank`) is the label that survives
clamping.

### 3.2 Resistance growth with aging (CC protocol, 10 s discharge, SOC 0.5)

| SOH | cycle | rank0 (~1C) | rank1 | rank2 | rank3 (~9C) | rank0/rank3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.995 | 40 | 13.59 | 13.17 | 12.75 | 12.50 | 1.09× |
| 0.917 | 600 | 15.78 | 15.67 | 15.24 | 15.03 | 1.05× |
| 0.822 | 1350 | 29.07 | 26.99 | 24.69 | 23.46 | 1.24× |
| 0.738 | 1725 | 78.56 | 60.19 | 50.55 | 41.24 | **1.90×** |

(mΩ)

Two things read out of this:

1. **The growth is strongly non-linear** — it accelerates below SOH 0.85. A
   linear SOH–resistance model is wrong late in life.
2. **The C-rate dependence widens with age** — 1.09× fresh becomes 1.90× at
   SOH 0.74. **SOP cannot come from a single resistance.**

---

## 4. Central finding — SOH alone does not determine resistance

10 s discharge resistance of six cells at the same SOH (rank0, SOC 0.5, mΩ):

| Protocol | 0.98 | 0.92 | 0.85 | 0.80 | 0.75 |
|---|---:|---:|---:|---:|---:|
| CC | 13.72 | 15.61 | 23.48 | 35.97 | **64.08** |
| BOOST | 13.69 | 15.93 | 22.10 | 32.85 | 55.49 |
| BOOST_NEGPULSE_1S | 13.64 | 15.96 | 22.23 | 32.04 | 55.76 |
| CC_CELL2 | 13.74 | 15.40 | 20.12 | 27.39 | 40.45 |
| BOOST_NEGPULSE | 13.61 | 15.73 | 19.84 | 24.50 | 32.91 |
| BOOST_REST | 13.98 | 15.59 | 18.60 | 22.52 | **26.63** |
| **max/min** | 1.03× | 1.04× | 1.26× | 1.60× | **2.41×** |

They agree within 2–4 % down to SOH 0.92 and diverge below it.

### 4.1 The cause — the control group answers it

It is tempting to conclude a protocol effect, but **CC and CC_CELL2 run the
same protocol and differ only in the cell.** Those two differ by **1.58×**
at SOH 0.75.

| SOH | CC | CC_CELL2 | ratio |
|---:|---:|---:|---:|
| 0.90 | 17.40 | 16.39 | 1.06× |
| 0.85 | 23.48 | 20.12 | 1.17× |
| 0.80 | 35.97 | 27.39 | 1.31× |
| 0.75 | 64.08 | 40.45 | **1.58×** |

So a substantial part of the observed 2.41× spread is **cell-to-cell
variation**. With one cell per protocol, protocol effect and cell variation
cannot be separated — and they need not be, because the conclusion is the
same either way.

Not an interpolation artefact either: all six protocols have measured points
down to SOH 0.69, and the low-SOH range in the table is inside the measured
range.

### 4.2 Implications for the design

**An SOP model conditioned on an SOH scalar cannot be trusted below SOH
0.90.** If two cells at the same SOH differ by 1.6× in resistance, an output
conditioned on SOH is wrong by that much.

Paradoxically this is **evidence that the prior work's approach is right.**
It learns voltage directly from SOC/T/P, and the voltage response carries
the cell's actual state. Reading the cell's present response from recent
history beats bolting SOH on as one more feature. For the extension design
see `docs/soh_extension_design.md`.

### 4.3 ECM decomposition — which component the spread comes from

`analysis/uypydj_ecm.py` → `uypydj_ecm.csv` (27,891 rows, 2RC Thevenin)

Unlike the file's median 1 s spacing, HPPC pulses are sampled at about **101
points over 10 s, with 13 of them in the first second**. The fast time
constant is observed rather than extrapolated.

The order was chosen from residuals — across four fresh currents, 1RC gives
0.30–2.33 mV, 2RC 0.09–0.47 mV, 3RC 0.09–0.17 mV. 2RC is a 5× improvement
over 1RC, and 3RC adds only at the highest current, so 2RC is the default.

**Aging appears in polarisation, not in the ohmic term** (CC cell, discharge
rank0, SOC 0.5):

| SOH | R0 | R1 | τ1 | R2 | τ2 | sum |
|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 8.56 | 0.69 | 0.86 | 8.45 | 13.75 | 17.70 |
| 0.90 | 9.94 | 1.38 | 0.16 | 9.14 | 9.65 | 20.46 |
| 0.82 | 11.51 | 8.33 | 0.23 | 12.60 | 7.75 | 32.44 |
| 0.74 | 12.76 | 32.01 | 0.72 | 33.47 | 5.50 | 78.23 |

While R0 grows 1.6× and R2 4–5×, R1 goes from 4 % of the fresh total
(0.7 mΩ) to 31 % at SOH 0.70 (39 mΩ). That phrasing is more accurate than
the multiple (25–67×, depending on current level) — the fresh value is
nearly zero, which inflates the ratio.

#### 4.3.1 Cell-to-cell spread is almost entirely R1

| SOH | R0 | R1 | R2 | total |
|---:|---:|---:|---:|---:|
| 0.95 | 1.04× | 1.44× | 1.04× | 1.02× |
| 0.85 | 1.05× | **3.60×** | 1.04× | 1.19× |
| 0.75 | 1.06× | **5.58×** | 1.91× | 2.16× |

**The ohmic resistance of all six cells agrees within 1.05× everywhere.**
The spread of §4.1 comes from charge-transfer resistance alone — 4.79
(BOOST_REST) to 26.75 mΩ (CC) at SOH 0.75.

This is the physical explanation of §4.2. Capacity fade and R1 growth follow
different degradation mechanisms — capacity reflects active-material and
lithium-inventory loss, R1 reflects the interface (SEI growth, surface
films). So one SOH number does not fix SOP.

It also makes concrete **what the context encoder has to capture.** R1's
time constant is short, 0.15–1.05 s, so it appears in the voltage response
as soon as load is applied and is observable from recent (V, I, T) history.

#### 4.3.2 Validation

- **Uniqueness of the decomposition**: refitting from four initialisations
  (τ seeds scattered 0.1–50 s) converges to the same solution for fresh,
  mid-life and aged. The R1 surge is not the two RC branches trading roles.
- **Agreement with an independent route**: Reff(10 s) reconstructed from the
  ECM against direct measurement gives **r = 0.99999**, median relative error
  0.22 %, deviation within 0.2 % across the whole SOH range (2,355 pairs).
- Median residual 0.466 mV, 95th percentile 2.12 mV. The 1,124 fits (3.9 %)
  with RMSE above 5 mV are excluded.

> **Analysis caveat:** the first pass took medians with `rate_rank` mixed
> and reported the R1 growth as 44×. Resistance differs by current level, so
> rank has to be held fixed (fixed: rank0 67×, rank3 25×).

### 4.4 The low SOC × low SOH measurement gap

The lowest SOC HPPC reaches (rated axis) rises with SOH:

| SOH | 0.95–1.00 | 0.90–0.95 | 0.85–0.90 | 0.80–0.85 | 0.75–0.80 | 0.65–0.75 |
|---|---:|---:|---:|---:|---:|---:|
| lowest SOC | 0.056 | 0.091 | 0.149 | 0.196 | 0.228 | **0.288** |

The protocol steps SOC down in *percent of capacity*, so on the rated axis
the same number of steps covers a progressively narrower band. All six cells
behave identically.

SOP is tightest at low SOC, so **the combination SOH < 0.80 with SOC < 0.29
has no data.** Predictions in that region must be marked extrapolated.

---

## 5. Status of the prior-work reproduction

### 5.1 Results

| | best val RMSE | epoch | note |
|---|---:|---:|---|
| baseline (drive cycles only) | 45.61 mV | 260 | schedule in an error state |
| full model (+SOP 2s/30s) | 43.93 mV | 160 | schedule in an error state |
| rerun (corrected schedule) | in progress | — | `--epochs 1000` |
| paper | 21.54 mV | — | 4 × A100, about 6 hours |

### 5.2 43.93 mV is not a converged value

The paper's MultiStepLR milestones are **at every 20 % of total epochs**,
with a budget of 5000 epochs. So lr = 1e-4 holds for the first 1000 epochs.

Running the same recipe with `--epochs 300` pulls the milestones to
60/120/180/240, reaching lr 1e-8 by epoch 240. The first run stopping at
44 mV was not convergence but **the optimiser dying.**

Early stopping was structurally unable to fire, too: patience 100 × a
validation interval of 10 epochs means 1000 epochs without improvement,
against a budget of 300.

> **Generalised lesson:** hyperparameters defined as a fraction of total
> epochs shrink with the run. Always check this before trusting an abridged
> reproduction.

After the fix the defaults are `--epochs 1000 / --patience 20`, and the
schedule and early-stopping conditions are printed at startup.

### 5.3 Two implementation errors that were blocking the reproduction

Even after fixing the LR schedule it stalled at 35.45 mV. Comparing against
the paper's specification line by line found **two errors on my side.**

#### (1) The FC head was 3.8× too small

Two statements in the paper contradict each other.

| Source | output-layer hidden units | head parameters |
|---|---|---:|
| Table 1 | "2⁸, 2⁴" = (256, 16) | 69,921 |
| Table 4 (parameter counts) | **(512, 256)** | **263,169** |

The parameter counts are right. With the head at (512, 256) all three
architectures match **exactly** — RNN 461,057 / GRU 858,369 / LSTM
1,054,721. Six- and seven-digit numbers do not coincide three times by
chance, so Table 1's "2⁴" is a typo.

(An incidental internal inconsistency: the RNN and LSTM entries are counted
with one input, the GRU entry with three. The text states three inputs —
SOC, temperature, power — so three was used.)

#### (2) The target was off by one step — the dominant cause

From the paper:

> "at each time step 𝑘 ... The model then estimates the battery voltage by
> integrating information from both the **present** measured inputs and memory
> states."

The label is the voltage at the **last instant of the input window**,
`V[i+W-1]`. My implementation predicted `V[i+W]`, one step later.

That is a completely different problem. Terminal voltage responds almost
instantly to power through the ohmic drop, so predicting one step ahead
means predicting **without having seen the power step that caused the
voltage change.**

Measured evidence:

| Cycle | RMS(1 s voltage change) | observed error | ratio |
|---|---:|---:|---:|
| HWFET | 26.2 mV | 23.3 | 0.89× |
| UDDS | 34.6 mV | 31.4 | 0.91× |
| LA92 | 45.1 mV | 42.5 | 0.94× |
| US06 | 62.9 mV | 45.2 | 0.72× |

Correlation r = 0.93, and the cycle ranking matches. The **error was the
unpredictable component itself.**

The clearest check:

| mean RMSE over 24 conditions | |
|---|---:|
| persistence predictor (Vₖ = Vₖ₋₁) | 42.22 mV |
| my model (one step ahead) | 35.62 mV |
| paper's LSTM | 21.54 mV |

**An 860 k-parameter LSTM was only 16 % better than the trivial answer
"voltage stays the same."** The model had not failed to learn physics; the
information to learn from was not in the input.

The fix was applied in `lstm_voltage.py` (two places), `eval_voltage.py` and
`windows.py`, and the label was confirmed to match the window's last sample.

> **Lesson:** when a reproduction fails, the training setup was the first
> suspect, but the real cause was the **task definition**. Comparing against
> a trivial baseline (the persistence predictor) first would have caught it
> much sooner. When a model fails to clearly beat a trivial baseline,
> suspect the problem statement rather than the hyperparameters.

### 5.4 Error by temperature (pre-fix results, therefore provisional)

Broken out over 24 conditions (4 named cycles × 6 temperatures):

| | −20 °C | 40 °C | ratio |
|---|---:|---:|---:|
| baseline | 72.4 mV | 27.9 mV | 2.6× |
| +SOP | 57.3 mV | 39.8 mV | 1.5× |

The SOP data improves the cold end by 19–21 % and degrades the hot end by
27–43 %. The likely cause is SOP windows outnumbering drive-cycle windows
6.4 : 1, and `--sop-ratio` was added to cap the ratio. Needs reconfirming on
the corrected schedule.

---

## 6. Training-loop performance

Measured (978,282 windows, 196 batches per epoch, one RTX 5090):

| | epoch | 1000 ep |
|---|---:|---:|
| original code (CPU tensors, `.item()` every batch) | 18.1 s | 5.0 h |
| training tensors resident in VRAM, syncs removed | **16.7 s** | **4.6 h** |

Numerical identity verified: train and val match the original exactly at
epochs 1, 10, 20 and 30.

**Precision was not touched.** bf16 cuts to 8.1 s/epoch but changes the
numbers, so it is off by default; TF32 is off for the same reason (with it
on, epoch 1 val moves 207.86 → 219.23 mV). The bottleneck is the 200-step ×
2-layer LSTM computation itself rather than data movement, so with precision
excluded there is not much more to take out.

---

## 7. Known data defects

| Source | Defect | Handling |
|---|---|---|
| Mendeley | `meas.Wh` is uint8 with values 44/45 — cannot be watt-hours | not read; integrate Power if needed |
| RPCWBY Test#3 | two byte-identical duplicate pairs, `-10degC` / `n10degC` | SHA256 de-duplication |
| RPCWBY Test#3 | one cell of the SOC column in `SOP_30T_July_14_-20degC_10s` holds the string `'s'` (1 row of 38,173) | set NaN, record and report the position |
| UYPYDJ | HPPC files lack SOH/CAP/SOC fields | interpolate linearly from drive-cycle files by cycle number; **exclude rather than extrapolate** outside the range |
| UYPYDJ | published SOC is on the aged-capacity basis | conversion in §2.1 |
| UYPYDJ | **8 of 583 drive cycles have a dead temperature channel** — 2 have `T` as a 0-dimensional NaN scalar, 6 oscillate near 0–10 °C throughout (or for the first 20 %) | rule-based discard and masking (§7.1) |
| UYPYDJ | 7 drive-cycle runs (1.16 % of samples) have SOH missing | interpolate from neighbouring runs by cycle number |

### 7.1 The temperature-channel defect — a threshold alone was not enough

Found by exhaustive scan (the first attempt, which looked only at the first
few files, missed all of them):

| Cell | cycle | Symptom |
|---|---|---|
| BOOST_NEGPULSE_1S | 7, 24 | `T` is a 0-dimensional NaN scalar rather than an array — this crashed the first cache build |
| CC | 1388, 1405 | `T` NaN throughout |
| CC | 1501, 1518 | `T` oscillates over −0.57 to 0.43 (718 distinct values) |
| BOOST | 1463, 1480 | same |
| BOOST_NEGPULSE | 488, 505 | same but only over the first ~20 %; the rest is 25–33 °C |

Voltage and current are normal throughout, so the runs themselves are real
measurements that lost only the temperature channel.

**A dead thermocouple does not read zero; it wanders.** The dead stretch of
cycle 488 moves between 9.71 and 10.36 °C, so a 10 °C floor cuts through the
middle of the defect and passes 398 samples as valid. Tightening the number
does not fix it — the genuine minimum across all six cells is 16.4 °C
(CC_CELL2), leaving almost no margin.

So **300 seconds on either side of an invalid sample are invalidated too**
(mask dilation). Values next to a failed sensor cannot be trusted whatever
number they show, and this removes any need to guess where the defect ends.
After applying it, BOOST_NEGPULSE's temperature floor rises from 10.0 to
17.0 °C, putting all six cells above their genuine minima. The cost is 0.6 %
of that cell.

---

### 7.2 Exhaustive temperature audit (2026-08-24) — where 7.1's mask did not reach

7.1 found the temperature-channel defects and built a mask, but it applied
**only to the drive cache (cache_t)**. The HPPC resistance table has no
temperature column at all, so the same filter could not be applied, and as a
result labels measured at 4 °C were setting the discharge SOP safety factor
(`sop_hybrid_spec.md` 26.5). So the raw data was scanned again in full
(`analysis/temp_audit_all.py`, `temp_audit_all.csv`).

**Scope**: 5,020 raw UYPYDJ files. 3,735 of them have a temperature channel;
1,285 are a different format without a `meas` struct (CAP and so on) and
have no temperature to begin with — not defects. The criterion is 7.1's:
outside 15–45 °C, or non-finite.

**Result: 66 defective files.**

| Cell | scanned | defective | total | partial |
|---|---:|---:|---:|---:|
| CC | 625 | 23 (3.7 %) | 23 | 0 |
| BOOST_NEGPULSE_1S | 680 | 24 (3.5 %) | 24 | 0 |
| BOOST | 625 | 11 (1.8 %) | 11 | 0 |
| **BOOST_NEGPULSE** | 603 | 5 (0.8 %) | 1 | **4** |
| CC_CELL2 | 721 | 3 (0.4 %) | 3 | 0 |
| BOOST_REST | 481 | **0** | 0 | 0 |

**They cluster in three blocks.**

1. Cycles 1383–1518 (CC) and 1458–1480 (BOOST) — the thermocouple died near
   0 °C (−0.85 to 0.02). The two cells are close in time, which reads as a
   chamber or instrumentation event.
2. Cycles 1–24 and 1314–1335 (BOOST_NEGPULSE_1S) — −222.9 / −199.8 /
   −100.0 °C. This is the cell where 7.1 caught some as "0-dimensional NaN
   scalars."
3. Cycles 483–505 (BOOST_NEGPULSE) — **the only four partial defects.** The
   median is a normal 25.6–26.1 °C with only part of the record at 0–4 °C,
   which is why no threshold caught them.

**Partial defects occur only in this one stretch.** The other 62 are total
and any filter catches them. What 7.1 missed is 483–505 alone; nothing else
is hiding.

**BOOST_REST is clean across all 481 files** — evidence that the defect is
an equipment problem in a particular period rather than a cell-specific one.

**And all six cases flowed into the downstream tables.** The guess that "if
the temperature channel dies, SOH and SOC will not come out and it will drop
out upstream" was wrong — voltage and current stay normal when temperature
dies, so the resistance calculation proceeds.

| Table | rows | defective rows | share |
|---|---:|---:|---:|
| `uypydj_hppc_resistance.csv` | 58,909 | 1,236 | 2.10 % |
| `uypydj_ecm.csv` | 28,486 | 593 | 2.08 % |
| `sop_label_measured.csv` | 7,406 | 156 | 2.11 % |
| `sop_label_charge.csv` | 6,946 | 146 | 2.10 % |

Among trustworthy labels (extrap ≤ 1.5) that is 43 discharge rows (2.6 %)
and 114 charge rows (2.4 %).

**Impact**: the point that sets the discharge lambda (τ = 10 s) is
BOOST_NEGPULSE#487 (6 of 12 combinations), and dropping it at the evaluation
stage relaxes 0.693 → 0.715. **Charge is unaffected** — its deciding point
is BOOST_NEGPULSE_1S#1881, whose temperature is normal.

**Open**: the ECM table and the trim were not rebuilt from upstream with
those six cycles removed. The present numbers exclude them at the evaluation
stage only.

### 7.3 Pipeline rebuild (2026-08-25) — and a prediction that missed

The six temperature defects found in 7.2 were filtered **at the extraction
stage** and everything rebuilt from upstream. `analysis/temp_defects.py`
reads the defect list from `temp_audit_all.csv`, and
`uypydj_hppc_resistance.py` / `uypydj_ecm.py` / `uypydj_ocv.py` skip them.

Whole cycles are discarded. The partial defect (BOOST_NEGPULSE#487, 19.6 %
invalid) sits only in the low-SOC range and half of it could be salvaged,
but it is not — 7.1 established that a dead thermocouple wanders rather than
reading zero, so the boundary of a defect cannot be trusted. Discarding
2.1 % is cheaper than guessing where the boundary lies.

| Artefact | before | after |
|---|---:|---:|
| `uypydj_hppc_resistance.csv` | 58,909 | **57,673** |
| `uypydj_ecm.csv` | 28,486 | **27,893** |
| `uypydj_ocv.csv` | — | 11,702 (one temperature defect excluded) |
| `sop_label_measured.csv` | 7,406 | **7,250** |
| `sop_label_charge.csv` | 6,946 | **6,800** |

The pool cache was cleared and the trim dataset and trim rebuilt in both
directions (`runs_trim_v2`, `runs_trim_chg_v2`). The trim's own score holds
— discharge +31.2 %, charge +31.6 % (before the rebuild, +33.6 / +34.0 %).

**One trap on the way.** `sop_label.py`'s `--out` default was a single
value independent of direction, so a `--direction charge` run **overwrote
the discharge label file entirely**, and inspection found 6,800 charge rows
inside `sop_label_measured.csv`. Fixed by making the default output
direction-dependent.

**Result: the safety factor does not relax.**

| | optimism | RMSE | worst | usable current | lambda |
|---|---:|---:|---:|---:|---|
| discharge, before | 61.1 % | 5.31 A | 0.89 A | 69 % | 0.693 / 0.489 |
| discharge, after | 69.4 % | 5.43 A | 1.19 A | 70 % | **0.679 / 0.462** |
| charge, before | 54.4 % | 2.23 A | 0.75 A | 59 % | 0.598 / 0.566 |
| charge, after | 61.7 % | 2.19 A | 0.58 A | 57 % | **0.567 / 0.544** |

The ECM is effectively unchanged (optimism 84.8 → 84.3 %, lambda the same).

**The prediction missed.** Excluding the defective rows at the evaluation
stage alone relaxed the discharge lambda 0.693 → **0.715**, so an upstream
rebuild was expected to lock that gain in. It **tightened to 0.679**
instead.

The reason is that the trim retrains. Excluding at the evaluation stage
removes labels while holding the trim fixed; a rebuild retrains the trim on
1,236 fewer rows as well. Change the training data and the trim changes, and
the extreme point that sets the safety factor moves with it. **"Remove the
defects and gain that much" does not hold** — the extreme point is not a
property attached to particular rows; a different row takes the role each
time.

**It is not reverted, though.** Labels measured at 4 °C sitting in a table
premised on 25 °C is wrong in itself, and the gap where 7.1's mask applied
only to driving is now filled. But the conclusion of `sop_hybrid_spec.md`
§26 stands — most of what the 41 % gives up is neither a data defect, nor a
model defect, nor insufficient conditioning.

**Runtimes (for whoever runs this again)**: resistance table ~12 min, ECM
table ~75 s, OCV ~50 s, **trim dataset 10–12 min per cell (~130 min for
both directions)**, trim training ~11 min per direction. The long pole is
dataset construction rather than training — each cell's whole drive record
is swept, simulating the pooled ECM at every sample, with three
`LinearNDInterpolator` calls per sample.

## 8. Open (as of 2026-08-15; updated in §10 below)

- whether the corrected-schedule rerun approaches 21.54 mV → **closed**: it
  stopped at 21.90 mV, with the remaining 0.36 mV concentrated at 25 and
  40 °C (`fig_repro_conditions.png`)
- the baseline also needs rerunning on the corrected schedule → closed
- best value for `--sop-ratio` → fixed at 0.5
- extension model design and validation protocol →
  `docs/soh_extension_design.md`

---

## 9. Adopted baselines (2026-08-22)

All three arms have been evaluated leave-one-cell-out. Adoption status:

| Status | Arm | Model | Score | Basis |
|---|---|---|---|---|
| **adopted** | SOP | hybrid linear **A8 (4 effective coefficients)** | usable current **69.6 %** discharge / **59.5 %** charge, 1 exceedance each | `sop_hybrid_spec.md` §32.7, §33, **§34.1** |
| **adopted** | SOH (charge) | **dQ/dV ridge (65 coefficients)** — the CNN is a baseline, §36 | SOH RMSE **0.0094**, bias −0.0005, worst cell 0.0130; 1D CNN 0.0135 / 0.0291 | `soh_extension_design.md`, §30.12, **§34.3** |
| held | SOH (driving) | (V,I) CNN (16,241) | **0.0191** per file | per-cell bias unresolved |
| reference | voltage | LSTM M2 (1.08 M) | **21.75 mV** on drive cycles | for conditioning comparison |

### Why the SOP arm is hybrid

On the same pulses, the same metric and equal information (both receive
current only), it beats 5 of 6 cells. Mean 44.8 against Full AI M2's
58.7 mV. What decides it is the **worst cell** — the hybrid never exceeds
1.4× its mean (62.6) while pure AI reaches 1.7× (99.6), and in the direction
that calls available power too high.

A 514-parameter MLP did not beat the 26 (45.4 against 44.8). Parameterisation
is doing the work, not capacity.

### Why there are two SOH arms

- **Charge-curve CNN, 0.0128** — accurate, but it produces a value only when
  there is a 1C charge passing through 3.55–4.05 V. Removing the magnitude
  information (area normalisation) holds it at 0.0122, so what carries the
  estimate is the **shape** of the incremental-capacity curve rather than
  partial capacity.
- **Driving-window CNN, 0.0191** — always available, but a per-cell bias of
  −1.4 to +1.6 %p remains and rotates with cycle (error-versus-cycle
  correlation −0.54 to +0.80). The absolute value cannot be trusted, so it is
  held. It is not a time-axis shortcut — the error-versus-position-in-file
  correlation is clean at −0.24 to +0.07.

### A defect common to every arm

**Current sensitivity is 0.62–0.67 × the physical expectation (I·R0).** The
sign is right in 100 % of all 12 combinations, but the magnitude is 33–38 %
low. Run an SOP binary search on it and it overstates the current needed to
reach the voltage limit, and therefore **answers optimistically about
available power.** Reproduced with P input and I input, in M1 and M2, and in
all six cells. It is the most robustly confirmed defect in this project and
the first thing to check when validating SOP in watts.

**SOP was measured in amperes (2026-08-22).** The V-I characteristic from
HPPC's four current levels was solved to V_min to make measured labels
(`sop_label_measured.csv`, 7,406 rows). Hybrid **4.94 A** against the
uncorrected ECM's 7.26 A, ahead on all six cells (+31.9 %). Rows optimistic
by more than 5 A fall 20.5 % → 8.2 %, so the tail improves more than the
RMSE. **The reference LSTM's binary search does not converge** — M1's slope
bends 6× flat beyond s > 2 and M2 flips sign to give 4.589 V. Details in
`sop_hybrid_spec.md` §11.

**And what holds SOP back is SOC estimation, not resistance correction
(2026-08-22).** The 4.94 A under oracle state becomes 14.23 A, three times
worse, with only a systematic 2 % SOC — larger than the hybrid correction's
gain (2.3 A). The cause was the missing Plett hysteresis term in the EKF
measurement model (residual-versus-h correlation +0.81 at SOH 0.69), and
adding a deterministic h improved SOC RMSE 0.0344 → 0.0261 (−24 %), and
−33 % over SOC 0–0.4 where SOP sensitivity peaks. See `ecm_kf_plan.md`.

What follows was written before that work.

**And SOP has never been measured in watts.** Every number is in mV.
`sop_reference.csv` is 84.8 % extrapolation beyond the measured range and
82.7 % clipped at the current ceiling, so it cannot serve as a label. HPPC
measured four current levels for every (cell, cycle, SOC group), so solving
the V-I line to V_min gives an I* label that does not pass through the ECM —
that is the next step.

---

## 10. Open items (as of 2026-08-23)

The document was swept to remove stale statements and keep only what is
genuinely open.

### Answerable by measurement

| Item | Status | Where |
|---|---|---|
| the **cause** of the 0.62–0.67 current sensitivity | unexplained — the training-current-distribution hypothesis was not checked | `sop_hybrid_spec.md` §7.6.6 |
| reading the slow branch during a **rest → load transition** | not attempted (only sustained load, which failed) | §13.7 |
| **actual RMSE** after int8 quantisation | **measured (2026-08-25)** — 6-fold 0.0128 → 0.0129, flash −61 %, speed +8.4 % | `sop_hybrid_spec.md` §27.9 |
| lengthening the context-z **refresh interval** | not measured — 16 ms cannot run every step | §7.7 |
| the **remaining 4 folds** at hidden 64 | only 2 cells run, and they disagreed in direction | §7.7 |
| **per-cell bias** of the driving SOH arm | held — −1.4 to +1.6 %p rotating with cycle | §9 |
| **two cells losing to the ECM on charge** | unresolved (−9 %, −31 %), both cells where the ECM is already good | `sop_hybrid_spec.md` §16.2 |
| the **−0.76 A bias** in discharge labels | sign and magnitude match an extrapolation bias, but not directly verified | §16.4 |
| the **temperature axis** | everything is 25 °C. Mendeley's six temperatures and RPCWBY Test#3 exist | — |
| discharge at **τ = 2 s** | zero interpolated rows. RPCWBY Test#3 has 2/10/30 s | §11.6 |

### Not answerable with this data

- **Six cells, one per protocol.** Repeats of the same protocol (CC versus
  CC_CELL2) diverge 15 % over life, so anything smaller than that cannot be
  distinguished.
- **Whether the LSTM's inversion failure is architectural or a matter of
  training distribution.** Adding loads above 30 A to training and inverting
  again would separate them, but that data exists only as HPPC pulses, which
  is a narrow distribution.
- **The residual 2.19 %p SOC error in aged cells.** Nine attempts, and all
  seven that tried to fix the model failed (`ecm_kf_plan.md`).

### 10.x The temperature axis (2026-08-23)

- **Temperature generalisation of resistance is validated.** On an external
  cell, 6 temperatures × 3 horizons over a 9× resistance range, the
  predicted/measured median is 0.908 with median relative error 13.7 %
  (559 rows). τ = 2 s and 30 s are validated for the first time.
- **Per-temperature SOP labels cannot be made.** Test#3 is an SOP search, so
  it converges, and on the warm side 30 A does not reach 2.55 V (SOP there is
  60–100 W).
- **The authors' measured SOP was scored, though.** On 342 voltage-limited
  rows the pooled ECM sees **21 % more usable current** (RMSE 6.44 A). That
  is the dangerous direction and it deepens with age. The source is
  resistance and a factor of 1.29 is needed — inside the trim's range
  (up to 1.60×).
- **Open**: standing up the trim feature pipeline on Test#2 (US06 driving
  with measured SOP, 10 and 25 °C) would let the hybrid be scored on an
  external cell against measured SOP. That is the strongest validation
  possible in this project and it has not been done.

### 10.y Asymmetry of SOP error (2026-08-23)

- **RMSE was hiding the direction.** On trustworthy labels (extrap ≤ 1.5,
  651 rows) the hybrid is **82.9 % optimistic** (calls the current higher
  than it is). The 42.1 % over all 5,995 rows is diluted by inflated
  high-extrapolation labels (median |I*| 95.6 A = 32C).
- **τ = 2 s breaks.** The median required R multiplier is 1.076 at
  τ = 10 s against **1.323 at τ = 2 s**, and the 90th percentile of 1.843
  exceeds the trim's k_f ceiling of 1.60. τ = 2 s has never been validated
  by interpolation in this dataset (the 143 rows with extrap ≤ 1.0 are all
  τ = 10 s).
- **Three independent tests point the same way**: internal 1.109, RPCWBY
  authors' SOP 1.29, Test#3 resistance 0.908. The pooled ECM reads resistance
  low.
- **On measured pulses it is unbiased, though** (ratio 0.98–1.08). The
  difference is the extrapolation convention beyond the fan — the label uses
  a chord, the model a tangent. Switching the label to lin2hi moves 82.9 →
  73.9 %, so 9 %p is that.
- **What fixed it**: (1) margin **per horizon** (usability 0.691 → 0.794),
  (2) aggregating the history snapshots with **max** — optimism 82.9 → 61.1 %
  without retraining, (3) an optional pinball quantile loss.
- **Recommended**: Huber + max aggregation + per-horizon derate. Targeting
  5 % gives an actual 5.1 %, worst 5.62 A, usability 0.826 — better safety
  and more output at once. The tighter the target, the larger the advantage.
- **mV is not the selection criterion for SOP**: mV RMSE is minimised at
  q = 0.50, ampere RMSE at q = 0.90.

### 10.z SOP margin does not cross datasets (2026-08-24)

- **Exceedance remains.** A 5 % target is exceeded as targeted, at 5.1 %
  (worst 5.62 A). Driving it to zero takes usability from 0.826 to 0.620 —
  38 % of the predicted current has to be thrown away.
- **It transfers between cells and not between labs.** On the six UYPYDJ
  cell holdouts, aiming at 5 % yields 5.1 %. Applying the same margin to the
  RPCWBY external cell (53 rows, 10 s) gives **13.2 %**. Without any margin
  92.5 % is exceeded.
- **So deployment needs a recalibration procedure.** "Exceedance can be held
  near target" holds only within one dataset.
- Blind spots: 651 of 5,995 rows are scoreable, τ = 2 s has no interpolated
  validation at all, everything internal is 25 °C, and the charge direction
  was not examined.

### 10.aa The hybrid validated on two external cells (2026-08-24)

- **Recomputing the features from the RPCWBY cells' own aging drives** scored
  the trim entirely from outside for the first time. Only the trim weights
  come from UYPYDJ; everything else is a different lab.
- **Error and tail halve on both cells.** Test#2 (US06 driving) RMSE
  4.32 → 1.76–2.26 A, worst overshoot 10.05 → 3.84–5.26 A. Test#1 (constant
  18 A) 6.92 → 3.57–3.97 A, 13.97 → 4.70–7.23 A.
- **The trim finds the required correction on its own**: median k_f
  1.14–1.28. §18.6 measured the required multiplier on this data as 1.29. The
  trim has never seen that label.
- **It works under constant-current excitation too** — expected to break
  without a drive cycle, and it does not.
- **The model transfers and the margin (lambda) does not.** Applying UYPYDJ's
  lambda directly exceeds on 16.1 % of Test#1. Matched on usability, the
  hybrid is equal or better on both cells (0.0 % against 14.3 % at 0.85;
  3.2 % against 6.5 %).
- Limits: n = 14/31, effectively a single point at 10 °C, all τ = 10 s.

### 10.bb The zero-overprediction configuration (2026-08-24)

- Setting the calibration target to **zero** rather than 5 %: internal
  holdout exceedance 3/651 = 0.46 % (95 % upper bound 1.19 %), worst
  overshoot 5.62 → **0.89 A**, usability 0.826 → 0.688.
  lambda(τ = 10 s) = 0.693, lambda(τ = 2 s) = 0.489.
- **That lambda transplants to both external cells unchanged**: 0/14 and
  0/31, worst 0.00 A, usability 0.740 / 0.761. (The 5 %-target lambda did not
  transplant — §20.4.)
- **At the same zero-exceedance standard the trim buys back 10–14 %p of pack
  peak current** (ECM 0.594/0.599/0.663 against hybrid 0.688/0.740/0.761).
  That, not RMSE, is the deployment argument.
- **τ is the right axis here too.** Adding SOH gains 2 %p of usability while
  the worst overshoot blows up 0.89 → 7.84 A (thin cells make lambda loose on
  a new cell).
- **The trim's spread across 12 blocks is not an uncertainty measure**:
  corr −0.087, and larger spread is if anything less optimistic. Rejected.
- Limits: "zero" is an observation, not a guarantee (95 % upper bound
  9.2–19.3 % externally). The real guarantee comes from undervoltage cutoff.
  External validation is all τ = 10 s and effectively one point at 10 °C.

### 10.cc The temperature axis (2026-08-24)

- The SOP summary workbook's **Test#3 sheet had never been parsed** (the
  parser only handled Test#1/#2). Reading it, the six temperature points
  exist **only at τ = 10 s**; τ = 2 s and 30 s are 25 °C only.
- **At −20 and −10 °C, all 14 rows are voltage-limited and cover the whole
  SOC 0.02–1.00 range** — the first case of voltage-limited SOP covering the
  entire SOC axis.
- **The margin depends on temperature**: the zero-exceedance lambda is 0.623
  at −20 °C, 0.657 at −10 °C, 0.810 at 0 °C, 0.808 at 10 °C. Monotone, and
  the 25 °C extrapolation is 1.02.
- **§21's lambda of 0.693 breaks below −10 °C** (4/10 exceed). **A single
  0.623 gives zero exceedance across all four datasets** — internal 0/504,
  Test#2 0/14, Test#1 0/31, Test#3 cold 0/26. The cost is internal usability
  0.697 → 0.627 (7 %p).
- **The SOH choice swings the conclusion 4×**: using block capacity as SOH
  flips the multiplier 1.121 → 0.275, reversing the sign. The adopted basis
  is (a) capacity moves up and down over time and therefore cannot be aging,
  and (b) g_temp already carries temperature, so putting it into SOH as well
  double-counts.
- **τ = 2 s cannot be measured with this equipment (established).** Digging
  into the raw files, only two SOC groups near SOC ≈ 0.05 bracket 2.55 V. The
  reason is physical — a short horizon has low resistance and needs more
  current to reach the floor, and at 25 °C SOC 0.5 the I* for τ = 2 s is
  **108.7 A** against a 30 A cycler. §21's lambda(τ = 2 s) = 0.489 remains
  unvalidated and conservative.
- Incidental: the existing `rpcwby_temp_pulses.csv` was discarding almost all
  search pulses because of a `MIN_REST_S = 20 s` gate (a search has a median
  2 s of preceding rest). That does not affect the resistance validation
  (18.2), but SOP must not be argued from that file.

### 10.dd Hysteresis effects — the trim's premise (2026-08-24)

- Test#8 varies **only the C-rate of the pulse immediately before the
  measurement**, 0–4C, on the same cell at the same 0 °C and the same SOC.
  The effect is real and large — 26.9 % of SOP amplitude at SOC 0.2, and the
  17 voltage-limited rows sit exactly in that range.
- **The effect is in V_pre, not in resistance.** At SOC 0.20, V_pre goes
  3.456 → 3.197 V (−29 % headroom) while resistance **falls** 38.8 →
  32.6 mΩ (self-heating). It is a voltage state left by diffusion
  polarisation.
- **The trim structurally cannot capture this** — its output is a resistance
  multiplier only, and with the loss defined on dV, OCV and RC initial state
  are excluded by design. Measurement agrees: k_f moves the **opposite way**
  (required multiplier −0.0136/C against k_f +0.0099/C).
- **The hybrid's SOP is right nonetheless — if V_pre is the measured value.**
  dI*/dC is −0.77 measured, −0.67 predicted with measured V_pre, and
  **+0.58** predicted with model OCV. **Design rule: V_pre in the SOP
  inversion must be the measured terminal voltage.**
- Limits: absolute level not established (at 0 °C the model reads I* half
  what it is, dominated by the SOH 0.90 assumption), and 0 °C is outside the
  trim's training temperature range.

### 10.ee Data design audit (2026-08-24)

- **The trim is a residual reader, not a history reader.** Correlation
  between k_f and dR_fast is +0.915; the load features are +0.03 to +0.08.
  dR_fast is the residual slope *now*, not an integral of history. That
  explains both why it works under constant-current excitation (20.3) and why
  it misses Test#8's hysteresis effect (23.3). **The claim has to narrow from
  "it reads history" to "it reads the resistance deviation of the pulse
  response."**
- **Closing the open item in 23.5**: the positive association of k_f with
  load appears in 25 °C UYPYDJ too (+0.0113 between the top and bottom 20 %
  of I_rms). Test#8's sign reversal is not a temperature-extrapolation
  artefact.
- **What the data teaches about history is invariance.** In **all 5,359**
  (cycle, SOC, rank) keys the 12 history windows share an **identical**
  label. And every labelled pulse in UYPYDJ follows a long rest — zero
  history contrast on the label side.
- **Contrast per axis**: SOH, SOC and rate are saturated. 26 labels carry
  temperature contrast (all τ = 10 s). **18 labels carry history contrast
  (all one cell, 0 °C).** Tens of thousands of rows, and this is where the
  contrast ends.
- **The answer to "there is so much data, why can it not teach this" is
  design, not volume** — these are aging tests, and an aging test moves SOH
  and holds everything else.

### 10.ff Safety in the charge direction (2026-08-24)

- **The label situation is better on charge**: 4,648 trustworthy rows
  (discharge 651). Externally 265 rows (discharge 53). The 4.2 V charge
  ceiling is close to OCV, so I* is small and lands inside the fan.
- **The max-aggregation recommendation reproduces**: the trim worsens
  optimism 72.9 → 81.6 %, and max aggregation brings it back to 54.4 % with
  RMSE also improving 2.52 → 2.23 A.
- **Charge fails in two different ways**: high SOC has large relative and
  small absolute error (exceedance 92.9 %, worst 2.09 A); low SOC is the
  reverse (29.4 %, **23.07 A**). **The yardstick must be absolute amperes,
  not a ratio.**
- **A 0.5 A tolerance is the knee**: usability 0.458 → **0.594**, actual
  worst 0.75 A. From 1.0 A the worst jumps to 3.35 A on a new cell and does
  not transfer. On discharge the curve is flat, so tightening to zero is
  cheap (0.688). **The optimum differs by direction.**
- **Margin must be multiplicative.** Additive alone kills the output
  entirely (usability 0.000); mixed gains 11 %p of usability while the worst
  blows up 0.03 → 11.11 A.
- **τ conditioning barely helps on charge** (0.458 against 0.447), in
  contrast to discharge, where τ = 2 s was specifically broken (19.3).
- **Validated on 265 external rows**: hybrid exceedance 8–10 %, ECM 34–42 %.
  Matched on usability, the hybrid's **tail is consistently less than half**
  (0.07 against 0.64 A, and so on). What the trim does on charge is not shift
  the distribution but **compress the tail**.
- **Charge has no temperature dependence** (10 and 25 °C identical), in
  contrast to discharge, where lambda had to tighten 0.693 → 0.623 in the
  cold.
- **§16.2's verdict flips (recounted per cell)**: 4/6 by RMSE but **6/6 by
  safety**, and **the two cells that lost on RMSE win the most on safety**
  (BOOST_NEGPULSE +16.7 %, CC_CELL2 +23.3 %). Not an aggregation artefact —
  counted with `last` the RMSE is still 4/6. **The reason: lambda is set by
  the tail.** The hybrid's worst is smaller in 4 of 6 and its p99 in 5 of 6,
  which loosens lambda by 25 % (0.465 against 0.582). CC_CELL2 has the ECM's
  highest RMSE of the six (1.36 A) but among the lowest usability (0.464) —
  **RMSE measures the middle, margin measures the end.** Robust at 6/6 across
  tolerances of 0, 0.5 and 1.0 A.
- **Adopted (2026-08-27, after switching the trim to A8)**: max history
  aggregation + multiplicative margin per τ + measured V_pre. Discharge λ
  **0.683/0.470**, charge λ **0.586/0.560**. Usable current **69.1 %**
  discharge, **59.6 %** charge. The ECM (A0) at the same safety level is
  59.3 / 50.7 % — the hybrid uses 9.8 %p and 8.9 %p more.
  **[Updated — §34.1]** Those λ are the median of six leave-one-cell-out
  fits applied to every cell. Calibrated strictly per held-out cell the λ
  span 0.683–0.708 (discharge 10 s), usable current is 69.6 / 59.5 %, and
  the exceedance is 1 or 2 per setting rather than zero.
  (The A3-era values were discharge 0.679/0.462, charge 0.567/0.544, and
  70.3 / 57.9 % current. The cold-guaranteed λ of 0.623 was measured on A3
  and has not been re-measured on A8 — §26.)

### 10.gg Where the charge margin's 41 % comes from (2026-08-24)

- **One point out of 2,092 sets the safety factor.** Removing the top 5 %
  relaxes lambda 0.598 → 0.768. The median ratio is 1.01, almost perfect.
- **The variance decomposition says why conditioning fails**: condition
  (32 cells) 0.122, cell 0.283, cell × condition 0.584. **Knowing the cell of
  the grid still leaves the cell explaining 0.461 more.** Which is why
  SOC/SOH conditioning, a median-bias grid and online per-cell calibration
  **all failed** (every one raised the worst overshoot).
- **Online per-cell calibration succeeded at the estimation itself**
  (between-cell correlation +0.962, error within 0.035). It gains nothing
  because lambda is set not by the cell mean but by outliers **within** a
  cell.
- **A genuine model defect found**: `ECMSurface.theta` interpolates the five
  RC parameters separately along the rank axis. Interpolating between fits
  where tau2 diverged to its ceiling (3000 s) produces combinations with no
  basis, and creates unphysical stretches where resistance **rises** with
  current (model 49.0 → 32.8 → 40.2 mΩ against measured 52.2 → 41.3 → 29.6).
  Fixed with `d_tau`.
- **Not adopted, though**: charge optimism 54.4 → 42.5 % while the holdout's
  actual worst goes 0.75 → 2.78 A. Fixing the middle and worsening the end —
  the recurring pattern in this project. Kept behind `--interp dtau`.
- **What sets the worst is real physics**: BOOST_NEGPULSE cycle 487, where
  resistance is 2–3× in both directions and at every rate, but only below
  SOC 0.3. The raw voltage is recorded that way (dV 0.408 → 0.708) and it
  recovers at the next characterisation. **The same kind of thing as Test#8's
  hysteresis effect in §23.** It is not removed.
- **Conclusion: much of the 41 % cannot be reduced by improving the model.**
  Recovering it requires the vehicle to observe that state, and the only
  channel is measured terminal voltage (23.4).

### 10.hh MCU measurements — the deployment question closes (2026-08-25)

The compute side of this project's opening question ("can AI-based SOH/SOP
run on an ordinary BMS?") was measured for the first time. NUCLEO-H563ZI,
Cortex-M33 at 250 MHz, float32 hard FPU, measured.

- **SOC + SOP in 217 µs per cycle.** 0.02 % at 1 Hz, 0.22 % at 10 Hz,
  2.17 % at 100 Hz, 4.34 % at 200 Hz. Effectively free for practical
  requirements.
- **SOH CNN 17.9 ms** (3 seeds). It runs once at the end of a charge, so the
  average load is 0.0005 %. It cannot hold a 100 Hz period, though, so it has
  to be spread out or the seeds reduced (5.97 ms with one).
- **Memory**: SOP alone 64.6 KB flash / 12.9 KB RAM; SOP+SOH 197.2 KB /
  24.8 KB; peak stack 624 B. Less than half a GRU-128 network (449 KB /
  82 KB).
- **1/884 of the network** (217 µs against 191,789 µs on the same board).
- **The cost of adding AI is 21 %** — pure physics 179.7 µs against the
  hybrid's 217.0 µs. In exchange, at the same safety level, it uses 11 %p
  more pack current on discharge and 6 %p more on charge.
- **The C implementation matches Python**: trim 1.1e-07, SOH 7.4e-10 (RMSE
  identical at 0.0123). SOP's 2.7e-03 is the intended gridding, not an
  implementation error.
- **The table stores D(2 s) and D(10 s) rather than RC** (32×16, 32 KB).
  Half the size, and it makes §26.3's tau2 divergence defect structurally
  impossible. The gridding cost is SOP optimism 70.6 → 70.9 %, buried in
  retraining spread.
- **One prediction missed**: table lookup was expected to dominate, but the
  trim is 2.2× more expensive (`expf` and `tanhf` twice each). The inversion
  is 87 % of the total, so the big picture was right.
- **Caution**: the same code moves 20 % depending on binary layout
  (ICACHE / flash placement). The "51×" against the previous EKF's 378.6 µs
  must not be used — precision (FP64 → float32) and structure changed
  together and the shares were not separated.
- **Traps hit**: SETUP.md's baud was stale (115200 → 921600); direct access
  to a float in a packed struct hard-faults on Cortex-M33 (UsageFault
  UNALIGNED); `soh_cnn.py` had never saved weights (`--save-model` added).
- **int8 quantisation (closed)**: per-channel int8 for SOH gives 6-fold RMSE
  0.0128 → **0.0129** (+1.7e-4), and for the ECM table, rank × horizon int8
  is smaller than the gridding error (0.23 % against 0.30 %). Measured
  **flash 197.2 → 76.2 KB (−61 %)** but **8.4 % slower** — on a Cortex-M33
  hard FPU a float32 multiply is one cycle while int8 adds unpacking and
  conversion. **The trade buys flash, not speed.** int4 collapses at 0.0216.
  The recommendation is float32 by default, int8 only when flash is tight.
- **Reducing SOH seeds (closed)**: 6-fold RMSE is 0.0128 with 3 seeds,
  **0.0129 with 2 (+1.0e-4, 11.9 ms)**, 0.0132 with 1 (+4.1e-4, 6.0 ms).
  **Two is the knee**; one swings ±8.4e-4 depending on which seed and is not
  recommended.
- **Full-integer path (closed, negative)**: written with int8 activations and
  SMLAD, it is **5 % slower than fp32 and less accurate** (6-fold 0.0136
  against 0.0128). The first version was 32 % slower because SIMD was applied
  only to dense1, which is 8.6 % of all MACs; applying it to conv2 (86 %)
  recovered 26 % but did not reverse it. **The reason is structural**: SMLAD
  does two int16 MACs per cycle while FPv5 does one float32 MAC, so the
  theoretical headroom is only 2× and requantisation eats it. Helium (M55)
  would be different.
- **EKF decomposition (closed)**: precision (float32 → double) **7.37×**,
  structure (folded 2RC → full 2RC) **1.16×**. 88 % of the difference is
  software double. **Moving a BMS's EKF to float32 with no algorithmic change
  is worth 7×.** **The target MCU is genuinely in that condition** — the
  S32K344 (BMU class, Cortex-M7 at 160 MHz, ASIL-D) has a single-precision
  FPU only (fpv5-sp-d16), so double is software-emulated, and it shares the
  FPU configuration of the H563 (M33, FPv5-SP) used for the measurement. The
  remaining 5× against the existing CEMA 378.6 µs cannot be attributed
  because the algorithm differs.
- **Remaining**: re-measure the integer path on a Helium core, and a
  like-for-like precision comparison of the same EKF.

### 10.ii Pack level (2026-08-25)

- SOP for a series pack is **min over cells**. If errors were independent and
  symmetric, min should make the pack safer than a cell, but **discharge goes
  the other way** (at lambda 0.70, exceedance N=1 4.1 % → N=96 **14.1 %**).
  Charge improves as expected (1.6 → 0.0 %).
- **Cause: the weaker the cell, the more optimistically it is estimated.**
  Within a condition group, corr(measured I*, predicted/measured) = −0.385
  (discharge) / −0.411 (charge); the ratio for the bottom 25 % of measured is
  1.054 / 1.136 while the top 25 % is 0.971 / 0.989. **min picks the weakest
  cell, and that is the cell estimated most optimistically.** Another face of
  §26's "cell-to-cell spread cannot be explained by conditions."
- **It does not show up in the adopted configuration** — at the
  zero-exceedance lambda, N = 1 through 192 all exceed zero times. It holds
  after the switch to A8 (discharge 0.683/0.470, charge 0.586/0.560), re-
  simulated (§31.2). Counting charge 10 s against a zero tolerance gives
  4.4 %, but the worst overshoot is 0.09 A, inside charge's design tolerance
  of 0.5 A. With enough margin the harm of the correlation fits inside it.
- **Conclusion: not "it works at pack level" but "it works at pack level
  because the margin was set at cell level."** Any attempt to reduce the
  margin has to be revalidated at pack level.
- Limits: pack cells were resampled from our six (wider spread than a real
  pack, so pessimistic, but the aging protocols differ so the spread
  structure differs too). A real pack measures every cell voltage and can
  therefore **observe** the limiting cell, which makes §23.4's "measured
  V_pre" stronger at pack level — untestable with this data.

## 10.jj Reducing the trim to one feature does not cost deployment performance (A8)

A trim using only dR_fast instead of the 12 EW features (A8) was actually
trained and measured.

On voltage it is 6.9 % worse (58.76 → 62.81 mV). Converted to current,
though, the two values that decide deployment are effectively the same:
safety factor λ 0.679 → 0.683, usable current 69.8 → 68.9 %. And before λ is
applied, the raw prediction is better on every count — optimism 69.4 →
62.8 %, RMSE 5.43 → 4.95 A, worst overshoot 1.19 → 0.97 A.

The reason is the change of units. Converting voltage error to current
divides by resistance, so the model that best reduces the voltage residual
is not guaranteed to have the smallest current error. **Choosing a trim by
voltage RMSE can pick the wrong deployment performance.**

Deployment gain: EW states 12 → 2, trim parameters 26 → 4.

Confirmed: the evaluation row set is identical in both versions (631 rows),
and **on 6/6 cells** A8 is better on both optimism (−2.3 to −8.3 %p) and
RMSE (−0.05 to −0.68 A). The gain is not concentrated in one cell.
**A8 adopted as the final configuration.**
