# Hybrid SOP / SOH / SOC for a production BMS

Samsung INR21700-30T. SOP from a nominal 2RC table plus a learned trim,
SOH from a partial-charge CNN, SOC from a 2RC EKF. Six cells.

Evaluation protocol is **not** uniform across the three arms, and saying so
matters more than the headline numbers:

* **SOP and SOH** are leave-one-cell-out: the model scoring cell *i* never
  saw cell *i*, and the safety factor λ is fitted with cell *i* removed too.
* **SOC** is *not*. Every filter reads its own cell's characterisation
  surface, so the EKF numbers describe a per-cell-calibrated deployment, not
  a transfer to an unseen cell.
* **What ships** is neither: the header on the board holds an all-cell fit
  (`--deployment`), because a product must not be a model trained without one
  of its own six cells. Leave-one-cell-out is how the cost of generalising is
  measured; it is not the artifact.

Timing measured on a NUCLEO-H563ZI (Cortex-M33, 250 MHz).

## Results

| Arm | Model | Size | Protocol | Result | Worst cell |
|---|---|---|---|---|---|
| SOP | 2RC table + trim (A8) | 4 effective coefficients, 2 EW states | leave-one-cell-out | usable current 69.6 % discharge, 59.5 % charge (τ = 10 s) | 59.9 % / 53.5 % (BOOST_REST) |
| SOH | dQ/dV ridge | 65 coefficients | leave-one-cell-out | RMSE 0.0094, bias −0.0005 | 0.0130 (BOOST_REST) |
| SOC | 2RC EKF, low-current gate | 3 states | **per-cell calibrated**, not held out | 2.14 %p over six sensor disturbances | 3.77 %p (current offset −0.10 A) |

The SOH arm was a 10,945-parameter CNN until the second audit round. Nested
selection — every candidate scored on the five training cells before the
held-out cell is touched — places that CNN **last on all six folds**, so it
was replaced by ridge: better on every cell, **6.50 µs against 19 442 µs** on
the board, and half the firmware (§36).

Safety factor λ is calibrated **per held-out cell**: cell *i* is scored under
a λ fitted with cell *i* removed entirely. Per-cell λ spans 0.683–0.708
(discharge 10 s). Observed exceedance is 1 of 491 rows on discharge 10 s and
1 of 2461 on charge 10 s, with one-sided 95 % upper bounds of 0.96 % and
0.19 %. **Zero observed exceedance is not zero risk**; the bound is the
number to quote. "Usable current" is the median of λ·predicted / measured.

Per-cycle cost on the board: SOP inversion 53.20 µs median, 81.09 µs worst;
SOC EKF 8.16; A8 feature update 6.79; SOH 6.50, once per charge rather than
per cycle. Deployment build text **70 796 B**. Measured on a NUCLEO-H563ZI
after flashing; `repro/run_parity.py` checks the C against Python to
9.2 × 10⁻⁶ and `mcu/bench_soh.py` checks the board's SOH against NumPy on 282
real curves.

Those SOP figures are 5.3 % slower than the same code in the previous
firmware, which held the CNN. Removing 71 kB the SOP path never calls moved
every hot function and changed eight alignments; with the instruction cache
disabled the ordering reverses. An embedded timing claim at this granularity
is a property of the image, not the algorithm (§36.4).

> Two audit rounds (2026-08-27 and 2026-08-31, branch
> `audit/etransportation-readiness`) revised several numbers above and
> contradicted others. `.paper_state/paper_map.yaml` lists every claim with
> its status; `.paper_state/evidence_ledger.yaml` carries the measurements;
> §34 and §35 of [docs/sop_hybrid_spec.md](docs/sop_hybrid_spec.md) record
> both rounds. In particular: ridge regression beats the SOH CNN on the same
> splits; 78 % of the discharge SOP labels are extrapolated rather than
> measured; and the numbers above are computed on the label's own SOC.

**What this work claims, and what it does not.** Not superiority — sequence
baselines match the trim on usable current at 1 100× the parameters, and
ridge beats the SOH CNN. The claim is *deployment-efficient equivalence under
a safety-aware current utility*: four effective coefficients and two
exponentially-weighted states reach the same usable current as far larger
models, at 50.52 µs and 142 kB on a Cortex-M33.

The production-readiness claim of the first draft is **withdrawn**. Every SOP
number above is computed on the label's own SOC, which a vehicle does not
have. Scored on identical pulses, estimated state costs about 4 %p of usable
current and moves discharge exceedance from 3 to 4 in 385 — but a wrong SOC
also corrupts the filter that decides which labels are trustworthy, and the
rows it wrongly admits carry a 24.3 % exceedance rate (§35.2). Oracle-state
validation overstates system safety.

**External validation, stated at its actual scope.** Six frozen UYPYDJ
leave-one-cell-out A8 models were transferred without refitting to the
in-hull portion of RPCWBY Test#2. On **discharge** the frozen safety factor
stayed conservative — 1.30–1.43× margin, zero exceedance in all six folds —
over the 54 % of calls the pooled hull covers, which excludes SOC below 0.30
and SOC = 1.0. On **charge** it did not transfer: the factor would have to
fall from 0.586 to 0.397, and as shipped it overshoots on 9–11 of 248 in-hull
rows per fold (§35.5).

## Reproduce

```bash
conda env create -f environment.yml && conda activate samsung30t
python3 repro/verify.py        # recompute the 68 published numbers
python3 repro/run.py --list    # stages, status, runtime
```

`verify.py` runs without the raw data: trained weights and result tables
are in the repository. To rebuild from the datasets, fetch the three DOIs
in [DATA.md](DATA.md) into `raw/` and run `python3 repro/run.py <stage>`.
Full rebuild is about 12 hours.

[REPRODUCE.md](REPRODUCE.md) maps every published number to the command
that produces it.

## Layout

```
repro/       stage graph, runner, verifier, QC, table and figure generators
analysis/    pipeline scripts; trained trim runs
mcu/         core shared by host and firmware; firmware for the H563ZI
docs/        design and measurement record (Korean)
```

Of 122 scripts in `analysis/`, 25 are on the critical path. The rest are
recorded exploration; `python3 repro/run.py --exploratory` lists them and
why they are excluded.

## Not included

Raw data (24 GB) and derived caches (5 GB) — rebuilt by the pipeline.
Third-party PDFs and dataset documentation — cited by DOI, not
redistributed.

## Method notes that affect how the numbers are read

- The SOC label is `1 + Ah/3.0` and the EKF prediction is
  `soc + I·dt/3600/3.0`. These are the same equation, so a benchmark that
  feeds the filter the label's own current and the exact initial SOC is
  circular: not using voltage wins by 12×. The SOC number above comes from
  a benchmark that perturbs the current the filter sees (initial SOC error,
  sensor offset, gain error) while the label keeps the true current.
- Voltage RMSE and usable current rank the six SOP methods differently in
  three of four direction × horizon settings, with the voltage metric taken
  at the same horizon as the current. They agree only at discharge
  τ = 10 s (Spearman +1.00).
- All three arms are also evaluated with estimated rather than oracle SOH:
  discharge 69.1 → 67.2 %, charge 59.6 → 56.6 %, SOC 2.05 → 2.17 %p.
- Cell-level λ does not transfer to a pack without recalibration. Keeping
  the oracle-SOH λ under estimated SOH gives 42 % pack exceedance at
  N = 192 for discharge τ = 2 s (worst overshoot 0.18 A). This is a
  **pack-level simulation sensitivity**, not a pack validation: it resamples
  single-cell evaluation rows and models no inter-cell correlation, shared
  current trajectory, thermal gradient or imbalance. There is no pack
  hardware and no HIL behind any number here.
- **The A8 adoption survives nested selection; the aggregation choice does
  not.** Choosing the rung and aggregation on inner splits that never see the
  test cell picks A8 in 16 of 22 folds, so that adoption is not an artifact
  of selecting on the evaluation. It picks `q75` in 10 folds against `max` in
  7, so the shipped `--trim-agg max` is one of two roughly equally supported
  settings, not an established one. Scored on the untouched outer cell the
  nested protocol gives 68.4 % discharge and 59.8 % charge at τ = 10 s,
  within 1.2 %p of the published figures — selecting on the evaluation was
  worth little (`analysis/results/tables/nested_selection.csv`).
- **Feeding the chain its own estimates changes the answer, and SOC is the
  term that matters.** Swapping oracle SOH for the SOH arm's own estimate
  costs about 1 %p of usable current. Swapping oracle SOC for the EKF's
  estimate takes discharge exceedance from 1 of 491 rows to 20 of 455 and,
  on charge, collapses λ from 0.586 to 0.404 and usable current from 59.5 %
  to 41.4 %. Those exceedances survive refitting λ per held-out cell on the
  same data, so they are not a bias the safety factor can price out. The
  both-estimated corner — the only one a vehicle can execute — gives 65.5 %
  discharge and 56.3 % charge at τ = 10 s
  (`analysis/results/tables/end_to_end.csv`).
- External validation of the **frozen** A8 on RPCWBY Test#2 (US06 drive plus
  SOP on the same cell, 375 rows, 10 and 25 °C, nothing refitted): on charge
  RMSE improves 8.36 → 7.21 W; on discharge RMSE worsens 2.25 → 3.65 W while
  the bias flips from +0.05 to about −2 W and over-prediction falls from
  51 % of rows to 21 %. Worse on RMSE, better on the criterion this project
  adopts by. Both belong in the paper.
- External, cross-laboratory: run on RPCWBY Test#3 (Chen et al.'s own SOP
  measurement set) through their published constant-power binary search, the
  physics model scores 1.7–4.2 W RMSE at 0–40 °C but 17.5 W at −10 °C and
  36.2 W at −20 °C, over-predicting available power by +29 W against a 30.8 W
  mean. Aged low-temperature behaviour is outside what this data supports.
- The SOP targets are **pulse-derived current-limit references**, not direct
  SOP measurements. A 30 A cycler cannot reach the discharge current the
  cell can take, so I\* is projected from a fit through four HPPC rates:
  78 % of discharge labels extrapolate past 1.5× the largest measured
  current. The A8-over-A0 result survives restricting to interpolated
  labels only (`analysis/results/tables/label_sensitivity.csv`).

## Licence

Code is [MIT](LICENSE). Derived data, tables, figures and documentation are
[CC BY 4.0](LICENSE-DATA).

Two of the three upstream datasets (RPCWBY, Mendeley) are CC BY 4.0, so
attribution passes through to anything derived from them — the citations are
in [LICENSE-DATA](LICENSE-DATA). UYPYDJ states no licence in its readme;
confirm with the depositor before redistributing derived material.
