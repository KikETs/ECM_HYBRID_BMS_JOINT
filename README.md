# Hybrid SOP / SOH / SOC for a production BMS

Samsung INR21700-30T. SOP from a nominal 2RC table plus a learned trim,
SOH from a partial-charge CNN, SOC from a 2RC EKF. Six cells,
leave-one-cell-out throughout. Timing measured on a NUCLEO-H563ZI
(Cortex-M33, 250 MHz).

## Results

| Arm | Model | Size | Leave-one-cell-out | Worst cell |
|---|---|---|---|---|
| SOP | 2RC table + trim (A8) | 4 effective coefficients, 2 EW states | usable current 69.6 % discharge, 59.5 % charge (τ = 10 s) | 59.9 % / 53.5 % (BOOST_REST) |
| SOH | dQ/dV CNN | 10,945 parameters | RMSE 0.0135, bias +0.0001 | 0.0293 (BOOST_REST) |
| SOC | 2RC EKF, low-current gate | 3 states | 2.14 %p over six sensor disturbances | 3.77 %p (current offset −0.10 A) |

Safety factor λ is calibrated **per held-out cell**: cell *i* is scored under
a λ fitted with cell *i* removed entirely. Per-cell λ spans 0.683–0.708
(discharge 10 s). Observed exceedance is 1 of 491 rows on discharge 10 s and
1 of 2461 on charge 10 s, with one-sided 95 % upper bounds of 0.96 % and
0.19 %. **Zero observed exceedance is not zero risk**; the bound is the
number to quote. "Usable current" is the median of λ·predicted / measured.

Per-cycle cost on the board: median 214.8 µs, worst case 307.1 µs (SOC EKF
7.1, A8 feature update 6.0, four SOP inversions 201.7). Deployment build
text 142 060 B = 138.7 KiB. Measured on a NUCLEO-H563ZI after flashing;
`repro/run_parity.py` checks the C against Python to 9.2 × 10⁻⁶.

> An audit on 2026-08-27 (branch `audit/etransportation-readiness`) revised
> several numbers above and contradicted others. `.paper_state/paper_map.yaml`
> lists every claim with its status; `.paper_state/evidence_ledger.yaml`
> carries the measurements. In particular: ridge regression beats the SOH
> CNN on the same splits, the frozen A8 has never been validated outside
> UYPYDJ, and 78 % of the discharge SOP labels are extrapolated rather than
> measured.

## Reproduce

```bash
conda env create -f environment.yml && conda activate samsung30t
python3 repro/verify.py        # recompute the 43 published numbers
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
