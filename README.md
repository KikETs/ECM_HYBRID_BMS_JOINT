# Hybrid SOP / SOH / SOC for a production BMS

Samsung INR21700-30T. SOP from a nominal 2RC table plus a learned trim,
SOH from a partial-charge CNN, SOC from a 2RC EKF. Six cells,
leave-one-cell-out throughout. Timing measured on a NUCLEO-H563ZI
(Cortex-M33, 250 MHz).

## Results

| Arm | Model | Size | Leave-one-cell-out |
|---|---|---|---|
| SOP | 2RC table + trim (A8) | 4 parameters, 2 EW states | usable current 69.1 % discharge, 59.6 % charge |
| SOH | dQ/dV CNN | 10,945 parameters | RMSE 0.0135, bias +0.0001 |
| SOC | 2RC EKF, low-current gate | 3 states | 2.05 %p under sensor perturbation |

Safety factor λ is calibrated leave-one-cell-out to zero exceedance:
discharge 0.683 (10 s) / 0.470 (2 s), charge 0.586 / 0.560 at a 0.5 A
tolerance. "Usable current" is the median of λ·predicted / measured.

Per-cycle cost on the board: 214.8 µs (SOC EKF 7.1, trim features 6.0,
four SOP inversions 201.7). Flash 142.1 KB for the deployment build.

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
  three of four direction × horizon settings. They agree only at discharge
  τ = 10 s.
- All three arms are also evaluated with estimated rather than oracle SOH:
  discharge 69.1 → 67.2 %, charge 59.6 → 56.6 %, SOC 2.05 → 2.17 %p.
- Cell-level λ does not transfer to a pack without recalibration. Keeping
  the oracle-SOH λ under estimated SOH gives 42 % pack exceedance at
  N = 192 for discharge τ = 2 s (worst overshoot 0.18 A).

## Licence

None set; all rights reserved.
