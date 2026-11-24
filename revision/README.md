# Revision 1 (November 2026): analyses for the reviewers of *Batteries* manuscript batteries-4720861

This directory holds every analysis added during the first revision and the firmware change that goes with it. All
analyses reuse the published datasets (see `DATA.md`) and the fitted assets of v1.0.0; no new cell was tested.

| Directory | Content |
|---|---|
| `review1/` | Analyses run on 2026-11-16 (scripts as run, logs, and result tables): reference-fit diagnostics, strict exclusion of the inner calibration cell (`strict_inner/`), estimated-state replay (`coupled_state/`, `followup/state/`), second external cell and cold test (`external/`, `followup/external/`), near-cutoff pulses (`followup/boundary/`), firmware timing (`mcu/`, `followup/wcet/`). |
| `review2/` | Analyses run on 2026-11-20 on top of the reproduced v1.0.0 baseline (`scripts/`, results in `out/E*`). |

## Reviewer comment -> analysis -> output

| Topic (manuscript location) | Script | Output |
|---|---|---|
| Unscored rows, causes, fallback outputs (Section 3.1, Table S8) | `review2/scripts/e2_solver_failure.py` | `review2/out/E2/` |
| Reference-fit sensitivity, extrapolation ratio, lowered boundary (Section 3.2, Table 4, Table S9A) | `review2/scripts/e1_reference.py` | `review2/out/E1/` |
| Leave-protocol-out, cell/cycle resampling, per-cell factors (Section 3.1, Table 3, Table S6) | `review2/scripts/e4_stats.py` | `review2/out/E4/` |
| Strict exclusion of the inner cell (Section 2.2, Table S6) | `review1/experiments/build_strict_inner.py`, `train_strict_inner.py`, `score_revision.py` | `review1/statistics/`, `review1/strict_inner/` |
| Quantile/conformal rules, influence, retained ratio (Section 3.3, Discussion, Table S10) | `review2/scripts/e6_calibration.py` | `review2/out/E6/` |
| Method comparison incl. HPPC pulse-resistance lookup, curves, cost (Section 3.3, Table 5, Figure 5, Table S10) | `review2/scripts/e5_rows.py`, `e5_tradeoff.py`, `e5_cost.py` | `review2/out/E5/` |
| Near-cutoff measured pulses vs all methods (Section 3.2, Table 4, Table S9B) | `review1/followup/scripts/holdout_boundary.py`, `review2/scripts/e5b_boundary.py`, `e5b_extra.py` | `review1/followup/boundary/`, `review2/out/E5b/` |
| Nested configuration selection on the current assets (Section 2.2, Table S7) | `review2/scripts/e3_nested_selection.py` | `review2/out/E3/` |
| External cells, charge decomposition (Section 3.5, Figure 8, Table S12) | `review1/followup/scripts/test1_external.py`, `cross_cell_calibration.py`, `test3_cold_hybrid.py`; `review2/scripts/e7_external.py`, `e7_extra.py` | `review1/followup/external/`, `review2/out/E7/` |
| Estimated-state replay on every eligible HPPC cycle (Section 2.5, Section 3.4, Table S15) | `review2/scripts/e13_replay_rows.py`, `e13_replay_drive.py`, `e13_summary.py` (the review-1 scripts `review1/experiments/coupled_state.py` and `review1/followup/scripts/replay_preceding_drive.py` without the eight-cycle subsampling) | `review2/out/E13/` |
| Record SOH target vs capacity-anchor SOH and conversion (Section 3.4, Discussion, Table S13) | `review2/scripts/e11_soh_map.py` | `review2/out/E11/` |
| Cycle-level performance and retained current in amperes (Section 3.1, Discussion, Table S14) | `review2/scripts/e12_cycle_table.py`, `e12_amps.py` | `review2/out/E12/` |
| Rated-capacity SOC axis anchored at full charge (Section 2.1, Section 3.4, Table S11) | `review2/scripts/e8b_soc_axis_full.py` (supersedes `e8_soc_axis.py`, which assumed an empty-anchored axis) | `review2/out/E8b/` |
| One-dimensional map, cyclewise maxima, epsilon (Section 2.4) | `review2/scripts/e9_minor.py`, `e9_eps.py` | `review2/out/E9/` |
| Firmware timing and derived pack budget (Section 4, Table 7) | `review1/followup/scripts/prepare_wcet.py`, `measure_wcet.py`; `review2/scripts/e10_budget.py` | `review1/followup/wcet/`, `review2/out/E10/` |
| Deployment image on the board: flagged solver vs host build, 13,440 inputs, timing (Section 4) | `review2/board/run_board.sh`, `board_deploy_check.py`, `host_board_ref.c` | `review2/board/out/` |
| Revised Figures 5 and 8 (journal versions; `make_figs.py` drew the first drafts) | `review2/scripts/make_journal_figs.py` | `review2/out/figs/` |

## Reproducing `review2`

```bash
# 1. baseline arms of v1.0.0 (stored predictions, no training; about 10 minutes)
export RCSOP_REPO=$PWD RCSOP_BUNDLE=$PWD/nested RCSOP_WORK=$PWD/revision/review2/baseline RCSOP_PYTHON=python3
bash nested/reproduce.sh score            # must report: 27 tables compared, 0 differ
# 2. revision analyses (outputs go to revision/review2/out unless RCSOP_REVISION_WORK is set)
cd revision/review2/scripts
python e2_solver_failure.py; python e1_reference.py; python e4_stats.py; python e6_calibration.py; python e9_minor.py
python e5_rows.py; python e5_tradeoff.py; python e5_cost.py; python e5b_boundary.py; python e5b_extra.py
python e3_nested_selection.py; python e7_external.py; python e7_extra.py; python e8_soc_axis.py; python e10_budget.py
python make_figs.py
```

`e9_eps.py` reads the strict-inner trim datasets, which are rebuilt by `review1/experiments/build_strict_inner.py`
(about 0.5 GB, not stored). `e4_stats.py` uses the stored strict-inner scored rows in
`review1/strict_inner/scored/strict_results_inner.pkl`. The `review1` scripts are kept as run; their paths follow the
2026-11-16 run directory, and `review1/run_*.sh` show the order.

## Firmware change

`mcu/sop_core.c` adds `sop_review_solve`, which masks queries outside the surface hull and returns a status code
(`SOP_REVIEW_OK`, `_INPUT`, `_HULL`, `_RESISTANCE`, `_ITERATION_CAP`, `_RANGE`) with 0 A on failure. The deployment
commands `SOP_CMD_SOLVE` and `SOP_CMD_FULL` in `mcu/fw_sop/Src/main.c` now use it and report the status code in the
`r_eff` field of the 32-byte reply. The benchmark commands under `SOP_BENCH_PACK` measured the coupled update with four
inversions reported in the manuscript. `mcu/tests/host_review_test.c` checks on the host that, wherever the flagged
solver reports success, it returns the same current as the legacy `sop_solve` (11,172 grid cases: 8,563 identical,
2,609 flagged with 0 A):

```bash
gcc -O2 -std=gnu11 -Imcu mcu/tests/host_review_test.c mcu/sop_core.c -lm -o host_review_test && ./host_review_test
```

Build sizes with the flagged solver (`make MODEL_ID=... EXTRA_CFLAGS=-DSOP_A8_ONLY` and the default build): text
72,220 B (A8 only) and 74,124 B (both arms), BSS 13,172 B. The legacy checks in `repro/expected.json` keep the v1.0.0
values (70,796 and 72,700 B), which describe the stored v1.0.0 build tables.
