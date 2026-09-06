# nested

Leave-one-cell-out evaluation in which every fitted asset used to calibrate a target cell's safety factor
(residual trim, drive-feature normalisation, pooled ECM/OCV surfaces, SOH estimator, SOC-error records)
is built without that cell, plus a segment-aware root solver for the current inversion.

- `scripts/` pipeline and verification scripts (`RCSOP_REPO`, `RCSOP_BUNDLE`, `RCSOP_WORK` environment variables;
  in this checkout `RCSOP_REPO` is the repository root and `RCSOP_BUNDLE` is this directory)
- `assets/` four-cell surface pools with provenance, inner SOH predictions, SOC-error records
- `final/` tables produced from these assets (`reproduce.sh score` regenerates them; the comparison-method
  inner predictions are not stored here, see COMMANDS.md and EXTERNAL_INPUTS.md)
- `results/` condition-level results, sensitivity, solver diagnostics, metric definitions
- `VERIFICATION.md` what was checked and how
- `COMMANDS.md`, `EXTERNAL_INPUTS.md`, `ENV.txt`, `reproduce.sh`
