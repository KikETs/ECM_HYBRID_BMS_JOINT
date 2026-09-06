#!/bin/bash
# Reproduction entry point for the nested-calibration results.
#
#   RCSOP_REPO   clean checkout of the base commit with code_patch/repo.patch applied
#   RCSOP_BUNDLE this directory
#   RCSOP_WORK   scratch directory for outputs (created)
#   RCSOP_PYTHON interpreter (default python3)
#
#   bash reproduce.sh patch        apply code_patch/repo.patch to $RCSOP_REPO (git apply)
#   bash reproduce.sh check-pools  rebuild the 30 inner pools from the five-cell CSVs,
#                                  compare with the bundled pools, and show that the
#                                  surface/prediction respond to the CSV content
#   bash reproduce.sh score        recompute the final tables from the bundled
#                                  checkpoints (no training): arms + six methods,
#                                  hull-aware solver, then compare with final/
#   bash reproduce.sh full         everything from the drive-cycle cache: inner
#                                  datasets, inner trims, comparison methods, SOC
#                                  errors, then score.  Needs analysis/cache_t and
#                                  analysis/cache/{trim,trim_chg,soh_charge.npz}
#                                  (tier 1-3 of REPRODUCE.md) and a GPU.  Hours.
set -eu
: "${RCSOP_REPO:?}" "${RCSOP_BUNDLE:?}" "${RCSOP_WORK:?}"
PY=${RCSOP_PYTHON:-python3}
S=$RCSOP_BUNDLE/scripts
mkdir -p "$RCSOP_WORK"
export RCSOP_REPO RCSOP_BUNDLE RCSOP_WORK RCSOP_PYTHON=$PY
case "${1:-}" in
  patch)
    cd "$RCSOP_REPO" && git apply --check "$RCSOP_BUNDLE/code_patch/repo.patch" && git apply "$RCSOP_BUNDLE/code_patch/repo.patch"
    echo "patch applied to $(git rev-parse --short HEAD)";;
  check-pools)
    $PY "$S/verify/check_pools.py";;
  score)
    bash "$0" score-arms && bash "$0" score-methods && $PY "$S/verify/compare_final.py";;
  score-arms)
    mkdir -p "$RCSOP_WORK/arms"
    cp "$RCSOP_BUNDLE/assets/soc_err/"*.npz "$RCSOP_WORK/arms/"
    cd "$S/nested"
    INNERPOOL=$RCSOP_BUNDLE/assets/innerpool INNERTRIM=$RCSOP_BUNDLE/assets/innertrim INNERSOH=$RCSOP_BUNDLE/assets/innersoh \
      SOCTAG=_fixpool SOLVER=hull OUTTAG=_level2_hull2 NPROC=${NPROC:-8} OUTDIR=$RCSOP_WORK/arms $PY eval_arms_nested.py replace
    OUTDIR=$RCSOP_WORK/arms $PY score_arms_nested.py replace replace_level2_hull2 hyb
    OUTDIR=$RCSOP_WORK/arms $PY score_arms_nested.py replace replace_level2_hull2 ecm;;
  score-methods)
    mkdir -p "$RCSOP_WORK/methods"; cd "$S/nested"
    INNERPOOL=$RCSOP_BUNDLE/assets/innerpool INNERTRIM=$RCSOP_BUNDLE/assets/innertrim BASETREES=$RCSOP_BUNDLE/assets/basetrees \
      INNERSOH=$RCSOP_BUNDLE/assets/innersoh SOLVER=hull OUT=$RCSOP_WORK/methods NPROC=${NPROC:-8} $PY score_methods_nested.py;;
  full)
    echo "full retrain: see COMMANDS.md (run_inner_datasets.sh, level2_chain.sh); not executed by this entry point" >&2; exit 2;;
  *) echo "usage: reproduce.sh {patch|check-pools|score|full}" >&2; exit 2;;
esac
