#!/bin/bash
# Reproduction entry point.  This checkout already contains the two code
# changes (analysis/ecm_pool.py forwards the CSVs to build(); analysis/
# sop_trim_dataset.py has --pool-src); code_patch/repo.patch is the same
# change against the commit named in BASE_COMMIT.
#   RCSOP_REPO   repository root      RCSOP_BUNDLE  this directory (nested/)
#   RCSOP_WORK   scratch (created)    RCSOP_PYTHON  interpreter (python3)   NPROC processes (8)
#   check-code    the two code changes are present
#   check-pools   30 inner pools rebuilt from assets/innerpool/no<c>/uypydj_{ecm,ocv}.csv == stored, != shipped; surfaces respond to the CSV
#   score         the 27 tables in final/ from the stored predictions (assets/), hull-aware solver, compared with final/
#   full          rebuild every inner asset and score (needs analysis/cache_t, analysis/cache/{trim,trim_chg}, a GPU; hours)
set -eu
: "${RCSOP_REPO:?}" "${RCSOP_BUNDLE:?}" "${RCSOP_WORK:?}"
PY=${RCSOP_PYTHON:-python3}; S=$RCSOP_BUNDLE/scripts; A=$RCSOP_BUNDLE/assets
mkdir -p "$RCSOP_WORK"
export RCSOP_REPO RCSOP_BUNDLE RCSOP_WORK RCSOP_PYTHON=$PY
case "${1:-}" in
  check-code)
    grep -q "ecm_csv=ecm_csv, ocv_csv=ocv_csv" "$RCSOP_REPO/analysis/ecm_pool.py" && grep -q -- "--pool-src" "$RCSOP_REPO/analysis/sop_trim_dataset.py" \
      && echo "code changes present" || { echo "code changes missing: apply code_patch/repo.patch" >&2; exit 1; };;
  prepare)
    # ecm_pool.surfaces() rebuilds a pool whose source CSV is newer than the pool file (a fresh checkout
    # gives arbitrary order) and parallel workers must never rebuild the same file: make the stored pools
    # newer than their CSVs and build the shipped five-cell pools once, serially.
    find "$A/innerpool" -path '*/pool/*' -type f -exec touch {} +
    cd "$RCSOP_REPO/analysis" && $PY -c "import ecm_pool; [ecm_pool.surfaces(c) for c in ['BOOST','BOOST_NEGPULSE','BOOST_NEGPULSE_1S','BOOST_REST','CC','CC_CELL2']]; print('shipped pools ready')";;
  check-pools) bash "$0" check-code && bash "$0" prepare && $PY "$S/verify/check_pools.py";;
  score) bash "$0" check-code && bash "$0" prepare && bash "$0" score-arms && bash "$0" score-methods && $PY "$S/verify/compare_final.py";;
  score-arms)
    mkdir -p "$RCSOP_WORK/arms"; cp "$A/soc_err/"*.npz "$RCSOP_WORK/arms/"; cd "$S/nested"
    INNERPOOL=$A/innerpool INNERTRIM=$A/innertrim INNERSOH=$A/innersoh SOCTAG=_fixpool SOLVER=hull OUTTAG=_level2_hull2 NPROC=${NPROC:-8} OUTDIR=$RCSOP_WORK/arms $PY eval_arms_nested.py replace
    OUTDIR=$RCSOP_WORK/arms $PY score_arms_nested.py replace replace_level2_hull2 hyb
    OUTDIR=$RCSOP_WORK/arms $PY score_arms_nested.py replace replace_level2_hull2 ecm;;
  score-methods)
    mkdir -p "$RCSOP_WORK/methods"; cd "$S/nested"
    INNERPOOL=$A/innerpool INNERTRIM=$A/innertrim BASETREES=$A/basetrees INNERSOH=$A/innersoh SOLVER=hull OUT=$RCSOP_WORK/methods NPROC=${NPROC:-8} $PY score_methods_nested.py;;
  full) bash "$0" check-code && bash "$S/nested/level2_chain.sh";;
  *) echo "usage: reproduce.sh {check-code|prepare|check-pools|score|score-arms|score-methods|full}" >&2; exit 2;;
esac
