#!/bin/bash
# Inner trim datasets: for outer target c and calibration cell j, the drive
# features and the nominal response NOM are computed against the pool of the
# four cells other than {c, j} (assets/innerpool/no<c>/pool, holdout = j).
#   RCSOP_REPO    checkout whose analysis/sop_trim_dataset.py supports --pool-src
#   RCSOP_BUNDLE  the nested/ directory (assets/innerpool)
#   RCSOP_WORK    scratch; writes $RCSOP_WORK/innerdata/{trim,trim_chg}_no<c>/trim_<j>.npz
# Needs analysis/cache_t (drive-cycle cache) and the HPPC tables in the repository.
set -u
: "${RCSOP_REPO:?}" "${RCSOP_BUNDLE:?}" "${RCSOP_WORK:?}"
PY=${RCSOP_PYTHON:-python3}
W=$RCSOP_WORK; POOLS=$RCSOP_BUNDLE/assets/innerpool
CELLS="BOOST BOOST_NEGPULSE BOOST_NEGPULSE_1S BOOST_REST CC CC_CELL2"
mkdir -p $W/innerdata $W/logs/data
jobs=()
for c in $CELLS; do for j in $CELLS; do [ $c = $j ] && continue; for d in discharge charge; do
  sub=trim; [ $d = charge ] && sub=trim_chg
  jobs+=("$c $j $d $sub")
done; done; done
export RCSOP_REPO POOLS W PY
printf '%s\n' "${jobs[@]}" | xargs -P ${NPROC:-12} -L 1 bash -c '
  c=$0; j=$1; d=$2; sub=$3
  mkdir -p $W/innerdata/${sub}_no$c
  cd $RCSOP_REPO/analysis && $PY sop_trim_dataset.py --only $j --direction $d \
     --pool-dir $POOLS/no$c/pool --pool-src $POOLS/no$c \
     --out $W/innerdata/${sub}_no$c > $W/logs/data/${sub}_no${c}_$j.log 2>&1
  echo "done $c $j $d $(date +%H:%M)"
'
n=$(ls $W/innerdata/*/trim_*.npz 2>/dev/null | wc -l)
echo "inner dataset files: $n (expect 60)"
[ "$n" -eq 60 ]
