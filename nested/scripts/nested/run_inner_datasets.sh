#!/bin/bash
# inner trim datasets: for outer c and calibration cell j, NOM and the drive
# features are computed against the pool of the four cells other than {c, j}.
F=$RCSOP_WORK/20_calib_fix
PY=${RCSOP_PYTHON:-python3}
CELLS="BOOST BOOST_NEGPULSE BOOST_NEGPULSE_1S BOOST_REST CC CC_CELL2"
mkdir -p $F/innerdata $F/logs_data
jobs=()
for c in $CELLS; do for j in $CELLS; do [ $c = $j ] && continue; for d in discharge charge; do
  sub=trim; [ $d = charge ] && sub=trim_chg
  jobs+=("$c $j $d $sub")
done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P 12 -L 1 bash -c '
  c=$0; j=$1; d=$2; sub=$3
  F='"$F"'; PY='"$PY"'
  mkdir -p $F/innerdata/${sub}_no$c
  cd $F/tree/analysis && $PY sop_trim_dataset.py --only $j --direction $d \
     --pool-dir $F/innerpool/no$c/pool --pool-src $F/innerpool/no$c \
     --out $F/innerdata/${sub}_no$c > $F/logs_data/${sub}_no${c}_$j.log 2>&1
  echo "done $c $j $d $(date +%H:%M)"
'
echo "ALLDONE inner datasets $(date +%H:%M)"
