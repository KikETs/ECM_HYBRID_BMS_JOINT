#!/bin/bash
# Full rebuild of every inner asset without the outer target, then scoring.
#   1. inner trim datasets   (CPU, ~75 min / 12 proc)  -> $W/innerdata
#   2. inner A8 trims         (GPU, ~25 min)            -> $W/innertrim
#   3. comparison methods     (GPU, ~35 min)            -> $W/basetrees/no<c>/analysis/runs_trim_*
#   4. SOC-error records      (CPU, minutes)            -> $W/soc_err
#   5. arms and six methods, hull-aware solver          -> $W/arms, $W/methods, then check_level2 + compare_final
# Inner pools are read from $RCSOP_BUNDLE/assets/innerpool (rebuild with scripts/nested/build_inner_pools.py).
# Requirements beyond the repository: analysis/cache_t, analysis/cache/{trim,trim_chg}, a CUDA GPU (steps 2-3).
# Training scripts run inside write-trees so the repository is never written.
set -eu
: "${RCSOP_REPO:?}" "${RCSOP_BUNDLE:?}" "${RCSOP_WORK:?}"
PY=${RCSOP_PYTHON:-python3}; R=$RCSOP_REPO; B=$RCSOP_BUNDLE; W=$RCSOP_WORK; S=$B/scripts/nested
CELLS="BOOST BOOST_NEGPULSE BOOST_NEGPULSE_1S BOOST_REST CC CC_CELL2"
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $W/logs
# ---- 1
if [ "$(ls $W/innerdata/*/trim_*.npz 2>/dev/null | wc -l)" -ne 60 ]; then bash $S/run_inner_datasets.sh > $W/logs/inner_datasets.log 2>&1; fi
log "inner datasets: $(ls $W/innerdata/*/trim_*.npz | wc -l) files"
# ---- 2
mkdir -p $W/innertrim $W/logs/trim
export R W PY
for c in $CELLS; do for sub in trim trim_chg; do echo "$c $sub"; done; done | xargs -P ${GPU_JOBS:-6} -L 1 bash -c '
  c=$0; sub=$1
  cd $R/analysis && $PY sop_trim.py --rung A8 --data $W/innerdata/${sub}_no$c --out $W/innertrim/${sub}_no$c --save-pred > $W/logs/trim/${sub}_no$c.log 2>&1
  echo "  trim done $c $sub $(date +%H:%M)"'
log "inner A8 trims: $(ls $W/innertrim/*/pred_A8_*.npz | wc -l) preds (expect 60)"
# ---- 3
mkdir -p $W/basetrees $W/logs/base
mktree() {
  T=$1; rm -rf $T; mkdir -p $T/analysis $T/repro
  for f in $R/analysis/*; do ln -sfn "$f" "$T/analysis/$(basename $f)"; done
  for f in $R/repro/*; do ln -sfn "$f" "$T/repro/$(basename $f)"; done
  ln -sfn $R/docs $T/docs; [ -e $R/raw ] && ln -sfn "$(readlink -f $R/raw)" $T/raw
  rm -f $T/analysis/results; cp -a $R/analysis/results $T/analysis/results
  for d in $T/analysis/runs_trim_*; do [ -L "$d" ] && rm -f "$d" && mkdir -p "$d"; done
  rm -f $T/analysis/__pycache__ $T/repro/__pycache__
  if find $T/analysis -maxdepth 1 -name 'runs_trim_*' -type l | grep -q .; then echo "FAIL links remain in $T" >&2; exit 1; fi
}
run_one() {
  c=$1; T=$W/basetrees/no$c; mktree $T
  for pair in "trim :" "trim_chg :_chg"; do
    set -- $pair; sub=$1; sfx=$2; [ "$sfx" = ":" ] && sfx="" || sfx="${sfx#:}"
    dd=$W/innerdata/${sub}_no$c
    ( cd $T/analysis && $PY sop_trim.py --rung A3 --data $dd --out $T/analysis/runs_trim_v2${sfx} --save-pred && \
      $PY sop_baseline_fill.py --data $dd --suffix "$sfx" && \
      $PY ../repro/run_sop_seq_baselines.py --data $dd --suffix "$sfx" ) >> $W/logs/base/no$c.log 2>&1
  done
  echo "  methods done no$c $(date +%H:%M)"
}
export -f mktree run_one
printf '%s\n' $CELLS | xargs -P ${GPU_JOBS_METHODS:-3} -I{} bash -c 'run_one {}'
log "comparison methods done"
# ---- 4
mkdir -p $W/soc_err
INNERPOOL=$B/assets/innerpool OUTTAG=_fixpool OUTDIR=$W/soc_err NPROC=${NPROC:-12} $PY $S/build_soc_err_nested.py replace > $W/logs/soc_err.log 2>&1
log "SOC errors done"
# ---- 5
mkdir -p $W/arms $W/methods; cp $W/soc_err/*.npz $W/arms/
cd $S
INNERPOOL=$B/assets/innerpool INNERTRIM=$W/innertrim INNERSOH=$B/assets/innersoh SOCTAG=_fixpool SOLVER=hull OUTTAG=_level2_hull2 NPROC=${NPROC:-12} OUTDIR=$W/arms $PY eval_arms_nested.py replace > $W/logs/eval_arms.log 2>&1
OUTDIR=$W/arms $PY score_arms_nested.py replace replace_level2_hull2 hyb >> $W/logs/eval_arms.log 2>&1
OUTDIR=$W/arms $PY score_arms_nested.py replace replace_level2_hull2 ecm >> $W/logs/eval_arms.log 2>&1
INNERPOOL=$B/assets/innerpool INNERTRIM=$W/innertrim BASETREES=$W/basetrees INNERSOH=$B/assets/innersoh SOLVER=hull OUT=$W/methods NPROC=${NPROC:-12} $PY score_methods_nested.py > $W/logs/methods.log 2>&1
INNERDATA=$W/innerdata INNERTRIM=$W/innertrim BASETREES=$W/basetrees SOCERR=$W/soc_err $PY $S/check_level2.py
$PY $B/scripts/verify/compare_final.py
log "LEVEL2 ALLDONE"
