#!/bin/bash
# Level 2: every inner asset built without the outer target.
#   1. wait for the inner trim datasets (features + NOM against the pool of
#      the four cells other than {c, j})
#   2. train the inner A8 trims (6 outer x 2 directions) on those datasets
#   3. train the comparison methods (A3, shrink/direct/rls fill, LSTM/GRU/FFRLS)
#      on the same datasets, one write-tree per outer target
#   4. inner rows for the four state-input arms (fp and hull solvers), scoring
#   5. six-method comparison (fp and hull)
# The repository is never written: training scripts run inside write-trees
# whose runs_trim_* directories are real directories, not links.
set -u
F=$RCSOP_WORK
X=$F/20_calib_fix
R=$RCSOP_REPO
PY=${RCSOP_PYTHON:-python3}
CELLS="BOOST BOOST_NEGPULSE BOOST_NEGPULSE_1S BOOST_REST CC CC_CELL2"
log() { echo "[$(date +%H:%M:%S)] $*"; }

until grep -q "ALLDONE inner datasets" $F/logs/inner_datasets.log 2>/dev/null; do sleep 60; done
log "datasets done"
n=$(ls $X/innerdata/*/trim_*.npz | wc -l); log "inner dataset files: $n (expect 60)"

# ---- 2. inner A8 trims
mkdir -p $X/innertrim $X/logs_trim
jobs=(); for c in $CELLS; do for sub in trim trim_chg; do jobs+=("$c $sub"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P 6 -L 1 bash -c '
  c=$0; sub=$1; X='"$X"'; R='"$R"'; PY='"$PY"'
  cd $R/analysis && $PY sop_trim.py --rung A8 --data $X/innerdata/${sub}_no$c --out $X/innertrim/${sub}_no$c --save-pred > $X/logs_trim/${sub}_no$c.log 2>&1
  echo "  trim done $c $sub $(date +%H:%M)"'
log "inner A8 trims done: $(ls $X/innertrim/*/pred_A8_*.npz | wc -l) preds (expect 60)"

# ---- 3. comparison methods, one write-tree per outer target
mkdir -p $X/basetrees $X/logs_base
mktree() {  # write-tree: analysis/* linked, runs_trim_* real dirs, fixed ecm_pool.py
  T=$1; rm -rf $T; mkdir -p $T/analysis $T/repro
  for f in $R/analysis/*; do ln -sfn "$f" "$T/analysis/$(basename $f)"; done
  for f in $R/repro/*; do ln -sfn "$f" "$T/repro/$(basename $f)"; done
  ln -sfn $R/docs $T/docs; ln -sfn "$(readlink -f $R/raw)" $T/raw
  rm -f $T/analysis/ecm_pool.py; cp $X/tree/analysis/ecm_pool.py $T/analysis/ecm_pool.py
  rm -f $T/analysis/results; cp -a $R/analysis/results $T/analysis/results
  for d in $T/analysis/runs_trim_*; do [ -L "$d" ] && rm -f "$d" && mkdir -p "$d"; done
  rm -f $T/analysis/__pycache__ $T/repro/__pycache__
  if find $T/analysis -maxdepth 1 -name 'runs_trim_*' -type l | grep -q .; then echo "FAIL links remain in $T" >&2; exit 1; fi
}
run_one() {
  c=$1; T=$X/basetrees/no$c; mktree $T
  for pair in "trim :" "trim_chg :_chg"; do
    set -- $pair; sub=$1; sfx=$2; [ "$sfx" = ":" ] && sfx="" || sfx="${sfx#:}"
    dd=$X/innerdata/${sub}_no$c
    ( cd $T/analysis && $PY sop_trim.py --rung A3 --data $dd --out $T/analysis/runs_trim_v2${sfx} --save-pred && \
      $PY sop_baseline_fill.py --data $dd --suffix "$sfx" && \
      $PY ../repro/run_sop_seq_baselines.py --data $dd --suffix "$sfx" ) >> $X/logs_base/no$c.log 2>&1
  done
  echo "  methods done no$c ($(ls -d $T/analysis/runs_trim_{v2,direct,shrink,rls,lstm,gru,ffrls}{,_chg} 2>/dev/null | wc -l) run dirs) $(date +%H:%M)"
}
export -f mktree run_one; export X R PY
printf '%s\n' $CELLS | xargs -P 3 -I{} bash -c 'run_one {}'
log "comparison methods done"

# ---- 4. arms
cd $F/20_stateinput_nested
for S in fp hull; do
  INNERPOOL=$X/innerpool INNERTRIM=$X/innertrim INNERSOH=$F/raudit_innersoh SOCTAG=_fixpool SOLVER=$S OUTTAG=_level2_$S NPROC=12 \
    $PY eval_arms_nested.py replace > $F/logs/eval_arms_level2_$S.log 2>&1
  $PY score_arms_nested.py replace replace_level2_$S hyb >> $F/logs/eval_arms_level2_$S.log 2>&1
  $PY score_arms_nested.py replace replace_level2_$S ecm >> $F/logs/eval_arms_level2_$S.log 2>&1
  log "arms level2 $S done"
done
# ---- 5. methods
cd $F/80_methods
for S in fp hull; do
  INNERPOOL=$X/innerpool INNERTRIM=$X/innertrim BASETREES=$X/basetrees INNERSOH=$F/raudit_innersoh SOLVER=$S OUT=$F/80_methods/level2_$S NPROC=12 \
    $PY score_methods_nested.py > $F/logs/methods_level2_$S.log 2>&1
  log "methods level2 $S done"
done
log "LEVEL2 ALLDONE"
