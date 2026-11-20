#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SOP_DATA_WORKERS=10
py="$HOME/anaconda3/envs/torch_env/bin/python"
trap 'echo "$?" > "$run/logs/strict_all.exit"' EXIT
"$py" -u "$run/experiments/build_strict_inner.py" > "$run/logs/strict_data_fast.log" 2>&1
"$py" -u "$run/experiments/train_strict_inner.py" > "$run/logs/strict_train.log" 2>&1
"$py" -u "$run/experiments/score_revision.py" strict > "$run/logs/strict_score.log" 2>&1
