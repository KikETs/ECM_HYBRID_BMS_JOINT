#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export run py="$HOME/anaconda3/envs/torch_env/bin/python"
trap 'echo "$?" > "$run/logs/external.exit"' EXIT
printf '%s\n' BOOST BOOST_NEGPULSE BOOST_NEGPULSE_1S BOOST_REST CC CC_CELL2 | xargs -P 6 -I @ bash -c '
"$py" -u "$run/repo/repro/run_external_a8.py" --surface "$1" --out "$run/external/$1.csv" > "$run/logs/external_$1.log" 2>&1' _ @
