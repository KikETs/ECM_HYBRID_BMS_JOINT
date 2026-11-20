#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
export root py="$HOME/anaconda3/envs/torch_env/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "$?" > "$root/logs/test1.exit"' EXIT
"$py" -u "$root/scripts/test1_external.py" features
printf '%s\n' BOOST BOOST_NEGPULSE BOOST_NEGPULSE_1S BOOST_REST CC CC_CELL2 | xargs -P 6 -I @ bash -c '
"$py" -u "$root/scripts/test1_external.py" "$1" > "$root/logs/test1_$1.log" 2>&1' _ @
