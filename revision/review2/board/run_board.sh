#!/bin/bash
# Board check of the deployment image (flagged solver, -DSOP_A8_ONLY build) on the STM32H563 board.
# Backs up the whole 2 MB flash, flashes the deployment build, runs board_deploy_check.py over 13,440 solver inputs,
# restores the backup and compares hashes. Needs STM32_Programmer_CLI (set CLI) and the board on /dev/ttyACM0.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
W=${BOARD_OUT:-$HERE/out}
CLI=${CLI:-STM32_Programmer_CLI}
PY=${PY:-python3}
mkdir -p "$W"; cd "$W"
make -s -C "$REPO/mcu/fw_sop" MODEL_ID=revision2_deploy EXTRA_CFLAGS=-DSOP_A8_ONLY -j8 > build.log 2>&1
cat "$REPO/mcu/fw_sop/Build/revision2_deploy/size.txt"
cp "$REPO/mcu/fw_sop/Build/revision2_deploy/sop_bench.bin" deploy.bin; sha256sum deploy.bin > deploy.sha256
gcc -O2 -std=gnu11 -DSOP_A8_ONLY -I"$REPO/mcu" "$HERE/host_board_ref.c" "$REPO/mcu/sop_core.c" -lm -o host_board_ref
"$CLI" -c port=SWD -r 0x08000000 0x200000 "$W/before.bin" > read_before.log 2>&1
sha256sum before.bin > before.sha256
"$CLI" -c port=SWD -w "$W/deploy.bin" 0x08000000 -v -rst > flash_deploy.log 2>&1
sleep 2
set +e
"$PY" "$HERE/board_deploy_check.py" "$W" "$W/host_board_ref" "$REPO/mcu" 2>&1 | tee measure.log
rc=${PIPESTATUS[0]}
set -e
"$CLI" -c port=SWD -w "$W/before.bin" 0x08000000 -v -rst > restore.log 2>&1
"$CLI" -c port=SWD -r 0x08000000 0x200000 "$W/after.bin" > read_after.log 2>&1
sha256sum before.bin after.bin | tee restoration.sha256
exit $rc
