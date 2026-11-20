#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$run/wcet"
prog="$HOME/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI"
conn=(port=SWD sn=0024003C3235511138363730)
test -z "$(fuser /dev/ttyACM0 2>/dev/null || true)"
build="$run/firmware/fw_sop/Build/review1_wcet/sop_bench.elf"
base="$run/wcet/before.bin"
"$prog" -c "${conn[@]}" mode=HotPlug -u 0x08000000 0x200000 "$base" > "$run/logs/backup.log" 2>&1
"$prog" -c "${conn[@]}" mode=HotPlug -u 0x08000000 0x200000 "$run/wcet/before_verify.bin" > "$run/logs/backup_verify.log" 2>&1
cmp "$base" "$run/wcet/before_verify.bin"
test "$(stat -c %s "$base")" = 2097152
restore() {
  result=$?
  trap - EXIT
  echo "$result" > "$run/logs/board_capture.exit"
  "$prog" -c "${conn[@]}" -w "$base" 0x08000000 -v -rst > "$run/logs/restore.log" 2>&1 || exit 31
  "$prog" -c "${conn[@]}" mode=HotPlug -u 0x08000000 0x200000 "$run/wcet/after.bin" > "$run/logs/restore_verify.log" 2>&1 || exit 32
  cmp "$base" "$run/wcet/after.bin" || exit 33
  sha256sum "$base" "$run/wcet/after.bin" > "$run/wcet/restoration.sha256"
  echo "$result" > "$run/logs/wcet.exit"
  exit "$result"
}
trap restore EXIT
"$prog" -c "${conn[@]}" -w "$build" -v -rst > "$run/logs/flash.log" 2>&1
sleep 2
"$HOME/anaconda3/envs/torch_env/bin/python" -u "$run/scripts/measure_wcet.py"
