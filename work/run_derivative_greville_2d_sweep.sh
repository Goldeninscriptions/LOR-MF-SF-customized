#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/run_derivative_greville_2d.sh"
root=${RESULT_DIR:-$script_dir/results/derivative_greville_2d}
read -r -a degrees <<< "${DEGREES_LIST:-2 3 4 6}"
read -r -a sizes <<< "${SIZES_LIST:-4 8 16 32}"
mkdir -p "$root/cases"
lock=${BATCH_LOCK_FILE:-$script_dir/.derivative_greville_explicit_batch.lock}
exec 9>"$lock"; flock -n 9 || { echo "Another explicit DMG batch is running" >&2; exit 4; }
summary="$root/summary.csv"; first=true
append_case() {
  local p=$1 ex=$2 ey=$3 file="$root/cases/p${1}_n${2}x${3}.csv"
  echo "[2D] p=$p elements=${ex}x${ey}"
  "$runner" "$p" "$ex" "$ey" "$file"
  if [[ "$first" == true ]]; then
    cp "$file" "$summary"; first=false
  else
    tail -n +2 "$file" >> "$summary"
  fi
}
for p in "${degrees[@]}"; do
  for n in "${sizes[@]}"; do append_case "$p" "$n" "$n"; done
done
# Required x/y-stride regression cases.  Skip duplicates if explicitly listed.
append_case 3 4 7
append_case 3 7 4
echo "[done] $summary"
