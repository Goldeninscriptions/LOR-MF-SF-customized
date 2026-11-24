#!/usr/bin/env bash
set -euo pipefail

# Sequential EPS sweep for both cond(T^T M_g T) and
# cond((T^T M_g T)^{-1} M_s). Full tensor-product mass spaces are retained,
# matching the reproduced GPT-conversation experiment.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
single_case="$script_dir/run_greville_mass_condition_2d.sh"
result_root=${RESULT_DIR:-$script_dir/results/eps_greville_mass_on_mass_batch_2d}
console_root=${CONSOLE_DIR:-$script_dir/logs/eps_greville_mass_on_mass_batch_2d}
summary_file=${SUMMARY_FILE:-$result_root/summary.tsv}
failure_file=${FAILURE_FILE:-$result_root/failures.tsv}
resume=${RESUME:-true}
continue_on_error=${CONTINUE_ON_ERROR:-false}
cooldown_seconds=${COOLDOWN_SECONDS:-0}
read -r -a degrees <<< "${DEGREES_LIST:-3 4 5 6 7}"
read -r -a sizes <<< "${SIZES_LIST:-4 8 16 32 64 128 256 512 1024}"
targets=(preconditioner preconditioned)
for value in "$resume" "$continue_on_error"; do
  [[ "$value" == true || "$value" == false ]] || {
    echo "RESUME and CONTINUE_ON_ERROR must be true or false" >&2; exit 2; }
done
[[ "$cooldown_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "COOLDOWN_SECONDS must be nonnegative" >&2; exit 2; }
for degree in "${degrees[@]}"; do
  [[ "$degree" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid degree: $degree" >&2; exit 2; }
done
for size in "${sizes[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid size: $size" >&2; exit 2; }
done
command -v flock >/dev/null || { echo "flock is required" >&2; exit 3; }
lock_file=${BATCH_LOCK_FILE:-$script_dir/.greville_mass_condition_batch.lock}
exec 9>"$lock_file"
flock -n 9 || {
  echo "Another Greville tensor batch is already running: $lock_file" >&2
  exit 4
}

mkdir -p "$result_root" "$console_root"
printf 'p\tn\toperator\ttarget\tmpi_ranks\tdofs\tlambda_min\tlambda_max\tcondition_number\teps_iterations_min\teps_iterations_max\tmin_error\tmax_error\ttime_eps_s\tresult_file\n' > "$summary_file"
printf 'p\tn\toperator\ttarget\texit_status\tconsole_file\tresult_file\n' > "$failure_file"
is_complete() {
  [[ -f "$1" ]] && grep -qx 'operator_type = mass' "$1" &&
    grep -qx "spectral_target = $2" "$1" &&
    grep -qx 'eigensolver_status = converged' "$1"
}
append_summary() {
  awk -F ' = ' -v path="$1" '
    $1=="p"{p=$2} $1=="n"{n=$2;sub(/,.*/,"",n)} $1=="operator_type"{op=$2} $1=="spectral_target"{target=$2}
    $1=="mpi_ranks"{r=$2} $1=="total_dofs"{d=$2}
    $1=="lambda_min"{a=$2} $1=="lambda_max"{b=$2} $1=="condition_number"{k=$2}
    $1=="eps_iterations_min"{imin=$2} $1=="eps_iterations_max"{imax=$2}
    $1=="min_eigenpair_error"{emin=$2} $1=="max_eigenpair_error"{emax=$2}
    $1=="time_eps_s"{t=$2}
    END{printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",p,n,op,target,r,d,a,b,k,imin,imax,emin,emax,t,path}' "$1" >> "$summary_file"
}

completed=0; failed=0; total=$(( ${#degrees[@]} * ${#sizes[@]} * ${#targets[@]} ))
for degree in "${degrees[@]}"; do
  for elements in "${sizes[@]}"; do
    for target in "${targets[@]}"; do
      case_dir="$result_root/p${degree}/n${elements}x${elements}"
      log_dir="$console_root/p${degree}/n${elements}x${elements}"
      result_file="$case_dir/${target}.log"; console_file="$log_dir/${target}.console.log"
      mkdir -p "$case_dir" "$log_dir"
      if [[ "$resume" == true ]] && is_complete "$result_file" "$target"; then
        echo "[skip] mass target=$target p=$degree n=${elements}x${elements}"
        append_summary "$result_file"; completed=$((completed+1)); continue
      fi
      echo "[batch $((completed+failed+1))/$total] mass target=$target p=$degree n=${elements}x${elements}"
      set +e
      "$single_case" "$target" "$degree" "$elements" "$result_file" 2>&1 | tee "$console_file"
      status=${PIPESTATUS[0]}; set -e
      if (( status == 0 )) && is_complete "$result_file" "$target"; then
        append_summary "$result_file"; completed=$((completed+1))
      else
        (( status != 0 )) || status=1; failed=$((failed+1))
        printf '%s\t%s\tmass\t%s\t%s\t%s\t%s\n' "$degree" "$elements" "$target" "$status" "$console_file" "$result_file" >> "$failure_file"
        [[ "$continue_on_error" == true ]] || exit "$status"
      fi
      [[ "$cooldown_seconds" == 0 ]] || sleep "$cooldown_seconds"
    done
  done
done
echo "[done] completed=$completed failed=$failed summary=$summary_file failures=$failure_file"
(( failed == 0 ))
