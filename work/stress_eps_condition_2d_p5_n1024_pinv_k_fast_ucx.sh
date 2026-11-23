#!/usr/bin/env bash
set -euo pipefail

# Repeated 56-rank stability/performance test for p=5, n=1024, P^{-1}K.
# Keep UCX's native high-speed shared-memory transport, but enlarge and drain
# its MM receive FIFO more aggressively to avoid the observed pending-queue
# assertion under PETSc GAMG communication bursts.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${STRESS_ROOT:-$script_dir/results/eps_condition_stress_2d_p5_n1024_pinv_k}
log_root=${STRESS_LOG_ROOT:-$script_dir/logs/eps_condition_stress_2d_p5_n1024_pinv_k}
summary=${STRESS_SUMMARY:-$root/summary.tsv}
repeats=${REPEATS:-3}
ranks=${MPI_RANKS:-56}
fifo_size=${UCX_MM_FIFO_SIZE:-4096}
fifo_max_poll=${UCX_MM_FIFO_MAX_POLL:-64}

for value in "$repeats" "$ranks" "$fifo_size" "$fifo_max_poll"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS, MPI_RANKS and UCX MM FIFO controls must be positive integers" >&2
    exit 2
  fi
done

# UCX requires a power-of-two MM FIFO size.
if (( (fifo_size & (fifo_size - 1)) != 0 )); then
  echo "UCX_MM_FIFO_SIZE must be a power of two" >&2
  exit 2
fi

mkdir -p "$root" "$log_root"
printf 'repeat\tstatus\tmpi_ranks\tucx_mm_fifo_size\tucx_mm_fifo_max_poll\tlambda_min\tlambda_max\tcondition_number\teps_iterations_min\teps_iterations_max\ttime_spectral_eps_s\ttime_case_total_s\tresult_file\tconsole_file\n' > "$summary"

# An inherited UCX_TLS would invalidate this test. Native transport selection
# is required so UCX can use its fast MM shared-memory path.
unset UCX_TLS
export UCX_MM_FIFO_SIZE=$fifo_size
export UCX_MM_FIFO_MAX_POLL=$fifo_max_poll
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

failures=0
for ((repeat=1; repeat<=repeats; ++repeat)); do
  tag=$(printf 'run_%02d_r%d_fifo%d_poll%d' "$repeat" "$ranks" "$fifo_size" "$fifo_max_poll")
  result_file="$root/$tag.log"
  console_file="$log_root/$tag.console.log"
  echo "[stress $repeat/$repeats] ranks=$ranks native_UCX MM_FIFO_SIZE=$fifo_size MM_FIFO_MAX_POLL=$fifo_max_poll"

  if MPI_RANKS=$ranks \
      "$script_dir/run_eps_condition_2d.sh" \
        5 1024 pinv_k "$result_file" \
        -eigmax_eps_monitor \
        -eigmin_eps_monitor \
        2>&1 | tee "$console_file"; then
    if [[ -f "$result_file" ]] &&
       grep -qx 'eigensolver_status = converged' "$result_file"; then
      status=converged
    else
      status=missing_result
      failures=$((failures + 1))
    fi
  else
    status=failed
    failures=$((failures + 1))
  fi

  awk -F ' = ' \
    -v repeat="$repeat" -v status="$status" -v ranks="$ranks" \
    -v fifo="$fifo_size" -v poll="$fifo_max_poll" \
    -v result="$result_file" -v console="$console_file" '
      $1=="lambda_min" {lmin=$2}
      $1=="lambda_max" {lmax=$2}
      $1=="eps_condition_number" {kappa=$2}
      $1=="eps_iterations_min" {imin=$2}
      $1=="eps_iterations_max" {imax=$2}
      $1=="time_spectral_eps_s" {eps_time=$2}
      $1=="time_case_total_s" {case_time=$2}
      END {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", repeat,status,ranks,fifo,poll,lmin,lmax,kappa,imin,imax,eps_time,case_time,result,console}
    ' "$result_file" 2>/dev/null >> "$summary" || \
    printf '%s\t%s\t%s\t%s\t%s\t\t\t\t\t\t\t\t%s\t%s\n' \
      "$repeat" "$status" "$ranks" "$fifo_size" "$fifo_max_poll" \
      "$result_file" "$console_file" >> "$summary"
done

echo "[stress done] repeats=$repeats failures=$failures summary=$summary"
(( failures == 0 ))
