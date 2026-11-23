#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
executable=${EXECUTABLE:-$project_dir/build/distributed_greville_mass_benchmark}
degree=${P:-3}
elements=${N:-1024}
case_name="distributed_greville_mass_p${degree}_n${elements}"
result_root=${RESULT_ROOT:-$script_dir/results/$case_name}
log_root=${LOG_ROOT:-$script_dir/logs/$case_name}
summary=${SUMMARY:-$result_root/summary.tsv}
repeats=${REPEATS:-3}
ranks_list=${RANKS_LIST:-"1 2 4 8 16 32 56"}
modes=${MODES:-"tensor_transpose distributed_band"}

if [[ ! -x "$executable" ]]; then
  echo "Missing executable: $executable" >&2
  exit 3
fi
if ! [[ "$repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$degree" =~ ^[1-9][0-9]*$ && "$elements" =~ ^[1-9][0-9]*$ ]]; then
  echo "P and N must be positive integers" >&2
  exit 2
fi

mkdir -p "$result_root" "$log_root"
printf 'mode\tranks\trepeat\tstatus\tp\tn\tdofs\titerations\treason\trelative_residual\trelative_solution_error\tsetup_s\tsolve_s\toperator_s\tpc_s\tpc_comm_s\tcommunicated_bytes\tfactor_bytes_per_rank\tworkspace_bytes_per_rank\tprocess_before_bytes\tprocess_after_bytes\tlog_file\n' > "$summary"

# Preserve UCX native transport selection while retaining the MM FIFO
# stabilization already validated by the large EPS experiments.
unset UCX_TLS
export UCX_MM_FIFO_SIZE=${UCX_MM_FIFO_SIZE:-4096}
export UCX_MM_FIFO_MAX_POLL=${UCX_MM_FIFO_MAX_POLL:-64}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

failures=0
for ranks in $ranks_list; do
  if ! [[ "$ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "Every RANKS_LIST entry must be a positive integer" >&2
    exit 2
  fi
  for mode in $modes; do
    case "$mode" in
      tensor_transpose|distributed_band) ;;
      *) echo "Unknown mode: $mode" >&2; exit 2 ;;
    esac
    for ((repeat=1; repeat<=repeats; ++repeat)); do
      tag="${mode}_r${ranks}_repeat${repeat}"
      log_file="$log_root/$tag.log"
      echo "[benchmark] mode=$mode ranks=$ranks repeat=$repeat/$repeats"
      set +e
      mpirun -n "$ranks" "$executable" -p "$degree" -n "$elements" \
        -mode "$mode" -rtol 1.0e-8 >"$log_file" 2>&1
      exit_status=$?
      set -e
      result_line=$(grep '^RESULT ' "$log_file" | tail -n 1 || true)
      if (( exit_status == 0 )) && [[ -n "$result_line" ]]; then
        status=converged
      else
        status=failed
        failures=$((failures + 1))
      fi
      awk -v mode="$mode" -v ranks="$ranks" -v repeat="$repeat" \
          -v status="$status" -v logfile="$log_file" '
        BEGIN {
          p=n=dofs=iterations=reason=residual=error=setup=solve=operator=pc=comm=bytes=factor=workspace=before=after=""
        }
        /^RESULT / {
          for (i=2; i<=NF; ++i) {
            split($i, field, "=")
            value[field[1]]=field[2]
          }
        }
        END {
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
            mode,ranks,repeat,status,value["p"],value["n"],value["dofs"],value["iterations"],value["reason"],value["relative_residual"],value["relative_solution_error"],value["setup_s"],value["solve_s"],value["operator_s"],value["pc_s"],value["pc_comm_s"],value["communicated_bytes"],value["factor_bytes_per_rank"],value["workspace_bytes_per_rank"],value["process_before_bytes"],value["process_after_bytes"],logfile
        }
      ' "$log_file" >> "$summary"
    done
  done
done

echo "[benchmark done] failures=$failures summary=$summary"
(( failures == 0 ))
