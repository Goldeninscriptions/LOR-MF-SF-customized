#!/usr/bin/env bash
set -uo pipefail

# Truncated-Cholesky solver/boundary sweep. This file is also the shared
# sequential engine used by run_trunc_chol_sweep_3d.sh.
#
# Safe defaults (the script only plans; it does not launch a solver):
#   BATCH_DRY_RUN=1 RESUME=1
#   SOLVER_MODES="left_gmres_cg_jacobi"
#     available: lu
#                left_gmres_cg_jacobi right_gmres_cg_jacobi
#                left_gmres_cg_gamg right_gmres_cg_gamg
#                cg_jacobi cg_gamg amg
#     left/right_gmres_cg_{jacobi,gamg} = ordinary GMRES + inner CG
#   BOUNDARY_PROTECTION_MODES="true false"
#   TRUNC_CHOL_K=auto MPI_RANK=default MAX_MPI_RANKS=
#   TRUNC_CHOL_MAX_MEMORY_BYTES=68719476736
#   COMPUTE_TRUNCATION_CERTIFICATE=true
#   COMPUTE_UNPRECONDITIONED_SOLUTION=false
#   COMPUTE_CONDITION_NUMBER=false
#   COMPUTE_SPECTRAL_CONDITION_NUMBER=false
#   VERIFY_TENSOR_IDENTITY=false
#   OUTER_RTOL=1e-6 OUTER_ATOL=1e-9 OUTER_DTOL=1e4 OUTER_MAX_IT=10000
#   INNER_RTOL=1e-6 INNER_ATOL=1e-6 INNER_DTOL=1e4 INNER_MAX_IT=10000
# Optional filters retain the complete defaults when unset:
#   P_VALUES="4 5" N_VALUES="4 8" SOLVER_MODES="amg"
# Operational controls:
#   RUN_ID=<stable-name> RUN_DIR=<absolute-or-relative-directory>
#   TIMEOUT_SEC=0 FAIL_FAST=0
#
# Manual launch sequence:
#   RUN_ID=trial BATCH_DRY_RUN=1 ./work/run_trunc_chol_sweep_2d.sh
#   RUN_ID=trial BATCH_DRY_RUN=0 ./work/run_trunc_chol_sweep_2d.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
dimension=${TRUNC_CHOL_SWEEP_DIMENSION:-2}
case "$dimension" in
  2)
    geometry_file="$script_dir/input/square_2d.txt"
    dimension_tag="2d"
    default_n_values="4 8 16 32 64 128 256 512 1024"
    ;;
  3)
    geometry_file="$script_dir/input/cube_3d.txt"
    dimension_tag="3d"
    default_n_values="4 8 16 32 64"
    ;;
  *) echo "TRUNC_CHOL_SWEEP_DIMENSION must be 2 or 3" >&2; exit 2 ;;
esac

degrees_text=${P_VALUES:-"4 5 6 7 8"}
n_values_text=${N_VALUES:-$default_n_values}
solver_modes_text=${SOLVER_MODES:-"left_gmres_cg_jacobi"}
boundary_modes_text=${BOUNDARY_PROTECTION_MODES:-"true false"}
read -r -a degrees <<<"$degrees_text"
read -r -a n_values <<<"$n_values_text"
read -r -a solver_modes <<<"$solver_modes_text"
read -r -a boundary_modes <<<"$boundary_modes_text"

trunc_chol_k=${TRUNC_CHOL_K:-auto}
mpi_rank=${MPI_RANK:-default}
maximum_mpi_ranks=${MAX_MPI_RANKS:-}
memory_limit=${TRUNC_CHOL_MAX_MEMORY_BYTES:-68719476736}
compute_certificate=${COMPUTE_TRUNCATION_CERTIFICATE:-true}
compute_unpreconditioned=${COMPUTE_UNPRECONDITIONED_SOLUTION:-false}
compute_svd=${COMPUTE_CONDITION_NUMBER:-false}
compute_eps=${COMPUTE_SPECTRAL_CONDITION_NUMBER:-false}
verify_tensor=${VERIFY_TENSOR_IDENTITY:-false}
dry_run=${BATCH_DRY_RUN:-1}
resume=${RESUME:-1}
timeout_sec=${TIMEOUT_SEC:-0}
fail_fast=${FAIL_FAST:-0}
outer_rtol=${OUTER_RTOL:-1.0e-6}
outer_atol=${OUTER_ATOL:-1.0e-9}
outer_dtol=${OUTER_DTOL:-1.0e4}
outer_max_it=${OUTER_MAX_IT:-10000}
inner_rtol=${INNER_RTOL:-1.0e-6}
inner_atol=${INNER_ATOL:-1.0e-6}
inner_dtol=${INNER_DTOL:-1.0e4}
inner_max_it=${INNER_MAX_IT:-10000}

for boolean in "$compute_certificate" "$compute_unpreconditioned" \
               "$compute_svd" "$compute_eps" "$verify_tensor"; do
  [[ "$boolean" == true || "$boolean" == false ]] || {
    echo "Boolean environment values must be true or false" >&2; exit 2;
  }
done
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || {
  echo "BATCH_DRY_RUN must be 0 or 1" >&2; exit 2;
}
[[ "$resume" == 0 || "$resume" == 1 ]] || {
  echo "RESUME must be 0 or 1" >&2; exit 2;
}
[[ "$fail_fast" == 0 || "$fail_fast" == 1 ]] || {
  echo "FAIL_FAST must be 0 or 1" >&2; exit 2;
}
[[ "$timeout_sec" =~ ^[0-9]+$ ]] || {
  echo "TIMEOUT_SEC must be a nonnegative integer" >&2; exit 2;
}
if (( timeout_sec > 0 )) && ! command -v timeout >/dev/null 2>&1; then
  echo "TIMEOUT_SEC requires the 'timeout' command" >&2
  exit 2
fi
[[ "$memory_limit" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "TRUNC_CHOL_MAX_MEMORY_BYTES must be positive" >&2; exit 2;
}
[[ "$trunc_chol_k" == auto || "$trunc_chol_k" =~ ^[0-9]+$ ]] || {
  echo "TRUNC_CHOL_K must be auto or a nonnegative integer" >&2; exit 2;
}
[[ "$mpi_rank" == default || "$mpi_rank" =~ ^[1-9][0-9]*$ ]] || {
  echo "MPI_RANK must be default or a positive integer" >&2; exit 2;
}
for mode in "${solver_modes[@]}"; do
  case "$mode" in
    lu|left_gmres_cg_jacobi|right_gmres_cg_jacobi|left_gmres_cg_gamg|right_gmres_cg_gamg|cg_jacobi|cg_gamg|amg) ;;
    *)
    echo "Unsupported SOLVER_MODES entry: $mode" >&2; exit 2;; esac
done
for protection in "${boundary_modes[@]}"; do
  [[ "$protection" == true || "$protection" == false ]] || {
    echo "BOUNDARY_PROTECTION_MODES entries must be true or false" >&2; exit 2;
  }
done

short_commit=$(git -C "$project_dir" rev-parse --short HEAD 2>/dev/null || printf unknown)
run_id=${RUN_ID:-"$(date +%Y%m%d-%H%M%S)-$short_commit"}
run_root=${RUN_DIR:-"$script_dir/results/trunc_chol_sweep_${dimension_tag}/$run_id"}
case_dir="$run_root/cases"
log_dir="$run_root/logs"
status_dir="$run_root/status"
manifest="$run_root/manifest.tsv"
summary="$run_root/summary.tsv"
mkdir -p "$case_dir" "$log_dir" "$status_dir"

hash_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print substr($1,1,12)}'
  else
    printf '%s' "$1" | shasum -a 256 | awk '{print substr($1,1,12)}'
  fi
}

result_value() {
  awk -F ' = ' -v key="$2" '$1 == key {print $2; exit}' "$1" 2>/dev/null
}

refresh_summary() {
  printf 'case_id\tstatus\treturn_code\tstart_time\tend_time\twall_seconds\tmpi_ranks\tdimension\tp\tn\tk_requested\tk_effective\tboundary_protection\tsolver_mode\tconfig_hash\ttotal_free_dofs\tnnz_A\tnnz_P\touter_iterations\tpc_applications\tinner_iterations_total\tinner_iterations_maximum\trelative_true_residual\tl2_error\tpc_setup_seconds\touter_solve_seconds\tpc_apply_seconds\tresult_file\tconsole_file\treason\n' > "$summary.tmp"
  while IFS= read -r status_file; do
    tail -n 1 "$status_file" >> "$summary.tmp"
  done < <(find "$status_dir" -type f -name '*.tsv' | sort)
  mv "$summary.tmp" "$summary"
}

write_status() {
  local file=$1
  shift
  printf '%b\n' "$*" > "$file.tmp"
  mv "$file.tmp" "$file"
  refresh_summary
}

config_base="commit=$short_commit;dim=$dimension;k=$trunc_chol_k;memory=$memory_limit;outer=$outer_rtol,$outer_atol,$outer_dtol,$outer_max_it;inner=$inner_rtol,$inner_atol,$inner_dtol,$inner_max_it;certificate=$compute_certificate;baseline=$compute_unpreconditioned;svd=$compute_svd;eps=$compute_eps;verify=$verify_tensor"
printf 'case_id\tdimension\tp\tn\tk_requested\tk_effective\tboundary_protection\tsolver_mode\tmpi_ranks\tconfig_hash\tmemory_limit_bytes\tcommand\n' > "$manifest.tmp"

declare -a planned_records=()
for degree in "${degrees[@]}"; do
  [[ "$degree" =~ ^[4-8]$ ]] || {
    echo "P_VALUES entries must be in 4..8: $degree" >&2; exit 2;
  }
  effective_k=$trunc_chol_k
  [[ "$effective_k" == auto ]] && effective_k=$(( (degree + 1) / 2 ))
  (( effective_k < degree - 1 )) || {
    echo "Invalid k for p=$degree: effective k=$effective_k must be < p-1" >&2; exit 2;
  }
  for n in "${n_values[@]}"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || {
      echo "N_VALUES entries must be positive integers: $n" >&2; exit 2;
    }
    if [[ "$dimension" == 2 ]]; then tuple="$n,$n"; else tuple="$n,$n,$n"; fi
    if [[ "$mpi_rank" == default ]]; then
      planner_args=("$project_dir/build/case_planner" -geometry "$geometry_file"
        -refinement k_refine -p "$degree" -n "$tuple" --recommended-ranks-only)
      [[ -n "$maximum_mpi_ranks" ]] && planner_args+=(--max-ranks "$maximum_mpi_ranks")
      if [[ ! -x "${planner_args[0]}" ]]; then
        echo "Missing planner: ${planner_args[0]}; build before planning" >&2
        exit 3
      fi
      ranks=$("${planner_args[@]}") || exit 3
    else
      ranks=$mpi_rank
    fi
    for protection in "${boundary_modes[@]}"; do
      for solver_mode in "${solver_modes[@]}"; do
        case "$solver_mode" in
          lu) lor_pc=left_cg_lu; inner_pc=not_applicable ;;
          left_gmres_cg_jacobi) lor_pc=left_gmres_cg; inner_pc=jacobi ;;
          right_gmres_cg_jacobi) lor_pc=right_gmres_cg; inner_pc=jacobi ;;
          left_gmres_cg_gamg) lor_pc=left_gmres_cg; inner_pc=gamg ;;
          right_gmres_cg_gamg) lor_pc=right_gmres_cg; inner_pc=gamg ;;
          cg_jacobi) lor_pc=right_fgmres_cg; inner_pc=jacobi ;;
          cg_gamg) lor_pc=right_fgmres_cg; inner_pc=gamg ;;
          amg) lor_pc=right_fgmres_amg; inner_pc=not_applicable ;;
        esac
        config="$config_base;p=$degree;n=$tuple;boundary=$protection;solver=$solver_mode;inner_pc=$inner_pc;ranks=$ranks"
        config_hash=$(hash_text "$config")
        case_id="${dimension_tag}_p${degree}_n${n}_k${trunc_chol_k}-eff${effective_k}_bp${protection}_${solver_mode}_${config_hash}"
        result_file="$case_dir/$case_id.result.log"
        console_file="$log_dir/$case_id.console.log"
        command_text="PRECONDITIONER_MATRIX_SOURCE=truncated_cholesky_tensor TRUNC_CHOL_K=$trunc_chol_k TRUNC_CHOL_BOUNDARY_PROTECTION=$protection TRUNC_CHOL_MAX_MEMORY_BYTES=$memory_limit COMPUTE_TRUNCATION_CERTIFICATE=$compute_certificate COMPUTE_SOLUTION=true COMPUTE_L2_ERROR=true COMPUTE_UNPRECONDITIONED_SOLUTION=$compute_unpreconditioned COMPUTE_CONDITION_NUMBER=$compute_svd COMPUTE_SPECTRAL_CONDITION_NUMBER=$compute_eps VERIFY_TENSOR_IDENTITY=$verify_tensor OUTER_RTOL=$outer_rtol OUTER_ATOL=$outer_atol OUTER_DTOL=$outer_dtol OUTER_MAX_IT=$outer_max_it"
        if [[ "$inner_pc" != not_applicable ]]; then
          command_text+=" INNER_PC=$inner_pc INNER_RTOL=$inner_rtol INNER_ATOL=$inner_atol INNER_DTOL=$inner_dtol INNER_MAX_IT=$inner_max_it"
        fi
        command_text+=" ./work/run_case.sh $ranks $geometry_file k_refine $degree $tuple greville $lor_pc $result_file"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$case_id" "$dimension" "$degree" "$tuple" "$trunc_chol_k" \
          "$effective_k" "$protection" "$solver_mode" "$ranks" \
          "$config_hash" "$memory_limit" "$command_text" >> "$manifest.tmp"
        planned_records+=("$case_id|$degree|$tuple|$effective_k|$protection|$solver_mode|$lor_pc|$inner_pc|$ranks|$config_hash|$result_file|$console_file")
        status_file="$status_dir/$case_id.tsv"
        if [[ ! -f "$status_file" ]]; then
          write_status "$status_file" "$case_id\tpending\tNA\tNA\tNA\tNA\t$ranks\t$dimension\t$degree\t$tuple\t$trunc_chol_k\t$effective_k\t$protection\t$solver_mode\t$config_hash\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t$result_file\t$console_file\tnot_started"
        fi
      done
    done
  done
done
mv "$manifest.tmp" "$manifest"
refresh_summary

echo "Planned ${#planned_records[@]} sequential cases in $run_root"
echo "Manifest: $manifest"
echo "Summary:  $summary"
if [[ "$dry_run" == 1 ]]; then
  echo "Dry run only. Set BATCH_DRY_RUN=0 with the same RUN_ID to execute."
  exit 0
fi

failed=0
for record in "${planned_records[@]}"; do
  IFS='|' read -r case_id degree tuple effective_k protection solver_mode \
    lor_pc inner_pc ranks config_hash result_file console_file <<<"$record"
  status_file="$status_dir/$case_id.tsv"
  success_marker="$status_dir/$case_id.success"
  if [[ "$resume" == 1 && -s "$result_file" && -f "$success_marker" &&
        "$(<"$success_marker")" == "$config_hash" ]]; then
    echo "[resume] $case_id already successful"
    continue
  fi

  result_tmp="$case_dir/.$case_id.result.tmp.$$"
  console_tmp="$log_dir/.$case_id.console.tmp.$$"
  start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  start_epoch=$(date +%s)
  env_args=(
    PRECONDITIONER_MATRIX_SOURCE=truncated_cholesky_tensor
    TRUNC_CHOL_K="$trunc_chol_k"
    TRUNC_CHOL_BOUNDARY_PROTECTION="$protection"
    TRUNC_CHOL_MAX_MEMORY_BYTES="$memory_limit"
    COMPUTE_TRUNCATION_CERTIFICATE="$compute_certificate"
    COMPUTE_SOLUTION=true COMPUTE_L2_ERROR=true
    COMPUTE_UNPRECONDITIONED_SOLUTION="$compute_unpreconditioned"
    COMPUTE_CONDITION_NUMBER="$compute_svd"
    COMPUTE_SPECTRAL_CONDITION_NUMBER="$compute_eps"
    VERIFY_TENSOR_IDENTITY="$verify_tensor"
    OUTER_RTOL="$outer_rtol" OUTER_ATOL="$outer_atol"
    OUTER_DTOL="$outer_dtol" OUTER_MAX_IT="$outer_max_it"
  )
  if [[ "$inner_pc" != not_applicable ]]; then
    env_args+=(INNER_PC="$inner_pc" INNER_RTOL="$inner_rtol"
      INNER_ATOL="$inner_atol" INNER_DTOL="$inner_dtol"
      INNER_MAX_IT="$inner_max_it")
  fi
  command=(env "${env_args[@]}" "$script_dir/run_case.sh" "$ranks"
    "$geometry_file" k_refine "$degree" "$tuple" greville "$lor_pc" "$result_tmp")
  printf '[start] %s\n[command]' "$case_id" > "$console_tmp"
  printf ' %q' "${command[@]}" >> "$console_tmp"
  printf '\n' >> "$console_tmp"
  if (( timeout_sec > 0 )); then
    timeout "$timeout_sec" "${command[@]}" >> "$console_tmp" 2>&1
  else
    "${command[@]}" >> "$console_tmp" 2>&1
  fi
  return_code=$?
  end_epoch=$(date +%s)
  end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  wall_seconds=$((end_epoch - start_epoch))

  status=failed
  reason="driver_returned_$return_code"
  if [[ $return_code -eq 124 ]]; then
    status=timeout; reason=timeout
  elif grep -q 'preflight conservative estimate.*exceeds configured limit' "$console_tmp"; then
    status=skipped_memory; reason=preconditioner_memory_preflight
  elif [[ $return_code -eq 0 && -s "$result_tmp" ]]; then
    status=success; reason=completed
  fi
  mv "$console_tmp" "$console_file"
  if [[ "$status" == success ]]; then
    mv "$result_tmp" "$result_file"
    printf '%s' "$config_hash" > "$success_marker"
  elif [[ -e "$result_tmp" ]]; then
    mv "$result_tmp" "$case_dir/$case_id.partial.log"
  fi

  source_result=$result_file
  [[ -s "$source_result" ]] || source_result=/dev/null
  dofs=$(result_value "$source_result" total_free_dofs); dofs=${dofs:-NA}
  nnz_a=$(result_value "$source_result" nnz_K); nnz_a=${nnz_a:-NA}
  nnz_p=$(result_value "$source_result" nnz_P); nnz_p=${nnz_p:-NA}
  outer=$(result_value "$source_result" outer_iterations); outer=${outer:-NA}
  applies=$(result_value "$source_result" pc_applications); applies=${applies:-NA}
  inner=$(result_value "$source_result" inner_iterations_total); inner=${inner:-NA}
  inner_max=$(result_value "$source_result" "inner_iterations_maximum($lor_pc)"); inner_max=${inner_max:-NA}
  if [[ "$solver_mode" == amg || "$solver_mode" == lu ]]; then
    inner=not_applicable
    inner_max=not_applicable
  fi
  residual=$(result_value "$source_result" relative_true_residual); residual=${residual:-NA}
  l2=$(result_value "$source_result" l2_error); l2=${l2:-NA}
  setup=$(result_value "$source_result" time_pc_ksp_setup_s); setup=${setup:-NA}
  outer_time=$(result_value "$source_result" time_outer_solve_s); outer_time=${outer_time:-NA}
  apply_time=$(result_value "$source_result" time_pc_apply_total_s); apply_time=${apply_time:-NA}
  write_status "$status_file" "$case_id\t$status\t$return_code\t$start_iso\t$end_iso\t$wall_seconds\t$ranks\t$dimension\t$degree\t$tuple\t$trunc_chol_k\t$effective_k\t$protection\t$solver_mode\t$config_hash\t$dofs\t$nnz_a\t$nnz_p\t$outer\t$applies\t$inner\t$inner_max\t$residual\t$l2\t$setup\t$outer_time\t$apply_time\t$result_file\t$console_file\t$reason"
  echo "[$status] $case_id (${wall_seconds}s, rc=$return_code)"
  if [[ "$status" == failed || "$status" == timeout ]]; then
    failed=$((failed + 1))
    [[ "$fail_fast" == 1 ]] && break
  fi
done

refresh_summary
echo "Sweep finished with $failed failed/timeout cases. Summary: $summary"
(( failed == 0 ))
