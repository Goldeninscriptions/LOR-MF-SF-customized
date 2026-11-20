#!/usr/bin/env bash
set -uo pipefail

# Shared selected-P comparison implementation for the dedicated 2D/3D entry
# scripts. INCLUDE_LEFT_GMRES controls whether A4/B5 are included.
# Safe default: create the manifest only; launch nothing until BATCH_DRY_RUN=0.
#
# A1: truncated P, right FGMRES, inner CG+GAMG
# A2: truncated P, right FGMRES, direct GAMG
# A3: truncated P, right FGMRES, direct diag(P)^-1 Jacobi
# B1: Greville LOR P, right FGMRES, inner CG+GAMG
# B2: Greville LOR P, right FGMRES, direct GAMG
# B3: Greville LOR P, right FGMRES, direct diag(P)^-1 Jacobi
# B4: Greville LOR P, right FGMRES, one-step inner CG+Jacobi
# A4: truncated P, left GMRES, direct GAMG
# B5: Greville LOR P, left GMRES, direct GAMG

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
benchmark_dimension=${BENCHMARK_DIMENSION:-2}
include_left_gmres=${INCLUDE_LEFT_GMRES:-1}
include_direct_k_baselines=${INCLUDE_DIRECT_K_BASELINES:-0}
comparison_method_set=${COMPARISON_METHOD_SET:-selected_p}
[[ "$comparison_method_set" == selected_p ||
   "$comparison_method_set" == direct_k_baseline ]] || {
  echo "COMPARISON_METHOD_SET must be selected_p or direct_k_baseline" >&2
  exit 2
}
if [[ "$benchmark_dimension" == 2 ]]; then
  geometry_file=${GEOMETRY_FILE:-"$script_dir/input/square_2d.txt"}
  default_element_cases="512,512 1024,1024"
  default_result_group=pc_comparison_2d
  default_run_id=p4-p7-n512-n1024-selected-p-nine-way
elif [[ "$benchmark_dimension" == 3 ]]; then
  geometry_file=${GEOMETRY_FILE:-"$script_dir/input/cube_3d.txt"}
  default_element_cases="32,32,32 64,64,64"
  default_result_group=pc_comparison_3d
  default_run_id=p4-p7-n32-n64-selected-p-with-k-baselines
else
  echo "BENCHMARK_DIMENSION must be 2 or 3" >&2
  exit 2
fi

read -r -a degrees <<<"${DEGREES:-4 5 6 7}"
read -r -a element_cases <<<"${ELEMENT_CASES:-$default_element_cases}"
mpi_rank_override=${MPI_RANK:-}
mpi_rank_512=${MPI_RANK_512:-27}
mpi_rank_1024=${MPI_RANK_1024:-107}
dry_run=${BATCH_DRY_RUN:-1}
resume=${RESUME:-1}
timeout_sec=${TIMEOUT_SEC:-0}
fail_fast=${FAIL_FAST:-0}
memory_limit=${TRUNC_CHOL_MAX_MEMORY_BYTES:-68719476736}
outer_rtol=${OUTER_RTOL:-1.0e-6}
outer_atol=${OUTER_ATOL:-1.0e-9}
outer_dtol=${OUTER_DTOL:-1.0e4}
outer_max_it=${OUTER_MAX_IT:-10000}
inner_rtol=${INNER_RTOL:-1.0e-3}
inner_atol=${INNER_ATOL:-1.0e-12}
inner_dtol=${INNER_DTOL:-1.0e4}
inner_max_it=${INNER_MAX_IT:-10000}
trunc_chol_k=${TRUNC_CHOL_K:-auto}
boundary_protection=${TRUNC_CHOL_BOUNDARY_PROTECTION:-true}
gamg_threshold=${GAMG_THRESHOLD:-0.0}
gamg_smooth_steps=${GAMG_SMOOTH_STEPS:-1}

for boolean in "$dry_run" "$resume" "$fail_fast" "$include_left_gmres" \
  "$include_direct_k_baselines"; do
  [[ "$boolean" == 0 || "$boolean" == 1 ]] || {
    echo "BATCH_DRY_RUN, RESUME, FAIL_FAST, INCLUDE_LEFT_GMRES, and INCLUDE_DIRECT_K_BASELINES must be 0 or 1" >&2
    exit 2
  }
done
rank_values=("$mpi_rank_512" "$mpi_rank_1024")
[[ -n "$mpi_rank_override" ]] && rank_values+=("$mpi_rank_override")
for rank_value in "${rank_values[@]}"; do
  [[ "$rank_value" == default || "$rank_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "MPI rank settings must be 'default' or positive integers" >&2
    exit 2
  }
done
[[ "$timeout_sec" =~ ^[0-9]+$ ]] || {
  echo "TIMEOUT_SEC must be a nonnegative integer" >&2; exit 2;
}
[[ "$gamg_smooth_steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "GAMG_SMOOTH_STEPS must be a positive integer" >&2; exit 2;
}
if (( timeout_sec > 0 )) && ! command -v timeout >/dev/null 2>&1; then
  echo "TIMEOUT_SEC requires the timeout command" >&2
  exit 2
fi
[[ "$boundary_protection" == true || "$boundary_protection" == false ]] || {
  echo "TRUNC_CHOL_BOUNDARY_PROTECTION must be true or false" >&2; exit 2;
}
[[ "$trunc_chol_k" == auto || "$trunc_chol_k" =~ ^[0-9]+$ ]] || {
  echo "TRUNC_CHOL_K must be auto or a nonnegative integer" >&2; exit 2;
}
for degree_value in "${degrees[@]}"; do
  [[ "$degree_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "DEGREES must contain positive integers" >&2; exit 2;
  }
done
for tuple_value in "${element_cases[@]}"; do
  if [[ "$benchmark_dimension" == 2 ]]; then
    [[ "$tuple_value" =~ ^[1-9][0-9]*,[1-9][0-9]*$ ]] || {
      echo "ELEMENT_CASES must contain comma-separated 2D tuples" >&2; exit 2;
    }
  else
    [[ "$tuple_value" =~ ^[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$ ]] || {
      echo "ELEMENT_CASES must contain comma-separated 3D tuples" >&2; exit 2;
    }
  fi
done

hash_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print substr($1,1,16)}'
  else
    printf '%s' "$1" | shasum -a 256 | awk '{print substr($1,1,16)}'
  fi
}

# Include relevant working-tree contents, not only HEAD, in resume identity.
solver_fingerprint=$(
  find "$project_dir/include" "$project_dir/src" -type f \
    \( -name '*.hpp' -o -name '*.cpp' \) -print0 |
    sort -z | xargs -0 git hash-object
  git hash-object "$script_dir/run_case.sh" \
    "$script_dir/run_pc_comparison_p7_n512_2d.sh"
)
solver_fingerprint=$(hash_text "$solver_fingerprint")
short_commit=$(git -C "$project_dir" rev-parse --short HEAD 2>/dev/null || printf unknown)
run_id=${RUN_ID:-$default_run_id}
result_group=${RESULT_GROUP:-$default_result_group}
run_root=${RUN_DIR:-"$script_dir/results/$result_group/$run_id"}
case_dir="$run_root/cases"
log_dir="$run_root/logs"
status_dir="$run_root/status"
manifest="$run_root/manifest.tsv"
summary="$run_root/summary.tsv"
mkdir -p "$case_dir" "$log_dir" "$status_dir"

result_value() {
  awk -F ' = ' -v key="$2" '$1 == key {print $2; exit}' "$1" 2>/dev/null
}

value_or_na() {
  local value
  value=$(result_value "$1" "$2")
  printf '%s' "${value:-NA}"
}

manifest_header='case_id\tconfiguration\tmatrix_source\touter_mode\tapplication\tinner_pc\tinner_max_it\tp\tn\tmpi_ranks\touter_rtol\touter_atol\tinner_rtol\tinner_atol\ttrunc_chol_k_effective\tboundary_protection\tgamg_threshold\tgamg_smooth_steps\tcommit\tsolver_fingerprint\tconfig_hash\tcommand'
summary_header='case_id\tconfiguration\tstatus\treturn_code\tmatrix_source\touter_mode\tapplication\tinner_pc\tinner_max_it\tp\tn\tmpi_ranks\ttotal_free_dofs\tnnz_K\tnnz_P\tmemory_bytes_K\tmemory_bytes_P\tallocated_nnz_K\tallocated_nnz_P\tallocation_calls_K\tallocation_calls_P\tT_K_assembly_s\tT_P_construction_s\tT_AMG_setup_s\tT_outer_setup_s\tT_outer_solve_s\tT_pc_apply_s\tT_inner_CG_s\tT_L2_s\tT_end_to_end_s\tN_outer\tN_pc_apply\tN_GAMG_apply\tN_inner_total\tN_inner_average\tN_inner_maximum\tN_inner_failed\tinner_last_reason\touter_reason\touter_failed\trelative_true_residual\tL2_error\tresult_file\tconsole_file\treason'
printf '%b\n' "$manifest_header" > "$manifest.tmp"
printf '%b\n' "$summary_header" > "$summary.tmp"

declare -a records=()
add_method() {
  local id=$1 source=$2 outer_mode=$3 application=$4 inner_pc=$5 method_inner_max=$6
  local source_label=$source
  if [[ "$application" == direct_k_jacobi ||
        "$application" == direct_k_gamg ]]; then
    source_label=high_order_K
  fi
  local gamg_options=not_applicable
  if [[ "$application" == cg_gamg || "$application" == direct_gamg ||
        "$application" == direct_k_gamg ]]; then
    gamg_options="threshold=$gamg_threshold,smooth=$gamg_smooth_steps"
  fi
  local config="fingerprint=$solver_fingerprint;method_set=$comparison_method_set;dim=$benchmark_dimension;geometry=$geometry_file;id=$id;source=$source;outer=$outer_mode;application=$application;inner_pc=$inner_pc;inner_max=$method_inner_max;p=$degree;n=$element_tuple;ranks=$mpi_rank;max_ranks=${MAX_MPI_RANKS:-};outer_tol=$outer_rtol,$outer_atol,$outer_dtol,$outer_max_it;inner_tol=$inner_rtol,$inner_atol,$inner_dtol;trunc_k=$effective_k;bp=$boundary_protection;gamg=$gamg_options;memory=$memory_limit"
  local config_hash
  config_hash=$(hash_text "$config")
  local tuple_label=${element_tuple//,/-}
  local case_id="${id}_p${degree}_n${tuple_label}_${config_hash}"
  local result_file="$case_dir/$case_id.result.log"
  local console_file="$log_dir/$case_id.console.log"
  local command_text="PRECONDITIONER_MATRIX_SOURCE=$source OUTER_RTOL=$outer_rtol OUTER_ATOL=$outer_atol"
  if [[ "$inner_pc" != not_applicable ]]; then
    command_text+=" INNER_PC=$inner_pc INNER_RTOL=$inner_rtol INNER_ATOL=$inner_atol INNER_MAX_IT=$method_inner_max"
  fi
  if [[ "$application" == cg_gamg || "$application" == direct_gamg ]]; then
    command_text+=" PETSC_OPTIONS='-inner_pc_gamg_threshold $gamg_threshold -inner_mg_levels_ksp_max_it $gamg_smooth_steps'"
  elif [[ "$application" == direct_k_gamg ]]; then
    command_text+=" PETSC_OPTIONS='-pc_gamg_threshold $gamg_threshold -mg_levels_ksp_max_it $gamg_smooth_steps'"
  fi
  command_text+=" ./work/run_case.sh $mpi_rank $geometry_file k_refine $degree $element_tuple greville $outer_mode $result_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$id" "$source_label" "$outer_mode" "$application" \
    "$inner_pc" "$method_inner_max" "$degree" "$element_tuple" \
    "$mpi_rank" "$outer_rtol" "$outer_atol" "$inner_rtol" \
    "$inner_atol" "$effective_k" "$boundary_protection" \
    "$gamg_threshold" "$gamg_smooth_steps" "$short_commit" \
    "$solver_fingerprint" "$config_hash" "$command_text" >> "$manifest.tmp"
  records+=("$case_id|$id|$source|$outer_mode|$application|$inner_pc|$method_inner_max|$degree|$element_tuple|$mpi_rank|$effective_k|$config_hash|$result_file|$console_file")
}

for degree in "${degrees[@]}"; do
  effective_k=$trunc_chol_k
  [[ "$effective_k" == auto ]] && effective_k=$(( (degree + 1) / 2 ))
  (( effective_k < degree - 1 )) || {
    echo "Invalid truncated-Cholesky k=$effective_k for p=$degree" >&2; exit 2;
  }
  for element_tuple in "${element_cases[@]}"; do
    if [[ -n "$mpi_rank_override" ]]; then
      mpi_rank=$mpi_rank_override
    elif [[ "$element_tuple" == 512,512 ]]; then
      mpi_rank=$mpi_rank_512
    elif [[ "$element_tuple" == 1024,1024 ]]; then
      mpi_rank=$mpi_rank_1024
    else
      echo "Set MPI_RANK for nonstandard ELEMENT_CASES tuple $element_tuple" >&2
      exit 2
    fi
    if [[ "$comparison_method_set" == direct_k_baseline ]]; then
      add_method K1 lor_q1 left_gmres_k_jacobi direct_k_jacobi not_applicable not_applicable
      add_method K2 lor_q1 left_gmres_k_gamg direct_k_gamg not_applicable not_applicable
    else
      add_method A1 truncated_cholesky_tensor right_fgmres_cg cg_gamg gamg "$inner_max_it"
      add_method A2 truncated_cholesky_tensor right_fgmres_amg direct_gamg not_applicable not_applicable
      add_method A3 truncated_cholesky_tensor right_fgmres_jacobi direct_jacobi not_applicable not_applicable
      if [[ "$include_left_gmres" == 1 ]]; then
        add_method A4 truncated_cholesky_tensor left_gmres_amg direct_gamg not_applicable not_applicable
      fi
      add_method B1 lor_q1 right_fgmres_cg cg_gamg gamg "$inner_max_it"
      add_method B2 lor_q1 right_fgmres_amg direct_gamg not_applicable not_applicable
      add_method B3 lor_q1 right_fgmres_jacobi direct_jacobi not_applicable not_applicable
      add_method B4 lor_q1 right_fgmres_cg cg_jacobi_one jacobi 1
      if [[ "$include_left_gmres" == 1 ]]; then
        add_method B5 lor_q1 left_gmres_amg direct_gamg not_applicable not_applicable
      fi
      if [[ "$include_direct_k_baselines" == 1 ]]; then
        add_method K1 lor_q1 left_gmres_k_jacobi direct_k_jacobi not_applicable not_applicable
        add_method K2 lor_q1 left_gmres_k_gamg direct_k_gamg not_applicable not_applicable
      fi
    fi
  done
done

mv "$manifest.tmp" "$manifest"
mv "$summary.tmp" "$summary"
echo "Planned ${#records[@]} selected-P cases for p={${degrees[*]}} and n={${element_cases[*]}}."
echo "Manifest: $manifest"
echo "Summary:  $summary"
if [[ "$dry_run" == 1 ]]; then
  echo "Dry run only. Re-run with the same RUN_ID and BATCH_DRY_RUN=0."
  exit 0
fi

append_summary() {
  local case_id=$1 id=$2 status=$3 return_code=$4 source=$5 outer_mode=$6
  local application=$7 inner_pc=$8 method_inner_max=$9 degree=${10}
  local element_tuple=${11} result_file=${12} console_file=${13} reason=${14}
  local input=$result_file
  [[ -s "$input" ]] || input=/dev/null

  local ranks dofs nnz_k nnz_p mem_k mem_p alloc_k alloc_p calls_k calls_p
  local t_k t_p t_amg t_setup t_solve t_apply t_inner t_l2 t_total
  local n_outer n_apply n_amg n_inner n_avg n_max n_failed inner_reason
  local outer_reason outer_failed residual l2
  ranks=$(value_or_na "$input" mpi_ranks)
  dofs=$(value_or_na "$input" total_free_dofs)
  nnz_k=$(value_or_na "$input" nnz_K)
  nnz_p=$(value_or_na "$input" nnz_P)
  mem_k=$(value_or_na "$input" memory_bytes_K)
  mem_p=$(value_or_na "$input" memory_bytes_P)
  alloc_k=$(value_or_na "$input" 'allocated_nnz(K)')
  alloc_p=$(value_or_na "$input" 'allocated_nnz(K_LOR)')
  calls_k=$(value_or_na "$input" 'allocation_calls(K)')
  calls_p=$(value_or_na "$input" 'allocation_calls(K_LOR)')
  t_k=$(value_or_na "$input" time_original_matrix_assembly_s)
  t_p=$(value_or_na "$input" time_preconditioner_matrix_construction_s)
  t_setup=$(value_or_na "$input" "setup_time_seconds($outer_mode)")
  t_solve=$(value_or_na "$input" time_outer_solve_s)
  t_apply=$(value_or_na "$input" time_pc_apply_total_s)
  t_l2=$(value_or_na "$input" time_l2_error_s)
  t_total=$(value_or_na "$input" time_case_total_s)
  n_outer=$(value_or_na "$input" outer_iterations)
  n_apply=$(value_or_na "$input" pc_applications)
  outer_reason=$(value_or_na "$input" ksp_reason)
  residual=$(value_or_na "$input" relative_true_residual)
  l2=$(value_or_na "$input" l2_error)
  if [[ "$application" == direct_k_jacobi ||
        "$application" == direct_k_gamg ]]; then
    # K is both system and PC operator; no separate P was assembled.
    source=high_order_K
    nnz_p=$nnz_k
    mem_p=$mem_k
    alloc_p=$alloc_k
    calls_p=$calls_k
    t_p=not_applicable
    n_apply=not_recorded
    t_apply=not_separately_timed
  fi
  outer_failed=NA
  if [[ "$outer_reason" =~ ^-?[0-9]+$ ]]; then
    (( outer_reason < 0 )) && outer_failed=true || outer_failed=false
  fi

  t_amg=not_applicable
  n_amg=not_applicable
  if [[ "$application" == cg_gamg || "$application" == direct_gamg ]]; then
    t_amg=$(value_or_na "$input" time_pc_ksp_setup_s)
    if [[ "$application" == direct_gamg ]]; then
      n_amg=$n_apply
    else
      n_amg=not_recorded
    fi
  elif [[ "$application" == direct_k_gamg ]]; then
    t_amg=not_separately_timed
    n_amg=not_recorded
  fi

  if [[ "$inner_pc" == not_applicable ]]; then
    t_inner=not_applicable
    n_inner=not_applicable
    n_avg=not_applicable
    n_max=not_applicable
    n_failed=not_applicable
    inner_reason=not_applicable
  else
    t_inner=$(value_or_na "$input" "inner_solve_time_seconds($outer_mode)")
    n_inner=$(value_or_na "$input" inner_iterations_total)
    n_avg=$(value_or_na "$input" "inner_iterations_average($outer_mode)")
    n_max=$(value_or_na "$input" "inner_iterations_maximum($outer_mode)")
    n_failed=$(value_or_na "$input" "inner_failed_applications($outer_mode)")
    inner_reason=$(value_or_na "$input" "inner_last_reason($outer_mode)")
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$id" "$status" "$return_code" "$source" "$outer_mode" \
    "$application" "$inner_pc" "$method_inner_max" "$degree" \
    "$element_tuple" "$ranks" "$dofs" \
    "$nnz_k" "$nnz_p" "$mem_k" "$mem_p" "$alloc_k" "$alloc_p" \
    "$calls_k" "$calls_p" "$t_k" "$t_p" "$t_amg" "$t_setup" \
    "$t_solve" "$t_apply" "$t_inner" "$t_l2" "$t_total" "$n_outer" \
    "$n_apply" "$n_amg" "$n_inner" "$n_avg" "$n_max" "$n_failed" \
    "$inner_reason" "$outer_reason" "$outer_failed" "$residual" "$l2" \
    "$result_file" "$console_file" "$reason" >> "$summary"
}

failed=0
for record in "${records[@]}"; do
  IFS='|' read -r case_id id source outer_mode application inner_pc \
    method_inner_max degree element_tuple mpi_rank effective_k config_hash \
    result_file console_file <<<"$record"
  success_marker="$status_dir/$case_id.success"
  if [[ "$resume" == 1 && -s "$result_file" && -f "$success_marker" &&
        "$(<"$success_marker")" == "$config_hash" ]]; then
    echo "[resume] $case_id"
    append_summary "$case_id" "$id" resumed 0 "$source" "$outer_mode" \
      "$application" "$inner_pc" "$method_inner_max" "$degree" \
      "$element_tuple" "$result_file" "$console_file" \
      existing_success_with_matching_hash
    continue
  fi

  result_tmp="$case_dir/.$case_id.result.tmp.$$"
  console_tmp="$log_dir/.$case_id.console.tmp.$$"
  env_args=(
    PRECONDITIONER_MATRIX_SOURCE="$source"
    TRUNC_CHOL_K="$trunc_chol_k"
    TRUNC_CHOL_BOUNDARY_PROTECTION="$boundary_protection"
    TRUNC_CHOL_MAX_MEMORY_BYTES="$memory_limit"
    COMPUTE_TRUNCATION_CERTIFICATE=true
    COMPUTE_SOLUTION=true COMPUTE_L2_ERROR=true
    COMPUTE_UNPRECONDITIONED_SOLUTION=false
    COMPUTE_CONDITION_NUMBER=false
    COMPUTE_SPECTRAL_CONDITION_NUMBER=false
    VERIFY_TENSOR_IDENTITY=false
    OUTER_RTOL="$outer_rtol" OUTER_ATOL="$outer_atol"
    OUTER_DTOL="$outer_dtol" OUTER_MAX_IT="$outer_max_it"
  )
  if [[ "$inner_pc" != not_applicable ]]; then
    env_args+=(INNER_PC="$inner_pc" INNER_RTOL="$inner_rtol"
      INNER_ATOL="$inner_atol" INNER_DTOL="$inner_dtol"
      INNER_MAX_IT="$method_inner_max")
  fi
  if [[ "$application" == cg_gamg || "$application" == direct_gamg ]]; then
    env_args+=(PETSC_OPTIONS="${PETSC_OPTIONS:-} -inner_pc_gamg_threshold $gamg_threshold -inner_mg_levels_ksp_max_it $gamg_smooth_steps")
  elif [[ "$application" == direct_k_gamg ]]; then
    env_args+=(PETSC_OPTIONS="${PETSC_OPTIONS:-} -pc_gamg_threshold $gamg_threshold -mg_levels_ksp_max_it $gamg_smooth_steps")
  fi
  command=(env "${env_args[@]}" "$script_dir/run_case.sh" "$mpi_rank"
    "$geometry_file" k_refine "$degree" "$element_tuple" greville
    "$outer_mode" "$result_tmp")
  printf '[command]' > "$console_tmp"
  printf ' %q' "${command[@]}" >> "$console_tmp"
  printf '\n' >> "$console_tmp"
  if (( timeout_sec > 0 )); then
    timeout "$timeout_sec" "${command[@]}" >> "$console_tmp" 2>&1
  else
    "${command[@]}" >> "$console_tmp" 2>&1
  fi
  return_code=$?

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
  append_summary "$case_id" "$id" "$status" "$return_code" "$source" \
    "$outer_mode" "$application" "$inner_pc" "$method_inner_max" \
    "$degree" "$element_tuple" "$result_file" "$console_file" "$reason"
  echo "[$status] $id"
  if [[ "$status" == failed || "$status" == timeout ]]; then
    failed=$((failed + 1))
    [[ "$fail_fast" == 1 ]] && break
  fi
done

echo "Comparison finished with $failed failed/timeout cases."
echo "Summary: $summary"
(( failed == 0 ))
