#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_case.sh MPI_RANK GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE \
#       [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]
# Automatic-rank example:
#   ./run_case.sh default input/square_2d.txt k_refine 5 16,16 greville right_fgmres_cg \
#     results/p5_n16x16_right_fgmres_cg.log 1e-10 1e-10 1e-4 1e-50
# Alternative exact-LU research mode (uncomment to use as a command):
# ./run_case.sh 8 input/square_2d.txt k_refine 5 16,16 greville left_cg_lu \
#     results/p5_n16x16_left_cg_lu.log 1e-10 1e-12 1e-10 1e-10

if [[ $# -ne 8 && $# -ne 12 ]]; then
  echo "Usage: $0 MPI_RANK GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]" >&2
  echo "MPI_RANK must be 'default' or a positive integer" >&2
  exit 2
fi

mpi_rank_setting=$1
geometry_file=$2
refinement=$3
degree=$4
element_tuple=$5
lor_points=$6
lor_pc=$7
output_file=$8
inner_pc=${INNER_PC:-jacobi}
fixed_jacobi_steps=${FIXED_JACOBI_STEPS:-4}
fixed_jacobi_omega=${FIXED_JACOBI_OMEGA:-0.6666666666666667}

# Accept either an explicit launch size or per-case automatic planning.
if [[ "$mpi_rank_setting" != "default" ]] &&
   ! [[ "$mpi_rank_setting" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_RANK must be 'default' or a positive integer" >&2
  exit 2
fi

# Reject unknown refinement policies before starting the executable.
case "$refinement" in
  p_refine|h_refine|k_refine) ;;
  *)
    echo "REFINEMENT must be p_refine, h_refine, or k_refine" >&2
    exit 2
    ;;
esac

# Accept only the two point-set strategies implemented by LORPointSet.
case "$lor_points" in
  greville|demko) ;;
  *)
    echo "LOR_POINTS must be greville or demko" >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver="$project_dir/build/assembled_driver"
planner="$project_dir/build/case_planner"

if [[ ! -x "$driver" ]]; then
  echo "Missing executable: $driver" >&2
  echo "Build it with: cd $project_dir/build && cmake ../examples && make -j" >&2
  exit 3
fi

# Refuse to mix a newly updated launcher/source tree with an older executable.
# This catches stale builds that silently ignore recently added PETSc options.
newer_build_input=$(find \
  "$project_dir/include" \
  "$project_dir/src" \
  "$project_dir/examples" \
  "$project_dir/conf" \
  -type f \
  \( -name '*.hpp' -o -name '*.cpp' -o -name 'CMakeLists.txt' -o -name '*.cmake' \) \
  -newer "$driver" -print -quit)
if [[ -n "$newer_build_input" || "$project_dir/CMakeLists.txt" -nt "$driver" ]]; then
  echo "Stale executable: $driver" >&2
  if [[ -n "$newer_build_input" ]]; then
    echo "Newer build input: $newer_build_input" >&2
  else
    echo "Newer build input: $project_dir/CMakeLists.txt" >&2
  fi
  echo "Rebuild with: cmake --build $project_dir/build -j" >&2
  exit 3
fi

# Automatic mode computes this case's exact post-refinement free-DOF count.
# MAX_MPI_RANKS optionally imposes a resource limit below detected hardware.
if [[ "$mpi_rank_setting" == "default" ]]; then
  if [[ ! -x "$planner" ]]; then
    echo "Missing executable: $planner" >&2
    echo "Build it with: cd $project_dir/build && cmake ../examples && make -j" >&2
    exit 3
  fi
  planner_command=(
    "$planner"
    -geometry "$geometry_file"
    -refinement "$refinement"
    -p "$degree"
    -n "$element_tuple"
    --recommended-ranks-only
  )
  if [[ -n ${MAX_MPI_RANKS:-} ]]; then
    planner_command+=(--max-ranks "$MAX_MPI_RANKS")
  fi
  mpi_ranks=$("${planner_command[@]}")
else
  mpi_ranks=$mpi_rank_setting
fi

mkdir -p "$(dirname "$output_file")"

# Every scheme name encodes its algebraic direction, outer Krylov method, and
# approximate-inverse implementation, so no independent side can conflict.
# Set only mode flags here; mode-specific options are appended later.
uses_inner_cg=false
uses_fixed_jacobi=false
case "$lor_pc" in
  left_cg_lu)
    preconditioner_side="left"
    outer_rtol="1.0e-12"
    outer_atol="1.0e-14"
    outer_dtol="1.0e8"
    ;;
  right_gmres_lu)
    preconditioner_side="right"
    outer_rtol="1.0e-10"
    outer_atol="1.0e-10"
    outer_dtol="1.0e4"
    ;;
  left_gmres_cg|right_gmres_cg|right_fgmres_cg)
    [[ "$lor_pc" == left_* ]] && preconditioner_side="left" || preconditioner_side="right"
    uses_inner_cg=true
    outer_rtol="1.0e-10"
    outer_atol="1.0e-10"
    outer_dtol="1.0e4"
    ;;
  left_gmres_fixed_jacobi)
    preconditioner_side="left"
    uses_fixed_jacobi=true
    outer_rtol="1.0e-10"
    outer_atol="1.0e-10"
    outer_dtol="1.0e4"
    ;;
  *)
    echo "LOR_PC must be left_cg_lu, right_gmres_lu, left_gmres_cg, right_gmres_cg, right_fgmres_cg, or left_gmres_fixed_jacobi" >&2
    exit 2
    ;;
esac
outer_max_it="10000"
inner_rtol="1.0e-10"
inner_atol="1.0e-10"
inner_dtol="1.0e4"
inner_max_it="10000"

# Environment overrides support batch sweeps without lengthening the stable
# positional interface; explicit trailing arguments still take precedence.
outer_rtol=${OUTER_RTOL:-$outer_rtol}
outer_atol=${OUTER_ATOL:-$outer_atol}
outer_dtol=${OUTER_DTOL:-$outer_dtol}
outer_max_it=${OUTER_MAX_IT:-$outer_max_it}
inner_rtol=${INNER_RTOL:-$inner_rtol}
inner_atol=${INNER_ATOL:-$inner_atol}
inner_dtol=${INNER_DTOL:-$inner_dtol}
inner_max_it=${INNER_MAX_IT:-$inner_max_it}

# SLEPc defaults may be overridden through environment variables without
# lengthening the positional single-case interface.
svd_tol=${SVD_TOL:-1.0e-8}
svd_max_it=${SVD_MAX_IT:-100000}
svd_ncv=${SVD_NCV:-100}

# Independent experiment features. Environment variables make the single-case
# launcher configurable without lengthening its positional interface.
compute_solution=${COMPUTE_SOLUTION:-true}
compute_l2_error=${COMPUTE_L2_ERROR:-true}
compute_condition_number=${COMPUTE_CONDITION_NUMBER:-false}
for feature_value in \
  "$compute_solution" "$compute_l2_error" "$compute_condition_number"; do
  if [[ "$feature_value" != "true" && "$feature_value" != "false" ]]; then
    echo "COMPUTE_SOLUTION, COMPUTE_L2_ERROR, and COMPUTE_CONDITION_NUMBER must be true or false" >&2
    exit 2
  fi
done
if [[ "$compute_l2_error" == "true" && "$compute_solution" != "true" ]]; then
  echo "COMPUTE_L2_ERROR=true requires COMPUTE_SOLUTION=true" >&2
  exit 2
fi

# Explicit command-line residuals belong only to nested-CG configurations.
if [[ $# -eq 12 ]]; then
  if [[ "$uses_inner_cg" != true ]]; then
    echo "Trailing INNER_RTOL/INNER_ATOL arguments require a *_cg LOR_PC" >&2
    exit 2
  fi
  outer_rtol=$9
  outer_atol=${10}
  inner_rtol=${11}
  inner_atol=${12}
fi

common_options=(
  -geometry "$geometry_file"
  -refinement "$refinement"
  -p "$degree"
  -n "$element_tuple"
  -lor_points "$lor_points"
  -lor_pc "$lor_pc"
  -preconditioner_side "$preconditioner_side"
  -compute_solution "$compute_solution"
  -compute_l2_error "$compute_l2_error"
  -compute_condition_number "$compute_condition_number"
  -outer_rtol "$outer_rtol"
  -outer_atol "$outer_atol"
  -outer_dtol "$outer_dtol"
  -outer_max_it "$outer_max_it"
  -svd_tol "$svd_tol"
  -svd_max_it "$svd_max_it"
  -svd_ncv "$svd_ncv"
  -output "$output_file"
)

if [[ "$uses_inner_cg" == true ]]; then
  if [[ "$inner_pc" != "jacobi" && "$inner_pc" != "gamg" ]]; then
    echo "INNER_PC must be jacobi or gamg for a *_cg LOR_PC" >&2
    exit 2
  fi
  common_options+=(
    -inner_pc "$inner_pc"
    -inner_rtol "$inner_rtol"
    -inner_atol "$inner_atol"
    -inner_dtol "$inner_dtol"
    -inner_max_it "$inner_max_it"
  )
elif [[ "$uses_fixed_jacobi" == true ]]; then
  common_options+=(
    -fixed_jacobi_steps "$fixed_jacobi_steps"
    -fixed_jacobi_omega "$fixed_jacobi_omega"
  )
fi

# Keep threaded numerical libraries from competing with the MPI ranks. Users
# can still override MPIEXEC when their cluster exposes another launcher name.
export OMP_NUM_THREADS=1
mpi_launcher=${MPIEXEC:-mpirun}
scheme_log=""
if [[ "$uses_inner_cg" == true ]]; then
  scheme_log=" inner_pc=$inner_pc inner_rtol=$inner_rtol inner_atol=$inner_atol inner_max_it=$inner_max_it"
elif [[ "$uses_fixed_jacobi" == true ]]; then
  scheme_log=" fixed_jacobi_steps=$fixed_jacobi_steps fixed_jacobi_omega=$fixed_jacobi_omega"
fi
echo "[run] mpi_ranks=$mpi_ranks refinement=$refinement requested_p=$degree requested_n=($element_tuple) point_set=$lor_points lor_pc=$lor_pc preconditioner_side=$preconditioner_side${scheme_log} solve=$compute_solution L2=$compute_l2_error condition=$compute_condition_number outer_rtol=$outer_rtol outer_atol=$outer_atol outer_dtol=$outer_dtol outer_max_it=$outer_max_it svd_tol=$svd_tol svd_max_it=$svd_max_it svd_ncv=$svd_ncv"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${common_options[@]}"
