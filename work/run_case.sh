#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_case.sh MPI_RANK GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE \
#       [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]
# Automatic-rank example:
#   ./run_case.sh default input/square_2d.txt k_refine 5 16,16 greville gmres_shell \
#     results/p5_n16x16_gmres_shell.log 1e-12 1e-14 1e-12 1e-14
# Alternative exact-LU research mode (uncomment to use as a command):
# ./run_case.sh 8 input/square_2d.txt k_refine 5 16,16 greville cg_lu \
#     results/p5_n16x16_cg_lu.log 1e-10 1e-12 1e-10 1e-10

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
preconditioner_side=${PRECONDITIONER_SIDE:-left}

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

# Keep the paper-compatible left action as the default while allowing focused
# right-preconditioning studies without changing the positional interface.
if [[ "$preconditioner_side" != "left" &&
      "$preconditioner_side" != "right" ]]; then
  echo "PRECONDITIONER_SIDE must be left or right" >&2
  exit 2
fi

# Mode-specific default residual settings.
if [[ "$lor_pc" == "cg_lu" ]]; then
  outer_rtol="1.0e-12"
  outer_atol="1.0e-14"
  outer_dtol="1.0e8"
elif [[ "$lor_pc" == "gmres_shell" ]]; then
  outer_rtol="1.0e-12"
  outer_atol="1.0e-14"
  outer_dtol="1.0e4"
else
  echo "LOR_PC must be cg_lu or gmres_shell" >&2
  exit 2
fi
outer_max_it="10000"
inner_rtol="1.0e-12"
inner_atol="1.0e-14"
inner_dtol="1.0e4"
inner_max_it="10000"

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

# Explicit command-line residuals override the mode defaults.
if [[ $# -eq 12 ]]; then
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

if [[ "$lor_pc" == "gmres_shell" ]]; then
  common_options+=(
    -inner_rtol "$inner_rtol"
    -inner_atol "$inner_atol"
    -inner_dtol "$inner_dtol"
    -inner_max_it "$inner_max_it"
  )
fi

# Keep threaded numerical libraries from competing with the MPI ranks. Users
# can still override MPIEXEC when their cluster exposes another launcher name.
export OMP_NUM_THREADS=1
mpi_launcher=${MPIEXEC:-mpirun}
echo "[run] mpi_ranks=$mpi_ranks refinement=$refinement requested_p=$degree requested_n=($element_tuple) point_set=$lor_points lor_pc=$lor_pc preconditioner_side=$preconditioner_side solve=$compute_solution L2=$compute_l2_error condition=$compute_condition_number outer_rtol=$outer_rtol outer_atol=$outer_atol inner_rtol=$inner_rtol inner_atol=$inner_atol svd_tol=$svd_tol svd_max_it=$svd_max_it svd_ncv=$svd_ncv"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${common_options[@]}"
