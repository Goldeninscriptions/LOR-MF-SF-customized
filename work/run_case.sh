#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_case.sh MPI_RANKS GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE \
#       [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]
# Example:
#   ./run_case.sh 4 input/square_2d.txt k_refine 5 16,16 greville gmres_shell \
#     results/p5_n16x16_gmres_shell.log 1e-10 1e-10 1e-10 1e-10
# Alternative exact-LU research mode (uncomment to use as a command):
# ./run_case.sh 4 input/square_2d.txt k_refine 5 16,16 greville cg_lu \
#     results/p5_n16x16_cg_lu.log 1e-10 1e-12 1e-10 1e-10

if [[ $# -ne 8 && $# -ne 12 ]]; then
  echo "Usage: $0 MPI_RANKS GEOMETRY_FILE REFINEMENT P N_TUPLE LOR_POINTS LOR_PC OUTPUT_FILE [OUTER_RTOL OUTER_ATOL INNER_RTOL INNER_ATOL]" >&2
  exit 2
fi

mpi_ranks=$1
geometry_file=$2
refinement=$3
degree=$4
element_tuple=$5
lor_points=$6
lor_pc=$7
output_file=$8

# MPI size is a launch-time choice, not a discretization option. Require an
# explicit positive rank count and pass it unchanged to mpirun -n.
if ! [[ "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_RANKS must be a positive integer" >&2
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

if [[ ! -x "$driver" ]]; then
  echo "Missing executable: $driver" >&2
  echo "Build it with: cd $project_dir/build && cmake ../examples && make -j" >&2
  exit 3
fi

mkdir -p "$(dirname "$output_file")"

# Mode-specific default residual settings.
if [[ "$lor_pc" == "cg_lu" ]]; then
  outer_rtol="1.0e-10"
  outer_atol="1.0e-12"
  outer_dtol="1.0e8"
elif [[ "$lor_pc" == "gmres_shell" ]]; then
  outer_rtol="1.0e-10"
  outer_atol="1.0e-10"
  outer_dtol="1.0e4"
else
  echo "LOR_PC must be cg_lu or gmres_shell" >&2
  exit 2
fi
outer_max_it="10000"
inner_rtol="1.0e-10"
inner_atol="1.0e-10"
inner_dtol="1.0e4"
inner_max_it="10000"

# SLEPc defaults may be overridden through environment variables without
# lengthening the positional single-case interface.
svd_tol=${SVD_TOL:-1.0e-8}
svd_max_it=${SVD_MAX_IT:-100000}
svd_ncv=${SVD_NCV:-100}

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
echo "[run] mpi_ranks=$mpi_ranks refinement=$refinement requested_p=$degree requested_n=($element_tuple) point_set=$lor_points lor_pc=$lor_pc svd_tol=$svd_tol svd_max_it=$svd_max_it svd_ncv=$svd_ncv"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${common_options[@]}"
