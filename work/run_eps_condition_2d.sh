#!/usr/bin/env bash
set -euo pipefail

# Run exactly one 2D EPS condition-number experiment.
#
# Usage:
#   run_eps_condition_2d.sh P N k|p|pinv_k RESULT_FILE [PETSc options...]
#
# k and p use standard Hermitian EPS problems. pinv_k solves Kx=lambda Px;
# neither an inverse matrix nor an identity matrix is formed.

if (( $# < 4 )); then
  echo "Usage: $0 P N k|p|pinv_k RESULT_FILE [PETSc options...]" >&2
  exit 2
fi

degree=$1
elements=$2
target=$3
result_file=$4
shift 4

if ! [[ "$degree" =~ ^[1-9][0-9]*$ ]]; then
  echo "P must be a positive integer" >&2
  exit 2
fi
if ! [[ "$elements" =~ ^[1-9][0-9]*$ ]]; then
  echo "N must be a positive integer" >&2
  exit 2
fi
case "$target" in
  k|p|pinv_k) ;;
  *) echo "Target must be k, p, or pinv_k" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/assembled_driver}
planner=${PLANNER:-$project_dir/build/case_planner}
mpi_launcher=${MPIEXEC:-mpirun}
mpi_rank_setting=${MPI_RANKS:-default}

if [[ ! -x "$driver" ]]; then
  echo "Missing executable: $driver" >&2
  echo "Build first with: cmake --build $project_dir/build -j" >&2
  exit 3
fi
newer_input=$(find "$project_dir/include" "$project_dir/src" \
  -type f \( -name '*.hpp' -o -name '*.cpp' \) \
  -newer "$driver" -print -quit)
if [[ -n "$newer_input" || "$project_dir/CMakeLists.txt" -nt "$driver" ]]; then
  echo "Stale executable: $driver" >&2
  echo "Rebuild with: cmake --build $project_dir/build -j" >&2
  exit 3
fi

if [[ "$mpi_rank_setting" == default ]]; then
  if [[ ! -x "$planner" ]]; then
    echo "Missing executable: $planner" >&2
    exit 3
  fi
  planner_command=(
    "$planner"
    -geometry "$script_dir/input/square_2d.txt"
    -refinement k_refine
    -p "$degree"
    -n "$elements,$elements"
    --recommended-ranks-only
  )
  if [[ -n ${MAX_MPI_RANKS:-} ]]; then
    planner_command+=(--max-ranks "$MAX_MPI_RANKS")
  fi
  mpi_ranks=$("${planner_command[@]}")
elif [[ "$mpi_rank_setting" =~ ^[1-9][0-9]*$ ]]; then
  mpi_ranks=$mpi_rank_setting
else
  echo "MPI_RANKS must be default or a positive integer" >&2
  exit 2
fi

eps_tol=${EPS_TOL:-1.0e-8}
eps_ncv=${EPS_NCV:-40}
eps_max_it=${EPS_MAX_IT:-10000}
inner_ksp_rtol=${INNER_KSP_RTOL:-1.0e-10}
inner_ksp_max_it=${INNER_KSP_MAX_IT:-10000}
stability_window=${EIGMIN_STABILITY_WINDOW:-100}
stability_tol=${EIGMIN_STABILITY_TOL:-5.0e-6}
stability_residual=${EIGMIN_STABILITY_RESIDUAL:-1.0e-2}
stability_min_it=${EIGMIN_STABILITY_MIN_IT:-300}

# Direct-K mode prevents P from being assembled in the k-only job. The other
# targets require the Q1-LOR matrix and therefore use an LOR-backed mode.
if [[ "$target" == k ]]; then
  lor_pc=left_gmres_k_jacobi
else
  lor_pc=left_cg_lu
fi

mkdir -p "$(dirname "$result_file")"
options=(
  -geometry "$script_dir/input/square_2d.txt"
  -refinement k_refine
  -p "$degree"
  -n "$elements,$elements"
  -lor_points greville
  -lor_pc "$lor_pc"
  -preconditioner_side left
  -preconditioner_matrix_source lor_q1
  -compute_solution false
  -compute_unpreconditioned_solution false
  -compute_l2_error false
  -compute_condition_number false
  -compute_spectral_condition_number true
  -spectral_condition_target "$target"
  -eps_tol "$eps_tol"
  -eps_max_it "$eps_max_it"
  -eps_ncv "$eps_ncv"
  -eigmax_st_type shift
  -eigmin_st_type shift
  -eigmin_eps_smallest_real
  -eigmin_eps_stability_stop true
  -eigmin_eps_stability_window "$stability_window"
  -eigmin_eps_stability_tol "$stability_tol"
  -eigmin_eps_stability_residual "$stability_residual"
  -eigmin_eps_stability_min_it "$stability_min_it"
  -output "$result_file"
)

# Only P^{-1}K has a nontrivial ST linear solve. Both extrema apply P^{-1}K,
# with GAMG-preconditioned CG solving systems in the sparse LOR matrix P.
if [[ "$target" == pinv_k ]]; then
  options+=(
    -eigmax_st_ksp_type cg
    -eigmax_st_ksp_rtol "$inner_ksp_rtol"
    -eigmax_st_ksp_max_it "$inner_ksp_max_it"
    -eigmax_st_ksp_error_if_not_converged true
    -eigmax_st_pc_type gamg
    -eigmin_st_ksp_type cg
    -eigmin_st_ksp_rtol "$inner_ksp_rtol"
    -eigmin_st_ksp_max_it "$inner_ksp_max_it"
    -eigmin_st_ksp_error_if_not_converged true
    -eigmin_st_pc_type gamg
  )
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# With MPICH/CH4:UCX, PETSc GAMG creates short, intense shared-memory message
# bursts.  Keep UCX's native transport selection (including the fast MM path),
# but give its receive FIFO more headroom and poll it more aggressively.  User
# settings always win, and an explicit UCX_TLS selection is left untouched.
if [[ -z ${UCX_TLS:-} ]]; then
  export UCX_MM_FIFO_SIZE=${UCX_MM_FIFO_SIZE:-4096}
  export UCX_MM_FIFO_MAX_POLL=${UCX_MM_FIFO_MAX_POLL:-64}
  for value in "$UCX_MM_FIFO_SIZE" "$UCX_MM_FIFO_MAX_POLL"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "UCX_MM_FIFO_SIZE and UCX_MM_FIFO_MAX_POLL must be positive integers" >&2
      exit 2
    fi
  done
  if (( (UCX_MM_FIFO_SIZE & (UCX_MM_FIFO_SIZE - 1)) != 0 )); then
    echo "UCX_MM_FIFO_SIZE must be a power of two" >&2
    exit 2
  fi
  ucx_description="native,mm_fifo=$UCX_MM_FIFO_SIZE,mm_max_poll=$UCX_MM_FIFO_MAX_POLL"
else
  ucx_description="tls=$UCX_TLS"
fi

echo "[run] 2D p=$degree n=${elements}x${elements} target=$target ranks=$mpi_ranks ucx=$ucx_description output=$result_file"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${options[@]}" "$@"
