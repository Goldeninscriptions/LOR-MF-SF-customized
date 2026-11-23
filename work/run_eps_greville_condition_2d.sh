#!/usr/bin/env bash
set -euo pipefail

# Run one matrix-free Greville tensor EPS condition-number experiment.
# Usage: run_eps_greville_condition_2d.sh P N p|pinv_k RESULT_FILE [PETSc options...]

if (( $# < 4 )); then
  echo "Usage: $0 P N p|pinv_k RESULT_FILE [PETSc options...]" >&2
  exit 2
fi

degree=$1
elements=$2
target=$3
result_file=$4
shift 4
[[ "$degree" =~ ^[1-9][0-9]*$ ]] || { echo "P must be positive" >&2; exit 2; }
[[ "$elements" =~ ^[1-9][0-9]*$ ]] || { echo "N must be positive" >&2; exit 2; }
case "$target" in p|pinv_k) ;; *) echo "Target must be p or pinv_k" >&2; exit 2;; esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/assembled_driver}
planner=${PLANNER:-$project_dir/build/case_planner}
mpi_launcher=${MPIEXEC:-mpirun}
mpi_rank_setting=${MPI_RANKS:-default}

[[ -x "$driver" ]] || { echo "Missing executable: $driver" >&2; exit 3; }
newer_input=$(find "$project_dir/include" "$project_dir/src" \
  -type f \( -name '*.hpp' -o -name '*.cpp' \) -newer "$driver" -print -quit)
if [[ -n "$newer_input" || "$project_dir/CMakeLists.txt" -nt "$driver" ]]; then
  echo "Stale executable: $driver; rebuild before running" >&2
  exit 3
fi

if [[ "$mpi_rank_setting" == default ]]; then
  if [[ "$target" == p ]]; then
    # eig(P_y x P_x) is obtained from two small one-dimensional EPS solves;
    # extra ranks would remain idle and only add launcher/UCX overhead.
    mpi_ranks=1
  else
    planner_command=("$planner" -geometry "$script_dir/input/square_2d.txt"
      -refinement k_refine -p "$degree" -n "$elements,$elements"
      --recommended-ranks-only --max-ranks "${MAX_MPI_RANKS:-56}")
    mpi_ranks=$("${planner_command[@]}")
  fi
elif [[ "$mpi_rank_setting" =~ ^[1-9][0-9]*$ ]]; then
  mpi_ranks=$mpi_rank_setting
else
  echo "MPI_RANKS must be default or a positive integer" >&2
  exit 2
fi

eps_tol=${EPS_TOL:-1.0e-8}
eps_ncv=${EPS_NCV:-40}
eps_max_it=${EPS_MAX_IT:-10000}
stability_window=${EIGMIN_STABILITY_WINDOW:-100}
stability_tol=${EIGMIN_STABILITY_TOL:-5.0e-6}
stability_residual=${EIGMIN_STABILITY_RESIDUAL:-1.0e-2}
stability_min_it=${EIGMIN_STABILITY_MIN_IT:-300}

mkdir -p "$(dirname "$result_file")"
options=(
  -geometry "$script_dir/input/square_2d.txt"
  -refinement k_refine -p "$degree" -n "$elements,$elements"
  -lor_points greville -lor_pc left_cg_lu -preconditioner_side left
  -preconditioner_matrix_source greville_mass_tensor
  -compute_solution false -compute_unpreconditioned_solution false
  -compute_l2_error false -compute_condition_number false
  -compute_spectral_condition_number true -spectral_condition_target "$target"
  -eps_tol "$eps_tol" -eps_max_it "$eps_max_it" -eps_ncv "$eps_ncv"
  -output "$result_file"
)
if [[ "$target" == pinv_k ]]; then
  options+=(
    -eigmax_st_type shift -eigmin_st_type shift -eigmin_eps_smallest_real
    -eigmin_eps_stability_stop true
    -eigmin_eps_stability_window "$stability_window"
    -eigmin_eps_stability_tol "$stability_tol"
    -eigmin_eps_stability_residual "$stability_residual"
    -eigmin_eps_stability_min_it "$stability_min_it"
  )
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [[ -z ${UCX_TLS:-} ]]; then
  export UCX_MM_FIFO_SIZE=${UCX_MM_FIFO_SIZE:-4096}
  export UCX_MM_FIFO_MAX_POLL=${UCX_MM_FIFO_MAX_POLL:-64}
  ucx_description="native,mm_fifo=$UCX_MM_FIFO_SIZE,mm_max_poll=$UCX_MM_FIFO_MAX_POLL"
else
  ucx_description="tls=$UCX_TLS"
fi

echo "[run] Greville 2D p=$degree n=${elements}x${elements} target=$target ranks=$mpi_ranks ucx=$ucx_description output=$result_file"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${options[@]}" "$@"
