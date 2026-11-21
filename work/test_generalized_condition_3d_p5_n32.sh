#!/usr/bin/env bash
set -euo pipefail

# Generalized condition number of the 3D p=5, 32x32x32 Greville Q1-LOR
# preconditioner. This script only runs matrix assembly and the two EPS solves.
# Override MPI_RANKS, MPIEXEC, EPS_TOL, EPS_NCV, INNER_KSP_RTOL, or the
# EIGMIN_STABILITY_* controls as needed.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
driver=${DRIVER:-$project_dir/build/assembled_driver}
mpi_launcher=${MPIEXEC:-mpirun}
mpi_ranks=${MPI_RANKS:-8}
eps_tol=${EPS_TOL:-1.0e-8}
eps_ncv=${EPS_NCV:-40}
eps_max_it=${EPS_MAX_IT:-10000}
inner_ksp_rtol=${INNER_KSP_RTOL:-1.0e-10}
inner_ksp_max_it=${INNER_KSP_MAX_IT:-10000}
stability_window=${EIGMIN_STABILITY_WINDOW:-100}
stability_tol=${EIGMIN_STABILITY_TOL:-5.0e-6}
stability_residual=${EIGMIN_STABILITY_RESIDUAL:-1.0e-2}
stability_min_it=${EIGMIN_STABILITY_MIN_IT:-300}

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
if ! [[ "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_RANKS must be a positive integer" >&2
  exit 2
fi

result_dir=${RESULT_DIR:-$script_dir/results/generalized_condition}
console_dir=${CONSOLE_DIR:-$script_dir/logs/generalized_condition}
mkdir -p "$result_dir" "$console_dir"
result_file=${OUTPUT_FILE:-$result_dir/3d_p5_n32x32x32.log}
console_file=${CONSOLE_FILE:-$console_dir/3d_p5_n32x32x32.console.log}

options=(
  -geometry "$script_dir/input/cube_3d.txt"
  -refinement k_refine
  -p 5
  -n 32,32,32
  -lor_points greville
  -lor_pc left_cg_lu
  -preconditioner_side left
  -preconditioner_matrix_source lor_q1
  -compute_solution false
  -compute_unpreconditioned_solution false
  -compute_l2_error false
  -compute_condition_number false
  -compute_spectral_condition_number true
  -eps_tol "$eps_tol"
  -eps_max_it "$eps_max_it"
  -eps_ncv "$eps_ncv"
  -eigmax_st_type shift
  -eigmax_st_ksp_type cg
  -eigmax_st_ksp_rtol "$inner_ksp_rtol"
  -eigmax_st_ksp_max_it "$inner_ksp_max_it"
  -eigmax_st_ksp_error_if_not_converged true
  -eigmax_st_pc_type gamg
  -eigmin_st_type shift
  -eigmin_eps_smallest_real
  -eigmin_eps_stability_stop true
  -eigmin_eps_stability_window "$stability_window"
  -eigmin_eps_stability_tol "$stability_tol"
  -eigmin_eps_stability_residual "$stability_residual"
  -eigmin_eps_stability_min_it "$stability_min_it"
  -eigmin_st_ksp_type cg
  -eigmin_st_ksp_rtol "$inner_ksp_rtol"
  -eigmin_st_ksp_max_it "$inner_ksp_max_it"
  -eigmin_st_ksp_error_if_not_converged true
  -eigmin_st_pc_type gamg
  -output "$result_file"
)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
echo "[run] 3D p=5 n=32x32x32 ranks=$mpi_ranks output=$result_file"
"$mpi_launcher" -n "$mpi_ranks" "$driver" "${options[@]}" "$@" \
  2>&1 | tee "$console_file"
