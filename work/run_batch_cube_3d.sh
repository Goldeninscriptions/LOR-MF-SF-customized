#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ----------------------- Cube 3D batch configuration ---------------------
batch_name="cube_3d"
geometry_file="$script_dir/input/cube_3d.txt"

# The input space is p=1, n=(1,1,1). Three-dimensional cost grows cubically,
# so the default n sequence is intentionally smaller than the square batch.
refinement="k_refine"
degrees=(3 4 5 6 7)
element_cases=("4,4,4" "8,8,8" "16,16,16" "32,32,32" "64,64,64")

result_dir="$script_dir/results/$batch_name"
console_dir="$script_dir/logs/$batch_name"
summary_file="$script_dir/batch_summary_${batch_name}.log"

lor_points="greville"
# lor_points="demko"

# Select exactly one complete solver block. Keep the active block uncommented
# and leave every unused block commented.

# Exact left LOR inverse; no inner iterative settings are used.
# lor_pc="left_cg_lu"
# outer_rtol="1.0e-12"
# outer_atol="1.0e-14"
# outer_dtol="1.0e8"
# outer_max_it="10000"

# Exact right LOR inverse; no inner iterative settings are used.
# lor_pc="right_gmres_lu"
# outer_rtol="1.0e-10"
# outer_atol="1.0e-10"
# outer_dtol="1.0e4"
# outer_max_it="10000"

# Standard left GMRES with tolerance-controlled inner CG.
# lor_pc="left_gmres_cg"
# outer_rtol="1.0e-10"
# outer_atol="1.0e-10"
# outer_dtol="1.0e4"
# outer_max_it="10000"
# inner_pc="jacobi"       # Alternative: "gamg"
# inner_rtol="1.0e-10"
# inner_atol="1.0e-10"
# inner_max_it="10000"

# Standard right GMRES with tolerance-controlled inner CG.
# lor_pc="right_gmres_cg"
# outer_rtol="1.0e-10"
# outer_atol="1.0e-10"
# outer_dtol="1.0e4"
# outer_max_it="10000"
# inner_pc="jacobi"       # Alternative: "gamg"
# inner_rtol="1.0e-10"
# inner_atol="1.0e-10"
# inner_max_it="10000"

# Scheme A1: flexible right GMRES permits an inexact, varying inner CG map.
# lor_pc="right_fgmres_cg"
# outer_rtol="1.0e-10"
# outer_atol="1.0e-10"
# outer_dtol="1.0e4"
# outer_max_it="10000"
# inner_pc="jacobi"       # Alternative: "gamg"
# inner_rtol="1.0e-4"
# inner_atol="1.0e-50"
# inner_max_it="10000"

# Reference: standard left GMRES with Jacobi formed directly from diag(K).
# lor_pc="left_gmres_k_jacobi"
# outer_rtol="1.0e-10"
# outer_atol="1.0e-10"
# outer_dtol="1.0e4"
# outer_max_it="10000"

# Scheme B (active/default): standard left GMRES with one fixed linear map.
lor_pc="left_gmres_fixed_jacobi"
outer_rtol="1.0e-10"
outer_atol="1.0e-10"
outer_dtol="1.0e4"
outer_max_it="10000"
fixed_jacobi_steps="4"
fixed_jacobi_omega="0.6666666666666667"

mpi_rank=${MPI_RANK:-default}
maximum_mpi_ranks=${MAX_MPI_RANKS:-}

compute_solution=true
compute_l2_error=true
compute_condition_number=false

svd_tol="1.0e-8"
svd_max_it="100000"
svd_ncv="100"
# -------------------------------------------------------------------------

source "$script_dir/run_batch_common.sh"
