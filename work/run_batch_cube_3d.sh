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
lor_pc="gmres_shell"
# lor_pc="cg_lu"
preconditioner_side=${PRECONDITIONER_SIDE:-left}
# preconditioner_side="right"
inner_pc=${INNER_PC:-jacobi}
# inner_pc="gamg"

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
