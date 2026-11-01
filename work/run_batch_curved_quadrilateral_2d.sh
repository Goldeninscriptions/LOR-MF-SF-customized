#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------- Curved quadrilateral 2D batch configuration ------------
batch_name="curved_quadrilateral_2d"
geometry_file="$script_dir/input/curved_quadrilateral_2d.txt"

# The exact polynomial geometry starts at p=3, n=(1,1). K-refinement preserves
# the geometry while producing the same analysis-space sweep as square_2d.
refinement="k_refine"
degrees=(3 4 5 6 7)
element_cases=("4,4" "8,8" "16,16" "32,32" "64,64" "128,128" "256,256" "512,512")

result_dir="$script_dir/results/$batch_name"
console_dir="$script_dir/logs/$batch_name"
summary_file="$script_dir/batch_summary_${batch_name}.log"

lor_points="greville"
# lor_points="demko"
lor_pc="gmres_shell"
# lor_pc="cg_lu"

mpi_rank=${MPI_RANK:-default}
maximum_mpi_ranks=${MAX_MPI_RANKS:-}

# Keep the project-wide default experiment policy consistent across geometries:
# assemble, solve, and measure L2 error, while leaving the expensive condition
# analysis disabled. The curved-domain manufactured solution supplies its
# exact source, nonzero affine boundary trace, and L2 reference function.
compute_solution=true
compute_l2_error=true
compute_condition_number=false

svd_tol="1.0e-8"
svd_max_it="100000"
svd_ncv="100"
# -------------------------------------------------------------------------

source "$script_dir/run_batch_common.sh"
