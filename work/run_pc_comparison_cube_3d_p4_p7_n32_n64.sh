#!/usr/bin/env bash
set -euo pipefail

# 3D cube comparison: seven selected-P modes plus two direct-K baselines.
# Safe default remains BATCH_DRY_RUN=1 in the shared implementation.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BENCHMARK_DIMENSION=3
export GEOMETRY_FILE="$script_dir/input/cube_3d.txt"
export DEGREES="${DEGREES:-4 5 6 7}"
export ELEMENT_CASES="${ELEMENT_CASES:-32,32,32 64,64,64}"
export INCLUDE_LEFT_GMRES=0
export INCLUDE_DIRECT_K_BASELINES=1
export MPI_RANK="${MPI_RANK:-default}"
export RESULT_GROUP="${RESULT_GROUP:-pc_comparison_cube_3d}"
export RUN_ID="${RUN_ID:-cube3d-p4-p7-n32-n64-nine-way}"

exec "$script_dir/run_pc_comparison_p7_n512_2d.sh" "$@"
