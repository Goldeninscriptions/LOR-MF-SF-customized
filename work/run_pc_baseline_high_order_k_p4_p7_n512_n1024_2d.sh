#!/usr/bin/env bash
set -euo pipefail

# High-order-K baselines matching the expanded 2D selected-P sweep:
#   K1: left GMRES + PCJACOBI(K)
#   K2: left GMRES + PCGAMG(K)
# p=4..7, n=512^2/1024^2 => 16 cases. Safe default remains dry-run.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BENCHMARK_DIMENSION=2
export GEOMETRY_FILE="$script_dir/input/square_2d.txt"
export DEGREES="${DEGREES:-4 5 6 7}"
export ELEMENT_CASES="${ELEMENT_CASES:-512,512 1024,1024}"
export COMPARISON_METHOD_SET=direct_k_baseline
export INCLUDE_LEFT_GMRES=0
export RESULT_GROUP="${RESULT_GROUP:-pc_baseline_high_order_k_2d}"
export RUN_ID="${RUN_ID:-high-order-k-p4-p7-n512-n1024}"

exec "$script_dir/run_pc_comparison_p7_n512_2d.sh" "$@"
