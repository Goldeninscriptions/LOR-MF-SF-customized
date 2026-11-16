#!/usr/bin/env bash
set -euo pipefail

# Descriptive entry point for the expanded 72-case selected-P comparison.
# The implementation remains at the earlier stable path so existing automation
# continues to work.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_pc_comparison_p7_n512_2d.sh" "$@"
