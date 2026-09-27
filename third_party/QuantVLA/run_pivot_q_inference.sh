#!/bin/bash
set -euo pipefail

# Start QuantVLA W4A8 with a trained PIVOT-Q-v2 PEFT adapter.
# Usage: CUDA_VISIBLE_DEVICES=0 ./run_pivot_q_inference.sh <suite> <adapter_dir>

TASK="${1:-libero_spatial}"
ADAPTER_PATH="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$ADAPTER_PATH" ]]; then
    echo "Usage: $0 <suite> <adapter_dir>" >&2
    exit 1
fi
if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "PIVOT-Q adapter directory not found: $ADAPTER_PATH" >&2
    exit 1
fi

export GR00T_ADAPTER_PATH="$(cd "$ADAPTER_PATH" && pwd)"
export GR00T_MODEL_VARIANT="groot-pivot_q-v2-w4a8"
exec "$REPO_ROOT/run_quantvla.sh" "$TASK"
