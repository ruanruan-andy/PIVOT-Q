#!/bin/bash
set -euo pipefail

# Usage: ./run_libero_eval.sh <suite> [--model-variant VARIANT] [evaluation args]

TASK="${1:-libero_10}"
shift || true
EXTRA_ARGS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
LIBERO_ROOT="${LIBERO_ROOT:-$REPO_ROOT/../LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/../../config/sparse_anchor_libero}"
PORT="${GR00T_PORT:-5556}"
MODEL_VARIANT="${EVAL_MODEL_VARIANT:-groot-fp16}"
MODEL_ARG_PRESENT=0
HEADLESS_FLAG="no"

for ((index=0; index<${#EXTRA_ARGS[@]}; index++)); do
    arg="${EXTRA_ARGS[$index]}"
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
    elif [[ "$arg" == "--model-variant" ]]; then
        MODEL_VARIANT="${EXTRA_ARGS[$((index + 1))]:-}"
        MODEL_ARG_PRESENT=1
    elif [[ "$arg" == --model-variant=* ]]; then
        MODEL_VARIANT="${arg#*=}"
        MODEL_ARG_PRESENT=1
    fi
done

case "$MODEL_VARIANT" in
    groot-fp16|groot-quantvla-w4a8|groot-pivot_q-v2-w4a8|groot-pivot-q-w4a8) ;;
    *) echo "Unsupported model variant: $MODEL_VARIANT" >&2; exit 1 ;;
esac
if [[ "$MODEL_ARG_PRESENT" == 0 ]]; then
    EXTRA_ARGS+=(--model-variant "$MODEL_VARIANT")
fi

EVAL_LOG_DIR="${LIBERO_EVAL_LOG_DIR:-$REPO_ROOT/output/libero/$MODEL_VARIANT/$TASK}"
source "$CONDA_SH"
conda activate libero_test
export PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export LIBERO_EVAL_LOG_DIR="$EVAL_LOG_DIR"
mkdir -p "$LIBERO_EVAL_LOG_DIR"
if [[ "$HEADLESS_FLAG" == "yes" ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

echo "=========================================="
echo "Running standard LIBERO evaluation"
echo "Suite: $TASK"
echo "Model: $MODEL_VARIANT"
echo "GR00T port: $PORT"
echo "Output: $LIBERO_EVAL_LOG_DIR"
echo "=========================================="

cd "$REPO_ROOT/examples/Libero/eval"
exec python run_libero_eval.py --task-suite-name "$TASK" --port "$PORT" "${EXTRA_ARGS[@]}"
