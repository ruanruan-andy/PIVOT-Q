#!/bin/bash
set -euo pipefail

# Usage: ./run_libero_plus_eval.sh <suite> [--model-variant VARIANT] [evaluation args]

TASK="${1:-libero_10}"
shift || true
EXTRA_ARGS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/path/to/workspace/LIBERO-plus}"
LIBERO_PLUS_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-$REPO_ROOT/configs/libero_plus}"
PORT="${GR00T_PORT:-5556}"
MODEL_VARIANT="${EVAL_MODEL_VARIANT:-groot-fp16}"
MODEL_ARG_PRESENT=0
SAMPLE_ARG_PRESENT=0

for ((index=0; index<${#EXTRA_ARGS[@]}; index++)); do
    arg="${EXTRA_ARGS[$index]}"
    if [[ "$arg" == "--model-variant" ]]; then
        MODEL_VARIANT="${EXTRA_ARGS[$((index + 1))]:-}"
        MODEL_ARG_PRESENT=1
    elif [[ "$arg" == --model-variant=* ]]; then
        MODEL_VARIANT="${arg#*=}"
        MODEL_ARG_PRESENT=1
    elif [[ "$arg" == "--sample-manifest" || "$arg" == --sample-manifest=* ]]; then
        SAMPLE_ARG_PRESENT=1
    fi
done

case "$TASK" in
    libero_spatial|libero_goal|libero_object|libero_10) ;;
    *) echo "Unsupported LIBERO-Plus suite: $TASK" >&2; exit 1 ;;
esac
case "$MODEL_VARIANT" in
    groot-fp16|groot-quantvla-w4a8|groot-pivot_q-v2-w4a8|groot-pivot-q-w4a8) ;;
    *) echo "Unsupported model variant: $MODEL_VARIANT" >&2; exit 1 ;;
esac
if [[ "$MODEL_ARG_PRESENT" == 0 ]]; then
    EXTRA_ARGS+=(--model-variant "$MODEL_VARIANT")
fi
if [[ "$SAMPLE_ARG_PRESENT" == 0 ]]; then
    EXTRA_ARGS+=(--sample-manifest "$REPO_ROOT/configs/libero_plus/shared560-first20.json")
fi

if [[ ! -d "$LIBERO_PLUS_ROOT/libero/libero/assets" ]]; then
    echo "LIBERO-Plus assets not found at $LIBERO_PLUS_ROOT/libero/libero/assets" >&2
    exit 1
fi
if [[ ! -f "$LIBERO_PLUS_CONFIG_PATH/config.yaml" ]]; then
    echo "LIBERO-Plus config not found at $LIBERO_PLUS_CONFIG_PATH/config.yaml" >&2
    exit 1
fi

OUTPUT_DIR="${LIBERO_PLUS_OUTPUT_DIR:-$REPO_ROOT/output/libero-plus/$MODEL_VARIANT/$TASK}"
source "$CONDA_SH"
conda activate libero_test
# Wand loads ImageMagick through ctypes.  Long-lived tmux shells may not have
# the active conda environment's native-library directory in the loader path,
# even though the Python environment itself is active.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAGICK_HOME="${MAGICK_HOME:-$CONDA_PREFIX}"
export PYTHONPATH="$LIBERO_PLUS_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_PATH"
export LIBERO_EVAL_LOG_DIR="$OUTPUT_DIR"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running LIBERO-Plus evaluation"
echo "Suite: $TASK"
echo "Model: $MODEL_VARIANT"
echo "GR00T port: $PORT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

cd "$REPO_ROOT"
exec python examples/LiberoPlus/eval/run_libero_plus_eval.py \
    --task-suite-name "$TASK" \
    --port "$PORT" \
    "${EXTRA_ARGS[@]}"
