#!/bin/bash
set -euo pipefail
# Script to run GR00T inference server for Libero evaluation
# Usage: ./run_inference_server.sh [task_suite_name]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

TASK=${1:-libero_10}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
PORT="${GR00T_PORT:-5556}"
MODEL_VARIANT="${GR00T_MODEL_VARIANT:-groot-fp16}"
HF_HOME="${HF_HOME:-$REPO_ROOT/model}"
HF_PROXY="${HF_PROXY:-}"

# Activate groot_test environment
source "$CONDA_SH"
conda activate groot_test
mkdir -p "$HF_HOME"
export HF_HOME
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ -n "$HF_PROXY" ]]; then
    export HTTP_PROXY="${HTTP_PROXY:-$HF_PROXY}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$HF_PROXY}"
fi
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Set model path and data config based on task. GR00T_MODEL_PATH may point to a
# local checkpoint or a Hugging Face model id for reproducible comparisons.
case $TASK in
    libero_spatial)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_goal)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
        ;;
    libero_object)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_90)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_10)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
        exit 1
        ;;
esac
MODEL_PATH="${GR00T_MODEL_PATH:-$MODEL_PATH}"

# Allow override of denoising steps via environment variable
DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}
ADAPTER_PATH=${GR00T_ADAPTER_PATH:-}
ADAPTER_ARGS=()
if [[ -n "$ADAPTER_PATH" ]]; then
    ADAPTER_ARGS+=(--adapter-path "$ADAPTER_PATH")
fi

echo "=========================================="
echo "Starting GR00T inference server for $TASK"
echo "Model variant: $MODEL_VARIANT"
echo "Model: $MODEL_PATH"
echo "Data Config: $DATA_CONFIG"
echo "Repository: $REPO_ROOT"
echo "Model cache: $HF_HOME"
echo "Port: $PORT"
echo "Denoising Steps: $DENOISING_STEPS"
echo "PEFT Adapter: ${ADAPTER_PATH:-none}"
echo "=========================================="

cd "$REPO_ROOT"

exec python scripts/inference_service.py \
    --model_path "$MODEL_PATH" \
    --server \
    --data_config "$DATA_CONFIG" \
    --denoising-steps "$DENOISING_STEPS" \
    --port "$PORT" \
    --embodiment-tag new_embodiment \
    "${ADAPTER_ARGS[@]}"
