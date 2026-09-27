#!/bin/bash
set -euo pipefail

# Train PIVOT-Q-v2 on the shared first-20 LIBERO-Plus manifest.
# Usage: CUDA_VISIBLE_DEVICES=0 ./run_pivot_q.sh libero_spatial [extra tyro args]

TASK="${1:-libero_spatial}"
shift || true
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
LIBERO_ROOT="${LIBERO_ROOT:-$REPO_ROOT/../LIBERO}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$REPO_ROOT/../LIBERO-plus}"
CLEAN_LIBERO_CONFIG_PATH="${CLEAN_LIBERO_CONFIG_PATH:-$REPO_ROOT/../../config/sparse_anchor_libero}"
LIBERO_PLUS_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-$REPO_ROOT/configs/libero_plus}"
SAMPLE_MANIFEST="${GAP_PIVOT_Q_SAMPLE_MANIFEST:-$REPO_ROOT/configs/libero_plus/shared560-first20.json}"
OOD_ENV_PORT="${GAP_PIVOT_Q_ENV_PORT:-5590}"
CLEAN_ENV_PORT="${GAP_PIVOT_Q_CLEAN_ENV_PORT:-5591}"

case "$TASK" in
    libero_spatial|libero_goal|libero_object|libero_10) ;;
    *) echo "Unsupported clean LIBERO suite: $TASK" >&2; exit 1 ;;
esac

OUTPUT_DIR="${GAP_PIVOT_Q_OUTPUT_DIR:-$REPO_ROOT/output/train/libero-plus/pivot_q-v2-s16-shared560-first20/seed-000/$TASK}"
mkdir -p "$OUTPUT_DIR/logs"

# PIVOT-Q adaptation environment. Train and eval intentionally share the same task IDs.
(
    source "$CONDA_SH"
    conda activate libero_test
    export PYTHONPATH="$REPO_ROOT:$LIBERO_PLUS_ROOT"
    export LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_PATH"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    exec python "$REPO_ROOT/scripts/libero_plus_env_service.py" \
        --task-suite-name "$TASK" \
        --sample-manifest "$SAMPLE_MANIFEST" \
        --port "$OOD_ENV_PORT"
) >"$OUTPUT_DIR/logs/plus_rollout_service.log" 2>&1 &
OOD_ENV_PID=$!

# Standard clean LIBERO supplies the optional IID preservation anchor.
(
    source "$CONDA_SH"
    conda activate libero_test
    export PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT"
    export LIBERO_CONFIG_PATH="$CLEAN_LIBERO_CONFIG_PATH"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    exec python "$REPO_ROOT/scripts/libero_iid_env_service.py" --port "$CLEAN_ENV_PORT"
) >"$OUTPUT_DIR/logs/iid_anchor_service.log" 2>&1 &
CLEAN_ENV_PID=$!
cleanup() {
    kill "$OOD_ENV_PID" "$CLEAN_ENV_PID" 2>/dev/null || true
    wait "$OOD_ENV_PID" 2>/dev/null || true
    wait "$CLEAN_ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source "$CONDA_SH"
conda activate groot_test

# The trainer imports neither simulator package; both stay in the services above.
export PYTHONPATH="$REPO_ROOT"
export HF_HOME="${HF_HOME:-$REPO_ROOT/model}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

cd "$REPO_ROOT"
python scripts/train_pivot_q.py \
    --task-suite-name "$TASK" \
    --output-dir "$OUTPUT_DIR" \
    --sample-manifest "$SAMPLE_MANIFEST" \
    --env-port "$OOD_ENV_PORT" \
    --clean-env-port "$CLEAN_ENV_PORT" \
    "$@"
