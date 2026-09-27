#!/bin/bash
set -euo pipefail

# Start a GR00T N1.5 server with QuantVLA's W4A8 fake-quantization path.
# Usage: CUDA_VISIBLE_DEVICES=4 GR00T_PORT=5570 ./run_quantvla.sh libero_spatial

TASK="${1:-libero_10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$TASK" in
    libero_spatial) CALIBRATION_NAME="spatial" ;;
    libero_goal) CALIBRATION_NAME="goal" ;;
    libero_object) CALIBRATION_NAME="object" ;;
    libero_10) CALIBRATION_NAME="long" ;;
    *)
        echo "Unsupported QuantVLA suite: $TASK" >&2
        echo "Available suites: libero_spatial, libero_goal, libero_object, libero_10" >&2
        exit 1
        ;;
esac

CALIBRATION_PATH="${GR00T_ATM_ALPHA_PATH:-$REPO_ROOT/atm_alpha_beta_${CALIBRATION_NAME}.json}"
PACK_DIR="${GR00T_DUQUANT_PACKDIR:-$REPO_ROOT/model/quantvla/groot-n1.5/$TASK/duquant_pack}"

if [[ ! -f "$CALIBRATION_PATH" ]]; then
    echo "QuantVLA ATM/OHB calibration file not found: $CALIBRATION_PATH" >&2
    exit 1
fi
mkdir -p "$PACK_DIR"

# QuantVLA layout: all language-model linear projections plus DiT MLPs.
# DiT Q/K/V/O projections remain floating point.
export GR00T_DUQUANT_DEBUG="${GR00T_DUQUANT_DEBUG:-1}"
export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)'
export GR00T_DUQUANT_WBITS_DEFAULT="${GR00T_DUQUANT_WBITS_DEFAULT:-4}"
export GR00T_DUQUANT_ABITS="${GR00T_DUQUANT_ABITS:-8}"
export GR00T_DUQUANT_BLOCK="${GR00T_DUQUANT_BLOCK:-64}"
export GR00T_DUQUANT_PERMUTE="${GR00T_DUQUANT_PERMUTE:-0}"
export GR00T_DUQUANT_ROW_ROT="${GR00T_DUQUANT_ROW_ROT:-restore}"
export GR00T_DUQUANT_ACT_PCT="${GR00T_DUQUANT_ACT_PCT:-99.9}"
export GR00T_DUQUANT_CALIB_STEPS="${GR00T_DUQUANT_CALIB_STEPS:-32}"
export GR00T_DUQUANT_LS="${GR00T_DUQUANT_LS:-0.15}"
export GR00T_DUQUANT_PACKDIR="$PACK_DIR"

export GR00T_ATM_ALPHA_PATH="$CALIBRATION_PATH"
export GR00T_ATM_ENABLE="${GR00T_ATM_ENABLE:-1}"
export GR00T_ATM_SCOPE="${GR00T_ATM_SCOPE:-dit}"
export GR00T_OHB_ENABLE="${GR00T_OHB_ENABLE:-1}"
export GR00T_OHB_FALLBACK="${GR00T_OHB_FALLBACK:-1.0}"
export GR00T_OHB_SCOPE="${GR00T_OHB_SCOPE:-dit}"

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
export GR00T_DENOISING_STEPS="${GR00T_DENOISING_STEPS:-8}"
export GR00T_MODEL_VARIANT="${GR00T_MODEL_VARIANT:-groot-quantvla-w4a8}"

echo "=========================================="
echo "Starting QuantVLA GR00T N1.5 W4A8"
echo "Suite: $TASK"
echo "Backend: fake quantization (W4A8)"
echo "ATM/OHB: $CALIBRATION_PATH"
echo "DuQuant pack: $PACK_DIR"
echo "Port: ${GR00T_PORT:-5556}"
echo "=========================================="

exec "$REPO_ROOT/run_inference_server.sh" "$TASK"
