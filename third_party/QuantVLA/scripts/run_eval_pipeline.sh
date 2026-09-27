#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/experiment_common.sh"

METHOD="${1:-}"
shift || true

usage() {
    cat <<'EOF'
Internal evaluator used by eval_fp16.sh, eval_quantvla.sh, and
eval_quantvla_pivot_q.sh.

Required/primary options:
  --benchmark NAME       libero or libero-plus (default: libero-plus)
  --suite NAME           libero_spatial|libero_goal|libero_object|libero_10
  --gpu INDEX            one physical GPU index (default: 0)
  --port PORT            inference server port (default: 5556)
  --output-root PATH     root for normalized outputs (default: ./output)
  --output-dir PATH      exact eval directory; overrides output-root/run-name
  --run-name NAME        optional final output component (default: none)
  --checkpoint PATH      base model/HF id; for PIVOT-Q, checkpoint or adapter dir
  --eval-seed N          fixed policy sampling seed (default: 2026)
  --train-seed N         PIVOT-Q train seed used in output naming

Evaluation options:
  --manifest PATH        LIBERO-Plus selection manifest
  --resume               resume an existing episodes.jsonl
  --save-video           save rollout videos (default: disabled)
  --no-headless          do not request headless EGL evaluation
  --offline              use only locally cached Hugging Face files
  --server-timeout SEC   server startup timeout (default: 900)
  --dry-run              validate and print commands without executing
  --help                 show this message
  -- ARGS...             forward remaining arguments to the evaluator
EOF
}

case "$METHOD" in
    fp16)
        MODEL_VARIANT="groot-fp16"
        ;;
    quantvla)
        MODEL_VARIANT="groot-quantvla-w4a8"
        ;;
    quantvla-pivot_q)
        MODEL_VARIANT="groot-pivot_q-v2-w4a8"
        ;;
    *) echo "Internal error: unknown method '$METHOD'" >&2; exit 2 ;;
esac

BENCHMARK="libero-plus"
SUITE="libero_spatial"
GPU="0"
PORT="5556"
OUTPUT_ROOT="$REPO_ROOT/output"
OUTPUT_DIR=""
RUN_NAME=""
CHECKPOINT=""
MANIFEST="$REPO_ROOT/configs/libero_plus/shared560-first20.json"
EVAL_SEED="2026"
TRAIN_SEED=""
RESUME=0
SAVE_VIDEO=0
HEADLESS=1
OFFLINE=0
SERVER_TIMEOUT=900
DRY_RUN=0
EXTRA_ARGS=()

while (($#)); do
    case "$1" in
        --benchmark) quantvla_require_value "$1" "${2:-}"; BENCHMARK="$2"; shift 2 ;;
        --suite) quantvla_require_value "$1" "${2:-}"; SUITE="$2"; shift 2 ;;
        --gpu) quantvla_require_value "$1" "${2:-}"; GPU="$2"; shift 2 ;;
        --port) quantvla_require_value "$1" "${2:-}"; PORT="$2"; shift 2 ;;
        --output-root) quantvla_require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
        --output-dir|--output) quantvla_require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --run-name) quantvla_require_value "$1" "${2:-}"; RUN_NAME="$2"; shift 2 ;;
        --checkpoint|--ckpt) quantvla_require_value "$1" "${2:-}"; CHECKPOINT="$2"; shift 2 ;;
        --manifest) quantvla_require_value "$1" "${2:-}"; MANIFEST="$2"; shift 2 ;;
        --eval-seed) quantvla_require_value "$1" "${2:-}"; EVAL_SEED="$2"; shift 2 ;;
        --train-seed) quantvla_require_value "$1" "${2:-}"; TRAIN_SEED="$2"; shift 2 ;;
        --server-timeout) quantvla_require_value "$1" "${2:-}"; SERVER_TIMEOUT="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --save-video) SAVE_VIDEO=1; shift ;;
        --no-headless) HEADLESS=0; shift ;;
        --offline) OFFLINE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

quantvla_validate_benchmark "$BENCHMARK"
quantvla_validate_suite "$SUITE"
quantvla_validate_gpu "$GPU"
quantvla_validate_port "$PORT"
[[ -z "$RUN_NAME" ]] || quantvla_validate_run_name "$RUN_NAME"
quantvla_validate_port "$SERVER_TIMEOUT"
if [[ ! "$EVAL_SEED" =~ ^[0-9]+$ || ( -n "$TRAIN_SEED" && ! "$TRAIN_SEED" =~ ^[0-9]+$ ) ]]; then
    echo "eval/train seeds must be non-negative integers" >&2
    exit 2
fi

OUTPUT_ROOT="$(quantvla_abs_path "$OUTPUT_ROOT")"
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$BENCHMARK" == "libero-plus" ]]; then
        METHOD_OUTPUT="$METHOD"
        [[ "$METHOD" == "quantvla-pivot_q" ]] && METHOD_OUTPUT="pivot_q-v2-s16/seed-$(printf '%03d' "${TRAIN_SEED:-0}")"
        OUTPUT_DIR="$OUTPUT_ROOT/eval/libero-plus/shared560-first20/$METHOD_OUTPUT/$SUITE"
    else
        OUTPUT_DIR="$OUTPUT_ROOT/eval/$BENCHMARK/$METHOD/$SUITE"
    fi
    [[ -z "$RUN_NAME" ]] || OUTPUT_DIR="$OUTPUT_DIR/$RUN_NAME"
else
    OUTPUT_DIR="$(quantvla_abs_path "$OUTPUT_DIR")"
fi
MANIFEST="$(quantvla_abs_path "$MANIFEST")"

if [[ "$BENCHMARK" == "libero-plus" && ! -f "$MANIFEST" ]]; then
    echo "LIBERO-Plus manifest not found: $MANIFEST" >&2
    exit 2
fi

ADAPTER_PATH=""
if [[ "$METHOD" == "quantvla-pivot_q" ]]; then
    if [[ -z "$CHECKPOINT" ]]; then
        echo "quantvla-pivot_q evaluation requires --checkpoint/--ckpt" >&2
        exit 2
    fi
    if [[ -z "$TRAIN_SEED" ]]; then
        echo "quantvla-pivot_q evaluation requires --train-seed for reproducible naming" >&2
        exit 2
    fi
    CHECKPOINT="$(quantvla_abs_path "$CHECKPOINT")"
    if [[ -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
        ADAPTER_PATH="$CHECKPOINT"
    elif [[ -f "$CHECKPOINT/adapter/adapter_model.safetensors" ]]; then
        ADAPTER_PATH="$CHECKPOINT/adapter"
    else
        echo "PIVOT-Q adapter_model.safetensors not found below: $CHECKPOINT" >&2
        exit 2
    fi
fi

if [[ -e "$OUTPUT_DIR/metrics/episodes.jsonl" && "$RESUME" == 0 ]]; then
    echo "Refusing to overwrite existing evaluation: $OUTPUT_DIR/metrics/episodes.jsonl" >&2
    echo "Use --resume or choose a new --run-name/--output-dir." >&2
    exit 2
fi

SERVER_ENV=(
    "CUDA_VISIBLE_DEVICES=$GPU"
    "PYTHONUNBUFFERED=1"
    "GR00T_PORT=$PORT"
    "GR00T_MODEL_VARIANT=$MODEL_VARIANT"
)
if [[ -n "$CHECKPOINT" && "$METHOD" != "quantvla-pivot_q" ]]; then
    SERVER_ENV+=("GR00T_MODEL_PATH=$CHECKPOINT")
fi

case "$METHOD" in
    fp16) SERVER_CMD=("$REPO_ROOT/run_inference_server.sh" "$SUITE") ;;
    quantvla) SERVER_CMD=("$REPO_ROOT/run_quantvla.sh" "$SUITE") ;;
    quantvla-pivot_q) SERVER_CMD=("$REPO_ROOT/run_pivot_q_inference.sh" "$SUITE" "$ADAPTER_PATH") ;;
esac

EVAL_ENV=(
    "CUDA_VISIBLE_DEVICES=$GPU"
    "GR00T_PORT=$PORT"
    "EVAL_MODEL_VARIANT=$MODEL_VARIANT"
    "LIBERO_EVAL_METRICS_DIR=$OUTPUT_DIR/metrics"
    "LIBERO_EVAL_LOGS_DIR=$OUTPUT_DIR/logs"
    "LIBERO_EVAL_VIDEO_DIR=$OUTPUT_DIR/videos"
)
if (( OFFLINE )); then
    OFFLINE_ENV=(
        "HF_HUB_OFFLINE=1"
        "TRANSFORMERS_OFFLINE=1"
        "HF_HUB_DISABLE_TELEMETRY=1"
    )
    SERVER_ENV+=("${OFFLINE_ENV[@]}")
    EVAL_ENV+=("${OFFLINE_ENV[@]}")
fi
if [[ "$BENCHMARK" == "libero-plus" ]]; then
    EVAL_ENV+=("LIBERO_PLUS_OUTPUT_DIR=$OUTPUT_DIR")
    EVAL_CMD=("$REPO_ROOT/run_libero_plus_eval.sh" "$SUITE" --model-variant "$MODEL_VARIANT" --sample-manifest "$MANIFEST" --policy-seed "$EVAL_SEED")
else
    EVAL_ENV+=("LIBERO_EVAL_LOG_DIR=$OUTPUT_DIR")
    EVAL_CMD=("$REPO_ROOT/run_libero_eval.sh" "$SUITE" --model-variant "$MODEL_VARIANT")
fi
(( HEADLESS )) && EVAL_CMD+=(--headless)
(( SAVE_VIDEO )) || EVAL_CMD+=(--no-save-video)
(( RESUME )) && EVAL_CMD+=(--resume)
EVAL_CMD+=("${EXTRA_ARGS[@]}")

echo "Method:       $METHOD"
echo "Benchmark:    $BENCHMARK"
echo "Suite:        $SUITE"
echo "GPU / port:   $GPU / $PORT"
echo "Checkpoint:   ${CHECKPOINT:-suite default}"
echo "Eval seed:    $EVAL_SEED"
[[ -n "$TRAIN_SEED" ]] && echo "Train seed:   $TRAIN_SEED"
echo "HF offline:   $OFFLINE"
echo "Output:       $OUTPUT_DIR"
echo "Server command:"
quantvla_print_command env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}"
echo "Evaluator command:"
quantvla_print_command env "${EVAL_ENV[@]}" "${EVAL_CMD[@]}"

if (( DRY_RUN )); then
    echo "Dry run complete; no process started and no output created."
    exit 0
fi

printf -v SERVER_COMMAND_TEXT '%q ' env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}"
printf -v EVALUATOR_COMMAND_TEXT '%q ' env "${EVAL_ENV[@]}" "${EVAL_CMD[@]}"
METADATA_MANIFEST=""
[[ "$BENCHMARK" == "libero-plus" ]] && METADATA_MANIFEST="$MANIFEST"
EVAL_ENV+=(
    "EVAL_METHOD=$METHOD"
    "EVAL_BENCHMARK=$BENCHMARK"
    "EVAL_TRAIN_SEED=$TRAIN_SEED"
    "EVAL_CHECKPOINT=$CHECKPOINT"
    "EVAL_MANIFEST=$METADATA_MANIFEST"
    "EVAL_SERVER_COMMAND=$SERVER_COMMAND_TEXT"
    "EVAL_EVALUATOR_COMMAND=$EVALUATOR_COMMAND_TEXT"
)
mkdir -p "$OUTPUT_DIR/metrics" "$OUTPUT_DIR/logs"
SERVER_LOG="$OUTPUT_DIR/logs/server.log"
PIPELINE_LOG="$OUTPUT_DIR/logs/pipeline.log"

env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

quantvla_wait_for_server "$SERVER_PID" "$SERVER_LOG" "$SERVER_TIMEOUT"
echo "Inference server ready (PID $SERVER_PID)."
set -o pipefail
env "${EVAL_ENV[@]}" "${EVAL_CMD[@]}" 2>&1 | tee -a "$PIPELINE_LOG"

if [[ ! -s "$OUTPUT_DIR/metrics/summary.json" ]]; then
    echo "Evaluation exited without a non-empty metrics/summary.json: $OUTPUT_DIR" >&2
    exit 1
fi
echo "Evaluation complete: $OUTPUT_DIR/metrics/summary.json"
