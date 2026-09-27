#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/scripts/experiment_common.sh"

usage() {
    cat <<'EOF'
Train the QuantVLA-PIVOT-Q v2 LoRA adapter on Shared-560 first-20 tasks.

Options:
  --suite NAME              suite to train (default: libero_spatial)
  --gpu INDEX               one physical GPU index (default: 0)
  --env-port PORT           LIBERO-Plus OOD service port (default: 5590)
  --clean-env-port PORT     clean LIBERO anchor service port (default: 5591)
  --output-root PATH        root for normalized outputs (default: ./output)
  --output-dir PATH         exact train directory; overrides output-root/run-name
  --run-name NAME           optional final output component (default: none)
  --manifest PATH           LIBERO-Plus selection manifest
  --resume-from PATH        explicit checkpoint directory
  --seed N                  training seed (default: 0)
  --episodes N              full rollout episodes (default: 140)
  --updates-per-episode N   optimizer updates per rollout (default: 5)
  --episode-horizon N       override suite horizon (default: suite protocol)
  --save-every-steps N      checkpoint interval (default: 70)
  --keep-last-checkpoints N keep newest N checkpoints; 0 keeps all (default: 2)
  --save-timestep-scores    save full per-timestep q/r arrays (default: off)
  --offline                 use only locally cached Hugging Face files
  --dry-run                 validate and print command without executing
  --help                    show this message
  -- ARGS...                forward advanced Tyro options to train_pivot_q.py
EOF
}

SUITE="libero_spatial"
GPU="0"
ENV_PORT="5590"
CLEAN_ENV_PORT="5591"
OUTPUT_ROOT="$REPO_ROOT/output"
OUTPUT_DIR=""
RUN_NAME=""
MANIFEST="$REPO_ROOT/configs/libero_plus/shared560-first20.json"
RESUME_FROM=""
SEED="0"
EPISODES="140"
UPDATES_PER_EPISODE="5"
EPISODE_HORIZON=""
SAVE_EVERY_STEPS="70"
KEEP_LAST_CHECKPOINTS="2"
SAVE_TIMESTEP_SCORES=0
OFFLINE=0
DRY_RUN=0
EXTRA_ARGS=()

while (($#)); do
    case "$1" in
        --suite) quantvla_require_value "$1" "${2:-}"; SUITE="$2"; shift 2 ;;
        --gpu) quantvla_require_value "$1" "${2:-}"; GPU="$2"; shift 2 ;;
        --env-port) quantvla_require_value "$1" "${2:-}"; ENV_PORT="$2"; shift 2 ;;
        --clean-env-port) quantvla_require_value "$1" "${2:-}"; CLEAN_ENV_PORT="$2"; shift 2 ;;
        --output-root) quantvla_require_value "$1" "${2:-}"; OUTPUT_ROOT="$2"; shift 2 ;;
        --output-dir|--output) quantvla_require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --run-name) quantvla_require_value "$1" "${2:-}"; RUN_NAME="$2"; shift 2 ;;
        --manifest) quantvla_require_value "$1" "${2:-}"; MANIFEST="$2"; shift 2 ;;
        --resume-from|--resume-from-checkpoint) quantvla_require_value "$1" "${2:-}"; RESUME_FROM="$2"; shift 2 ;;
        --seed) quantvla_require_value "$1" "${2:-}"; SEED="$2"; shift 2 ;;
        --episodes) quantvla_require_value "$1" "${2:-}"; EPISODES="$2"; shift 2 ;;
        --updates-per-episode) quantvla_require_value "$1" "${2:-}"; UPDATES_PER_EPISODE="$2"; shift 2 ;;
        --episode-horizon) quantvla_require_value "$1" "${2:-}"; EPISODE_HORIZON="$2"; shift 2 ;;
        --save-every-steps) quantvla_require_value "$1" "${2:-}"; SAVE_EVERY_STEPS="$2"; shift 2 ;;
        --keep-last-checkpoints) quantvla_require_value "$1" "${2:-}"; KEEP_LAST_CHECKPOINTS="$2"; shift 2 ;;
        --save-timestep-scores) SAVE_TIMESTEP_SCORES=1; shift ;;
        --offline) OFFLINE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

quantvla_validate_suite "$SUITE"
quantvla_validate_gpu "$GPU"
quantvla_validate_port "$ENV_PORT"
quantvla_validate_port "$CLEAN_ENV_PORT"
[[ -z "$RUN_NAME" ]] || quantvla_validate_run_name "$RUN_NAME"
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$EPISODES" =~ ^[1-9][0-9]*$ || \
      ! "$UPDATES_PER_EPISODE" =~ ^[1-9][0-9]*$ || \
      ! "$SAVE_EVERY_STEPS" =~ ^[1-9][0-9]*$ || \
      ! "$KEEP_LAST_CHECKPOINTS" =~ ^[0-9]+$ ]]; then
    echo "seed and training counts must be non-negative/positive integers" >&2
    exit 2
fi
if [[ -n "$EPISODE_HORIZON" && ! "$EPISODE_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
    echo "--episode-horizon must be a positive integer" >&2
    exit 2
fi

OUTPUT_ROOT="$(quantvla_abs_path "$OUTPUT_ROOT")"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$OUTPUT_ROOT/train/libero-plus/pivot_q-v2-s16-shared560-first20/seed-$(printf '%03d' "$SEED")/$SUITE"
    [[ -z "$RUN_NAME" ]] || OUTPUT_DIR="$OUTPUT_DIR/$RUN_NAME"
else
    OUTPUT_DIR="$(quantvla_abs_path "$OUTPUT_DIR")"
fi
MANIFEST="$(quantvla_abs_path "$MANIFEST")"
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 2; }
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_FROM="$(quantvla_abs_path "$RESUME_FROM")"
    [[ -d "$RESUME_FROM" ]] || { echo "Resume checkpoint not found: $RESUME_FROM" >&2; exit 2; }
fi

TRAIN_ENV=(
    "CUDA_VISIBLE_DEVICES=$GPU"
    "GAP_PIVOT_Q_ENV_PORT=$ENV_PORT"
    "GAP_PIVOT_Q_CLEAN_ENV_PORT=$CLEAN_ENV_PORT"
    "GAP_PIVOT_Q_OUTPUT_DIR=$OUTPUT_DIR"
    "GAP_PIVOT_Q_SAMPLE_MANIFEST=$MANIFEST"
)
if (( OFFLINE )); then
    TRAIN_ENV+=(
        "HF_HUB_OFFLINE=1"
        "TRANSFORMERS_OFFLINE=1"
        "HF_HUB_DISABLE_TELEMETRY=1"
    )
fi
TRAIN_CMD=(
    "$REPO_ROOT/run_pivot_q.sh" "$SUITE"
    --seed "$SEED"
    --num-rollout-episodes "$EPISODES"
    --updates-per-episode "$UPDATES_PER_EPISODE"
    --save-every-steps "$SAVE_EVERY_STEPS"
    --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
)
(( SAVE_TIMESTEP_SCORES )) && TRAIN_CMD+=(--save-timestep-scores)
[[ -n "$EPISODE_HORIZON" ]] && TRAIN_CMD+=(--episode-horizon "$EPISODE_HORIZON")
[[ -n "$RESUME_FROM" ]] && TRAIN_CMD+=(--resume-from-checkpoint "$RESUME_FROM")
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

echo "Method:       quantvla-pivot_q-v2-s16"
echo "Train data:   libero-plus"
echo "Suite:        $SUITE"
echo "GPU:          $GPU"
echo "Env ports:    $ENV_PORT / $CLEAN_ENV_PORT"
echo "Train seed:   $SEED"
echo "Episodes:     $EPISODES x $UPDATES_PER_EPISODE updates"
echo "HF offline:   $OFFLINE"
echo "Output:       $OUTPUT_DIR"
echo "Training command:"
quantvla_print_command env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}"

if (( DRY_RUN )); then
    echo "Dry run complete; no process started and no output created."
    exit 0
fi

printf -v LAUNCH_COMMAND '%q ' env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}"
TRAIN_ENV+=("GAP_PIVOT_Q_LAUNCH_COMMAND=$LAUNCH_COMMAND")
mkdir -p "$OUTPUT_DIR/logs"
set -o pipefail
env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/logs/train.log"
