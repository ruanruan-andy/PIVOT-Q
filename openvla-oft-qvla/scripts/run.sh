#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
command_name="${1:?Usage: run.sh COMMAND [arguments]}"
shift
case "$command_name" in
  eval-fp) module=evaluation.evaluate; config=fp ;;
  eval-qvla) module=evaluation.evaluate; config=qvla ;;
  train-pivot-q) module=training.train; config=pivot_q ;;
  train-full-distill) module=training.train; config=full_distill ;;
  eval-pivot-q) module=evaluation.evaluate; config=pivot_q ;;
  eval-full-distill) module=evaluation.evaluate; config=full_distill ;;
  collect-calibration) exec "${PYTHON:-python}" -m integration.calibrate collect --config configs/qvla.yaml "$@" ;;
  prepare-qvla) exec "${PYTHON:-python}" -m integration.calibrate quantize --config configs/qvla.yaml "$@" ;;
  *) echo "Unknown command: $command_name" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" -m "$module" --config "configs/$config.yaml" "$@"
