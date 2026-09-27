#!/bin/bash
set -euo pipefail

# Default: write a partial-safe three-model report under output/summary.
# Use --require-complete for a final report.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COLLECT_PYTHON:-python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python
fi
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/collect_eval_metrics.py" \
    --repo-root "$REPO_ROOT" "$@"
