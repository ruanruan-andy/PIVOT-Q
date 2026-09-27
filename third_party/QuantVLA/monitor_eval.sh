#!/bin/bash
set -euo pipefail

# Default: FP16 vs QuantVLA vs PIVOT-Q-v2 on the disjoint Test-560 split.
# All additional arguments are forwarded to scripts/eval_monitor.py.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MONITOR_PYTHON:-python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python
fi
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/eval_monitor.py" \
    --repo-root "$REPO_ROOT" "$@"
