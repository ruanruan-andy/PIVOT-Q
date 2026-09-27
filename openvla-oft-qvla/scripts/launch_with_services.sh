#!/usr/bin/env bash
set -euo pipefail

# Start the suite-specific LIBERO services, wait for health, then run train/eval.
# The model process may use oft_qvla while services run in libero_test.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:?usage: launch_with_services.sh train|eval --suite SUITE ...}"
shift

SUITE=""
PLUS_PORT=""
CLEAN_PORT=""
# Do not probe `command -v python` under `set -e`: non-interactive compute
# shells may not expose a PATH-level python.  The explicit option or
# LIBERO_SERVICE_PYTHON remains authoritative.
SERVICE_PYTHON="${LIBERO_SERVICE_PYTHON:-${PYTHON:-python}}"
PLUS_ROOT="${LIBERO_PLUS_ROOT:-${ROOT}/../third_party/LIBERO-plus}"
CLEAN_ROOT="${LIBERO_ROOT:-${ROOT}/../third_party/LIBERO}"
MANIFEST="${ROOT}/../manifests/libero_plus_first20.json"
MODEL_CMD=()
SERVICE_PIDS=()
cleanup() {
  local pid
  for pid in "${SERVICE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SERVICE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while (($#)); do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --plus-port) PLUS_PORT="$2"; shift 2 ;;
    --clean-port) CLEAN_PORT="$2"; shift 2 ;;
    --service-python) SERVICE_PYTHON="$2"; shift 2 ;;
    --plus-root) PLUS_ROOT="$2"; shift 2 ;;
    --clean-root) CLEAN_ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --) shift; MODEL_CMD=("$@"); break ;;
    *) MODEL_CMD+=("$1"); shift ;;
  esac
done

[[ -n "$SUITE" && -n "$PLUS_PORT" && -n "$CLEAN_PORT" && ${#MODEL_CMD[@]} -gt 0 ]] || {
  echo "missing --suite/--plus-port/--clean-port or command" >&2; exit 2;
}

module_cmd=("$SERVICE_PYTHON" -m evaluation.env_service)
start_service() {
  local domain="$1" port="$2" root="$3"
  local log="$ROOT/logs/service_${SUITE}_${domain}_${port}.log"
  local cfgdir="$ROOT/cache/runtime/${SUITE}_${domain}"
  mkdir -p "$cfgdir"
  if [[ ! -f "$cfgdir/config.yaml" ]]; then
    cat >"$cfgdir/config.yaml" <<YAML
benchmark_root: $root/libero/libero
bddl_files: $root/libero/libero/bddl_files
init_states: $root/libero/libero/init_files
datasets: $root/datasets
assets: $root/libero/libero/assets
YAML
  fi
  if "$SERVICE_PYTHON" - "$port" <<'PY' >/dev/null 2>&1
import socket,sys
s=socket.socket(); s.settimeout(.2)
try: s.connect(("127.0.0.1", int(sys.argv[1]))); raise SystemExit(0)
except OSError: raise SystemExit(1)
PY
  then
    return
  fi
  mkdir -p "$ROOT/logs"
  if [[ "$domain" == plus ]]; then
    (cd "$ROOT" && exec env PYTHONPATH="$ROOT:$PLUS_ROOT" LIBERO_CONFIG_PATH="$cfgdir" MUJOCO_GL=egl \
      "${module_cmd[@]}" --suite "$SUITE" --manifest "$MANIFEST" --port "$port" --seed 0) \
      >"$log" 2>&1 &
  else
    (cd "$ROOT" && exec env PYTHONPATH="$ROOT:$CLEAN_ROOT" LIBERO_CONFIG_PATH="$cfgdir" MUJOCO_GL=egl \
      "${module_cmd[@]}" --suite "$SUITE" --port "$port" --seed 0) \
      >"$log" 2>&1 &
  fi
  SERVICE_PIDS+=("$!")
}
start_service plus "$PLUS_PORT" "$PLUS_ROOT"
start_service clean "$CLEAN_PORT" "$CLEAN_ROOT"

for _ in $(seq 1 60); do
  if "$SERVICE_PYTHON" - "$PLUS_PORT" "$CLEAN_PORT" <<'PY' >/dev/null 2>&1
import socket,sys
for p in sys.argv[1:]:
 s=socket.socket(); s.settimeout(.2); s.connect(("127.0.0.1",int(p))); s.close()
PY
  then break; fi
  sleep 1
done
"$SERVICE_PYTHON" - "$PLUS_PORT" "$CLEAN_PORT" <<'PY'
import socket,sys
for p in sys.argv[1:]:
 s=socket.socket(); s.settimeout(.5); s.connect(("127.0.0.1",int(p))); s.close()
print("LIBERO services reachable:", ", ".join(sys.argv[1:]))
PY
"${MODEL_CMD[@]}"
