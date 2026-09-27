#!/bin/bash

# Shared helpers for the user-facing experiment launchers.
# This file is sourced; it is not an experiment entry point.

quantvla_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}
quantvla_require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "$option requires a value" >&2
        exit 2
    fi
}

quantvla_validate_suite() {
    case "$1" in
        libero_spatial|libero_goal|libero_object|libero_10) ;;
        *)
            echo "Unsupported suite: $1" >&2
            echo "Available suites: libero_spatial, libero_goal, libero_object, libero_10" >&2
            exit 2
            ;;
    esac
}

quantvla_validate_benchmark() {
    case "$1" in
        libero|libero-plus) ;;
        *) echo "Unsupported benchmark: $1 (use libero or libero-plus)" >&2; exit 2 ;;
    esac
}

quantvla_validate_run_name() {
    if [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "Invalid run name: $1" >&2
        echo "Use letters, digits, dot, underscore, or hyphen; do not use path separators." >&2
        exit 2
    fi
}

quantvla_validate_port() {
    if [[ ! "$1" =~ ^[0-9]+$ ]] || (( "$1" < 1 || "$1" > 65535 )); then
        echo "Invalid TCP port: $1" >&2
        exit 2
    fi
}

quantvla_validate_gpu() {
    if [[ ! "$1" =~ ^[0-9]+$ ]]; then
        echo "--gpu must name one physical GPU index, got: $1" >&2
        exit 2
    fi
}

quantvla_abs_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$PWD" "$path"
    fi
}

quantvla_print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

quantvla_wait_for_server() {
    local pid="$1"
    local log_path="$2"
    local timeout="$3"
    local elapsed=0
    while (( elapsed < timeout )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Inference server exited before becoming ready: $log_path" >&2
            tail -n 80 "$log_path" >&2 || true
            return 1
        fi
        if grep -q "Server is ready and listening" "$log_path" 2>/dev/null; then
            return 0
        fi
        sleep 1
        ((elapsed += 1))
    done
    echo "Timed out after ${timeout}s waiting for inference server: $log_path" >&2
    return 1
}
