from __future__ import annotations

import json
import math
import os
import pathlib
import shlex
import socket
import statistics
from collections import OrderedDict
from datetime import datetime
from typing import Any


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
DEFAULT_PLUS_MANIFEST = pathlib.Path("configs/libero_plus/shared560-first20.json")
DISPLAY_SUITE = {
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_10": "libero_10",
}
MODELS = OrderedDict(
    (
        (
            "groot-fp16",
            {
                "label": "GR00T N1.5 · FP16",
                "precision": "FP16",
                "quantization_backend": "none",
            },
        ),
        (
            "groot-quantvla-w4a8",
            {
                "label": "GR00T N1.5 · QuantVLA W4A8",
                "precision": "W4A8",
                "quantization_backend": "fake-quant",
            },
        ),
        (
            "groot-pivot_q-v2-w4a8",
            {
                "label": "GR00T N1.5 · QuantVLA-PIVOT-Q W4A8",
                "precision": "W4A8 + PEFT",
                "quantization_backend": "fake-quant + PIVOT-Q",
            },
        ),
    )
)
PRIMARY_MODELS = ("groot-fp16", "groot-quantvla-w4a8", "groot-pivot_q-v2-w4a8")
METHOD_TO_MODEL = OrderedDict(
    (
        ("fp16", "groot-fp16"),
        ("quantvla", "groot-quantvla-w4a8"),
        ("quantvla-pivot_q", "groot-pivot_q-v2-w4a8"),
    )
)
MODEL_TO_METHOD = {model: method for method, model in METHOD_TO_MODEL.items()}
BENCHMARKS = OrderedDict(
    (
        ("libero", {"label": "Standard LIBERO"}),
        ("libero-plus", {"label": "LIBERO-Plus · Shared-560 first-20"}),
    )
)
CATEGORY_ORDER = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)
CATEGORY_LABEL = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes <= 0:
        return 0.0, 0.0
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def load_sample_manifest(
    repo_root: pathlib.Path, manifest_path: pathlib.Path | None = None
) -> dict[str, Any]:
    path = manifest_path or DEFAULT_PLUS_MANIFEST
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    result["path"] = str(path)
    return result


def manifest_suite_quota(manifest: dict[str, Any], suite: str) -> int:
    value = manifest.get("per_suite_per_category")
    if isinstance(value, dict):
        if suite not in value:
            raise KeyError(f"manifest is missing per-suite quota for {suite}")
        return int(value[suite])
    if value is None:
        total = int(manifest["total_per_category"])
        if total % len(SUITES):
            raise ValueError("total_per_category is not divisible by the number of suites")
        return total // len(SUITES)
    return int(value)


def benchmark_totals(
    repo_root: pathlib.Path, manifest_path: pathlib.Path | None = None
) -> dict[str, dict[str, int]]:
    manifest = load_sample_manifest(repo_root, manifest_path)
    category_count = len(manifest["categories"])
    return {
        "libero": {suite: 50 for suite in SUITES},
        "libero-plus": {
            suite: manifest_suite_quota(manifest, suite) * category_count for suite in SUITES
        },
    }


def _record_key(benchmark: str, record: dict[str, Any]) -> tuple[int, ...]:
    task_index = int(record["task_index"])
    if benchmark == "libero":
        return task_index, int(record.get("episode_index", 0))
    return (task_index,)


def load_effective_records(
    path: pathlib.Path, benchmark: str
) -> tuple[list[dict[str, Any]], int, int]:
    if not path.exists():
        return [], 0, 0
    effective: dict[tuple[int, ...], tuple[int, dict[str, Any]]] = {}
    malformed = 0
    valid_lines = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for sequence, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = _record_key(benchmark, record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed += 1
                continue
            valid_lines += 1
            effective[key] = (sequence, record)
    ordered = [item[1] for item in sorted(effective.values(), key=lambda item: item[0])]
    return ordered, malformed, valid_lines - len(ordered)


def model_result_dir(
    repo_root: pathlib.Path,
    model: str,
    benchmark: str,
    suite: str,
    *,
    output_root: pathlib.Path | None = None,
    run_name: str = "default",
    pivot_q_train_seed: int = 0,
    legacy_fallback: bool = True,
) -> pathlib.Path:
    root = (output_root or (repo_root / "output")).resolve()
    method = MODEL_TO_METHOD.get(model, model)
    if benchmark == "libero-plus":
        method_path = (
            pathlib.Path("pivot_q-v2-s16") / f"seed-{pivot_q_train_seed:03d}"
            if method == "quantvla-pivot_q"
            else pathlib.Path(method)
        )
        suite_root = (
            root
            / "eval"
            / "libero-plus"
            / "shared560-first20"
            / method_path
            / suite
        )
    else:
        suite_root = root / "eval" / benchmark / method / suite
    normalized = suite_root if run_name in ("", "default") else suite_root / run_name
    old_default = suite_root / "default"
    has_new_direct_result = (suite_root / "metrics" / "episodes.jsonl").exists() or (
        suite_root / "episodes.jsonl"
    ).exists()
    if (
        normalized == suite_root
        and old_default.exists()
        and not has_new_direct_result
    ):
        return old_default
    if normalized.exists() or not legacy_fallback:
        return normalized
    if old_default.exists():
        return old_default
    return repo_root / "output" / benchmark / model / suite


def model_episode_paths(
    repo_root: pathlib.Path,
    model: str,
    benchmark: str,
    suite: str,
    *,
    output_root: pathlib.Path | None = None,
    run_name: str = "default",
    pivot_q_train_seed: int = 0,
    legacy_fallback: bool = True,
) -> list[pathlib.Path]:
    """Return the direct result file or category-specific result files for a suite.

    Baseline evaluations write one ``episodes.jsonl`` directly under a suite.
    Category-by-category ablations write one file per category below that suite.
    Prefer the direct file so a resumed baseline is never double counted.
    """
    suite_dir = model_result_dir(
        repo_root,
        model,
        benchmark,
        suite,
        output_root=output_root,
        run_name=run_name,
        pivot_q_train_seed=pivot_q_train_seed,
        legacy_fallback=legacy_fallback,
    )
    direct_paths = [suite_dir / "metrics" / "episodes.jsonl", suite_dir / "episodes.jsonl"]
    for direct_path in direct_paths:
        if direct_path.exists():
            return [direct_path]
    category_paths = list(suite_dir.glob("*/metrics/episodes.jsonl"))
    category_paths.extend(suite_dir.glob("*/episodes.jsonl"))
    return sorted(set(category_paths))


def load_model_records(
    paths: list[pathlib.Path], benchmark: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Load per-file effective records without conflating task ids across categories."""
    records: list[dict[str, Any]] = []
    malformed = 0
    duplicates = 0
    for path in paths:
        file_records, file_malformed, file_duplicates = load_effective_records(path, benchmark)
        records.extend(file_records)
        malformed += file_malformed
        duplicates += file_duplicates
    return records, malformed, duplicates


def _safe_number(record: dict[str, Any], field: str) -> float | None:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_json_object(path: pathlib.Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} is not a JSON object"
    return value, None


def suite_metrics(
    repo_root: pathlib.Path,
    model: str,
    benchmark: str,
    suite: str,
    total: int,
    eta_window: int,
    *,
    output_root: pathlib.Path | None = None,
    run_name: str = "default",
    pivot_q_train_seed: int = 0,
    legacy_fallback: bool = True,
) -> dict[str, Any]:
    paths = model_episode_paths(
        repo_root,
        model,
        benchmark,
        suite,
        output_root=output_root,
        run_name=run_name,
        pivot_q_train_seed=pivot_q_train_seed,
        legacy_fallback=legacy_fallback,
    )
    records, malformed, duplicates = load_model_records(paths, benchmark)
    completed_records = [record for record in records if record.get("error") is None]
    error_records = [record for record in records if record.get("error") is not None]
    successes = sum(bool(record.get("success")) for record in completed_records)
    durations = [
        value
        for record in completed_records
        if (value := _safe_number(record, "duration_seconds")) is not None and value > 0
    ]
    steps = [
        value
        for record in completed_records
        if (value := _safe_number(record, "steps")) is not None and value >= 0
    ]
    recent = durations[-eta_window:] if eta_window > 0 else durations
    typical = statistics.median(recent) if recent else None
    attempted = len(records)
    evaluated = len(completed_records)
    low, high = wilson_interval(successes, evaluated)
    suite_dir = model_result_dir(
        repo_root,
        model,
        benchmark,
        suite,
        output_root=output_root,
        run_name=run_name,
        pivot_q_train_seed=pivot_q_train_seed,
        legacy_fallback=legacy_fallback,
    )
    summary_path = suite_dir / "metrics" / "summary.json"
    if not summary_path.exists():
        summary_path = suite_dir / "summary.json"
    summary, summary_error = _load_json_object(summary_path)
    summary_completed = None if summary is None else summary.get("completed_episodes")
    warnings: list[str] = []
    if malformed:
        warnings.append(f"{malformed} malformed episode record(s)")
    if duplicates:
        warnings.append(f"{duplicates} duplicate/resumed episode record(s)")
    if summary_error:
        warnings.append(summary_error)
    if summary_completed is not None:
        try:
            summary_completed = int(summary_completed)
        except (TypeError, ValueError):
            warnings.append("summary completed_episodes is not an integer")
            summary_completed = None
    if summary_completed is not None and summary_completed != attempted:
        warnings.append(
            f"episodes/summary mismatch: episodes={attempted}, summary={summary_completed}"
        )
    if attempted > total:
        warnings.append(f"episode count exceeds manifest total: {attempted}>{total}")
    if not paths:
        data_state = "missing"
    elif attempted == 0:
        data_state = "empty"
    elif attempted < total:
        data_state = "partial"
    elif attempted == total:
        data_state = "complete"
    else:
        data_state = "overfull"
    return {
        "model": model,
        "model_label": MODELS[model]["label"],
        "precision": MODELS[model]["precision"],
        "quantization_backend": MODELS[model]["quantization_backend"],
        "benchmark": benchmark,
        "benchmark_label": BENCHMARKS[benchmark]["label"],
        "suite": suite,
        "suite_label": DISPLAY_SUITE[suite],
        "completed": attempted,
        "evaluated": evaluated,
        "total": total,
        "completion_rate": attempted / total if total else 0.0,
        "successes": successes,
        "success_rate": successes / evaluated if evaluated else None,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": len(error_records),
        "malformed_records": malformed,
        "resume_duplicates": duplicates,
        "data_state": data_state,
        "runtime_state": "unknown",
        "warnings": warnings,
        "summary_path": str(summary_path),
        "summary_completed": summary_completed,
        "typical_seconds": typical,
        "eta_seconds": max(0, total - attempted) * typical if typical is not None else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
        "mean_steps": statistics.fmean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "records": records,
        "episodes_path": ",".join(str(path) for path in paths),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sum(row["completed"] for row in rows)
    evaluated = sum(row.get("evaluated", row["completed"] - row["errors"]) for row in rows)
    total = sum(row["total"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    low, high = wilson_interval(successes, evaluated)
    incomplete_rows = [row for row in rows if row["completed"] < row["total"]]
    if not incomplete_rows:
        aggregate_eta: float | None = 0.0
    elif any(row["eta_seconds"] is None for row in incomplete_rows):
        aggregate_eta = None
    else:
        aggregate_eta = max(float(row["eta_seconds"]) for row in incomplete_rows)
    return {
        "completed": completed,
        "evaluated": evaluated,
        "total": total,
        "completion_rate": completed / total if total else 0.0,
        "successes": successes,
        "success_rate": successes / evaluated if evaluated else None,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": sum(row["errors"] for row in rows),
        "malformed_records": sum(row.get("malformed_records", 0) for row in rows),
        "resume_duplicates": sum(row.get("resume_duplicates", 0) for row in rows),
        "eta_seconds": aggregate_eta,
        "warnings": [warning for row in rows for warning in row.get("warnings", [])],
    }


def _difficulty_label(value: Any) -> str:
    return "Unspecified" if value is None else str(value)


def _selected_plus_metadata(
    repo_root: pathlib.Path, manifest_path: pathlib.Path | None = None
) -> tuple[pathlib.Path, list[dict[str, Any]]]:
    plus_root = repo_root.parent / "LIBERO-plus"
    path = plus_root / "libero" / "libero" / "benchmark" / "task_classification.json"
    with path.open(encoding="utf-8") as stream:
        classification = json.load(stream)
    manifest = load_sample_manifest(repo_root, manifest_path)
    categories = [str(value) for value in manifest["categories"]]
    selected: list[dict[str, Any]] = []
    if manifest.get("selection") == "explicit_task_ids":
        metadata_by_suite = manifest.get("task_metadata_by_suite")
        if not isinstance(metadata_by_suite, dict):
            raise ValueError("explicit manifest is missing task_metadata_by_suite")
        for suite in SUITES:
            for item in metadata_by_suite[suite]:
                selected.append(
                    {
                        "suite": suite,
                        "category": str(item["category"]),
                        "difficulty": _difficulty_label(item.get("difficulty_level")),
                    }
                )
        return path, selected
    for suite in SUITES:
        rows = classification[suite]
        per_group = manifest_suite_quota(manifest, suite)
        for category in categories:
            candidates = [item for item in rows if str(item["category"]) == category]
            for item in candidates[:per_group]:
                selected.append(
                    {
                        "suite": suite,
                        "category": category,
                        "difficulty": _difficulty_label(item.get("difficulty_level")),
                    }
                )
    return path, selected


def _aggregate_group(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    completed_records = [record for record in records if record.get("error") is None]
    successes = sum(bool(record.get("success")) for record in completed_records)
    durations = [
        value
        for record in completed_records
        if (value := _safe_number(record, "duration_seconds")) is not None and value > 0
    ]
    steps = [
        value
        for record in completed_records
        if (value := _safe_number(record, "steps")) is not None and value >= 0
    ]
    attempted = len(records)
    evaluated = len(completed_records)
    low, high = wilson_interval(successes, evaluated)
    return {
        "completed": attempted,
        "evaluated": evaluated,
        "total": total,
        "completion_rate": attempted / total if total else 0.0,
        "successes": successes,
        "success_rate": successes / evaluated if evaluated else None,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": sum(record.get("error") is not None for record in records),
        "mean_steps": statistics.fmean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
    }


def plus_group_metrics(
    repo_root: pathlib.Path,
    suite_rows: list[dict[str, Any]],
    manifest_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    classification_path, metadata = _selected_plus_metadata(repo_root, manifest_path)
    records_by_suite = {row["suite"]: row["records"] for row in suite_rows}
    all_records = [record for row in suite_rows for record in row["records"]]
    categories = []
    suite_categories = []
    for category in CATEGORY_ORDER:
        expected = sum(item["category"] == category for item in metadata)
        records = [record for record in all_records if record.get("category") == category]
        categories.append(
            {
                "category": category,
                "category_label": CATEGORY_LABEL[category],
                **_aggregate_group(records, expected),
            }
        )
        for suite in SUITES:
            suite_expected = sum(
                item["suite"] == suite and item["category"] == category for item in metadata
            )
            suite_records = [
                record
                for record in records_by_suite[suite]
                if record.get("category") == category
            ]
            suite_categories.append(
                {
                    "suite": suite,
                    "category": category,
                    "category_label": CATEGORY_LABEL[category],
                    **_aggregate_group(suite_records, suite_expected),
                }
            )
    difficulty_order = ["1", "2", "3", "4", "5", "Unspecified"]
    difficulties = []
    suite_difficulties = []
    category_difficulties = []
    for difficulty in difficulty_order:
        expected = sum(item["difficulty"] == difficulty for item in metadata)
        if expected:
            records = [
                record
                for record in all_records
                if _difficulty_label(record.get("difficulty_level")) == difficulty
            ]
            difficulties.append({"difficulty": difficulty, **_aggregate_group(records, expected)})
        for suite in SUITES:
            suite_expected = sum(
                item["suite"] == suite and item["difficulty"] == difficulty for item in metadata
            )
            if suite_expected:
                records = [
                    record
                    for record in records_by_suite[suite]
                    if _difficulty_label(record.get("difficulty_level")) == difficulty
                ]
                suite_difficulties.append(
                    {"suite": suite, "difficulty": difficulty, **_aggregate_group(records, suite_expected)}
                )
        for category in CATEGORY_ORDER:
            category_expected = sum(
                item["category"] == category and item["difficulty"] == difficulty
                for item in metadata
            )
            if category_expected:
                records = [
                    record
                    for record in all_records
                    if record.get("category") == category
                    and _difficulty_label(record.get("difficulty_level")) == difficulty
                ]
                category_difficulties.append(
                    {
                        "category": category,
                        "category_label": CATEGORY_LABEL[category],
                        "difficulty": difficulty,
                        **_aggregate_group(records, category_expected),
                    }
                )
    return {
        "classification_path": str(classification_path),
        "categories": categories,
        "suite_categories": suite_categories,
        "difficulties": difficulties,
        "suite_difficulties": suite_difficulties,
        "category_difficulties": category_difficulties,
    }


def _argument(tokens: list[str], name: str) -> str | None:
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def _process_environment(pid: int) -> dict[str, str]:
    try:
        raw = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def _suite_from_model_path(value: str | None) -> str | None:
    if not value:
        return None
    if "libero-spatial" in value:
        return "libero_spatial"
    if "libero-goal" in value:
        return "libero_goal"
    if "libero-object" in value:
        return "libero_object"
    if "libero-long" in value or "libero-10" in value:
        return "libero_10"
    return None


def _discover_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for proc_dir in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
            tokens = [
                item.decode(errors="replace")
                for item in (proc_dir / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (OSError, ValueError):
            continue
        if not tokens:
            continue
        is_plus_eval = any(token.endswith("run_libero_plus_eval.py") for token in tokens)
        is_standard_eval = any(token.endswith("run_libero_eval.py") for token in tokens)
        is_eval = is_plus_eval or is_standard_eval
        is_server = any(token.endswith("scripts/inference_service.py") for token in tokens) and "--server" in tokens
        if not is_eval and not is_server:
            continue
        environment = _process_environment(pid)
        if is_eval:
            kind = "eval"
            model = _argument(tokens, "--model-variant") or environment.get("EVAL_MODEL_VARIANT")
            suite = _argument(tokens, "--task-suite-name")
        else:
            kind = "server"
            adapter = _argument(tokens, "--adapter-path")
            model = environment.get("GR00T_MODEL_VARIANT")
            if adapter and "pivot-q" in adapter:
                model = "groot-pivot-q-w4a8"
            suite = _suite_from_model_path(_argument(tokens, "--model_path"))
        try:
            port = int(_argument(tokens, "--port") or environment.get("GR00T_PORT", "0"))
        except ValueError:
            port = 0
        gpu_text = environment.get("CUDA_VISIBLE_DEVICES")
        processes.append(
            {
                "kind": kind,
                "model": model or "unknown",
                "benchmark": (
                    "libero-plus" if is_plus_eval else "libero" if is_standard_eval else "unknown"
                ),
                "suite": suite or "unknown",
                "pid": pid,
                "gpu": gpu_text or "unknown",
                "port": port,
                "alive": True,
                "command": " ".join(shlex.quote(token) for token in tokens),
            }
        )
    return processes


def runtime_status(repo_root: pathlib.Path) -> dict[str, Any]:
    manifest = repo_root / "runtime_logs" / "eval" / "processes.tsv"
    processes = _discover_processes()
    server_ports = sorted(
        {item["port"] for item in processes if item["kind"] == "server" and item["port"] > 0}
    )
    ports: dict[str, bool] = {}
    for port in server_ports:
        connection = socket.socket()
        connection.settimeout(0.15)
        ports[str(port)] = connection.connect_ex(("127.0.0.1", port)) == 0
        connection.close()
    return {
        "manifest": str(manifest),
        "processes": processes,
        "servers_alive": sum(item["alive"] for item in processes if item["kind"] == "server"),
        "servers_total": sum(item["kind"] == "server" for item in processes),
        "evaluators_alive": sum(item["alive"] for item in processes if item["kind"] == "eval"),
        "evaluators_total": sum(item["kind"] == "eval" for item in processes),
        "ports_alive": sum(ports.values()),
        "ports_total": len(ports),
        "ports": ports,
    }


def _decorate_runtime_states(models: dict[str, Any], runtime: dict[str, Any]) -> None:
    for model, model_data in models.items():
        for benchmark, rows in model_data["benchmarks"].items():
            for row in rows:
                eval_alive = any(
                    process["kind"] == "eval"
                    and process["model"] == model
                    and process["benchmark"] == benchmark
                    and process["suite"] == row["suite"]
                    for process in runtime["processes"]
                )
                server_alive = any(
                    process["kind"] == "server"
                    and process["model"] == model
                    and process["suite"] == row["suite"]
                    for process in runtime["processes"]
                )
                if row["data_state"] == "complete":
                    state = "done"
                elif eval_alive:
                    state = "running"
                elif row["data_state"] in {"partial", "overfull"}:
                    state = "stopped"
                elif row["data_state"] == "empty" and (server_alive or eval_alive):
                    state = "starting"
                elif row["data_state"] == "empty":
                    state = "empty"
                else:
                    state = "not-started"
                row["runtime_state"] = state
                row["server_alive"] = server_alive
                row["evaluator_alive"] = eval_alive


def build_snapshot(
    repo_root: pathlib.Path,
    eta_window: int = 10,
    manifest_path: pathlib.Path | None = None,
    selected_models: tuple[str, ...] | list[str] | None = None,
    selected_benchmarks: tuple[str, ...] | list[str] | None = None,
    output_root: pathlib.Path | None = None,
    run_name: str = "default",
    pivot_q_train_seed: int = 0,
    legacy_fallback: bool = True,
) -> dict[str, Any]:
    selected_models = tuple(selected_models or PRIMARY_MODELS)
    selected_benchmarks = tuple(selected_benchmarks or ("libero-plus",))
    unknown_models = sorted(set(selected_models) - set(MODELS))
    unknown_benchmarks = sorted(set(selected_benchmarks) - set(BENCHMARKS))
    if unknown_models:
        raise ValueError(f"unknown model(s): {', '.join(unknown_models)}")
    if unknown_benchmarks:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown_benchmarks)}")
    totals_config = benchmark_totals(repo_root, manifest_path)
    models: dict[str, Any] = OrderedDict()
    for model in selected_models:
        benchmark_rows: dict[str, list[dict[str, Any]]] = OrderedDict()
        totals: dict[str, dict[str, Any]] = OrderedDict()
        for benchmark in selected_benchmarks:
            rows = [
                suite_metrics(
                    repo_root,
                    model,
                    benchmark,
                    suite,
                    totals_config[benchmark][suite],
                    eta_window,
                    output_root=output_root,
                    run_name=run_name,
                    pivot_q_train_seed=pivot_q_train_seed,
                    legacy_fallback=legacy_fallback,
                )
                for suite in SUITES
            ]
            benchmark_rows[benchmark] = rows
            totals[benchmark] = aggregate_rows(rows)
        plus_rows = benchmark_rows.get("libero-plus")
        models[model] = {
            "config": MODELS[model],
            "benchmarks": benchmark_rows,
            "totals": totals,
            "libero_plus_groups": (
                plus_group_metrics(repo_root, plus_rows, manifest_path) if plus_rows is not None else None
            ),
        }
    runtime = runtime_status(repo_root)
    _decorate_runtime_states(models, runtime)
    warnings = []
    for model, model_data in models.items():
        for rows in model_data["benchmarks"].values():
            for row in rows:
                warnings.extend(
                    f'{model}/{row["benchmark"]}/{row["suite"]}: {warning}'
                    for warning in row.get("warnings", [])
                )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "output_root": str((output_root or (repo_root / "output")).resolve()),
        "run_name": run_name,
        "pivot_q_train_seed": pivot_q_train_seed,
        "legacy_fallback": legacy_fallback,
        "eta_window": eta_window,
        "sample_manifest": load_sample_manifest(repo_root, manifest_path),
        "selected_models": list(selected_models),
        "selected_benchmarks": list(selected_benchmarks),
        "models": models,
        "runtime": runtime,
        "warnings": warnings,
    }
