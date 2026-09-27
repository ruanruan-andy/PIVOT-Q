from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Iterable

from rich.console import Console

from eval_metrics import (
    BENCHMARKS,
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    METHOD_TO_MODEL,
    MODELS,
    PRIMARY_MODELS,
    build_snapshot,
    format_duration,
    wilson_interval,
)


def _atomic_text(path: pathlib.Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: pathlib.Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _number(record: dict[str, Any], field: str) -> float | None:
    try:
        return float(record[field])
    except (KeyError, TypeError, ValueError):
        return None


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("error") is None]
    successes = sum(bool(record.get("success")) for record in completed)
    durations = [value for record in completed if (value := _number(record, "duration_seconds"))]
    steps = [
        value
        for record in completed
        if (value := _number(record, "steps")) is not None and value >= 0
    ]
    low, high = wilson_interval(successes, len(completed))
    return {
        "completed": len(completed),
        "successes": successes,
        "success_rate": successes / len(completed) if completed else 0.0,
        "success_ci95_low": low,
        "success_ci95_high": high,
        "errors": sum(record.get("error") is not None for record in records),
        "mean_steps": statistics.fmean(steps) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
    }


def _suite_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_data in snapshot["models"].items():
        for benchmark, suites in model_data["benchmarks"].items():
            for row in suites:
                rows.append({key: value for key, value in row.items() if key != "records"})
    return rows


def _episode_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_data in snapshot["models"].items():
        config = MODELS[model]
        for benchmark, suites in model_data["benchmarks"].items():
            for suite_row in suites:
                for record in suite_row["records"]:
                    rows.append(
                        {
                            "model": model,
                            "precision": config["precision"],
                            "quantization_backend": config["quantization_backend"],
                            "benchmark": benchmark,
                            "suite": suite_row["suite"],
                            "task_index": record.get("task_index"),
                            "episode_index": record.get("episode_index"),
                            "classification_id": record.get("classification_id"),
                            "task_name": record.get("task_name"),
                            "task_description": record.get("task_description"),
                            "category": record.get("category"),
                            "difficulty_level": record.get("difficulty_level"),
                            "success": record.get("success"),
                            "steps": record.get("steps"),
                            "duration_seconds": record.get("duration_seconds"),
                            "error": record.get("error"),
                            "video_path": record.get("video_path"),
                        }
                    )
    return rows


def _plus_dimension_rows(snapshot: dict[str, Any], field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, model_data in snapshot["models"].items():
        groups = model_data["libero_plus_groups"]
        if groups is None:
            continue
        for row in groups[field]:
            rows.append(
                {
                    "model": model,
                    "precision": MODELS[model]["precision"],
                    "quantization_backend": MODELS[model]["quantization_backend"],
                    **row,
                }
            )
    return rows


def _standard_task_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for model, model_data in snapshot["models"].items():
        if "libero" not in model_data["benchmarks"]:
            continue
        for suite_row in model_data["benchmarks"]["libero"]:
            for record in suite_row["records"]:
                groups[(model, suite_row["suite"], int(record["task_index"]))].append(record)
    return [
        {
            "model": model,
            "precision": MODELS[model]["precision"],
            "suite": suite,
            "task_index": task_index,
            **_aggregate_records(records),
        }
        for (model, suite, task_index), records in sorted(groups.items())
    ]


def _comparison_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if not all(model in snapshot["models"] for model in PRIMARY_MODELS):
        return []
    fp16 = snapshot["models"][PRIMARY_MODELS[0]]
    quant = snapshot["models"][PRIMARY_MODELS[1]]
    pivot_q = snapshot["models"][PRIMARY_MODELS[2]]
    rows: list[dict[str, Any]] = []
    for benchmark in snapshot["selected_benchmarks"]:
        fp_rows = {row["suite"]: row for row in fp16["benchmarks"][benchmark]}
        quant_rows = {row["suite"]: row for row in quant["benchmarks"][benchmark]}
        pivot_q_rows = {row["suite"]: row for row in pivot_q["benchmarks"][benchmark]}
        for suite in fp_rows:
            base = fp_rows[suite]
            quant_row = quant_rows[suite]
            pivot_q_row = pivot_q_rows[suite]
            rows.append(
                {
                    "dimension": "suite",
                    "benchmark": benchmark,
                    "name": suite,
                    "fp16_completed": base["completed"],
                    "fp16_total": base["total"],
                    "fp16_success_rate": base["success_rate"],
                    "quantvla_completed": quant_row["completed"],
                    "quantvla_total": quant_row["total"],
                    "quantvla_success_rate": quant_row["success_rate"],
                    "pivot_q_completed": pivot_q_row["completed"],
                    "pivot_q_total": pivot_q_row["total"],
                    "pivot_q_success_rate": pivot_q_row["success_rate"],
                    "pivot_q_vs_fp16_delta_points": _rate_delta(pivot_q_row, base),
                    "pivot_q_vs_quantvla_delta_points": _rate_delta(pivot_q_row, quant_row),
                    "all_complete": all(
                        row["completed"] == row["total"] for row in (base, quant_row, pivot_q_row)
                    ),
                }
            )
    if "libero-plus" not in snapshot["selected_benchmarks"]:
        return rows
    fp_categories = {
        row["category_label"]: row for row in fp16["libero_plus_groups"]["categories"]
    }
    quant_categories = {
        row["category_label"]: row for row in quant["libero_plus_groups"]["categories"]
    }
    pivot_q_categories = {
        row["category_label"]: row for row in pivot_q["libero_plus_groups"]["categories"]
    }
    for name, base in fp_categories.items():
        quant_row = quant_categories[name]
        pivot_q_row = pivot_q_categories[name]
        rows.append(
            {
                "dimension": "category",
                "benchmark": "libero-plus",
                "name": name,
                "fp16_completed": base["completed"],
                "fp16_total": base["total"],
                "fp16_success_rate": base["success_rate"],
                "quantvla_completed": quant_row["completed"],
                "quantvla_total": quant_row["total"],
                "quantvla_success_rate": quant_row["success_rate"],
                "pivot_q_completed": pivot_q_row["completed"],
                "pivot_q_total": pivot_q_row["total"],
                "pivot_q_success_rate": pivot_q_row["success_rate"],
                "pivot_q_vs_fp16_delta_points": _rate_delta(pivot_q_row, base),
                "pivot_q_vs_quantvla_delta_points": _rate_delta(pivot_q_row, quant_row),
                "all_complete": all(
                    row["completed"] == row["total"] for row in (base, quant_row, pivot_q_row)
                ),
            }
        )
    return rows


def _matched_task_comparison_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if "libero-plus" not in snapshot["selected_benchmarks"]:
        return []
    if not all(model in snapshot["models"] for model in PRIMARY_MODELS):
        return []
    records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for model in PRIMARY_MODELS:
        model_records: dict[tuple[str, int], dict[str, Any]] = {}
        for suite_row in snapshot["models"][model]["benchmarks"]["libero-plus"]:
            for record in suite_row["records"]:
                if record.get("error") is not None:
                    continue
                try:
                    key = (suite_row["suite"], int(record["task_index"]))
                except (KeyError, TypeError, ValueError):
                    continue
                model_records[key] = record
        records[model] = model_records

    def build_row(dimension: str, name: str, keys: set[tuple[str, int]], expected: int) -> dict[str, Any]:
        shared = set.intersection(*(set(records[model]) & keys for model in PRIMARY_MODELS))
        rates: dict[str, float | None] = {}
        successes: dict[str, int] = {}
        for model in PRIMARY_MODELS:
            successes[model] = sum(bool(records[model][key].get("success")) for key in shared)
            rates[model] = successes[model] / len(shared) if shared else None
        return {
            "dimension": dimension,
            "name": name,
            "matched_tasks": len(shared),
            "expected_tasks": expected,
            "fp16_successes": successes[PRIMARY_MODELS[0]],
            "fp16_success_rate": rates[PRIMARY_MODELS[0]],
            "quantvla_successes": successes[PRIMARY_MODELS[1]],
            "quantvla_success_rate": rates[PRIMARY_MODELS[1]],
            "pivot_q_successes": successes[PRIMARY_MODELS[2]],
            "pivot_q_success_rate": rates[PRIMARY_MODELS[2]],
            "pivot_q_vs_fp16_delta_points": (
                None if rates[PRIMARY_MODELS[2]] is None or rates[PRIMARY_MODELS[0]] is None
                else (rates[PRIMARY_MODELS[2]] - rates[PRIMARY_MODELS[0]]) * 100
            ),
            "pivot_q_vs_quantvla_delta_points": (
                None if rates[PRIMARY_MODELS[2]] is None or rates[PRIMARY_MODELS[1]] is None
                else (rates[PRIMARY_MODELS[2]] - rates[PRIMARY_MODELS[1]]) * 100
            ),
        }

    universe = set().union(*(set(value) for value in records.values()))
    suite_totals = {
        row["suite"]: row["total"]
        for row in snapshot["models"][PRIMARY_MODELS[0]]["benchmarks"]["libero-plus"]
    }
    rows = [
        build_row(
            "suite",
            suite,
            {key for key in universe if key[0] == suite},
            suite_totals[suite],
        )
        for suite in suite_totals
    ]
    category_totals = {
        row["category"]: row["total"]
        for row in snapshot["models"][PRIMARY_MODELS[0]]["libero_plus_groups"]["categories"]
    }
    for category in CATEGORY_ORDER:
        keys = {
            key
            for key in universe
            if any(records[model].get(key, {}).get("category") == category for model in PRIMARY_MODELS)
        }
        rows.append(build_row("category", CATEGORY_LABEL[category], keys, category_totals[category]))
    return rows


def _rate_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> float | None:
    candidate_rate = candidate.get("success_rate")
    reference_rate = reference.get("success_rate")
    if candidate_rate is None or reference_rate is None:
        return None
    return (candidate_rate - reference_rate) * 100


def _rate_text(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def _markdown_model(snapshot: dict[str, Any], model: str) -> str:
    data = snapshot["models"][model]
    lines = [f'### {MODELS[model]["label"]}', ""]
    for benchmark in snapshot["selected_benchmarks"]:
        config = BENCHMARKS[benchmark]
        lines.extend(
            [
                f'#### {config["label"]}',
                "",
                "| Suite | Progress | Success | Success rate | Errors | Typical | ETA |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in data["benchmarks"][benchmark]:
            lines.append(
                f'| {row["suite_label"]} | {row["completed"]}/{row["total"]} | '
                f'{row["successes"]} | {_rate_text(row["success_rate"])} | '
                f'{row["errors"]} | {format_duration(row["typical_seconds"])} | '
                f'{format_duration(row["eta_seconds"])} |'
            )
        total = data["totals"][benchmark]
        lines.append(
            f'| **Total** | **{total["completed"]}/{total["total"]}** | '
            f'**{total["successes"]}** | **{_rate_text(total["success_rate"])}** | '
            f'**{total["errors"]}** | — | **{format_duration(total["eta_seconds"])}** |'
        )
        lines.append("")
    if data["libero_plus_groups"] is None:
        return "\n".join(lines)
    lines.extend(
        [
            "#### LIBERO-Plus categories",
            "",
            "| Category | Progress | Success | Success rate | Errors |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["libero_plus_groups"]["categories"]:
        lines.append(
            f'| {row["category_label"]} | {row["completed"]}/{row["total"]} | '
            f'{row["successes"]} | {_rate_text(row["success_rate"])} | {row["errors"]} |'
        )
    return "\n".join(lines)


def _markdown_comparison(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Cross-model comparison",
        "",
        "`*` means that model/group is incomplete; deltas involving partial groups are descriptive, not final.",
        "",
        "| Dimension | Name | FP16 | QuantVLA | QuantVLA-PIVOT-Q | PIVOT-Q-FP16 | PIVOT-Q-Quant | Complete |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        fp_marker = "" if row["fp16_completed"] == row["fp16_total"] else "*"
        quant_marker = "" if row["quantvla_completed"] == row["quantvla_total"] else "*"
        pivot_q_marker = "" if row["pivot_q_completed"] == row["pivot_q_total"] else "*"
        fp16 = f'{_rate_text(row["fp16_success_rate"])}{fp_marker} ({row["fp16_completed"]}/{row["fp16_total"]})'
        quant = f'{_rate_text(row["quantvla_success_rate"])}{quant_marker} ({row["quantvla_completed"]}/{row["quantvla_total"]})'
        pivot_q = f'{_rate_text(row["pivot_q_success_rate"])}{pivot_q_marker} ({row["pivot_q_completed"]}/{row["pivot_q_total"]})'
        delta_fp = "N/A" if row["pivot_q_vs_fp16_delta_points"] is None else f'{row["pivot_q_vs_fp16_delta_points"]:+.1f} pp'
        delta_quant = "N/A" if row["pivot_q_vs_quantvla_delta_points"] is None else f'{row["pivot_q_vs_quantvla_delta_points"]:+.1f} pp'
        lines.append(
            f'| {row["dimension"]} | {row["name"]} | {fp16} | {quant} | {pivot_q} | '
            f'{delta_fp} | {delta_quant} | {"yes" if row["all_complete"] else "no"} |'
        )
    return "\n".join(lines)


def _markdown_matched_comparison(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Matched-task robustness check",
        "",
        "Only task IDs with valid episode records in all three models are included.",
        "",
        "| Dimension | Name | Matched | FP16 | QuantVLA | QuantVLA-PIVOT-Q | PIVOT-Q-Quant |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["pivot_q_vs_quantvla_delta_points"]
        delta_text = "N/A" if delta is None else f"{delta:+.1f} pp"
        lines.append(
            f'| {row["dimension"]} | {row["name"]} | '
            f'{row["matched_tasks"]}/{row["expected_tasks"]} | '
            f'{_rate_text(row["fp16_success_rate"])} | '
            f'{_rate_text(row["quantvla_success_rate"])} | '
            f'{_rate_text(row["pivot_q_success_rate"])} | {delta_text} |'
        )
    return "\n".join(lines)


def _technical_summary(snapshot: dict[str, Any], partial: bool, inconsistent: bool) -> str:
    lines = ["## Technical summary", ""]
    benchmark = "libero-plus" if "libero-plus" in snapshot["selected_benchmarks"] else snapshot["selected_benchmarks"][0]
    for model in snapshot["selected_models"]:
        total = snapshot["models"][model]["totals"][benchmark]
        lines.append(
            f'- **{MODELS[model]["label"]}:** {total["completed"]}/{total["total"]} episodes, '
            f'{total["successes"]} successes, {_rate_text(total["success_rate"])} success rate, '
            f'{total["errors"]} errors.'
        )
    if partial:
        lines.append("- **Interpretation:** this is a partial snapshot; deltas involving incomplete groups are not final claims.")
    else:
        lines.append("- **Interpretation:** all selected model/benchmark groups satisfy the manifest episode totals.")
    if inconsistent:
        lines.append("- **Data quality:** consistency warnings are present; strict final collection is blocked.")
    else:
        lines.append("- **Data quality:** no malformed, duplicate, overfull, or episodes/summary mismatch warning was detected.")
    return "\n".join(lines)


def _scope_and_definitions(snapshot: dict[str, Any]) -> str:
    manifest = snapshot["sample_manifest"]
    categories = len(manifest.get("categories", []))
    total_per_category = manifest.get("total_per_category", "manifest-defined")
    return "\n".join(
        [
            "## Scope, source, and metric definitions",
            "",
            f'- **Snapshot timestamp:** `{snapshot["generated_at"]}`.',
            f'- **Source of truth:** per-suite `episodes.jsonl`; manifest `{manifest["path"]}`.',
            f'- **Evaluation grain:** one episode per selected task; {categories} categories and {total_per_category} tasks per category.',
            "- **Progress denominator:** manifest-selected episodes. Error episodes count toward progress and are reported separately.",
            "- **Success-rate denominator:** parsed episodes whose `error` field is null.",
            "- **Comparison unit:** percentage points (pp). A `*` marks an incomplete model/group.",
            "- **Visualization choice:** exact tables are used because the evidence consists of four suites and seven discrete categories, not a time series.",
        ]
    )


def _methodology_and_limits(snapshot: dict[str, Any], partial: bool, inconsistent: bool) -> str:
    lines = [
        "## Methodology, limitations, and next steps",
        "",
        "- Repeated task IDs from resume are deduplicated within each JSONL file by keeping the last record.",
        "- Success-rate uncertainty is exported as a 95% Wilson interval in CSV/JSON outputs.",
        "- Matched-task comparisons use only `(suite, task_index)` keys with valid records in all three primary models.",
        "- `summary.json` is a consistency check, not a fallback for missing or truncated episode records.",
    ]
    if partial:
        lines.append("- Complete the remaining episodes before using ordinary suite/category deltas as final evidence.")
    if inconsistent:
        lines.append("- Resolve every data warning, then rerun collection with `--require-complete`.")
    else:
        lines.append("- Archive this report together with its manifest and CSV outputs; rerun with `--require-complete` for final publication.")
    lines.extend(
        [
            "",
            "### Further questions",
            "",
            "- Do the final full-set deltas agree with the matched-task deltas?",
            "- Are any remaining gaps concentrated in a suite/category interaction or difficulty level?",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect QuantVLA evaluation metrics")
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--eta-window", type=int, default=10)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("configs/libero_plus/shared560-first20.json"),
    )
    parser.add_argument(
        "--methods",
        "--models",
        dest="methods",
        nargs="+",
        choices=tuple(METHOD_TO_MODEL) + tuple(MODELS),
        default=list(METHOD_TO_MODEL),
        help="Methods to collect; legacy groot-* model ids are also accepted",
    )
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(BENCHMARKS), default=["libero-plus"])
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
        help="Normalized output root (default: <repo>/output)",
    )
    parser.add_argument("--run-name", default="default", help="Run name below each suite")
    parser.add_argument("--pivot_q-train-seed", type=int, default=0)
    parser.add_argument(
        "--no-legacy-fallback",
        action="store_true",
        help="Read only normalized output/eval paths",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail before writing if any selected result is incomplete or inconsistent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.manifest is not None and not args.manifest.is_absolute():
        args.manifest = repo_root / args.manifest
    output_root = (args.output_root or (repo_root / "output")).resolve()
    default_report_dir = (
        output_root
        / "reports"
        / "libero-plus"
        / "shared560-first20"
        / f"pivot_q-seed-{args.pivot_q_train_seed:03d}"
    )
    if args.run_name not in ("", "default"):
        default_report_dir /= args.run_name
    output_dir = (args.output_dir or default_report_dir).resolve()
    selected_models = [METHOD_TO_MODEL.get(value, value) for value in args.methods]
    snapshot = build_snapshot(
        repo_root,
        args.eta_window,
        manifest_path=args.manifest,
        selected_models=selected_models,
        selected_benchmarks=args.benchmarks,
        output_root=output_root,
        run_name=args.run_name,
        pivot_q_train_seed=args.pivot_q_train_seed,
        legacy_fallback=not args.no_legacy_fallback,
    )
    partial = any(
        total["completed"] < total["total"]
        for model_data in snapshot["models"].values()
        for total in model_data["totals"].values()
    )
    inconsistent = bool(snapshot["warnings"])
    if args.require_complete and (partial or inconsistent):
        reasons = []
        if partial:
            reasons.append("one or more selected results are incomplete")
        if inconsistent:
            reasons.append("data consistency warnings are present")
        raise SystemExit("Refusing final collection: " + "; ".join(reasons))
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = _suite_rows(snapshot)
    episodes = _episode_rows(snapshot)
    categories = _plus_dimension_rows(snapshot, "categories")
    suite_categories = _plus_dimension_rows(snapshot, "suite_categories")
    difficulties = _plus_dimension_rows(snapshot, "difficulties")
    suite_difficulties = _plus_dimension_rows(snapshot, "suite_difficulties")
    category_difficulties = _plus_dimension_rows(snapshot, "category_difficulties")
    standard_tasks = _standard_task_rows(snapshot)
    comparisons = _comparison_rows(snapshot)
    matched_comparisons = _matched_task_comparison_rows(snapshot)

    clean_summary = {
        "generated_at": snapshot["generated_at"],
        "repo_root": snapshot["repo_root"],
        "output_root": snapshot["output_root"],
        "run_name": snapshot["run_name"],
        "sample_manifest": snapshot["sample_manifest"],
        "partial": partial,
        "inconsistent": inconsistent,
        "warnings": snapshot["warnings"],
        "models": snapshot["models"],
        "comparisons": comparisons,
        "matched_task_comparisons": matched_comparisons,
        "runtime": snapshot["runtime"],
    }
    for model_data in clean_summary["models"].values():
        for rows in model_data["benchmarks"].values():
            for row in rows:
                row.pop("records", None)
    _atomic_text(output_dir / "summary.json", json.dumps(clean_summary, indent=2, ensure_ascii=False) + "\n")

    suite_fields = [
        "model", "model_label", "precision", "quantization_backend", "benchmark", "suite",
        "completed", "total", "completion_rate", "successes", "success_rate",
        "evaluated", "data_state", "runtime_state", "server_alive", "evaluator_alive",
        "success_ci95_low", "success_ci95_high", "errors", "malformed_records",
        "resume_duplicates", "typical_seconds", "eta_seconds", "mean_steps", "median_steps",
        "mean_duration_seconds", "median_duration_seconds", "episodes_path", "summary_path",
        "summary_completed", "warnings",
    ]
    _write_csv(output_dir / "suites.csv", suite_fields, suites)
    _write_csv(
        output_dir / "episodes.csv",
        [
            "model", "precision", "quantization_backend", "benchmark", "suite", "task_index",
            "episode_index", "classification_id", "task_name", "task_description", "category",
            "difficulty_level", "success", "steps", "duration_seconds", "error", "video_path",
        ],
        episodes,
    )
    group_fields = [
        "model", "precision", "quantization_backend", "suite", "category", "category_label",
        "difficulty", "completed", "total", "completion_rate", "successes", "success_rate",
        "evaluated",
        "success_ci95_low", "success_ci95_high", "errors", "mean_steps", "median_steps",
        "mean_duration_seconds", "median_duration_seconds",
    ]
    _write_csv(output_dir / "libero_plus_by_category.csv", group_fields, categories)
    _write_csv(output_dir / "libero_plus_by_suite_category.csv", group_fields, suite_categories)
    _write_csv(output_dir / "libero_plus_by_difficulty.csv", group_fields, difficulties)
    _write_csv(output_dir / "libero_plus_by_suite_difficulty.csv", group_fields, suite_difficulties)
    _write_csv(output_dir / "libero_plus_by_category_difficulty.csv", group_fields, category_difficulties)
    _write_csv(
        output_dir / "standard_libero_by_task.csv",
        [
            "model", "precision", "suite", "task_index", "completed", "successes", "success_rate",
            "success_ci95_low", "success_ci95_high", "errors", "mean_steps", "median_steps",
            "mean_duration_seconds", "median_duration_seconds",
        ],
        standard_tasks,
    )
    comparison_fields = [
        "dimension", "benchmark", "name",
        "fp16_completed", "fp16_total", "fp16_success_rate",
        "quantvla_completed", "quantvla_total", "quantvla_success_rate",
        "pivot_q_completed", "pivot_q_total", "pivot_q_success_rate",
        "pivot_q_vs_fp16_delta_points", "pivot_q_vs_quantvla_delta_points", "all_complete",
    ]
    _write_csv(output_dir / "model_comparisons.csv", comparison_fields, comparisons)
    _write_csv(
        output_dir / "matched_task_comparisons.csv",
        [
            "dimension", "name", "matched_tasks", "expected_tasks",
            "fp16_successes", "fp16_success_rate",
            "quantvla_successes", "quantvla_success_rate",
            "pivot_q_successes", "pivot_q_success_rate",
            "pivot_q_vs_fp16_delta_points", "pivot_q_vs_quantvla_delta_points",
        ],
        matched_comparisons,
    )

    report = "\n\n".join(
        [
            "# GR00T FP16 vs QuantVLA vs QuantVLA-PIVOT-Q Evaluation Report",
            _technical_summary(snapshot, partial, inconsistent),
            _scope_and_definitions(snapshot),
            "## Results by model",
            *(_markdown_model(snapshot, model) for model in snapshot["selected_models"]),
            _markdown_comparison(comparisons),
            _markdown_matched_comparison(matched_comparisons),
            _methodology_and_limits(snapshot, partial, inconsistent),
        ]
    )
    _atomic_text(output_dir / "report.md", report + "\n")

    console = Console()
    console.print("[bold green]Evaluation metrics collected[/bold green]")
    for model, model_data in snapshot["models"].items():
        console.print(f'  [bold]{MODELS[model]["label"]}[/bold]')
        for benchmark, total in model_data["totals"].items():
            console.print(
                f'    {BENCHMARKS[benchmark]["label"]}: {total["completed"]}/{total["total"]}, '
                f'success {_rate_text(total["success_rate"])}, errors {total["errors"]}'
            )
    if snapshot["warnings"]:
        console.print("  [bold yellow]Data warnings:[/bold yellow]")
        for warning in snapshot["warnings"]:
            console.print(f"    - {warning}")
    console.print(f"  Report directory: [cyan]{output_dir}[/cyan]")


if __name__ == "__main__":
    main()
