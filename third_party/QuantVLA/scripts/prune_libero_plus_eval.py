"""Prune LIBERO-Plus evaluation outputs to a shared balanced prefix subset.

The canonical task IDs are taken from the existing FP16 evaluation, which was
generated with the original first-by-task-index manifest.  Dry-run is the
default; pass ``--apply`` to rewrite JSONL/summary files and remove videos that
belong only to removed records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


@dataclass
class Config:
    output_root: str = "output/libero-plus"
    per_suite_per_category: int = 6
    apply: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def _group_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(field)
        groups.setdefault("Unspecified" if value is None else str(value), []).append(record)
    result = {}
    for key, items in sorted(groups.items()):
        successes = sum(bool(item.get("success")) for item in items)
        result[key] = {
            "episodes": len(items),
            "successes": successes,
            "success_rate": successes / len(items),
            "errors": sum(item.get("error") is not None for item in items),
        }
    return result


def _canonical_allowed_ids(
    output_root: Path, quota: int
) -> dict[str, dict[str, set[int]]]:
    allowed: dict[str, dict[str, set[int]]] = {}
    expected_categories: set[str] | None = None
    for suite in SUITES:
        canonical_path = output_root / "groot-fp16" / suite / "episodes.jsonl"
        records = _read_jsonl(canonical_path)
        by_category: dict[str, set[int]] = {}
        for record in records:
            category = str(record["category"])
            task_id = int(record["task_index"])
            by_category.setdefault(category, set()).add(task_id)
        if expected_categories is None:
            expected_categories = set(by_category)
        elif set(by_category) != expected_categories:
            raise ValueError(f"category mismatch in canonical file {canonical_path}")
        too_short = {key: len(value) for key, value in by_category.items() if len(value) < quota}
        if too_short:
            raise ValueError(f"canonical file {canonical_path} lacks tasks: {too_short}")
        allowed[suite] = {
            category: set(sorted(task_ids)[:quota])
            for category, task_ids in by_category.items()
        }
    return allowed


def _update_summary(path: Path, records: list[dict[str, Any]]) -> None:
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    successes = sum(bool(record.get("success")) for record in records)
    summary.update(
        {
            "selected_tasks": len(records),
            "completed_episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records) if records else 0.0,
            "errors": sum(record.get("error") is not None for record in records),
            "by_category": _group_summary(records, "category") if records else {},
            "by_difficulty": _group_summary(records, "difficulty_level") if records else {},
        }
    )
    selection = summary.get("selection")
    if isinstance(selection, dict):
        selection["sample_manifest"] = "configs/libero_plus/first_24_per_category.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prune(config: Config) -> None:
    if config.per_suite_per_category <= 0:
        raise ValueError("per_suite_per_category must be positive")
    output_root = Path(config.output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    allowed = _canonical_allowed_ids(output_root, config.per_suite_per_category)

    changed_files = 0
    removed_records = 0
    video_paths: set[Path] = set()
    retained_video_paths: set[Path] = set()
    for episodes_path in sorted(output_root.rglob("episodes.jsonl")):
        records = _read_jsonl(episodes_path)
        if not records:
            continue
        suites = {str(record.get("suite")) for record in records}
        if len(suites) != 1 or next(iter(suites)) not in allowed:
            raise ValueError(f"unexpected suite values in {episodes_path}: {sorted(suites)}")
        suite = next(iter(suites))
        kept = []
        removed = []
        for record in records:
            category = str(record.get("category"))
            task_id = int(record.get("task_index", -1))
            if category in allowed[suite] and task_id in allowed[suite][category]:
                kept.append(record)
            else:
                removed.append(record)
        for record in kept:
            raw_path = record.get("video_path")
            if raw_path:
                retained_video_paths.add(Path(raw_path).expanduser().resolve())
        if not removed:
            continue
        changed_files += 1
        removed_records += len(removed)
        for record in removed:
            raw_path = record.get("video_path")
            if raw_path:
                video_path = Path(raw_path).expanduser().resolve()
                if not video_path.is_relative_to(output_root):
                    raise ValueError(f"refusing to remove video outside {output_root}: {video_path}")
                video_paths.add(video_path)
        print(f"{episodes_path}: keep={len(kept)} remove={len(removed)}")
        if config.apply:
            temporary = episodes_path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept),
                encoding="utf-8",
            )
            temporary.replace(episodes_path)
            summary_path = episodes_path.with_name("summary.json")
            if summary_path.exists():
                _update_summary(summary_path, kept)

    orphan_videos = {
        path.resolve()
        for path in output_root.rglob("*.mp4")
        if path.resolve() not in retained_video_paths
    }
    video_paths.update(orphan_videos)
    existing_videos = [path for path in video_paths if path.is_file()]
    bytes_to_remove = sum(path.stat().st_size for path in existing_videos)
    print(
        json.dumps(
            {
                "apply": config.apply,
                "changed_episode_files": changed_files,
                "removed_records": removed_records,
                "existing_videos_to_remove": len(existing_videos),
                "unreferenced_videos_to_remove": len(orphan_videos),
                "video_bytes_to_remove": bytes_to_remove,
            },
            indent=2,
        )
    )
    if config.apply:
        for path in existing_videos:
            path.unlink()


if __name__ == "__main__":
    prune(tyro.cli(Config))
