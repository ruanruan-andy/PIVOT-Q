"""Build the shared first-N LIBERO-Plus train/eval manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from libero.libero import benchmark, get_libero_path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


@dataclass
class ManifestConfig:
    output_path: str = "configs/libero_plus/shared560-first20.json"
    per_suite_category: int = 20
    overwrite: bool = False


def _serialize_task(
    task_id: int,
    *,
    task_names: list[str],
    metadata_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    metadata = metadata_by_id[task_id]
    return {
        "task_id": task_id,
        "task_name": task_names[task_id],
        "classification_id": int(metadata["id"]),
        "category": str(metadata["category"]),
        "difficulty_level": metadata.get("difficulty_level"),
    }


def build(config: ManifestConfig) -> None:
    if config.per_suite_category <= 0:
        raise ValueError("per_suite_category must be positive")
    output_path = Path(config.output_path).expanduser().resolve()
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"manifest exists: {output_path}; pass --overwrite to replace it")

    classification_path = (
        Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    benchmark_dict = benchmark.get_benchmark_dict()
    task_ids_by_suite: dict[str, list[int]] = {}
    task_metadata_by_suite: dict[str, list[dict[str, Any]]] = {}

    for suite in SUITES:
        task_names = benchmark_dict[suite]().get_task_names()
        metadata_by_name = {str(item["name"]): item for item in classification[suite]}
        missing = [name for name in task_names if name not in metadata_by_name]
        if missing:
            raise RuntimeError(f"classification missing {suite} task {missing[0]!r}")
        metadata_by_id = {
            task_id: metadata_by_name[name] for task_id, name in enumerate(task_names)
        }
        selected: list[int] = []
        for category in CATEGORIES:
            candidates = sorted(
                task_id
                for task_id, metadata in metadata_by_id.items()
                if str(metadata["category"]) == category
            )
            if len(candidates) < config.per_suite_category:
                raise ValueError(
                    f"{suite}/{category} has {len(candidates)} tasks; "
                    f"requires {config.per_suite_category}"
                )
            selected.extend(candidates[: config.per_suite_category])
        expected_suite_total = len(CATEGORIES) * config.per_suite_category
        if len(selected) != expected_suite_total or len(set(selected)) != expected_suite_total:
            raise AssertionError(f"invalid shared manifest selection for {suite}")
        task_ids_by_suite[suite] = selected
        task_metadata_by_suite[suite] = [
            _serialize_task(
                task_id,
                task_names=task_names,
                metadata_by_id=metadata_by_id,
            )
            for task_id in selected
        ]

    total_tasks = len(SUITES) * len(CATEGORIES) * config.per_suite_category
    manifest = {
        "version": 4,
        "protocol": "libero-plus-shared-first-n-v1",
        "train_eval_relation": "same_task_ids",
        "selection": "explicit_task_ids",
        "selection_order": "first_by_task_index",
        "per_suite_per_category": config.per_suite_category,
        "total_tasks": total_tasks,
        "initial_state_ids": [0],
        "categories": list(CATEGORIES),
        "suites": list(SUITES),
        "task_ids_by_suite": task_ids_by_suite,
        "task_metadata_by_suite": task_metadata_by_suite,
    }
    if sum(map(len, task_ids_by_suite.values())) != total_tasks:
        raise AssertionError("shared manifest total is inconsistent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "protocol": manifest["protocol"],
                "tasks": total_tasks,
                "per_suite": len(CATEGORIES) * config.per_suite_category,
                "per_suite_category": config.per_suite_category,
                "train_eval_relation": manifest["train_eval_relation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build(tyro.cli(ManifestConfig))
