"""Shared LIBERO-Plus task-manifest selection utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def _quota_for_suite(manifest: dict[str, Any], suite_name: str) -> int:
    quota = manifest["per_suite_per_category"]
    if isinstance(quota, dict):
        if suite_name not in quota:
            raise ValueError(f"manifest has no quota for suite {suite_name!r}")
        quota = quota[suite_name]
    quota = int(quota)
    if quota <= 0:
        raise ValueError("per-suite category quota must be positive")
    return quota


def select_manifest_task_ids(
    manifest: dict[str, Any],
    *,
    suite_name: str,
    task_names: list[str],
    metadata_by_name: dict[str, dict[str, Any]],
    candidate_task_ids: Iterable[int] | None = None,
) -> list[int]:
    """Resolve legacy quota and v2 explicit manifests to zero-based task IDs."""
    candidate_ids = (
        list(range(len(task_names)))
        if candidate_task_ids is None
        else list(dict.fromkeys(int(value) for value in candidate_task_ids))
    )
    invalid_candidates = [value for value in candidate_ids if not 0 <= value < len(task_names)]
    if invalid_candidates:
        raise ValueError(f"invalid candidate task IDs: {invalid_candidates[:5]}")
    candidate_set = set(candidate_ids)

    selection = manifest.get("selection")
    if selection == "first_by_task_index":
        categories = [str(value) for value in manifest["categories"]]
        quota = _quota_for_suite(manifest, suite_name)
        selected: list[int] = []
        for category in categories:
            matches = [
                task_id
                for task_id in candidate_ids
                if str(metadata_by_name[task_names[task_id]]["category"]) == category
            ]
            if len(matches) < quota:
                raise ValueError(
                    f"suite {suite_name} has {len(matches)} tasks in {category!r}; "
                    f"requested {quota}"
                )
            selected.extend(matches[:quota])
        return selected

    if selection == "explicit_task_ids":
        by_suite = manifest.get("task_ids_by_suite", {})
        if suite_name not in by_suite:
            raise ValueError(f"manifest has no explicit task IDs for suite {suite_name!r}")
        selected = [int(value) for value in by_suite[suite_name]]
        if len(selected) != len(set(selected)):
            raise ValueError(f"manifest contains duplicate task IDs for suite {suite_name}")
        invalid = [value for value in selected if value not in candidate_set]
        if invalid:
            raise ValueError(f"manifest task IDs are outside candidates: {invalid[:5]}")

        expected = int(manifest["per_suite_per_category"])
        categories = [str(value) for value in manifest["categories"]]
        counts = Counter(
            str(metadata_by_name[task_names[task_id]]["category"]) for task_id in selected
        )
        unexpected = {
            category: counts.get(category, 0)
            for category in categories
            if counts.get(category, 0) != expected
        }
        if unexpected:
            raise ValueError(
                f"suite {suite_name} violates per-category quota {expected}: {unexpected}"
            )
        if set(counts) != set(categories):
            raise ValueError(f"suite {suite_name} contains categories outside the manifest")
        return selected

    raise ValueError(f"unsupported manifest selection: {selection!r}")
