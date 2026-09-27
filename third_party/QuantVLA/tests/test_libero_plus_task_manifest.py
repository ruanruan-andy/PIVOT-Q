from __future__ import annotations

import pytest

from examples.LiberoPlus.task_manifest import select_manifest_task_ids


CATEGORIES = ("camera", "language")
TASK_NAMES = ["camera-0", "language-0", "camera-1", "language-1"]
METADATA = {
    name: {"category": name.split("-", maxsplit=1)[0]} for name in TASK_NAMES
}


def test_explicit_manifest_preserves_order_and_quota() -> None:
    manifest = {
        "selection": "explicit_task_ids",
        "categories": list(CATEGORIES),
        "per_suite_per_category": 1,
        "task_ids_by_suite": {"libero_spatial": [3, 2]},
    }
    assert select_manifest_task_ids(
        manifest,
        suite_name="libero_spatial",
        task_names=TASK_NAMES,
        metadata_by_name=METADATA,
    ) == [3, 2]


def test_explicit_manifest_rejects_overlap_outside_candidates() -> None:
    manifest = {
        "selection": "explicit_task_ids",
        "categories": list(CATEGORIES),
        "per_suite_per_category": 1,
        "task_ids_by_suite": {"libero_spatial": [0, 1]},
    }
    with pytest.raises(ValueError, match="outside candidates"):
        select_manifest_task_ids(
            manifest,
            suite_name="libero_spatial",
            task_names=TASK_NAMES,
            metadata_by_name=METADATA,
            candidate_task_ids=[2, 3],
        )


def test_legacy_manifest_still_supported() -> None:
    manifest = {
        "selection": "first_by_task_index",
        "categories": list(CATEGORIES),
        "per_suite_per_category": 1,
    }
    assert select_manifest_task_ids(
        manifest,
        suite_name="libero_spatial",
        task_names=TASK_NAMES,
        metadata_by_name=METADATA,
    ) == [0, 1]
