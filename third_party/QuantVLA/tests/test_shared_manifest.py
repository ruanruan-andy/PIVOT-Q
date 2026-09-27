from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "libero_plus" / "shared560-first20.json"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def test_shared_manifest_has_exact_first20_quotas() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "libero-plus-shared-first-n-v1"
    assert manifest["train_eval_relation"] == "same_task_ids"
    assert manifest["selection"] == "explicit_task_ids"
    assert manifest["selection_order"] == "first_by_task_index"
    assert manifest["per_suite_per_category"] == 20
    assert manifest["total_tasks"] == 560
    assert manifest["initial_state_ids"] == [0]
    assert tuple(manifest["suites"]) == SUITES
    assert len(manifest["categories"]) == 7

    total = 0
    for suite in SUITES:
        task_ids = manifest["task_ids_by_suite"][suite]
        metadata = manifest["task_metadata_by_suite"][suite]
        assert len(task_ids) == len(set(task_ids)) == 140
        assert [item["task_id"] for item in metadata] == task_ids
        by_category: dict[str, list[int]] = defaultdict(list)
        for item in metadata:
            by_category[item["category"]].append(int(item["task_id"]))
        assert set(by_category) == set(manifest["categories"])
        for category in manifest["categories"]:
            assert len(by_category[category]) == 20
            assert by_category[category] == sorted(by_category[category])
        total += len(task_ids)
    assert total == manifest["total_tasks"]
