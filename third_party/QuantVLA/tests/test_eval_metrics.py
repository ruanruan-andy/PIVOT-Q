from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_metrics import (  # noqa: E402
    aggregate_rows,
    benchmark_totals,
    load_effective_records,
    suite_metrics,
)


class EvalMetricsTest(unittest.TestCase):
    def test_aggregate_eta_is_unknown_if_an_incomplete_suite_has_no_samples(self) -> None:
        rows = [
            {
                "completed": 42,
                "evaluated": 42,
                "total": 42,
                "successes": 40,
                "errors": 0,
                "malformed_records": 0,
                "resume_duplicates": 0,
                "eta_seconds": 0.0,
                "warnings": [],
            },
            {
                "completed": 0,
                "evaluated": 0,
                "total": 42,
                "successes": 0,
                "errors": 0,
                "malformed_records": 0,
                "resume_duplicates": 0,
                "eta_seconds": None,
                "warnings": [],
            },
        ]
        self.assertIsNone(aggregate_rows(rows)["eta_seconds"])

    def test_first_24_manifest_dict_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = root / "configs" / "libero_plus" / "first_24_per_category.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "categories": [f"category-{index}" for index in range(7)],
                        "per_suite_per_category": {
                            "libero_spatial": 6,
                            "libero_goal": 6,
                            "libero_object": 6,
                            "libero_10": 6,
                        },
                    }
                ),
                encoding="utf-8",
            )
            totals = benchmark_totals(root, manifest)
            self.assertEqual(totals["libero-plus"], {
                "libero_spatial": 42,
                "libero_goal": 42,
                "libero_object": 42,
                "libero_10": 42,
            })

    def test_resume_duplicate_keeps_latest_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "episodes.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"task_index": 4, "success": False, "error": None}),
                        "not-json",
                        json.dumps({"task_index": 4, "success": True, "error": None}),
                        json.dumps({"task_index": 5, "success": False, "error": None}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            records, malformed, duplicates = load_effective_records(path, "libero-plus")
            self.assertEqual(malformed, 1)
            self.assertEqual(duplicates, 1)
            self.assertEqual([record["task_index"] for record in records], [4, 5])
            self.assertTrue(records[0]["success"])

    def test_summary_episode_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            suite_dir = root / "output" / "libero-plus" / "groot-fp16" / "libero_spatial"
            suite_dir.mkdir(parents=True)
            (suite_dir / "episodes.jsonl").write_text(
                json.dumps(
                    {
                        "task_index": 1,
                        "success": True,
                        "error": None,
                        "duration_seconds": 3.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (suite_dir / "summary.json").write_text(
                json.dumps({"completed_episodes": 42}), encoding="utf-8"
            )
            row = suite_metrics(
                root,
                "groot-fp16",
                "libero-plus",
                "libero_spatial",
                total=42,
                eta_window=10,
            )
            self.assertEqual(row["completed"], 1)
            self.assertEqual(row["data_state"], "partial")
            self.assertEqual(row["success_rate"], 1.0)
            self.assertTrue(any("episodes/summary mismatch" in value for value in row["warnings"]))

    def test_missing_data_uses_na_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = suite_metrics(
                pathlib.Path(directory),
                "groot-pivot_q-v2-w4a8",
                "libero-plus",
                "libero_goal",
                total=42,
                eta_window=10,
            )
            self.assertEqual(row["data_state"], "missing")
            self.assertIsNone(row["success_rate"])

    def test_normalized_output_path_and_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output_root = root / "custom-output"
            suite_dir = (
                output_root
                / "eval"
                / "libero-plus"
                / "shared560-first20"
                / "fp16"
                / "libero_spatial"
                / "paper-v1"
            )
            metrics_dir = suite_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "episodes.jsonl").write_text(
                json.dumps({"task_index": 1, "success": True, "error": None}) + "\n",
                encoding="utf-8",
            )
            row = suite_metrics(
                root,
                "groot-fp16",
                "libero-plus",
                "libero_spatial",
                total=42,
                eta_window=10,
                output_root=output_root,
                run_name="paper-v1",
                legacy_fallback=False,
            )
            self.assertEqual(row["completed"], 1)
            self.assertEqual(row["data_state"], "partial")
            self.assertEqual(row["episodes_path"], str(metrics_dir / "episodes.jsonl"))


if __name__ == "__main__":
    unittest.main()
