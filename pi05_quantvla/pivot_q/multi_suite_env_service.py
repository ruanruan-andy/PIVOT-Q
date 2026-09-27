"""LIBERO-Plus service for the paired-eight-GPU full-distillation ablation.

Unlike the original service, one process may visit all four suites in order.
The simulator still owns at most one live rollout environment at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from libero.libero import benchmark, get_libero_path
from examples.LiberoPlus.task_manifest import select_manifest_task_ids
from scripts.libero_plus_env_service import LiberoPlusService, ServiceConfig, SUPPORTED_SUITES


@dataclass
class MultiSuiteConfig:
    sample_manifest: str
    host: str = "127.0.0.1"
    port: int = 6500


class MultiSuiteService(LiberoPlusService):
    def __init__(self, config: MultiSuiteConfig):
        super().__init__(ServiceConfig(
            task_suite_name="libero_10",
            sample_manifest=config.sample_manifest,
            host=config.host,
            port=config.port,
        ))
        self.manifest = json.loads(
            Path(config.sample_manifest).expanduser().resolve().read_text(encoding="utf-8")
        )
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.classifications = json.loads(
            (
                Path(get_libero_path("benchmark_root"))
                / "benchmark"
                / "task_classification.json"
            ).read_text(encoding="utf-8")
        )
        self._active_suite = "libero_10"

    def activate(self, suite: str) -> None:
        if suite not in SUPPORTED_SUITES:
            raise ValueError(f"unsupported LIBERO-Plus suite: {suite}")
        if suite == self._active_suite:
            return
        # The prior rollout is finished before the trainer requests a new suite.
        if self.env is not None:
            self.env.close()
            self.env = None
        task_suite = self.benchmark_dict[suite]()
        task_names = task_suite.get_task_names()
        metadata_by_name = {
            str(item["name"]): item for item in self.classifications[suite]
        }
        selected = select_manifest_task_ids(
            self.manifest,
            suite_name=suite,
            task_names=task_names,
            metadata_by_name=metadata_by_name,
        )
        self.task_suite = task_suite
        self.task_names = task_names
        self.selected_tasks = [
            {
                "task_id": task_id,
                "task_name": task_names[task_id],
                "category": str(metadata_by_name[task_names[task_id]]["category"]),
            }
            for task_id in selected
        ]
        self.selected_ids = {item["task_id"] for item in self.selected_tasks}
        self.config.task_suite_name = suite
        self._active_suite = suite

    def dispatch(self, request: dict):
        endpoint = request.get("endpoint")
        data = request.get("data", {})
        if endpoint in {"list_tasks", "reset"}:
            self.activate(str(data["suite"]))
            if endpoint == "list_tasks":
                return self.selected_tasks
        return super().dispatch(request)


if __name__ == "__main__":
    MultiSuiteService(tyro.cli(MultiSuiteConfig)).run()
