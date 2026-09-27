"""Trusted-localhost environment service for PIVOT-Q OOD calibration.

This process runs with the LIBERO-Plus package in ``libero_test``.  The model
trainer remains in ``groot_test`` and communicates through a localhost RPC, so
the two incompatible Python dependency sets never share one interpreter.
"""

from __future__ import annotations

import json
import pickle
import traceback
from dataclasses import dataclass
from pathlib import Path

import tyro
import zmq
from libero.libero import benchmark, get_libero_path

from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env
from examples.LiberoPlus.task_manifest import select_manifest_task_ids


SUPPORTED_SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


@dataclass
class ServiceConfig:
    task_suite_name: str = "libero_spatial"
    sample_manifest: str = "configs/libero_plus/shared560-first20.json"
    host: str = "127.0.0.1"
    port: int = 5590


class LiberoPlusService:
    def __init__(self, config: ServiceConfig):
        if config.task_suite_name not in SUPPORTED_SUITES:
            raise ValueError(f"unsupported LIBERO-Plus suite: {config.task_suite_name}")
        self.config = config
        manifest_path = Path(config.sample_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.categories = [str(value) for value in manifest["categories"]]

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[config.task_suite_name]()
        self.task_names = self.task_suite.get_task_names()
        classification_path = (
            Path(get_libero_path("benchmark_root"))
            / "benchmark"
            / "task_classification.json"
        )
        classifications = json.loads(classification_path.read_text(encoding="utf-8"))[
            config.task_suite_name
        ]
        metadata_by_name = {str(item["name"]): item for item in classifications}
        missing = [name for name in self.task_names if name not in metadata_by_name]
        if missing:
            raise RuntimeError(f"classification metadata missing for task {missing[0]!r}")

        selected = select_manifest_task_ids(
            manifest,
            suite_name=config.task_suite_name,
            task_names=self.task_names,
            metadata_by_name=metadata_by_name,
        )
        self.selected_tasks = [
            {
                "task_id": task_id,
                "task_name": self.task_names[task_id],
                "category": str(metadata_by_name[self.task_names[task_id]]["category"]),
            }
            for task_id in selected
        ]
        self.selected_ids = {item["task_id"] for item in self.selected_tasks}

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{config.host}:{config.port}")
        self.env = None
        self.running = True

    def reset(
        self,
        *,
        suite: str,
        task_id: int,
        initial_state_id: int,
        num_steps_wait: int,
        horizon: int,
    ) -> dict:
        if suite != self.config.task_suite_name:
            raise ValueError(f"service is configured for {self.config.task_suite_name}, got {suite}")
        if task_id not in self.selected_ids:
            raise ValueError(f"task ID {task_id} is outside the configured shared manifest")
        if self.env is not None:
            self.env.close()
            self.env = None

        task = self.task_suite.get_task(task_id)
        initial_states = self.task_suite.get_task_init_states(task_id)
        if not 0 <= initial_state_id < len(initial_states):
            raise ValueError(
                f"initial-state ID {initial_state_id} is outside count {len(initial_states)}"
            )
        self.env, language = get_libero_env(task, resolution=256, horizon=horizon)
        language = str(getattr(self.env, "language_instruction", language))
        observation = self.env.reset()
        observation = self.env.set_init_state(initial_states[initial_state_id])
        done = False
        for _ in range(num_steps_wait):
            observation, _, done, _ = self.env.step(get_libero_dummy_action())
            if done:
                break
        return {"observation": observation, "language": language, "done": bool(done)}

    def step(self, *, action: list[float]) -> dict:
        if self.env is None:
            raise RuntimeError("reset must be called before step")
        observation, reward, done, info = self.env.step(action)
        return {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }

    def dispatch(self, request: dict):
        endpoint = request.get("endpoint")
        data = request.get("data", {})
        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint == "list_tasks":
            return self.selected_tasks
        if endpoint == "reset":
            return self.reset(**data)
        if endpoint == "step":
            return self.step(**data)
        if endpoint == "close":
            if self.env is not None:
                self.env.close()
                self.env = None
            return {"status": "closed"}
        if endpoint == "kill":
            self.running = False
            return {"status": "stopping"}
        raise ValueError(f"unknown endpoint: {endpoint}")

    def run(self) -> None:
        print(
            f"LIBERO-Plus service listening on tcp://{self.config.host}:{self.config.port}; "
            f"suite={self.config.task_suite_name}, tasks={len(self.selected_tasks)}",
            flush=True,
        )
        try:
            while self.running:
                request = pickle.loads(self.socket.recv())
                try:
                    response = {"result": self.dispatch(request)}
                except Exception as error:
                    traceback.print_exc()
                    response = {"error": f"{type(error).__name__}: {error}"}
                self.socket.send(pickle.dumps(response))
        finally:
            if self.env is not None:
                self.env.close()
            self.socket.close(linger=0)
            self.context.term()


if __name__ == "__main__":
    LiberoPlusService(tyro.cli(ServiceConfig)).run()
