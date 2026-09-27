"""Trusted-localhost environment service for standard clean LIBERO.

This process runs in ``libero_test`` while PIVOT-Q training runs in
``groot_test``.  Keeping the environments separate avoids mixing the target
LIBERO-Plus package into IID training and avoids dependency conflicts.
"""

from __future__ import annotations

import pickle
import traceback
from dataclasses import dataclass

import tyro
import zmq
from libero.libero import benchmark

from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 5590


class CleanLiberoService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{config.host}:{config.port}")
        self.env = None
        self.suites = {}
        self.running = True

    def _suite(self, name: str):
        if name not in self.suites:
            benchmark_dict = benchmark.get_benchmark_dict()
            if name not in benchmark_dict:
                raise ValueError(f"unknown clean LIBERO suite: {name}")
            self.suites[name] = benchmark_dict[name]()
        return self.suites[name]

    def reset(
        self,
        *,
        suite: str,
        task_id: int,
        initial_state_id: int,
        num_steps_wait: int,
        horizon: int,
    ) -> dict:
        if self.env is not None:
            self.env.close()
            self.env = None
        task_suite = self._suite(suite)
        if not 0 <= task_id < task_suite.n_tasks:
            raise ValueError(f"task ID {task_id} is outside suite size {task_suite.n_tasks}")
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if not 0 <= initial_state_id < len(initial_states):
            raise ValueError(
                f"initial-state ID {initial_state_id} is outside count {len(initial_states)}"
            )
        self.env, language = get_libero_env(task, resolution=256, horizon=horizon)
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
            f"Clean LIBERO environment service listening on "
            f"tcp://{self.config.host}:{self.config.port}",
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
    CleanLiberoService(tyro.cli(ServiceConfig)).run()
