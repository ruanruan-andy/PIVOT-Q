"""Simulator RPC service using an unmodified LIBERO or LIBERO-Plus checkout."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import zmq


class Service:
    def __init__(self, manifest=None, seed=0, suite=None):
        from libero.libero import benchmark, get_libero_path
        self.benchmarks = benchmark.get_benchmark_dict()
        self.manifest = json.loads(Path(manifest).read_text()) if manifest else None
        classification = Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"
        self.classes = json.loads(classification.read_text()) if manifest else None
        self.seed = seed
        self.active = None
        self.env = None
        if suite:
            self.activate(suite)

    def activate(self, suite):
        if self.active == suite:
            return
        self.suite = self.benchmarks[suite]()
        names = self.suite.get_task_names()
        ids = self.manifest["task_ids_by_suite"][suite] if self.manifest else range(len(names))
        classes = {r["name"]: r["category"] for r in self.classes[suite]} if self.classes else {}
        self.tasks = [dict(task_id=int(i), task_name=names[i], category=classes.get(names[i], "clean"))
                      for i in ids]
        expected = self.manifest.get("task_metadata_by_suite", {}).get(suite, []) if self.manifest else []
        tasks = {r["task_id"]: r for r in self.tasks}
        for record in expected:
            for key in ("task_name", "category"):
                if key in record and record[key] != tasks[record["task_id"]][key]:
                    raise ValueError("Installed benchmark differs from the task manifest")
        if self.env is not None:
            self.env.close()
            self.env = None
        self.active = suite

    def dispatch(self, endpoint, data):
        if endpoint == "ping":
            return dict(status="ok", domain="plus" if self.manifest else "clean", seed=self.seed)
        if endpoint in ("list_tasks", "reset"):
            self.activate(data.get("suite") or self.active)
        if endpoint == "list_tasks":
            return self.tasks
        if endpoint == "reset":
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            task_id, init_id = data["task_id"], data["initial_state_id"]
            if task_id not in {t["task_id"] for t in self.tasks}:
                raise ValueError("Task is outside the selected benchmark")
            states = self.suite.get_task_init_states(task_id)
            task = self.suite.get_task(task_id)
            if self.env is not None:
                self.env.close()
            path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            self.env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256,
                                         camera_widths=256, horizon=data["horizon"])
            self.env.seed(data.get("seed", self.seed))
            self.env.reset()
            observation = self.env.set_init_state(states[init_id])
            done = False
            for _ in range(data["num_steps_wait"]):
                observation, _, done, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
                if done:
                    break
            return dict(observation=observation, language=str(getattr(self.env, "language_instruction", task.language)),
                        done=bool(done))
        if endpoint == "step":
            observation, reward, done, info = self.env.step(data["action"])
            return dict(observation=observation, reward=float(reward), done=bool(done), info=info)
        raise ValueError(f"Unknown RPC endpoint: {endpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    service = Service(args.manifest, args.seed, args.suite)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{args.port}")
    print("Simulator ready", flush=True)
    try:
        while True:
            request = pickle.loads(socket.recv())
            try:
                reply = {"result": service.dispatch(request["endpoint"], request.get("data", {}))}
            except Exception as error:
                reply = {"error": repr(error)}
            socket.send(pickle.dumps(reply))
    finally:
        if service.env is not None:
            service.env.close()
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
