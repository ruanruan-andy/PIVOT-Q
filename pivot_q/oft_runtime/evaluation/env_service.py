"""Run separately in a clean LIBERO or LIBERO-Plus Python environment."""
from __future__ import annotations

import argparse
import os
import json
import pickle
from pathlib import Path
import traceback

os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(__file__).resolve().parents[1] / "configs/libero_plus"))


class Service:
    def __init__(self, args):
        from libero.libero import benchmark, get_libero_path
        self.args = args
        self.suite = benchmark.get_benchmark_dict()[args.suite]()
        self.env = None
        names = self.suite.get_task_names()
        self.tasks = [{"task_id": i, "task_name": name, "category": "clean"} for i, name in enumerate(names)]
        if args.manifest:
            manifest = json.loads(Path(args.manifest).read_text())
            ids = manifest["task_ids_by_suite"][args.suite]
            classification = Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"
            classes = {item["name"]: item for item in json.loads(classification.read_text())[args.suite]}
            self.tasks = [{"task_id": i, "task_name": names[i], "category": classes[names[i]]["category"]} for i in ids]
            expected = {item["task_id"]: item for item in manifest.get("task_metadata_by_suite", {}).get(args.suite, [])}
            for task in self.tasks:
                for key in ("task_name", "category"):
                    if task["task_id"] in expected and task[key] != expected[task["task_id"]][key]:
                        raise ValueError("manifest task identity does not match the installed benchmark")
        self.ids = {item["task_id"] for item in self.tasks}

    def reset(self, suite, task_id, initial_state_id, num_steps_wait, horizon):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        if suite != self.args.suite or task_id not in self.ids:
            raise ValueError("task outside configured suite/manifest")
        states = self.suite.get_task_init_states(task_id)
        if not 0 <= initial_state_id < len(states):
            raise ValueError("invalid initial state index")
        if self.env is not None:
            self.env.close()
        task = self.suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                      camera_widths=256, horizon=horizon)
        self.env.seed(self.args.seed)
        self.env.reset()
        obs = self.env.set_init_state(states[initial_state_id])
        done = False
        for _ in range(num_steps_wait):
            obs, _, done, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                break
        return {"observation": obs, "language": str(getattr(self.env, "language_instruction", task.language)), "done": bool(done)}

    def dispatch(self, endpoint, data):
        if endpoint == "ping":
            return {"status": "ok", "suite": self.args.suite, "seed": self.args.seed,
                    "domain": "plus" if self.args.manifest else "clean"}
        if endpoint == "list_tasks":
            return self.tasks
        if endpoint == "reset":
            return self.reset(**data)
        if endpoint == "step":
            if self.env is None:
                raise RuntimeError("reset before step")
            obs, reward, done, info = self.env.step(data["action"])
            return {"observation": obs, "reward": float(reward), "done": bool(done), "info": info}
        raise ValueError(f"unsupported endpoint: {endpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--manifest", help="omit for clean LIBERO service")
    parser.add_argument("--port", type=int, default=5590)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import zmq
    service = Service(args)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"Ready: {args.suite}, {len(service.tasks)} tasks, port {args.port}", flush=True)
    try:
        while True:
            request = pickle.loads(socket.recv())
            try:
                reply = {"result": service.dispatch(request["endpoint"], request.get("data", {}))}
            except Exception as error:
                traceback.print_exc()
                reply = {"error": f"{type(error).__name__}: {error}"}
            socket.send(pickle.dumps(reply))
    finally:
        if service.env is not None:
            service.env.close()
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
