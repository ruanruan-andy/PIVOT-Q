"""Trusted localhost RPC compatible with the existing PIVOT_Q environment services."""
from __future__ import annotations

import json
import pickle

from integration.config import resolve


class EnvClient:
    def __init__(self, cfg, *, clean=False):
        import zmq
        e = cfg["environment"]
        self.cfg, self.clean = cfg, clean
        if e["host"] not in {"127.0.0.1", "localhost"}:
            raise ValueError("pickle RPC is restricted to localhost")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, e["timeout_ms"])
        self.socket.setsockopt(zmq.SNDTIMEO, e["timeout_ms"])
        self.socket.setsockopt(zmq.LINGER, 0)
        port = e["clean_port"] if clean else e["port"]
        self.socket.connect(f"tcp://{e['host']}:{port}")

    def call(self, endpoint, **data):
        self.socket.send(pickle.dumps({"endpoint": endpoint, "data": data}))
        reply = pickle.loads(self.socket.recv())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self):
        try:
            info = self.call("ping")
            if info.get("suite", self.cfg["suite"]) != self.cfg["suite"]:
                raise ValueError("environment service suite mismatch")
            if info.get("seed", self.cfg["environment"]["seed"]) != self.cfg["environment"]["seed"]:
                raise ValueError("environment service seed mismatch")
            if self.clean and info.get("domain", "clean") != "clean":
                raise ValueError("clean anchor/calibration port is connected to a Plus service")
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *_):
        self.close()


def episode_specs(cfg):
    path = resolve(cfg["paths"]["manifest"])
    manifest = json.loads(path.read_text())
    suite = cfg["suite"]
    ids = manifest["task_ids_by_suite"][suite]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("manifest task IDs must be non-empty and unique")
    metadata = {int(item["task_id"]): item for item in manifest.get("task_metadata_by_suite", {}).get(suite, [])}
    init_ids = cfg["evaluation"]["initial_state_ids"]
    if not init_ids or len(set(init_ids)) != len(init_ids) or min(init_ids) < 0:
        raise ValueError("initial state IDs must be unique nonnegative integers")
    if "initial_state_ids" in manifest and init_ids != manifest["initial_state_ids"]:
        raise ValueError("initial states differ from the shared manifest")
    return [dict(metadata.get(task_id, {}), task_id=task_id, initial_state_id=init_id)
            for task_id in ids for init_id in init_ids]


def validate_service(client, cfg):
    info = client.call("ping")
    if "suite" in info and info["suite"] != cfg["suite"]:
        raise ValueError("environment service suite differs from configuration")
    if "seed" in info and info["seed"] != cfg["environment"]["seed"]:
        raise ValueError("environment service seed differs from configuration")
    tasks = client.call("list_tasks")
    remote = {int(item["task_id"]): item for item in tasks}
    for spec in episode_specs(cfg):
        if spec["task_id"] not in remote:
            raise ValueError("environment service does not contain the configured manifest")
        for key in ("task_name", "category"):
            if key in spec and spec[key] != remote[spec["task_id"]].get(key):
                raise ValueError(f"environment/manifest mismatch: {spec['task_id']}/{key}")


def reset(client, cfg, spec, horizon):
    return client.call("reset", suite=cfg["suite"], task_id=spec["task_id"],
                       initial_state_id=spec["initial_state_id"],
                       num_steps_wait=cfg["environment"]["num_steps_wait"],
                       horizon=horizon + cfg["environment"]["num_steps_wait"] + 1)
