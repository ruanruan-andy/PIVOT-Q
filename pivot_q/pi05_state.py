"""State and simulator interface shared by π0.5 training and evaluation."""
from __future__ import annotations
import pickle, time, math, json
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import torch
import numpy as np
import zmq
from .pi05_flow import sample, make_noise, FrozenPrefixCache
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if denominator < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * (2.0 * math.acos(float(quat[3])) / denominator)

def prepare_image(image: np.ndarray) -> np.ndarray:
    """Match LeRobot's LIBERO image path: flipped CHW float32 in [0, 1]."""
    return np.ascontiguousarray(image[::-1, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0

class EnvClient:
    def __init__(self, port: int, timeout_ms: int = 120_000) -> None:
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(self.endpoint)

    def call(self, endpoint: str, **data):
        self.socket.send(pickle.dumps({"endpoint": endpoint, "data": data}))
        response = pickle.loads(self.socket.recv())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def wait(self, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.call("ping")
                self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                return
            except (zmq.ZMQError, RuntimeError):
                if time.time() >= deadline:
                    raise TimeoutError(f"environment service unavailable at {self.endpoint}")
                self.socket.close(linger=0)
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, 1000)
                self.socket.setsockopt(zmq.SNDTIMEO, 1000)
                self.socket.connect(self.endpoint)
                time.sleep(0.5)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()

@dataclass
class Bundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    scales: AttentionStatistics | None = None

@dataclass
class State:
    raw_sample: dict[str, Any]
    noise: torch.Tensor
    teacher_action: torch.Tensor
    student_action: torch.Tensor

def raw_sample(observation: dict, language: str) -> dict[str, Any]:
    return {
        "observation.images.image": torch.from_numpy(prepare_image(observation["agentview_image"])),
        "observation.images.image2": torch.from_numpy(prepare_image(observation["robot0_eye_in_hand_image"])),
        "observation.state": torch.from_numpy(np.concatenate((
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ))),
        "task": language,
    }

def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
