#!/usr/bin/env python3
"""Run clean LIBERO or LIBERO-Plus rollouts against a π0.5 inference server."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from collections import deque
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROOT_PARENT = ROOT / "third_party"

SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


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


class RemotePI05Policy:
    def __init__(self, *, host: str, port: int, authkey: str) -> None:
        self.connection = Client((host, port), authkey=authkey.encode())
        self.queue: deque[np.ndarray] = deque()
        self.last_language: str | None = None

    def get_action(self, observation: dict, language: str, policy_seed: int | None = None) -> np.ndarray:
        if language != self.last_language:
            self.queue.clear()
            self.last_language = language
        if not self.queue:
            request = {
                "image": prepare_image(observation["agentview_image"]),
                "wrist_image": prepare_image(observation["robot0_eye_in_hand_image"]),
                "state": np.concatenate((
                    np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
                    axis_angle(observation["robot0_eef_quat"]),
                    np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
                )),
                "task": language,
                "seed": 0 if policy_seed is None else int(policy_seed),
            }
            self.connection.send(request)
            self.queue.extend(np.asarray(self.connection.recv()["actions"], dtype=np.float32))
        return self.queue.popleft()

    def __del__(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass


def run_suite(args: argparse.Namespace, suite: str) -> None:
    if args.output_dir is not None:
        output_dir = args.output_dir / suite if args.suite == "all" else args.output_dir
    elif args.output_root is None:
        method = {'fp16': 'original', 'quantvla': 'quantized', 'pivot_q': 'pivot_q', 'pivot_q': 'pivot_q'}.get(args.method, args.method)
        output_dir = ROOT / 'outputs/quantvla/pi05' / method / 'eval' / f'seed-{args.seed:03d}'
        if args.benchmark != 'libero-plus':
            output_dir /= args.benchmark
        output_dir /= suite
    else:
        output_dir = args.output_root / args.method / f"seed-{args.seed:03d}" / suite
    environment = {
        "LIBERO_EVAL_LOG_DIR": str(output_dir),
        "LIBERO_EVAL_METRICS_DIR": str(output_dir / "metrics"),
        "LIBERO_EVAL_LOGS_DIR": str(output_dir / "logs"),
        "LIBERO_EVAL_VIDEO_DIR": str(output_dir / "videos"),
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        sys.path.insert(0, str(ROOT_PARENT / ("LIBERO-plus" if args.benchmark == "libero-plus" else "LIBERO")))
        sys.path.insert(0, str(ROOT_PARENT / "QuantVLA"))
        if args.benchmark == "libero-plus":
            from examples.LiberoPlus.eval import run_libero_plus_eval as libero_eval

            libero_eval.GR00TPolicy = lambda **_: RemotePI05Policy(
                host=args.server_host, port=args.server_port, authkey=args.authkey
            )
            cfg = libero_eval.LiberoPlusEvalConfig(
                task_suite_name=suite,
                headless=True,
                sample_manifest=str(args.manifest),
                policy_seed=args.seed,
                save_video=args.save_video,
                resume=args.resume,
                model_variant=f"pi05-{args.method}",
            )
            libero_eval.evaluate(cfg)
        else:
            from examples.Libero.eval import run_libero_eval as libero_eval

            libero_eval.video_di = environment["LIBERO_EVAL_VIDEO_DIR"]
            libero_eval.GR00TPolicy = lambda **_: RemotePI05Policy(
                host=args.server_host, port=args.server_port, authkey=args.authkey
            )
            cfg = libero_eval.GenerateConfig(
                task_suite_name=suite,
                num_trials_per_task=args.trials_per_task,
                port=args.server_port,
                headless=True,
                task_ids=args.task_ids,
                save_video=args.save_video,
                resume=args.resume,
                model_variant=f"pi05-{args.method}",
                policy_seed=args.seed,
            )
            libero_eval.eval_libero(cfg)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: "pivot_q" if value in ("pivot-q", "pivot_q") else value, choices=("fp16", "quantvla", "pivot_q"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero-plus")
    parser.add_argument("--suite", choices=(*SUITES, "all"), required=True)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/libero_plus_first20.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        help="exact suite output directory; with --suite all, one suite subdirectory is added",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5700)
    parser.add_argument("--authkey", default="pi05")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "libero-plus" and not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    for suite in SUITES if args.suite == "all" else (args.suite,):
        run_suite(args, suite)


if __name__ == "__main__":
    main()
