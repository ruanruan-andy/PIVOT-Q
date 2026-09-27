#!/usr/bin/env python3
"""Evaluate an already-running Ω-QVLA GR00T server on LIBERO or LIBERO-Plus."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python

from paths import load_config


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: "pivot_q" if value in ("pivot-q", "pivot_q") else value, choices=("fp16", "omega_qvla", "pivot_q"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero-plus")
    parser.add_argument("--suite", choices=(*SUITES, "all"), required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=root / "config" / "omega_qvla.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="exact suite output directory; with --suite all, one suite subdirectory is added",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def client_environment(config: dict) -> dict[str, str]:
    paths = config["paths"]
    environment = os.environ.copy()
    libero_prefix = conda_python("libero_test").resolve().parent.parent
    libero_lib = libero_prefix / "lib"
    pythonpath = [str(paths["quantvla_root"]), str(paths["libero_plus_root"])]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": ":".join(pythonpath),
            "LIBERO_CONFIG_PATH": str(paths["libero_plus_config_path"]),
            "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
            "LD_LIBRARY_PATH": f"{libero_lib}:{environment.get('LD_LIBRARY_PATH', '')}",
            "MAGICK_HOME": str(libero_prefix),
        }
    )
    return environment


def probe(port: int, environment: dict[str, str]) -> None:
    code = (
        "from gr00t.eval.service import ExternalRobotInferenceClient; "
        f"assert ExternalRobotInferenceClient(host='127.0.0.1', port={port}).ping()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def evaluate_suite(args: argparse.Namespace, config: dict, suite: str) -> None:
    paths = config["paths"]
    seed = int(config["evaluation"]["policy_seed"] if args.seed is None else args.seed)
    method_dir = args.method
    benchmark_dir = "libero_plus" if args.benchmark == "libero-plus" else "libero"
    if args.output_dir is not None:
        output_dir = args.output_dir / suite if args.suite == "all" else args.output_dir
    elif args.output_root is None:
        method = {'fp16': 'original', 'omega_qvla': 'quantized', 'pivot_q': 'pivot_q'}[args.method]
        output_dir = Path(__file__).resolve().parents[2] / 'outputs/holoq_vla/groot_n1_5' / method / 'eval' / f'seed-{seed:03d}'
        if args.benchmark != 'libero-plus':
            output_dir /= args.benchmark
        output_dir /= suite
    else:
        output_root = args.output_root or Path(paths["results_root"]) / "omega_qvla"
        output_dir = output_root / benchmark_dir / method_dir / f"seed-{seed:03d}" / suite
    environment = client_environment(config)
    environment.update(
        {
            "LIBERO_EVAL_LOG_DIR": str(output_dir),
            "LIBERO_EVAL_METRICS_DIR": str(output_dir / "metrics"),
            "LIBERO_EVAL_LOGS_DIR": str(output_dir / "logs"),
            "LIBERO_EVAL_VIDEO_DIR": str(output_dir / "videos"),
        }
    )
    quantvla_root = Path(paths["quantvla_root"])
    variant = {
        "fp16": "groot-fp16",
        "omega_qvla": "omega-qvla-w4a4",
        "pivot_q": "omega-pivot_q",
    }[args.method]
    if args.benchmark == "libero-plus":
        count = int(config["evaluation"]["libero_plus_num"])
        manifest = Path(paths["manifests_root"]) / f"libero_plus_first{count}.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"LIBERO-Plus manifest not found: {manifest}")
        command = [
            sys.executable,
            str(quantvla_root / "examples" / "LiberoPlus" / "eval" / "run_libero_plus_eval.py"),
            "--task-suite-name", suite, "--port", str(args.server_port), "--headless",
            "--model-variant", variant, "--sample-manifest", str(manifest), "--policy-seed", str(seed),
        ]
        if args.resume:
            command.append("--resume")
        if not args.save_video:
            command.append("--no-save-video")
    else:
        command = [
            sys.executable,
            str(quantvla_root / "examples" / "Libero" / "eval" / "run_libero_eval.py"),
            "--task-suite-name", suite, "--port", str(args.server_port), "--headless",
            "--model-variant", variant, "--policy-seed", str(seed),
        ]
        if args.resume:
            command.append("--resume")
        if not args.save_video:
            command.append("--no-save-video")
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=quantvla_root, env=environment)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    environment = client_environment(config)
    if args.probe:
        probe(args.server_port, environment)
        return
    suites = SUITES if args.suite == "all" else (args.suite,)
    for suite in suites:
        evaluate_suite(args, config, suite)


if __name__ == "__main__":
    main()
