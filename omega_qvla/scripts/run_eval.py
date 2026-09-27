#!/usr/bin/env python3
"""Start an Ω-QVLA server, evaluate one suite, then stop that server."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker, stop

import yaml


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: "pivot_q" if value in ("pivot-q", "pivot_q") else value, choices=("fp16", "omega_qvla", "pivot_q"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero-plus")
    parser.add_argument("--suite", choices=("libero_spatial", "libero_goal", "libero_object", "libero_10", "all"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=Path, default=root / "config" / "omega_qvla.yaml")
    parser.add_argument(
        "--adapter-path", type=Path,
        help="trained PIVOT_Q adapter directory (required only for --method pivot_q)",
    )
    parser.add_argument("--startup-timeout", type=float, default=None)
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    if args.suite == "all":
        raise SystemExit(
            "Ω-QVLA uses suite-specific checkpoints and packs; start one runner per suite"
        )
    if args.method == "pivot_q":
        if args.adapter_path is None:
            raise SystemExit("--adapter-path is required for --method pivot-q")
        args.adapter_path = args.adapter_path.expanduser().resolve()
        if not args.adapter_path.is_dir():
            raise SystemExit(f"adapter directory does not exist: {args.adapter_path}")
    elif args.adapter_path is not None:
        raise SystemExit("--adapter-path is only valid with --method pivot_q")
    scripts = Path(__file__).resolve().parent
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    groot_python = sys.executable
    libero_python = str(conda_python("libero_test"))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    server_command = [
        groot_python, str(scripts / "inference_server.py"), "--method", args.method,
        "--suite", "libero_spatial" if args.suite == "all" else args.suite,
        "--port", str(args.port), "--config", str(args.config),
    ]
    if args.adapter_path is not None:
        server_command += ["--adapter-path", str(args.adapter_path)]
    print("server:", " ".join(server_command), flush=True)
    with managed_processes() as processes:
        server = subprocess.Popen(server_command, env=environment, start_new_session=True)
        processes.append(server)
        started = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"inference server exited with code {server.returncode}")
            probe = [libero_python, str(scripts / "eval_client.py"), "--method", args.method,
                     "--server-port", str(args.port), "--config", str(args.config), "--probe", "--suite", "libero_spatial"]
            probe_process = subprocess.Popen(probe, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            processes.append(probe_process)
            try:
                ready = probe_process.wait(timeout=10) == 0
            except subprocess.TimeoutExpired:
                ready = False
            finally:
                stop(probe_process)
                processes.remove(probe_process)
            if ready:
                break
            if args.startup_timeout is not None and time.monotonic() - started >= args.startup_timeout:
                raise TimeoutError(f"inference server did not become ready within {args.startup_timeout:.0f}s")
            time.sleep(2)
        evaluation = [
            libero_python, str(scripts / "eval_client.py"), "--method", args.method,
            "--benchmark", args.benchmark, "--suite", args.suite, "--server-port", str(args.port),
            "--seed", str(args.seed), "--config", str(args.config), *extra,
        ]
        run_worker(evaluation, env=environment, processes=processes)


if __name__ == "__main__":
    main()
