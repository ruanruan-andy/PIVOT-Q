#!/usr/bin/env python3
"""Start a local π0.5 server, run one LIBERO evaluation, then clean up."""

from __future__ import annotations

import argparse
import os
import sys
import shlex
import subprocess
import time
from multiprocessing.connection import Client
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_PYTHON = Path(sys.executable)
DEFAULT_EVAL_PYTHON = None


def wait_for_server(
    process: subprocess.Popen[object], host: str, port: int, authkey: str, timeout: float | None
) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"inference server exited with code {process.returncode}")
        try:
            connection = Client((host, port), authkey=authkey.encode())
            connection.close()
            return
        except (EOFError, OSError):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"inference server did not open {host}:{port} within {timeout:.0f}s")
            time.sleep(1)


def stop_server(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="CUDA device visible to the inference server")
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: "pivot_q" if value in ("pivot-q", "pivot_q") else value, choices=("fp16", "quantvla", "pivot_q"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero-plus")
    parser.add_argument("--suite", choices=("libero_spatial", "libero_goal", "libero_object", "libero_10", "all"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--port", type=int, default=5700)
    parser.add_argument("--config", type=Path, default=ROOT / "pi05_quantvla/config/quantvla.yaml")
    parser.add_argument("--server-python", type=Path, default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--eval-python", type=Path, default=DEFAULT_EVAL_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--authkey", default="pi05")
    parser.add_argument("--startup-timeout", type=float, help="optional maximum server startup time in seconds")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        help="exact suite output directory; with --suite all, one suite subdirectory is added",
    )
    parser.add_argument(
        "--adapter-path", type=Path,
        help="trained PIVOT_Q adapter directory (required only for --method pivot_q)",
    )
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    args.eval_python = args.eval_python or conda_python("libero_test")

    if args.method == "pivot_q":
        if args.adapter_path is None:
            parser.error("--adapter-path is required for --method pivot-q")
        args.adapter_path = args.adapter_path.expanduser().resolve()
        if not args.adapter_path.is_dir():
            parser.error(f"adapter directory does not exist: {args.adapter_path}")
    elif args.adapter_path is not None:
        parser.error("--adapter-path is only valid with --method pivot_q")
    if args.output_root is not None and args.output_dir is not None:
        parser.error("--output-root and --output-dir are mutually exclusive")

    server_command = [
        str(args.server_python),
        "pi05_quantvla/scripts/inference_server.py",
        "--method", args.method,
        "--config", str(args.config),
        "--host", args.host,
        "--port", str(args.port),
        "--authkey", args.authkey,
    ]
    if args.adapter_path is not None:
        server_command += ["--adapter-path", str(args.adapter_path)]
    eval_command = [
        str(args.eval_python),
        "pi05_quantvla/scripts/eval_client.py",
        "--method", args.method,
        "--benchmark", args.benchmark,
        "--suite", args.suite,
        "--seed", str(args.seed),
        "--server-host", args.host,
        "--server-port", str(args.port),
        "--authkey", args.authkey,
        "--trials-per-task", str(args.trials_per_task),
    ]
    if args.manifest is not None:
        eval_command += ["--manifest", str(args.manifest)]
    if args.output_root is not None:
        eval_command += ["--output-root", str(args.output_root)]
    if args.output_dir is not None:
        eval_command += ["--output-dir", str(args.output_dir)]
    if args.task_ids is not None:
        eval_command += ["--task-ids", *(str(task_id) for task_id in args.task_ids)]
    if args.resume:
        eval_command.append("--resume")
    if args.save_video:
        eval_command.append("--save-video")

    server_environment = os.environ.copy()
    server_environment.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONPATH": f"{ROOT}:{server_environment.get('PYTHONPATH', '')}",
        "HF_HOME": server_environment.get("HF_HOME", str(ROOT / ".cache/huggingface")),
        "HF_HUB_OFFLINE": server_environment.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": server_environment.get("TRANSFORMERS_OFFLINE", "1"),
    })
    eval_environment = os.environ.copy()
    eval_prefix = args.eval_python.expanduser().resolve().parent.parent
    eval_pythonpath = [str(ROOT / "third_party" / "QuantVLA"), str(ROOT / "third_party" / "LIBERO-plus")]
    if eval_environment.get("PYTHONPATH"):
        eval_pythonpath.append(eval_environment["PYTHONPATH"])
    eval_environment.update({
        "PYTHONPATH": ":".join(eval_pythonpath),
        "MUJOCO_GL": eval_environment.get("MUJOCO_GL", "egl"),
        "LD_LIBRARY_PATH": (
            f"{eval_prefix / 'lib'}:{eval_environment.get('LD_LIBRARY_PATH', '')}"
        ),
        "MAGICK_HOME": str(eval_prefix),
    })
    if args.benchmark == "libero-plus":
        eval_environment["LIBERO_CONFIG_PATH"] = str(
            ROOT / "third_party" / "QuantVLA/configs/libero_plus"
        )
    print("server:", shlex.join(server_command), flush=True)
    with managed_processes() as processes:
        process = subprocess.Popen(server_command, cwd=ROOT, env=server_environment, start_new_session=True)
        processes.append(process)
        wait_for_server(process, args.host, args.port, args.authkey, args.startup_timeout)
        print("eval:", shlex.join(eval_command), flush=True)
        run_worker(eval_command, processes=processes, cwd=ROOT, env=eval_environment)


if __name__ == "__main__":
    main()
