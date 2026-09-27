#!/usr/bin/env python3
"""Start a local π0.5 server, run one LIBERO evaluation, then clean up."""

from __future__ import annotations

import argparse
import os
import sys
import json
import shlex
import subprocess
import time
from multiprocessing.connection import Client
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from pi05_omegavla.common import load_config, resolve_paths, inside, atomic_json, sha256, checkpoint_identity
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
    parser.add_argument("--method", choices=("fp_original", "omega_original", "full_distill", "pivot_q"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "libero-plus"), default="libero-plus")
    parser.add_argument("--suite", choices=("libero_spatial", "libero_goal", "libero_object", "libero_10", "all"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--port", type=int, default=7700)
    parser.add_argument("--config", type=Path, default=None)
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
        help="trained PIVOT_Q adapter directory (required for full_distill/pivot_q)",
    )
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.config = (args.config or ROOT / f"pi05_omegavla/config/{args.method}.yaml").resolve()
    raw = resolve_paths(load_config(args.config))
    if raw["mode"] != args.method:
        parser.error("method and config mode differ")
    if args.output_root is not None:
        parser.error("use --output-dir to specify an isolated run directory")
    method_dir = {'fp_original': 'original', 'omega_original': 'quantized'}.get(args.method, args.method)
    args.output_dir = inside(args.output_dir or ROOT / "outputs/holoq_vla/pi05" / method_dir / "eval" / f"seed-{args.seed:03d}" / args.benchmark)
    args.manifest = args.manifest or Path(raw["paths"]["manifest"])
    args.eval_python = args.eval_python or conda_python("libero_test")

    if args.method in {"full_distill", "pivot_q"}:
        if args.adapter_path is None:
            parser.error("--adapter-path is required for full_distill/pivot_q")
        args.adapter_path = args.adapter_path.expanduser().resolve()
        if not args.dry_run and not args.adapter_path.is_dir():
            parser.error(f"adapter directory does not exist: {args.adapter_path}")
    elif args.adapter_path is not None:
        parser.error("--adapter-path is only valid for full_distill/pivot_q")
    if args.output_root is not None and args.output_dir is not None:
        parser.error("--output-root and --output-dir are mutually exclusive")

    server_command = [
        str(args.server_python),
        "pi05_omegavla/evaluation/inference_server.py",
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
        "pi05_omegavla/evaluation/eval_client.py",
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
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{ROOT}:{server_environment.get('PYTHONPATH', '')}",
        "HF_HOME": server_environment.get("HF_HOME", str(ROOT / ".cache/huggingface")),
        "HF_HUB_OFFLINE": server_environment.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": server_environment.get("TRANSFORMERS_OFFLINE", "1"),
    })
    eval_environment = os.environ.copy()
    eval_environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
    if args.dry_run:
        print(json.dumps({"server": server_command, "eval": eval_command, "output": str(args.output_dir)}, indent=2))
        return
    metadata = {"mode": args.method, "config": raw, "seed": args.seed, "benchmark": args.benchmark,
                "adapter": str(args.adapter_path) if args.adapter_path else None,
                "inference": raw["inference"], "trials_per_task": args.trials_per_task,
                "task_ids": args.task_ids, "manifest": str(args.manifest)}
    metadata["discrepancy"] = {"enabled": False, "schema_version": 1,
        "sampling": "every_executed_state", "action_space": "normalized_first_action_7d",
        "shared_noise": True, "teacher": "fp_original", "seed_encoding": "task*1000000+episode*10000+t+seed"}
    metadata["checkpoint"] = checkpoint_identity(raw["paths"]["checkpoint"])
    if args.method != "fp_original":
        metadata["pack_sha256"] = sha256(Path(raw["paths"]["quant_pack_dir"]) / "manifest.json")
    if args.adapter_path:
        metadata["adapter_sha256"] = sha256(args.adapter_path / "adapter_model.pt")
    metadata_path = args.output_dir / "protocol.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("evaluation output already has a different protocol")
    if metadata_path.exists() and not args.resume:
        raise ValueError("evaluation already exists; use --resume or a new --output-dir")
    atomic_json(metadata_path, metadata)
    print("server:", shlex.join(server_command), flush=True)
    with managed_processes() as processes:
        process = subprocess.Popen(server_command, cwd=ROOT, env=server_environment, start_new_session=True)
        processes.append(process)
        wait_for_server(process, args.host, args.port, args.authkey, args.startup_timeout)
        print("eval:", shlex.join(eval_command), flush=True)
        run_worker(eval_command, processes=processes, cwd=ROOT, env=eval_environment)


if __name__ == "__main__":
    main()
