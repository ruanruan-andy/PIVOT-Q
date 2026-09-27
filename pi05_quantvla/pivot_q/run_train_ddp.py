#!/usr/bin/env python3
"""Launch four-suite synchronous π0.5-PIVOT_Q training and simulator services."""

from __future__ import annotations

import argparse
import json
import os
import sys
import shlex
import signal
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import start, stop, managed_processes, run_worker

import yaml

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=4, default=["4", "5", "6", "7"])
    parser.add_argument("--ports", nargs=4, type=int)
    parser.add_argument("--clean-ports", nargs=4, type=int)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=root / "pi05_quantvla/pivot_q/config/pivot_q.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, training, experiment, selection = (
        config["paths"], config["training"], config["experiment"], config["selection"]
    )
    method_name = str(experiment["name"])
    selection_method = str(selection.get("method", "pivot_q"))
    if selection_method not in {"pivot_q", "full_distill"}:
        raise ValueError(f"unsupported selection method: {selection_method}")
    distributed = config["distributed"]
    if tuple(distributed["suites"]) != SUITES:
        raise ValueError(f"distributed.suites must use this order: {SUITES}")
    ports = args.ports or [int(port) for port in distributed["rollout_ports"]]
    clean_ports = args.clean_ports or [int(port) for port in distributed["clean_anchor_ports"]]
    master_port = int(args.master_port or distributed["master_port"])
    all_ports = [*ports, *clean_ports, master_port]
    if len(set(all_ports)) != len(all_ports):
        parser.error("rollout, anchor, and torchrun master ports must all be distinct")
    workspace = Path(paths["root_dir"]).expanduser().resolve()
    pivot_q_root = resolve(workspace, paths["pivot_q_dir"])
    quantvla_root = resolve(workspace, paths["quantvla_dir"])
    libero_root = resolve(workspace, paths["libero_dir"])
    libero_plus_root = resolve(workspace, paths["libero_plus_dir"])
    manifest = resolve(workspace, paths["manifest"])
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))["task_ids_by_suite"]
    episode_counts = {suite: len(manifest_data[suite]) for suite in SUITES}
    if len(set(episode_counts.values())) != 1:
        raise RuntimeError(f"balanced DDP requires equal suite sizes: {episode_counts}")
    episodes = next(iter(episode_counts.values()))
    seed = int(training["seed"] if args.seed is None else args.seed)
    output = resolve(workspace, paths["results_dir"]) / f"seed-{seed:03d}" / "shared"

    libero_python = str(conda_python("libero_test"))
    plus_env = os.environ.copy()
    plus_env.update({
        "PYTHONPATH": f"{quantvla_root}:{libero_plus_root}",
        "LIBERO_CONFIG_PATH": str(quantvla_root / "configs/libero_plus"),
        "MUJOCO_GL": plus_env.get("MUJOCO_GL", "egl"),
    })
    clean_env = os.environ.copy()
    clean_env.update({
        "PYTHONPATH": f"{quantvla_root}:{libero_root}",
        "LIBERO_CONFIG_PATH": str(pivot_q_root / "config/sparse_anchor_libero"),
        "MUJOCO_GL": clean_env.get("MUJOCO_GL", "egl"),
    })
    services: list[tuple[list[str], dict[str, str], Path]] = []
    for rank, suite in enumerate(SUITES):
        services.append(([
            libero_python, str(quantvla_root / "scripts/libero_plus_env_service.py"),
            "--task-suite-name", suite, "--sample-manifest", str(manifest),
            "--port", str(ports[rank]),
        ], plus_env, output / f"logs/{suite}_plus_rollout_service.log"))
        if float(experiment["anchor_lambda"]) > 0:
            services.append(([
                libero_python, str(quantvla_root / "scripts/libero_iid_env_service.py"),
                "--port", str(clean_ports[rank]),
            ], clean_env, output / f"logs/{suite}_clean_anchor_service.log"))

    python = sys.executable
    command = [
        python, "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc-per-node=4", "--master-port", str(master_port),
        str(pivot_q_root / "pi05_quantvla/pivot_q/train_ddp.py"),
        "--suites", *SUITES,
        "--env-ports", *(str(port) for port in ports),
        "--clean-env-ports", *(str(port) for port in clean_ports),
        "--checkpoint", str(resolve(workspace, paths["checkpoint"])),
        "--quant-pack-dir", str(resolve(workspace, paths["quant_pack_dir"])),
        "--output-dir", str(output), "--manifest", str(manifest),
        "--episodes-per-suite", str(episodes), "--seed", str(seed),
        "--method-name", method_name,
        "--selection-method", selection_method,
        "--updates-per-episode", str(training["updates_per_episode"]),
        "--learning-rate", str(training["learning_rate"]),
        "--lora-rank", str(training["lora_rank"]),
        "--lora-alpha", str(training["lora_alpha"]),
        "--lora-dropout", str(training["lora_dropout"]),
        "--lambda-anchor", str(experiment["anchor_lambda"]),
        "--anchor-replay-size", str(training["anchor_replay_size"]),
        "--anchor-batch-size", str(training["anchor_batch_size"]),
        "--save-every-steps", str(training["save_every_steps"]),
        "--keep-last-checkpoints", str(training["keep_last_checkpoints"]),
        "--temporal-horizon", str(selection["temporal_horizon"]),
        "--temporal-discount", str(selection["temporal_discount"]),
        "--alpha-q", str(selection["alpha_q"]), "--beta-r", str(selection["beta_r"]),
        "--phase-bins", str(selection["phase_bins"]),
        "--top-per-phase", str(selection["top_per_phase"]),
        "--min-temporal-gap", str(selection["min_temporal_gap"]),
    ]
    if not training["resume"]:
        command.append("--no-resume")
    if args.dry_run:
        print(json.dumps({
            "services": [shlex.join(item[0]) for item in services],
            "train": shlex.join(command),
            "cuda_visible_devices": ",".join(args.gpus),
            "output_dir": str(output),
        }, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    with managed_processes() as processes:
        for service in services:
            processes.append(start(*service))
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": ",".join(args.gpus),
            "PYTHONPATH": str(pivot_q_root),
            "PIVOT_Q_LAUNCH_COMMAND": shlex.join(command),
            "HF_HOME": environment.get("HF_HOME", str(pivot_q_root / ".cache/huggingface")),
            "HF_HUB_OFFLINE": environment.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": environment.get("TRANSFORMERS_OFFLINE", "1"),
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "8"),
        })
        run_worker(command, processes=processes, env=environment, cwd=pivot_q_root)


if __name__ == "__main__":
    main()
