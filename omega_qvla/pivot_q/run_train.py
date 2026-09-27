#!/usr/bin/env python3
"""Launch one Ω-PIVOT_Q suite with isolated LIBERO simulator services."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker, start

import yaml

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clean-port", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=root / "omega_qvla/pivot_q/config/pivot_q.yaml")
    args = parser.parse_args()
    if args.port == args.clean_port:
        parser.error("rollout and clean ports must differ")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, training, experiment = config["paths"], config["training"], config["experiment"]
    workspace = Path(paths["root_dir"]).expanduser().resolve()
    pivot_q_root = resolve(workspace, paths["pivot_q_dir"])
    quantvla_root = resolve(workspace, paths["quantvla_dir"])
    omega_root = resolve(workspace, paths["omega_source"])
    libero_root = resolve(workspace, paths["libero_dir"])
    libero_plus_root = resolve(workspace, paths["libero_plus_dir"])
    manifest = resolve(workspace, paths["manifest"])
    output = resolve(workspace, paths["results_dir"]) / f"seed-{args.seed if args.seed is not None else training['seed']:03d}" / args.suite
    seed = int(training["seed"] if args.seed is None else args.seed)
    task_count = len(json.loads(manifest.read_text(encoding="utf-8"))["task_ids_by_suite"][args.suite])
    python = sys.executable
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
    output.mkdir(parents=True, exist_ok=True)
    services = [([
        libero_python, str(quantvla_root / "scripts/libero_plus_env_service.py"),
        "--task-suite-name", args.suite, "--sample-manifest", str(manifest),
        "--port", str(args.port),
    ], plus_env, output / "logs/plus_rollout_service.log")]
    if float(experiment["anchor_lambda"]) > 0:
        services.append(([
            libero_python, str(quantvla_root / "scripts/libero_iid_env_service.py"),
            "--port", str(args.clean_port),
        ], clean_env, output / "logs/clean_anchor_service.log"))

    with managed_processes() as processes:
        for service in services:
            processes.append(start(*service))
        train_env = os.environ.copy()
        train_env.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": f"{omega_root}:{pivot_q_root}:{quantvla_root}",
            "PIVOT_Q_LAUNCH_COMMAND": " ".join(sys.argv),
            "HF_HOME": train_env.get("HF_HOME", str(quantvla_root / "model")),
            "HF_HUB_OFFLINE": train_env.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": train_env.get("TRANSFORMERS_OFFLINE", "1"),
            "TORCH_COMPILE_DISABLE": "1",
        })
        command = [
            python, str(pivot_q_root / "omega_qvla/pivot_q/train.py"),
            "--task-suite-name", args.suite,
            "--output-dir", str(output),
            "--sample-manifest", str(manifest),
            "--num-rollout-episodes", str(task_count),
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
            "--method-name", str(experiment["name"]),
            "--model-path", str(resolve(workspace, config["models"][args.suite])),
            "--quant-pack-dir", str(resolve(workspace, config["packs"][args.suite])),
            "--denoising-steps", str(experiment["denoising_steps"]),
            "--env-port", str(args.port), "--clean-env-port", str(args.clean_port),
            "--seed", str(seed),
        ]
        selection = config["selection"]
        command += [
            "--temporal-horizon", str(selection["temporal_horizon"]),
            "--temporal-discount", str(selection["temporal_discount"]),
            "--alpha-q", str(selection["alpha_q"]),
            "--beta-r", str(selection["beta_r"]),
            "--phase-bins", str(selection["phase_bins"]),
            "--priority-per-phase", str(selection["top_per_phase"]),
            "--random-per-phase", "0",
            "--min-temporal-gap", str(selection["min_temporal_gap"]),
        ]
        if not training["resume"]:
            command.append("--no-resume")
        if training["save_timestep_scores"]:
            command.append("--save-timestep-scores")
        run_worker(command, env=train_env, processes=processes)

    adapters = sorted((output / "checkpoints").glob("step-*/adapter"))
    if not adapters:
        raise RuntimeError(f"training wrote no adapter under {output}")
    final_adapter = output / "final_adapter"
    if final_adapter.is_symlink():
        final_adapter.unlink()
    elif final_adapter.exists():
        raise RuntimeError(f"refusing to overwrite non-link {final_adapter}")
    final_adapter.symlink_to(adapters[-1])


if __name__ == "__main__":
    main()
