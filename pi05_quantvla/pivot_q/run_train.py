#!/usr/bin/env python3
"""Launch one π0.5-PIVOT_Q suite and its isolated simulator services."""

from __future__ import annotations

import argparse
import json
import os
import sys
import signal
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker

import yaml

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def start(command: list[str], environment: dict[str, str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        command, stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT,
        env=environment, start_new_session=True,
    )


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clean-port", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=root / "pi05_quantvla/pivot_q/config/pivot_q.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, training, experiment, selection = (
        config["paths"], config["training"], config["experiment"], config["selection"]
    )
    workspace = Path(paths["root_dir"]).expanduser().resolve()
    pivot_q_root = resolve(workspace, paths["pivot_q_dir"])
    quantvla_root = resolve(workspace, paths["quantvla_dir"])
    libero_root = resolve(workspace, paths["libero_dir"])
    libero_plus_root = resolve(workspace, paths["libero_plus_dir"])
    manifest = resolve(workspace, paths["manifest"])
    seed = int(training["seed"] if args.seed is None else args.seed)
    output = resolve(workspace, paths["results_dir"]) / f"seed-{seed:03d}" / args.suite
    episodes = len(json.loads(manifest.read_text(encoding="utf-8"))["task_ids_by_suite"][args.suite])
    output.mkdir(parents=True, exist_ok=True)

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
    libero_python = str(conda_python("libero_test"))
    plus_command = [
        libero_python, str(quantvla_root / "scripts/libero_plus_env_service.py"),
        "--task-suite-name", args.suite, "--sample-manifest", str(manifest),
        "--port", str(args.port),
    ]
    clean_command = [
        libero_python, str(quantvla_root / "scripts/libero_iid_env_service.py"),
        "--port", str(args.clean_port),
    ]
    with managed_processes() as processes:
        processes.append(start(plus_command, plus_env, output / "logs/plus_rollout_service.log"))
        if experiment["anchor_lambda"]:
            processes.append(start(clean_command, clean_env, output / "logs/clean_anchor_service.log"))
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(pivot_q_root),
            # π0.5's preprocessor references the upstream PaliGemma tokenizer by
            # repository ID.  The artifacts are already cached locally; forcing
            # offline mode avoids a slow network HEAD request on every resume.
            "HF_HOME": environment.get(
                "HF_HOME", str(pivot_q_root / ".cache/huggingface")
            ),
            "HF_HUB_OFFLINE": environment.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": environment.get("TRANSFORMERS_OFFLINE", "1"),
        })
        command = [
            sys.executable, str(pivot_q_root / "pi05_quantvla/pivot_q/train.py"),
            "--suite", args.suite, "--checkpoint", str(resolve(workspace, paths["checkpoint"])),
            "--quant-pack-dir", str(resolve(workspace, paths["quant_pack_dir"])),
            "--output-dir", str(output), "--manifest", str(manifest),
            "--episodes", str(episodes), "--seed", str(seed),
            "--env-port", str(args.port), "--clean-env-port", str(args.clean_port),
            "--updates-per-episode", str(training["updates_per_episode"]),
            "--learning-rate", str(training["learning_rate"]),
            "--lora-rank", str(training["lora_rank"]), "--lora-alpha", str(training["lora_alpha"]),
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
        run_worker(command, processes=processes, env=environment)
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
