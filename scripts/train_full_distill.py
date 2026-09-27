"""Full-distillation service launcher and command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.conda_utils import conda_python
from env.processes import start, managed_processes, run_worker

import yaml


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODEL_DIRS = {
    "libero_spatial": "gr00t-n1.5-libero-spatial-posttrain",
    "libero_object": "gr00t-n1.5-libero-object-posttrain",
    "libero_goal": "gr00t-n1.5-libero-goal-posttrain",
    "libero_10": "gr00t-n1.5-libero-long-posttrain",
}


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _manifest_task_count(path: Path, suite: str) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data["task_ids_by_suite"][suite])


def run_round(args, config: dict, experiment: dict, core: str, manifest: Path, output: Path, initial_adapter: Path | None = None) -> Path:
    paths, training = config["paths"], config["training"]
    root_dir = Path(paths["root_dir"])
    qv_root = resolve(root_dir, paths["quantvla_dir"])
    libero_root = resolve(root_dir, paths["libero_dir"])
    libero_plus_root = resolve(root_dir, paths["libero_plus_dir"])
    pivot_q_root = resolve(root_dir, paths["pivot_q_dir"])
    rollout_env = os.environ.copy()
    rollout_env.update({
        "PYTHONPATH": f"{qv_root}:{libero_plus_root}",
        "LIBERO_CONFIG_PATH": str(qv_root / "configs/libero_plus"),
        "MUJOCO_GL": rollout_env.get("MUJOCO_GL", "egl"),
    })
    clean_env = os.environ.copy()
    clean_env.update({
        "PYTHONPATH": f"{qv_root}:{libero_root}",
        "LIBERO_CONFIG_PATH": str(pivot_q_root / "config/sparse_anchor_libero"),
        "MUJOCO_GL": clean_env.get("MUJOCO_GL", "egl"),
    })
    services = [([
        "conda", "run", "--no-capture-output", "-n", "libero_test", "python",
        str(qv_root / "scripts/libero_plus_env_service.py"),
        "--task-suite-name", args.suite, "--sample-manifest", str(manifest), "--port", str(args.port),
    ], rollout_env, output / "logs/plus_rollout_service.log")]
    if float(experiment["anchor_lambda"]) > 0:
        services.append(([
            "conda", "run", "--no-capture-output", "-n", "libero_test", "python",
            str(qv_root / "scripts/libero_iid_env_service.py"), "--port", str(args.clean_port),
        ], clean_env, output / "logs/clean_anchor_service.log"))
    with managed_processes() as processes:
        for service in services:
            processes.append(start(*service))
        train_env = os.environ.copy()
        train_env.update({
            "PYTHONPATH": f"{qv_root}:{pivot_q_root}",
            "PIVOT_Q_QUANTVLA_ROOT": str(qv_root),
            "PIVOT_Q_LAUNCH_COMMAND": " ".join(sys.argv),
            "HF_HOME": train_env.get("HF_HOME", str(qv_root / "model")),
            "HF_HUB_OFFLINE": train_env.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": train_env.get("TRANSFORMERS_OFFLINE", "1"),
            "TORCH_COMPILE_DISABLE": "1",
        })
        command = [
            sys.executable,
            str(pivot_q_root / "pivot_q" / core),
            "--task-suite-name", args.suite,
            "--output-dir", str(output),
            "--sample-manifest", str(manifest),
            "--num-rollout-episodes", str(_manifest_task_count(manifest, args.suite)),
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
            "--method-name", args.experiment,
            "--model-path", str(pivot_q_root / "models" / MODEL_DIRS[args.suite]),
            "--env-port", str(args.port), "--clean-env-port", str(args.clean_port),
            "--seed", str(args.seed),
        ]
        if not training["resume"]:
            command.append("--no-resume")
        if training["save_timestep_scores"]:
            command.append("--save-timestep-scores")
        if initial_adapter is not None:
            command += ["--initial-adapter-path", str(initial_adapter)]
        run_worker(command, env=train_env, processes=processes)
    checkpoints = sorted((output / "checkpoints").glob("step-*/adapter"))
    if not checkpoints:
        raise RuntimeError(f"training wrote no adapter under {output}")
    final_adapter = output / "final_adapter"
    if final_adapter.is_symlink():
        final_adapter.unlink()
    elif final_adapter.exists():
        raise RuntimeError(f"refusing to overwrite non-link final adapter: {final_adapter}")
    final_adapter.symlink_to(checkpoints[-1])
    return final_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clean-port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/full_distill.yaml"))
    args = parser.parse_args()
    if args.port == args.clean_port:
        parser.error("target and clean ports must differ")
    args.experiment = "full_distill"
    config = load_config(args.config)
    args.seed = config["training"]["seed"] if args.seed is None else args.seed
    root = Path(config["paths"]["root_dir"])
    run_round(args, config, config["experiment"], "full_distill.py", resolve(root, config["paths"]["manifest"]), resolve(root, config["paths"]["results_dir"]) / f"seed-{args.seed:03d}" / args.suite)


if __name__ == "__main__":
    main()
