#!/usr/bin/env python3
"""Launch the eight-GPU paired full-distillation ablation.

Four two-GPU pairs process four rollouts per round. All pairs visit the same
suite before moving to the next suite; no GPU is permanently suite-assigned.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pi05_quantvla.pivot_q.run_train_ddp import resolve, start, stop


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=root / "pi05_quantvla/pivot_q/config/paired8_full_distill.yaml",
    )
    parser.add_argument("--gpus", nargs=8, default=[str(i) for i in range(8)])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from pi05_quantvla.pivot_q.train_paired8 import ORDER
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    training = config["training"]
    experiment = config["experiment"]
    distributed = config["distributed"]
    if tuple(distributed["suite_order"]) != ORDER:
        raise ValueError(f"expected suite order {ORDER}")
    ports = list(map(int, distributed["rollout_ports"]))
    clean_ports = list(map(int, distributed["clean_anchor_ports"]))
    master_port = int(distributed["master_port"])
    if len(ports) != 4 or len(clean_ports) != 4:
        raise ValueError("four rollout and four clean-anchor ports are required")
    if len(set([*ports, *clean_ports, master_port])) != 9:
        raise ValueError("service and master ports must be distinct")
    workspace = Path(paths["root_dir"]).expanduser().resolve()
    pivot_q_root = resolve(workspace, paths["pivot_q_dir"])
    quantvla_root = resolve(workspace, paths["quantvla_dir"])
    libero_root = resolve(workspace, paths["libero_dir"])
    libero_plus_root = resolve(workspace, paths["libero_plus_dir"])
    manifest = resolve(workspace, paths["manifest"])
    checkpoint = resolve(workspace, paths["checkpoint"])
    quant_pack = resolve(workspace, paths["quant_pack_dir"])
    output = args.output_dir or (
        resolve(workspace, paths["results_dir"])
        / f"seed-{int(training['seed']):03d}"
        / "shared"
    )
    output = output.expanduser().resolve()
    for name in (
        "config.json", "model.safetensors",
        "policy_preprocessor.json", "policy_postprocessor.json",
    ):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not (quant_pack / "atm_ohb.json").is_file():
        raise FileNotFoundError(quant_pack / "atm_ohb.json")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    hf_home = Path(os.environ.get("HF_HOME", str(pivot_q_root / ".cache/huggingface")))
    if not (hf_home / "hub/models--google--paligemma-3b-pt-224/snapshots").is_dir():
        raise FileNotFoundError(
            f"offline PaliGemma tokenizer cache missing below {hf_home}"
        )

    libero_python = str(conda_python("libero_test"))
    plus_env = os.environ.copy()
    plus_env.update({
        "PYTHONPATH": f"{pivot_q_root}:{quantvla_root}:{libero_plus_root}",
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
    for pair in range(4):
        services.append((
            [
                libero_python,
                str(pivot_q_root / "pi05_quantvla/pivot_q/multi_suite_env_service.py"),
                "--sample-manifest", str(manifest),
                "--port", str(ports[pair]),
            ],
            plus_env,
            output / f"logs/pair-{pair}_plus_service.log",
        ))
        if float(experiment["anchor_lambda"]) > 0:
            services.append((
                [
                    libero_python,
                    str(quantvla_root / "scripts/libero_iid_env_service.py"),
                    "--port", str(clean_ports[pair]),
                ],
                clean_env,
                output / f"logs/pair-{pair}_clean_service.log",
            ))

    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc-per-node=8", "--master-port", str(master_port),
        str(pivot_q_root / "pi05_quantvla/pivot_q/train_paired8.py"),
        "--suites", "libero_spatial", "libero_object", "libero_goal", "libero_10",
        "--suite-order", *ORDER,
        "--env-ports", *(str(value) for value in ports),
        "--clean-env-ports", *(str(value) for value in clean_ports),
        "--checkpoint", str(checkpoint),
        "--quant-pack-dir", str(quant_pack),
        "--output-dir", str(output),
        "--manifest", str(manifest),
        "--episodes-per-suite", str(training["episodes_per_suite"]),
        "--seed", str(training["seed"]),
        "--method-name", str(experiment["name"]),
        "--selection-method", "full_distill",
        "--updates-per-episode", str(training["updates_per_episode"]),
        "--learning-rate", str(training["learning_rate"]),
        "--lora-rank", str(training["lora_rank"]),
        "--lora-alpha", str(training["lora_alpha"]),
        "--lora-dropout", str(training["lora_dropout"]),
        "--lambda-anchor", str(experiment["anchor_lambda"]),
        "--anchor-replay-size", str(training["anchor_replay_size"]),
        "--anchor-batch-size", str(training["anchor_batch_size"]),
        "--state-microbatch", str(training["state_microbatch"]),
        "--save-every-steps", str(training["save_every_steps"]),
        "--keep-last-checkpoints", str(training["keep_last_checkpoints"]),
    ]
    if args.max_rounds:
        command.extend(["--max-rounds", str(args.max_rounds)])
    if not training["resume"]:
        command.append("--no-resume")
    if args.dry_run:
        print(json.dumps({
            "services": [shlex.join(item[0]) for item in services],
            "train": shlex.join(command),
            "cuda_visible_devices": ",".join(args.gpus),
            "output_dir": str(output),
            "hf_home": str(hf_home),
            "offline": True,
        }, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    with managed_processes() as processes:
        for item in services:
            processes.append(start(*item))
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": ",".join(args.gpus),
            "PYTHONPATH": str(pivot_q_root),
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
            "PIVOT_Q_LAUNCH_COMMAND": shlex.join(command),
        })
        run_worker(command, processes=processes, env=environment, cwd=pivot_q_root)


if __name__ == "__main__":
    main()
