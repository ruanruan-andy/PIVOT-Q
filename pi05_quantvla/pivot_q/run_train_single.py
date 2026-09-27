#!/usr/bin/env python3
"""Launch single-GPU sequential-suite recovery, with owned simulator services."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ORDER = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]


def main(backend="quantvla"):
    import yaml
    from env.conda_utils import conda_python
    from env.processes import managed_processes, run_worker
    from pi05_quantvla.pivot_q.run_train_ddp import resolve, start, stop

    parser = argparse.ArgumentParser(description=__doc__)
    default = "pi05_omegavla/config/pivot_q.yaml" if backend == "holoq" else "pi05_quantvla/pivot_q/config/pivot_q.yaml"
    parser.add_argument("--config", type=Path, default=ROOT / default)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--port", type=int, default=19500)
    parser.add_argument("--clean-port", type=int, default=19501)
    parser.add_argument("--max-rounds", type=int, default=0, help="Stop after this many additional groups, save a resumable checkpoint; 0 = full budget")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.port == args.clean_port or args.max_rounds < 0 or "," in args.gpu:
        parser.error("Use one GPU, distinct service ports, and a nonnegative round limit")
    if backend == "holoq":
        from pi05_omegavla.common import load_config
        raw = load_config(args.config)
        method = raw["mode"]
    else:
        raw = yaml.safe_load(args.config.read_text())
        method = raw["selection"].get("method", "pivot_q")
    if method not in {"pivot_q", "full_distill"}:
        parser.error("Only pivot_q and full_distill are supported")
    paths, training, selection = raw["paths"], raw["training"], raw["selection"]
    workspace = Path(paths["root_dir"]).expanduser().resolve()
    root = resolve(workspace, paths["pivot_q_dir"]).resolve()
    manifest = resolve(workspace, paths["manifest"])
    counts = {s: len(v) for s, v in json.loads(manifest.read_text())["task_ids_by_suite"].items() if s in ORDER}
    if set(counts) != set(ORDER) or len(set(counts.values())) != 1 or next(iter(counts.values())) % 4:
        raise ValueError("Four suites with equal counts divisible by four are required")
    episodes = next(iter(counts.values()))
    if "episodes_per_suite" in training and int(training["episodes_per_suite"]) != episodes:
        raise ValueError("episodes_per_suite differs from manifest")
    seed = int(training["seed"] if args.seed is None else args.seed)
    output = (args.output_dir or (resolve(workspace, paths["results_dir"]) / f"seed-{seed:03d}" / "single_gpu")).resolve()
    if output.name == "shared":
        raise ValueError("Do not reuse a multi-GPU shared output directory")

    # Defaults match the multi-GPU Config; optional YAML values override them.
    defaults = dict(initial_state_ids=[0], updates_per_episode=5, learning_rate=5e-5,
        weight_decay=0.01, gradient_clip_norm=1.0, lora_rank=16, lora_alpha=32,
        lora_dropout=0.05, anchor_replay_size=256, anchor_batch_size=4,
        clean_anchor_horizon=4, save_every_steps=25, keep_last_checkpoints=1)
    for key in defaults:
        if key in training:
            defaults[key] = training[key]
    scoring = dict(temporal_horizon=4, temporal_discount=0.9, alpha_q=0.5, beta_r=0.5,
                   phase_bins=4, top_per_phase=4, min_temporal_gap=4, weight_min=0.5, weight_max=2.0)
    for key in scoring:
        if key in selection:
            scoring[key] = selection[key]
    values = dict(**defaults, **scoring, backend=backend, schedule="sequential",
        suites=["libero_spatial", "libero_object", "libero_goal", "libero_10"], suite_order=ORDER,
        selection_method=method, method_name=raw["experiment"]["name"], seed=seed,
        lambda_anchor=float(raw["experiment"]["anchor_lambda"]), device="cuda:0",
        episodes_per_suite=episodes, env_port=args.port, clean_env_port=args.clean_port,
        checkpoint=str(resolve(workspace, paths["checkpoint"])), quant_pack_dir=str(resolve(workspace, paths["quant_pack_dir"])),
        omega_config=str(args.config.resolve()), output_dir=str(output), manifest=str(manifest),
        resume=bool(training.get("resume", True)) and not args.no_resume, max_rounds=args.max_rounds)
    # Store protocol values, not just a path that may later be edited in place.
    values["model_protocol"] = ({"quantization": raw["quantization"], "inference": raw["inference"]}
        if backend == "holoq" else yaml.safe_load((root / "pi05_quantvla/config/quantvla.yaml").read_text()))
    if values["updates_per_episode"] <= 0 or values["save_every_steps"] <= 0 or values["lambda_anchor"] < 0:
        raise ValueError("Invalid update/checkpoint/anchor settings")
    quant = resolve(workspace, paths["quantvla_dir"])
    libero = resolve(workspace, paths["libero_dir"])
    plus = resolve(workspace, paths["libero_plus_dir"])
    python = str(conda_python("libero_test"))
    libero_prefix = Path(python).resolve().parent.parent
    common = os.environ.copy()
    common.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen",
                  MUJOCO_GL=common.get("MUJOCO_GL", "egl"), CUDA_VISIBLE_DEVICES=args.gpu)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        common.setdefault(variable, "8")
    plus_env = dict(common, PYTHONPATH=f"{root}:{quant}:{plus}", LIBERO_CONFIG_PATH=str(quant / "configs/libero_plus"))
    plus_env.update(MAGICK_HOME=str(libero_prefix),
                    LD_LIBRARY_PATH=f"{libero_prefix / 'lib'}:{common.get('LD_LIBRARY_PATH', '')}")
    clean_env = dict(common, PYTHONPATH=f"{quant}:{libero}", LIBERO_CONFIG_PATH=str(root / "config/sparse_anchor_libero"))
    services = [([python, str(root / "pi05_quantvla/pivot_q/multi_suite_env_service.py"),
                  "--sample-manifest", str(manifest), "--port", str(args.port)], plus_env, output / "logs/plus_service.log")]
    if values["lambda_anchor"] > 0:
        services.append(([python, str(quant / "scripts/libero_iid_env_service.py"), "--port", str(args.clean_port)],
                         clean_env, output / "logs/clean_service.log"))
    runtime = output / "single_runtime.json"
    command = [sys.executable, str(root / "pi05_quantvla/pivot_q/train_single.py"), "--runtime-config", str(runtime)]
    if args.dry_run:
        print(json.dumps({"train": shlex.join(command), "config": values,
                          "services": [shlex.join(s[0]) for s in services], "gpu": args.gpu,
                          "rollouts": episodes * 4, "optimizer_steps": episodes * values["updates_per_episode"]}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    # Linux training hosts: a held lock prevents simultaneous writers/resumes.
    import fcntl
    lock = (output / ".single_train.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another single-device launcher owns {output}") from None
    if (output / "checkpoints").exists():
        for checkpoint in (output / "checkpoints").glob("step-*"):
            marker = checkpoint / "complete.json"
            if marker.is_file() and json.loads(marker.read_text()).get("format") != "pivot_q_single_v1":
                raise RuntimeError("Multi-GPU checkpoints require explicit conversion, which is not supported")
    runtime.write_text(json.dumps(values, indent=2))
    with managed_processes() as processes:
        # Fail before launching if any selected port is already owned.
        import socket
        for port in [args.port] + ([args.clean_port] if values["lambda_anchor"] > 0 else []):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        for service in services:
            processes.append(start(*service))
        model_env = dict(common, PYTHONPATH=f"{root}:{root / 'third_party/lerobot/src'}",
                         HF_HOME=common.get("HF_HOME", str(root / ".cache/huggingface")))
        run_worker(command, processes=processes, env=model_env, cwd=root)


if __name__ == "__main__":
    main()
