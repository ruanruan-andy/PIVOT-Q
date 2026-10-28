"""Run GR00T suites sequentially on one GPU, with owned simulator services."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from env.conda_utils import conda_python
from env.processes import managed_processes, start, run_worker
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
MODEL = dict(zip(SUITES, ["spatial", "object", "goal", "long"]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=SUITES)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--port", type=int, default=19600)
    parser.add_argument("--clean-port", type=int, default=19601)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "," in args.gpu or args.port == args.clean_port:
        parser.error("Use one GPU and distinct service ports")
    raw = yaml.safe_load(args.config.read_text())
    paths, training, experiment = [raw[k] for k in ("paths", "training", "experiment")]
    experiment = dict(experiment)
    experiment.setdefault("name", "full_distill")
    selection = raw.get("selection", dict(temporal_horizon=4, temporal_discount=0.9, alpha_q=0.5, beta_r=0.5, phase_bins=4, top_per_phase=4, min_temporal_gap=4))
    root = Path(paths["root_dir"]).resolve()
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path
    backend = "holoq" if experiment["name"].startswith("omega_") else "quantvla"
    quant = resolve(paths["quantvla_dir"])
    plus, clean = resolve(paths["libero_plus_dir"]), resolve(paths["libero_dir"])
    manifest = resolve(paths["manifest"])
    seed = training["seed"] if args.seed is None else args.seed
    python = str(conda_python("libero_test"))
    common = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen", MUJOCO_GL="egl")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        common.setdefault(name, "8")
    common["LD_LIBRARY_PATH"] = str(Path(python).parent.parent / "lib") + ":" + common.get("LD_LIBRARY_PATH", "")
    for suite in args.suites:
        output = resolve(paths["results_dir"]) / f"seed-{seed:03d}" / "single_gpu" / suite
        command = [sys.executable, str(ROOT / "pivot_q/train_single_gpu.py"),
            "--task-suite-name", suite, "--output-dir", str(output), "--sample-manifest", str(manifest),
            "--num-rollout-episodes", str(len(json.loads(manifest.read_text())["task_ids_by_suite"][suite])),
            "--env-port", str(args.port), "--clean-env-port", str(args.clean_port),
            "--seed", str(seed), "--method-name", experiment["name"],
            "--model-path", str(resolve(raw["models"][suite]) if backend == "holoq" else ROOT / "models" / f"gr00t-n1.5-libero-{MODEL[suite]}-posttrain"),
            "--lambda-anchor", str(experiment["anchor_lambda"]), "--priority-per-phase", str(selection["top_per_phase"]),
            "--random-per-phase", "0"]
        for key in ("updates_per_episode", "learning_rate", "lora_rank", "lora_alpha", "lora_dropout",
                    "anchor_replay_size", "anchor_batch_size", "save_every_steps", "keep_last_checkpoints"):
            command += ["--" + key.replace("_", "-"), str(training[key])]
        for key in ("temporal_horizon", "temporal_discount", "alpha_q", "beta_r", "phase_bins", "min_temporal_gap",
                    "selection_method", "selection_weighting"):
            if key in selection:
                command += ["--" + key.replace("_", "-"), str(selection[key])]
        if backend == "holoq":
            command += ["--quant-pack-dir", str(resolve(raw["packs"][suite])), "--denoising-steps", str(experiment["denoising_steps"])]
        if training["save_timestep_scores"]:
            command += ["--save-timestep-scores"]
        if not training["resume"]:
            command += ["--no-resume"]
        if args.dry_run:
            print(json.dumps({"suite": suite, "command": command, "optimizer_steps": len(json.loads(manifest.read_text())["task_ids_by_suite"][suite]) * training["updates_per_episode"], "output": str(output)}))
            continue
        output.mkdir(parents=True, exist_ok=True)
        import fcntl
        with (output / ".single_gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ports = [args.port] + ([args.clean_port] if experiment["anchor_lambda"] else [])
            for port in ports:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", port))
            services = [([python, str(quant / "scripts/libero_plus_env_service.py"), "--task-suite-name", suite,
                          "--sample-manifest", str(manifest), "--port", str(args.port)],
                         dict(common, PYTHONPATH=f"{quant}:{plus}", LIBERO_CONFIG_PATH=str(quant / "configs/libero_plus")),
                         output / "logs/plus.log")]
            if experiment["anchor_lambda"]:
                services.append(([python, str(quant / "scripts/libero_iid_env_service.py"), "--port", str(args.clean_port)],
                    dict(common, PYTHONPATH=f"{quant}:{clean}", LIBERO_CONFIG_PATH=str(ROOT / "config/sparse_anchor_libero")), output / "logs/clean.log"))
            model_path = f"{ROOT}:{quant}"
            if backend == "holoq":
                model_path = f"{resolve(paths['omega_source'])}:" + model_path
            model_env = dict(common, PYTHONPATH=model_path, PIVOT_Q_SINGLE_GPU_BACKEND=backend,
                PIVOT_Q_QUANTVLA_ROOT=str(quant), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_COMPILE_DISABLE="1")
            with managed_processes() as processes:
                for service in services:
                    processes.append(start(*service))
                run_worker(command, env=model_env, processes=processes, cwd=ROOT)
            adapters = sorted((output / "checkpoints").glob("step-*/adapter"))
            if adapters:
                final = output / "final_adapter"
                if final.is_symlink():
                    final.unlink()
                final.symlink_to(adapters[-1].relative_to(output))
if __name__ == "__main__":
    main()
