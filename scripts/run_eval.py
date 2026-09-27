#!/usr/bin/env python3
"""Run one deterministic LIBERO-Plus evaluation via the standalone QuantVLA."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.conda_utils import conda_python
from env.processes import managed_processes, run_worker

import yaml


METHOD_RUNNER = {
    "fp16": "eval_fp16.sh",
    "quantvla": "eval_quantvla.sh",
    "pivot_q": None,
}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def infer_adapter_variant(adapter: Path) -> str:
    value = str(adapter).lower()
    if "full_distill" in value:
        return "groot-full-distill-w4a8"
    if "random_sparse" in value:
        return "groot-random-sparse-w4a8"
    if "uniform_sparse" in value:
        return "groot-uniform-sparse-w4a8"
    return "groot-pivot-q-w4a8"


def validate_manifest(path: Path, count: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}; run scripts/build_manifest.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_total = 4 * 7 * count
    if data.get("per_suite_per_category") != count or data.get("total_tasks") != expected_total:
        raise RuntimeError(f"manifest does not match LIBERO_PLUS_NUM={count}: {path}")


def run_pivot_q_direct(args, config, paths, manifest, adapter, quantvla_root, output_dir):
    """Evaluate a PIVOT-Q adapter using Python entrypoints, without shell wrappers."""
    suite = args.suite
    model = resolve(Path(paths["root_dir"]), config["models"][suite]).resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"original model not found: {model}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {adapter}")
    calibration_name = {"libero_spatial": "spatial", "libero_object": "object",
                        "libero_goal": "goal", "libero_10": "long"}[suite]
    calibration = quantvla_root / f"atm_alpha_beta_{calibration_name}.json"
    if not calibration.is_file():
        raise FileNotFoundError(f"QuantVLA calibration not found: {calibration}")
    pack_dir = quantvla_root / "model/quantvla/groot-n1.5" / suite / "duquant_pack"
    if not pack_dir.is_dir():
        raise FileNotFoundError(f"QuantVLA pack not found: {pack_dir}")
    if (output_dir / "metrics/episodes.jsonl").exists() and not args.resume:
        raise FileExistsError(f"evaluation already exists: {output_dir}; use --resume")

    variant = args.model_variant or infer_adapter_variant(adapter)
    data_config = ("examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
                   if suite == "libero_goal" else
                   "examples.Libero.custom_data_config:LiberoDataConfig")
    server_command = [
        sys.executable, str(quantvla_root / "scripts/inference_service.py"),
        "--model_path", str(model), "--server", "--data_config", data_config,
        "--denoising-steps", "8", "--port", str(args.port),
        "--embodiment-tag", "new_embodiment", "--adapter-path", str(adapter),
    ]
    evaluator_command = [
        str(conda_python("libero_test")),
        str(quantvla_root / "examples/LiberoPlus/eval/run_libero_plus_eval.py"),
        "--task-suite-name", suite, "--port", str(args.port),
        "--model-variant", variant, "--sample-manifest", str(manifest),
        "--policy-seed", str(config["evaluation"]["policy_seed"]),
        "--headless", "--no-save-video",
    ]
    if args.resume:
        evaluator_command.append("--resume")

    server_env = os.environ.copy()
    server_env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONUNBUFFERED": "1",
        "GR00T_PORT": str(args.port), "GR00T_MODEL_VARIANT": variant,
        "PYTHONPATH": f"{quantvla_root}:{server_env.get('PYTHONPATH', '')}",
        "HF_HUB_DISABLE_XET": "1", "NO_PROXY": "127.0.0.1,localhost",
    })
    quant_defaults = {
        "GR00T_DUQUANT_DEBUG": "1",
        "GR00T_DUQUANT_SCOPE": "",
        "GR00T_DUQUANT_INCLUDE": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*",
        "GR00T_DUQUANT_EXCLUDE": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
        "GR00T_DUQUANT_WBITS_DEFAULT": "4", "GR00T_DUQUANT_ABITS": "8",
        "GR00T_DUQUANT_BLOCK": "64", "GR00T_DUQUANT_PERMUTE": "0",
        "GR00T_DUQUANT_ROW_ROT": "restore", "GR00T_DUQUANT_ACT_PCT": "99.9",
        "GR00T_DUQUANT_CALIB_STEPS": "32", "GR00T_DUQUANT_LS": "0.15",
        "GR00T_DUQUANT_PACKDIR": str(pack_dir),
        "GR00T_ATM_ALPHA_PATH": str(calibration), "GR00T_ATM_ENABLE": "1",
        "GR00T_ATM_SCOPE": "dit", "GR00T_OHB_ENABLE": "1",
        "GR00T_OHB_FALLBACK": "1.0", "GR00T_OHB_SCOPE": "dit",
        "TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1",
        "TORCH_CUDA_GRAPH_DISABLE": "1", "TORCHINDUCTOR_DISABLE_CUDAGRAPHS": "1",
    }
    for name, value in quant_defaults.items():
        server_env.setdefault(name, value)

    libero_plus_root = resolve(Path(paths["root_dir"]), paths["libero_plus_dir"]).resolve()
    libero_config = quantvla_root / "configs/libero_plus"
    evaluator_env = os.environ.copy()
    evaluator_env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "GR00T_PORT": str(args.port),
        "EVAL_MODEL_VARIANT": variant, "EVAL_METHOD": "quantvla-pivot_q",
        "EVAL_BENCHMARK": config["evaluation"]["benchmark"],
        "EVAL_TRAIN_SEED": str(args.train_seed), "EVAL_CHECKPOINT": str(adapter),
        "EVAL_MANIFEST": str(manifest),
        "EVAL_SERVER_COMMAND": shlex.join(server_command),
        "EVAL_EVALUATOR_COMMAND": shlex.join(evaluator_command),
        "LIBERO_PLUS_OUTPUT_DIR": str(output_dir),
        "LIBERO_EVAL_METRICS_DIR": str(output_dir / "metrics"),
        "LIBERO_EVAL_LOGS_DIR": str(output_dir / "logs"),
        "LIBERO_EVAL_VIDEO_DIR": str(output_dir / "videos"),
        "LIBERO_EVAL_LOG_DIR": str(output_dir),
        "LIBERO_CONFIG_PATH": str(libero_config), "MUJOCO_GL": "egl",
        "PYTHONPATH": f"{libero_plus_root}:{quantvla_root}:{evaluator_env.get('PYTHONPATH', '')}",
        "LD_LIBRARY_PATH": f"{conda_python('libero_test').parent.parent / 'lib'}:{evaluator_env.get('LD_LIBRARY_PATH', '')}",
        "MAGICK_HOME": str(conda_python("libero_test").parent.parent),
    })
    if args.offline:
        for environment in (server_env, evaluator_env):
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                "HF_HUB_DISABLE_TELEMETRY": "1"})
    print("Server:", shlex.join(server_command), flush=True)
    print("Evaluator:", shlex.join(evaluator_command), flush=True)
    if args.dry_run:
        return

    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    server_log = output_dir / "logs/server.log"
    with managed_processes() as processes:
        with server_log.open("w", encoding="utf-8") as stream:
            server = subprocess.Popen(server_command, cwd=quantvla_root, env=server_env,
                                      stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        processes.append(server)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"inference server exited: {server_log}")
            if "Server is ready and listening" in server_log.read_text(
                encoding="utf-8", errors="replace"
            ):
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"inference server did not become ready: {server_log}")
        print(f"Inference server ready on port {args.port}", flush=True)
        with (output_dir / "logs/pipeline.log").open("a", encoding="utf-8") as stream:
            run_worker(evaluator_command, cwd=quantvla_root, env=evaluator_env,
                       stdout=stream, stderr=subprocess.STDOUT, processes=processes)
        if not (output_dir / "metrics/summary.json").is_file():
            raise RuntimeError(f"evaluator exited without summary: {output_dir}")
        print(f"Evaluation complete: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: {"pivot-q": "pivot_q", "pivot_q": "pivot_q", "quantized": "quantvla"}.get(value, value), choices=METHOD_RUNNER, required=True)
    parser.add_argument("--suite", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "eval.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-mode", choices=("standard",),
                        default="standard")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="final PIVOT-Q LoRA adapter directory (required for --method pivot-q)")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="Override the evaluation seed without editing YAML")
    parser.add_argument("--model-variant", default=None,
                        help="optional result label; normally inferred")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.policy_seed is not None:
        config["evaluation"]["policy_seed"] = args.policy_seed
    paths = config["paths"]
    evaluation = config["evaluation"]
    count = int(config["LIBERO_PLUS_NUM"])
    root = Path(paths["root_dir"])
    manifest = resolve(root, paths["manifests_dir"]) / f"libero_plus_first{count}.json"
    if args.method == "pivot_q":
        if args.adapter_path is None:
            parser.error("--adapter-path is required for --method pivot-q")
        checkpoint = args.adapter_path.expanduser().resolve()
    else:
        checkpoint = resolve(root, config["models"][args.suite])
    quantvla_root = resolve(root, paths["quantvla_dir"])
    output_dir = (
        args.output_dir or (Path(__file__).resolve().parents[1] / 'outputs/quantvla/groot_n1_5'
                            / {'fp16': 'original', 'quantvla': 'quantized', 'pivot_q': 'pivot_q'}[args.method]
                            / 'eval' / f'seed-{int(evaluation["policy_seed"]):03d}' / args.suite)
    ).expanduser().resolve()

    validate_manifest(manifest, count)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.method == "pivot_q":
        run_pivot_q_direct(args, config, paths, manifest, checkpoint,
                           quantvla_root, output_dir)
        return

    runner = quantvla_root / METHOD_RUNNER[args.method]
    if not runner.is_file():
        raise FileNotFoundError(f"QuantVLA runner not found: {runner}")

    command = [
        str(runner), "--benchmark", evaluation["benchmark"], "--suite", args.suite,
        "--gpu", str(args.gpu), "--port", str(args.port), "--checkpoint", str(checkpoint),
        "--manifest", str(manifest), "--eval-seed", str(evaluation["policy_seed"]),
        "--output-dir", str(output_dir),
    ]
    if args.method == "pivot_q":
        command += ["--train-seed", str(args.train_seed)]
    if args.resume:
        command.append("--resume")
    if args.offline:
        command.append("--offline")
    if args.dry_run:
        command.append("--dry-run")
    print(" ".join(command))
    environment = os.environ | {"LIBERO_PLUS_ROOT": str(resolve(root, paths["libero_plus_dir"]))}
    if args.method == "pivot_q":
        environment["GR00T_MODEL_PATH"] = str(resolve(root, config["models"][args.suite]).resolve())
    with managed_processes(grace_seconds=120) as processes:
        run_worker(command, cwd=quantvla_root, env=environment, processes=processes)


if __name__ == "__main__":
    main()
