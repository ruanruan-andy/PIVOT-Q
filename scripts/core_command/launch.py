#!/usr/bin/env python3
"""Command adapters for the five existing experiment implementations."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.processes import managed_processes, run_worker
from env.protocol import assert_same, file_value
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FAMILIES = ("groot_quantvla", "pi05_quantvla", "groot_holoqvla", "pi05_holoqvla", "openvla_oft_qvla")
SETTINGS = ("fp_original", "quant_original", "full_distill", "pivot_q")
ABLATION_DIRS = {
    'pivot_q_current_only': 'current_only', 'pivot_q_no_anchor': 'no_anchor',
    'pivot_q_a03_b07': 'score_a03_b07', 'pivot_q_a04_b06': 'score_a04_b06',
    'pivot_q_a07_b03': 'score_a07_b03',
    'pivot_q_a03_b07_no_anchor': 'score_a03_b07_no_anchor',
}

def result_root(family, setting):
    roots = {
        "groot_quantvla": ["outputs/quantvla/groot_n1_5/original", "outputs/quantvla/groot_n1_5/quantized",
                           "outputs/quantvla/groot_n1_5/full_distill", "outputs/quantvla/groot_n1_5/pivot_q"],
        "pi05_quantvla": ["outputs/quantvla/pi05/original", "outputs/quantvla/pi05/quantized",
                          "outputs/quantvla/pi05/full_distill", "outputs/quantvla/pi05/pivot_q"],
        "groot_holoqvla": ["outputs/holoq_vla/groot_n1_5/original",
                           "outputs/holoq_vla/groot_n1_5/quantized",
                           "outputs/holoq_vla/groot_n1_5/full_distill", "outputs/holoq_vla/groot_n1_5/pivot_q"],
        "pi05_holoqvla": ["outputs/holoq_vla/pi05/" + m for m in
                          ("original", "quantized", "full_distill", "pivot_q")],
        "openvla_oft_qvla": ["outputs/qvla/openvla_oft/" + m for m in
                            ("original", "quantized", "full_distill", "pivot_q")],
    }
    return ROOT / roots[family][SETTINGS.index(setting)]

def eval_root(base, family, setting):
    return base / "eval"

def eval_directory(base, family, setting, seed, suite):
    path = eval_root(base, family, setting) / f"seed-{seed:03d}"
    if family == "pi05_holoqvla":
        path /= "libero-plus"
    return path / suite

def read_config(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    parent = data.pop("extends", None)
    if parent:
        base = read_config(path.parent / parent)
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
            return a
        data = merge(base, data)
    return data

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=FAMILIES)
    p.add_argument("setting", choices=SETTINGS)
    p.add_argument("action", choices=("train", "eval", "calibrate", "service_plus", "service_clean"))
    p.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    p.add_argument("--gpu", default="0")
    p.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
    p.add_argument("--seed", type=int)
    p.add_argument("--port", type=int, default=7700)
    p.add_argument("--clean-port", type=int, default=7701)
    p.add_argument("--adapter-path", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--results-root", type=Path)
    p.add_argument("--python", dest="python")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ablation", choices=("random_sparse", "uniform_sparse", "global_random_sparse",
        "pivot_q_current_only", "pivot_q_no_anchor", "pivot_q_a03_b07", "pivot_q_a04_b06",
        "pivot_q_a07_b03", "pivot_q_a03_b07_no_anchor"))
    p.add_argument("--resume-path", type=Path)
    p.add_argument("--dry-run", action="store_true", help="print commands without starting processes or writing files")
    a, extra = p.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    family, setting, action = a.family, a.setting, a.action
    if a.ablation and (family, setting, action) != ("groot_quantvla", "pivot_q", "eval"):
        p.error("--ablation requires groot_quantvla pivot_q eval and its matching adapter")
    if any(v in extra for v in ("--output-dir", "--output-root", "--benchmark", "--method", "--manifest")):
        p.error("output/method/benchmark/manifest are managed by this command; use --results-root")
    for i, value in enumerate(extra):
        if value == "--set" and i + 1 < len(extra):
            key = extra[i+1].split("=", 1)[0]
            if key in ("paths.output", "seed", "evaluation.seed", "suite", "method"):
                p.error(f"{key} is managed by the command parameters")
    if a.seed is not None and a.seed < 0:
        p.error("seed must be nonnegative")
    recovered = setting in ("full_distill", "pivot_q")
    if action == "train" and not recovered:
        p.error("original baselines have no train command")
    if action == "eval" and recovered and a.adapter_path is None:
        p.error("recovery evaluation requires --adapter-path (use {suite} for suite-specific adapters)")
    if a.adapter_path is not None and (action != "eval" or not recovered):
        p.error("--adapter-path is only valid for recovery evaluation")
    if a.resume and action != "eval":
        p.error("--resume is evaluation-only; training follows config (OFT uses --resume-path)")
    if a.resume_path and (family != "openvla_oft_qvla" or action != "train"):
        p.error("--resume-path is only supported for OFT training")
    if a.seed is None:
        a.seed = 2026 if action == "eval" else 0
    base = (a.results_root or result_root(family, setting)).expanduser().resolve()
    if a.ablation and a.results_root is None:
        base = ROOT / "outputs/quantvla/groot_n1_5/ablations" / ABLATION_DIRS.get(a.ablation, a.ablation)
    environment = "oft_qvla" if family == "openvla_oft_qvla" else ("lerobot_pi05" if family.startswith("pi05") else "groot_test")
    python = a.python or sys.executable  # use the environment selected by conda activate
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    if not (action == "train" and family.startswith("pi05")):
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    def run(cmd, cwd=ROOT, overrides=None):
        current = dict(env, **(overrides or {}))
        print(f"cd {shlex.quote(str(cwd))} && " +
              " ".join(f"{k}={shlex.quote(v)}" for k,v in ({**({"CUDA_VISIBLE_DEVICES": current["CUDA_VISIBLE_DEVICES"]} if "CUDA_VISIBLE_DEVICES" in current else {}), **(overrides or {})}).items()) +
              " " + shlex.join(list(map(str, cmd))), flush=True)
        if not a.dry_run:
            # Give nested launchers time to clean their independently owned services.
            with managed_processes(grace_seconds=300) as processes:
                run_worker(list(map(str, cmd)), cwd=cwd, env=current, processes=processes)
    snapshots = {}
    def config(source, updates=None):
        data = read_config(a.config or ROOT / source)
        if "paths" in data and "root_dir" in data["paths"]:
            data["paths"]["root_dir"] = str(ROOT)
        if updates:
            updates(data)
        content = yaml.safe_dump(data, sort_keys=False)
        target = base / "configs" / (hashlib.sha256(content.encode()).hexdigest()[:16] + ".yaml")
        snapshots[str(target)] = copy.deepcopy(data)
        if a.dry_run:
            print(json.dumps({"generated_config": str(target), "values": data}, ensure_ascii=False))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_text() != content:
                raise ValueError(f"config collision: {target}")
            target.write_text(content)
        return target
    def receipt(out, suite, protocol):
        data = {"family": family, "setting": f"ablations/{ABLATION_DIRS.get(a.ablation, a.ablation)}" if a.ablation else setting, "seed": a.seed, "suite": suite,
                "adapter": str(Path(str(a.adapter_path).replace('{suite}', suite)).expanduser().resolve()) if a.adapter_path else None,
                "extra": extra, "benchmark": "libero-plus", "protocol": protocol}
        path = out / "core_command.json"
        if not a.dry_run:
            if path.exists():
                assert_same(json.loads(path.read_text()), data, f"Evaluation identity {path}")
                if not a.resume:
                    raise ValueError(f"run exists; use --resume or a new seed/results-root: {out}")
            elif any((out / name).exists() for name in ('metrics', 'eval', 'episodes.jsonl', 'run.json')):
                raise ValueError(f"Existing evaluation lacks a protocol receipt; cannot establish compatibility: {out}")
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n")
    suites = SUITES if a.suite == "all" else (a.suite,)
    if action.startswith("service_"):
        if family != "openvla_oft_qvla" or a.suite == "all":
            p.error("OFT service requires one --suite")
        module = ROOT / "openvla-oft-qvla"
        plus = action == "service_plus"
        label, upstream = ("libero_plus", "LIBERO-plus") if plus else ("libero", "LIBERO")
        service_python = a.python or sys.executable  # service commands activate libero_test
        service_config = module / "cache" / label
        if not a.dry_run and not (service_config / "config.yaml").is_file():
            raise FileNotFoundError(f"prepare simulator paths first: {service_config}/config.yaml; see openvla-oft-qvla/commands.md")
        cmd = [service_python, "-m", "evaluation.env_service", "--suite", a.suite,
               "--port", str(a.port if plus else a.clean_port), "--seed", str(a.seed)]
        if plus:
            cmd += ["--manifest", str(ROOT / "manifests/libero_plus_first20.json")]
        run(cmd + extra, module, {"PYTHONPATH": str(ROOT / "third_party" / upstream),
            "LIBERO_CONFIG_PATH": str(service_config), "MUJOCO_GL": "egl"})
        return
    if family == "openvla_oft_qvla":
        if a.suite == "all":
            p.error("OFT requires one --suite per matching simulator service")
        module = ROOT / "openvla-oft-qvla"
        mapped = {"fp_original": "fp", "quant_original": "qvla"}.get(setting, setting)
        out = eval_directory(base, family, setting, a.seed, a.suite) if action == "eval" else base / "train" / f"seed-{a.seed:03d}" / a.suite
        flags = ["--suite", a.suite, "--set", f"environment.port={a.port}",
                 "--set", f"environment.clean_port={a.clean_port}"]
        if a.config:
            p.error("OFT uses its setting config; pass --set key=value overrides after --")
        if action == "calibrate":
            if setting != "quant_original":
                p.error("calibration requires quant_original")
            for cmd in ("collect-calibration", "prepare-qvla"):
                run([python, "-m", "integration.calibrate",
                     "collect" if cmd == "collect-calibration" else "quantize",
                     "--config", "configs/qvla.yaml", *flags, "--set", f"seed={a.seed}", *extra], module)
        else:
            flags += ["--set", f"paths.output={out}", "--set", f"seed={a.seed}"]
            if action == "eval":
                effective = read_config(module / f"configs/{mapped}.yaml")
                for index, value in enumerate(extra):
                    if value == '--set':
                        key, val = extra[index + 1].split('=', 1)
                        node = effective
                        parts = key.split('.')
                        for part in parts[:-1]:
                            node = node[part]
                        node[parts[-1]] = yaml.safe_load(val)
                manifest_path = Path(effective['paths']['manifest']).expanduser()
                if not manifest_path.is_absolute():
                    manifest_path = module / manifest_path
                receipt(out, a.suite, {"config": effective,
                    "manifest": file_value(manifest_path)})
                flags += ["--set", f"evaluation.seed={a.seed}"]
                if recovered:
                    flags += ["--set", f"paths.adapter={a.adapter_path.expanduser().resolve()}"]
            if a.resume_path:
                flags += ["--set", f"training.resume={a.resume_path.expanduser().resolve()}"]
            run([python, "-m", "evaluation.evaluate" if action == "eval" else "training.train",
                 "--config", f"configs/{mapped}.yaml", *flags, *extra], module)
        return
    if action == "calibrate":
        if setting != "quant_original":
            p.error("calibration requires quant_original")
        if family == "pi05_holoqvla":
            cfg = config("pi05_omegavla/config/omega_original.yaml")
            run([python, "pi05_omegavla/scripts/calibrate.py", "--config", cfg, *extra])
        elif family == "pi05_quantvla":
            cfg = config("pi05_quantvla/config/quantvla.yaml")
            run([python, "pi05_quantvla/scripts/build.py", "--config", cfg])
            run([python, "pi05_quantvla/scripts/calibrate.py", "--config", cfg,
                 "--buffer", ROOT / "pi05_quantvla/data/calibration/libero_clean_32_seed-000.pt",
                 "--seed", str(a.seed), *extra])
        else:
            p.error("this family uses existing packs; no standalone calibration wrapper")
        return
    if action == "train":
        sources = {
            "groot_quantvla": f"config/{'full_distill' if setting == 'full_distill' else 'pivot_q'}.yaml",
            "groot_holoqvla": f"omega_qvla/pivot_q/config/{'full_distill' if setting == 'full_distill' else 'pivot_q'}.yaml",
            "pi05_quantvla": f"pi05_quantvla/pivot_q/config/{'full_distill' if setting == 'full_distill' else 'pivot_q'}.yaml",
            "pi05_holoqvla": f"pi05_omegavla/config/{setting}.yaml",
        }
        def train_update(c):
            c["training"]["seed"] = a.seed
            c["paths"]["results_dir"] = str(base / "train")
        cfg = config(sources[family], train_update)
        if family.startswith("pi05"):
            if a.suite != "all":
                p.error("pi05 training jointly covers all four suites; use --suite all")
            entry = "pi05_omegavla/training/run_train.py" if family == "pi05_holoqvla" else "pi05_quantvla/pivot_q/run_train_ddp.py"
            run([python, entry, "--config", cfg, "--gpus", *a.gpus, *extra])
        else:
            entry = ("scripts/train_full_distill.py" if setting == "full_distill" else "scripts/train_pivot_q.py") if family == "groot_quantvla" else "omega_qvla/pivot_q/run_train.py"
            for suite in suites:
                cmd = [python, entry, "--config", cfg, "--suite", suite, "--seed", str(a.seed),
                       "--port", str(a.port), "--clean-port", str(a.clean_port)]
                if family == "groot_holoqvla":
                    cmd += ["--gpu", a.gpu]
                run(cmd + extra)
        return
    sources = {"groot_quantvla": "config/eval.yaml", "groot_holoqvla": "omega_qvla/config/omega_qvla.yaml",
               "pi05_quantvla": "pi05_quantvla/config/quantvla.yaml",
               "pi05_holoqvla": f"pi05_omegavla/config/{'omega_original' if setting == 'quant_original' else setting}.yaml"}
    def eval_update(c):
        if family == "groot_quantvla":
            c["evaluation"]["policy_seed"] = a.seed
    cfg = config(sources[family], eval_update)
    entries = {"groot_quantvla":"scripts/run_eval.py", "groot_holoqvla":"omega_qvla/scripts/run_eval.py",
               "pi05_quantvla":"pi05_quantvla/scripts/run_eval.py", "pi05_holoqvla":"pi05_omegavla/evaluation/run_eval.py"}
    if family == "pi05_holoqvla":
        method = "omega_original" if setting == "quant_original" else setting
    else:
        method = "pivot_q" if recovered else ("fp16" if setting == "fp_original" else ("omega_qvla" if family == "groot_holoqvla" else "quantvla"))
    for suite in suites:
        out = eval_directory(base, family, setting, a.seed, suite)
        cmd = [python, entries[family], "--config", cfg, "--method", method, "--suite", suite,
               "--gpu", a.gpu, "--port", str(a.port), "--output-dir", out]
        if family != "groot_quantvla":
            cmd += ["--seed", str(a.seed)]
        if recovered:
            cmd += ["--adapter-path", str(Path(str(a.adapter_path).replace("{suite}", suite)).expanduser().resolve())]
        if a.resume:
            cmd.append("--resume")
        protocol_config = copy.deepcopy(snapshots[str(cfg)])
        # These affect the execution location, not policy behavior.
        protocol_config.get('evaluation', {}).pop('port', None)
        manifest_value = protocol_config.get('paths', {}).get('manifest')
        manifest_path = Path(manifest_value) if manifest_value else ROOT / 'manifests/libero_plus_first20.json'
        if not manifest_path.is_absolute():
            manifest_path = Path(protocol_config.get('paths', {}).get('root_dir', ROOT)) / manifest_path
        receipt(out, suite, {"config": protocol_config, "manifest": file_value(manifest_path)})
        run(cmd + extra)

if __name__ == "__main__":
    main()
