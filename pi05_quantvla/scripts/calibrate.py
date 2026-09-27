#!/usr/bin/env python3
"""Calibrate ATM/OHB from a frozen clean-LIBERO buffer."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pi05_quantvla.scripts.atm_ohb import AttentionStatistics, compute_scales
from pi05_quantvla.scripts.policy import load_policy
from pi05_quantvla.scripts.paths import resolve_config
from pi05_quantvla.scripts.quantvla import PI05QuantVLAConfig, apply_quantvla_layout

def batches(policy, samples, checkpoint, device, seed):
    from lerobot.policies import make_pre_post_processors
    pre, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    for i, sample in enumerate(samples):
        batch = pre(sample)
        generator = torch.Generator(device=device).manual_seed(seed + i)
        noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), device=device, generator=generator)
        yield batch, noise

def collect(policy, samples, checkpoint, device, seed):
    observer = AttentionStatistics(policy.model)
    with torch.no_grad():
        for batch, noise in batches(policy, samples, checkpoint, device, seed):
            policy.predict_action_chunk(batch, noise=noise, num_steps=policy.config.num_inference_steps)
    stats = observer.finalize(); observer.close(); return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "pi05_quantvla/config/quantvla.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import yaml
    raw = resolve_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    checkpoint = Path(raw["paths"]["checkpoint"])
    pack = Path(raw["paths"]["quant_pack_dir"])
    samples = torch.load(args.buffer, weights_only=False)["samples"]

    teacher = load_policy(ROOT, checkpoint, device=args.device)
    teacher_stats = collect(teacher, samples, checkpoint, args.device, args.seed)
    del teacher
    torch.cuda.empty_cache()

    student = load_policy(ROOT, checkpoint, device=args.device)
    quant = PI05QuantVLAConfig(**raw["quantization"])
    apply_quantvla_layout(student.model, quant, pack_dir=pack)
    student_stats = collect(student, samples, checkpoint, args.device, args.seed)
    result = {
        "buffer": str(args.buffer),
        "seed": args.seed,
        "scales": compute_scales(
            teacher_stats,
            student_stats,
            log_clip=raw["quantization"]["log_scale_clip"],
            neutral=raw["quantization"]["neutrality_band"],
        ),
    }
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "atm_ohb.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_path = pack / "quantvla_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["calibration_required"] = False
        manifest["calibration"] = {"buffer": str(args.buffer), "seed": args.seed, "scale_file": "atm_ohb.json"}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pack / "atm_ohb.json")
if __name__ == "__main__": main()
