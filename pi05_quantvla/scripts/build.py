#!/usr/bin/env python3
"""Build the paper-aligned π0.5 QuantVLA W4A8 pack inside PIVOT_Q."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pi05_quantvla.scripts.policy import load_policy, validate_checkpoint
from pi05_quantvla.scripts.paths import resolve_config
from pi05_quantvla.scripts.quantvla import PI05QuantVLAConfig, apply_quantvla_layout, assert_paper_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "pi05_quantvla/config/quantvla.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="list target layers without replacing them")
    args = parser.parse_args()

    raw = resolve_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    checkpoint = Path(raw["paths"]["checkpoint"])
    pack_dir = Path(raw["paths"]["quant_pack_dir"])
    checkpoint_config = validate_checkpoint(checkpoint)
    quant = PI05QuantVLAConfig(**raw["quantization"])
    quant.validate()

    policy = load_policy(REPO_ROOT, checkpoint, device=args.device)
    targets = apply_quantvla_layout(policy.model, quant, pack_dir=pack_dir, dry_run=args.dry_run)
    assert_paper_layout(targets)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_inference_steps": checkpoint_config["num_inference_steps"],
        "checkpoint_action_steps": checkpoint_config["n_action_steps"],
        "quantization": quant.to_dict(),
        "target_count": len(targets),
        "targets": targets,
        "calibration_required": True,
        "note": "Run calibration before using this pack for evaluation.",
    }
    if not args.dry_run:
        pack_dir.mkdir(parents=True, exist_ok=True)
        (pack_dir / "quantvla_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
