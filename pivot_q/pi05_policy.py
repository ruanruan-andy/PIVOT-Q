"""Local loader for the unchanged LeRobot π0.5 checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_FEATURES = {
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
}


def validate_checkpoint(checkpoint: Path) -> dict[str, object]:
    """Validate the local LeRobot checkpoint before allocating the model."""
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise FileNotFoundError(f"incomplete π0.5 checkpoint at {checkpoint}: missing {missing}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05":
        raise ValueError(f"expected a LeRobot pi05 checkpoint, got {config.get('type')!r}")
    features = set(config.get("input_features", {}))
    if not REQUIRED_FEATURES.issubset(features):
        raise ValueError(f"π0.5 checkpoint is missing LIBERO features: {REQUIRED_FEATURES - features}")
    if config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        raise ValueError("π0.5 LIBERO policy must emit 7-D actions")
    return config


def load_policy(repo_root: Path, checkpoint: Path, *, device: str):
    """Load LeRobot without modifying its checkout or requiring global installation."""
    validate_checkpoint(checkpoint)
    source_root = repo_root.resolve() / "third_party" / "lerobot" / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"LeRobot source is missing: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from lerobot.policies.pi05 import PI05Config, PI05Policy

    config = PI05Config.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = device
    config.compile_model = False
    config.gradient_checkpointing = False
    policy = PI05Policy.from_pretrained(str(checkpoint), config=config, local_files_only=True)
    policy.to(device)
    policy.eval()
    return policy
