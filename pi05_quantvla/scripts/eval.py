#!/usr/bin/env python3
"""Evaluate local π0.5 FP16 or QuantVLA on the LIBERO-Plus manifest."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROOT_PARENT = ROOT / "third_party"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT_PARENT / "lerobot" / "src"))
sys.path.insert(0, str(ROOT_PARENT / "LIBERO-plus"))
sys.path.insert(0, str(ROOT_PARENT / "QuantVLA"))

from pi05_quantvla.scripts.atm_ohb import AttentionStatistics
from pi05_quantvla.scripts.policy import load_policy
from pi05_quantvla.scripts.paths import resolve_config
from pi05_quantvla.scripts.quantvla import PI05QuantVLAConfig, apply_quantvla_layout


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quaternion[3]) ** 2))
    if denominator < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return quaternion[:3] * (2.0 * math.acos(float(quaternion[3])) / denominator)


class PI05LiberoPolicy:
    """Local π0.5 policy with the interface expected by the LIBERO-Plus evaluator."""

    def __init__(self, *, checkpoint: Path, method: str, quant_config: dict, device: str) -> None:
        self.device = device
        self.policy = load_policy(ROOT, checkpoint, device=device)
        self.preprocessor, self.postprocessor = self._processors(checkpoint)
        self._scale_hooks = None
        if method == "quantvla":
            pack_dir = Path(quant_config["paths"]["quant_pack_dir"])
            apply_quantvla_layout(
                self.policy.model,
                PI05QuantVLAConfig(**quant_config["quantization"]),
                pack_dir=pack_dir,
            )
            scale_path = pack_dir / "atm_ohb.json"
            if not scale_path.is_file():
                raise FileNotFoundError(f"QuantVLA calibration is missing: {scale_path}")
            scales = json.loads(scale_path.read_text(encoding="utf-8"))["scales"]
            self._scale_hooks = AttentionStatistics(self.policy.model, scales=scales, collect=False)

    def _processors(self, checkpoint: Path):
        from lerobot.policies import make_pre_post_processors

        return make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def get_action(self, observation: dict, language: str, policy_seed: int | None = None) -> np.ndarray:
        main = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
                axis_angle(observation["robot0_eef_quat"]),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        sample = {
            "observation.images.image": torch.from_numpy(main).permute(2, 0, 1),
            "observation.images.image2": torch.from_numpy(wrist).permute(2, 0, 1),
            "observation.state": torch.from_numpy(state),
            "task": language,
        }
        batch = self.preprocessor(sample)
        seed = 0 if policy_seed is None else int(policy_seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            (1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
            device=self.device,
            generator=generator,
        )
        with torch.inference_mode():
            action = self.policy.predict_action_chunk(
                batch,
                noise=noise,
                num_steps=self.policy.config.num_inference_steps,
            )[:, 0]
            action = self.postprocessor(action)
        return action[0].detach().float().cpu().numpy()


def load_config(path: Path) -> dict:
    return resolve_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def run_suite(args: argparse.Namespace, suite: str, quant_config: dict) -> None:
    from examples.LiberoPlus.eval import run_libero_plus_eval as libero_eval

    checkpoint = Path(quant_config["paths"]["checkpoint"])
    policy = lambda **_: PI05LiberoPolicy(
        checkpoint=checkpoint,
        method=args.method,
        quant_config=quant_config,
        device=args.device,
    )
    libero_eval.GR00TPolicy = policy
    output_dir = args.output_root / args.method / f"seed-{args.seed:03d}" / suite
    cfg = libero_eval.LiberoPlusEvalConfig(
        task_suite_name=suite,
        headless=True,
        sample_manifest=str(args.manifest),
        policy_seed=args.seed,
        save_video=args.save_video,
        resume=args.resume,
        model_variant=f"pi05-{args.method}",
    )
    previous_output = libero_eval.os.environ.get("LIBERO_EVAL_LOG_DIR")
    libero_eval.os.environ["LIBERO_EVAL_LOG_DIR"] = str(output_dir)
    try:
        libero_eval.evaluate(cfg)
    finally:
        if previous_output is None:
            libero_eval.os.environ.pop("LIBERO_EVAL_LOG_DIR", None)
        else:
            libero_eval.os.environ["LIBERO_EVAL_LOG_DIR"] = previous_output
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fp16", "quantvla"), required=True)
    parser.add_argument("--suite", choices=(*SUITES, "all"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "pi05_quantvla/config/quantvla.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/libero_plus_first20.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/motivation/pi05")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    quant_config = load_config(args.config)
    for suite in SUITES if args.suite == "all" else (args.suite,):
        run_suite(args, suite, quant_config)


if __name__ == "__main__":
    main()
