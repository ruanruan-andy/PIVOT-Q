#!/usr/bin/env python3
"""Serve local π0.5 FP16 or QuantVLA actions to a LIBERO evaluator process."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from multiprocessing.connection import Listener
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pi05_quantvla.scripts.atm_ohb import AttentionStatistics
from pi05_quantvla.scripts.policy import load_policy
from pi05_quantvla.scripts.paths import resolve_config
from pi05_quantvla.scripts.quantvla import PI05QuantVLAConfig, apply_quantvla_layout

class PI05Server:
    def __init__(
        self, method: str, config: dict, device: str, adapter_path: Path | None = None
    ) -> None:
        checkpoint = Path(config["paths"]["checkpoint"])
        self.device = device
        self.policy = load_policy(ROOT, checkpoint, device=device)
        from lerobot.policies import make_pre_post_processors

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.scale_hooks = None
        if method in {"quantvla", "pivot_q"}:
            pack_dir = Path(config["paths"]["quant_pack_dir"])
            apply_quantvla_layout(
                self.policy.model,
                PI05QuantVLAConfig(**config["quantization"]),
                pack_dir=pack_dir,
            )
            scales_path = pack_dir / "atm_ohb.json"
            if not scales_path.is_file():
                raise FileNotFoundError(f"QuantVLA calibration is missing: {scales_path}")
            scales = json.loads(scales_path.read_text(encoding="utf-8"))["scales"]
            self.scale_hooks = AttentionStatistics(self.policy.model, scales=scales, collect=False)
        if method == "pivot_q":
            if adapter_path is None:
                raise ValueError("adapter_path is required for PIVOT_Q inference")
            from peft import PeftModel

            self.policy.model = PeftModel.from_pretrained(
                self.policy.model, adapter_path, is_trainable=False
            )
            self.policy.model.eval()

    def actions(self, request: dict) -> torch.Tensor:
        sample = {
            "observation.images.image": torch.from_numpy(request["image"]),
            "observation.images.image2": torch.from_numpy(request["wrist_image"]),
            "observation.state": torch.from_numpy(request["state"]),
            "task": request["task"],
        }
        batch = self.preprocessor(sample)
        generator = torch.Generator(device=self.device).manual_seed(int(request["seed"]))
        noise = torch.randn(
            (1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
            device=self.device,
            generator=generator,
        )
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(
                batch, noise=noise, num_steps=self.policy.config.num_inference_steps
            )[0, : self.policy.config.n_action_steps]
            return self.postprocessor(actions).detach().float().cpu()

    def close(self) -> None:
        if self.scale_hooks is not None:
            self.scale_hooks.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: "pivot_q" if value in ("pivot-q", "pivot_q") else value, choices=("fp16", "quantvla", "pivot_q"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "pi05_quantvla/config/quantvla.yaml")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5700)
    parser.add_argument("--authkey", default="pi05")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.method == "pivot_q":
        if args.adapter_path is None:
            parser.error("--adapter-path is required for --method pivot-q")
        args.adapter_path = args.adapter_path.expanduser().resolve()
        if not args.adapter_path.is_dir():
            parser.error(f"adapter directory does not exist: {args.adapter_path}")
    elif args.adapter_path is not None:
        parser.error("--adapter-path is only valid with --method pivot_q")
    config = resolve_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    server = PI05Server(args.method, config, args.device, args.adapter_path)
    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    print(f"π0.5 {args.method} inference server listening on {args.host}:{args.port}", flush=True)
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    if request.get("type") == "shutdown":
                        return
                    connection.send({"actions": server.actions(request).numpy()})
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        server.close()


if __name__ == "__main__":
    main()
