#!/usr/bin/env python3
"""Serve an Ω-QVLA GR00T-N1.5 policy without modifying the Ω-QVLA checkout."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from paths import load_config


SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


def clear_quant_environment(environment: dict[str, str]) -> None:
    for name in tuple(environment):
        if name.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "GR00T_ATM", "GR00T_OHB")):
            environment.pop(name, None)


def suite_data_config(suite: str) -> str:
    if suite == "libero_goal":
        return "examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
    return "examples.Libero.custom_data_config:LiberoDataConfig"


def enable_pack(environment: dict[str, str], config: dict, suite: str) -> None:
    pack = Path(config["packs"][suite])
    if not pack.is_file():
        raise FileNotFoundError(
            f"Ω-QVLA pack not found: {pack}. Download the matching GR00T W4A4 pack first."
        )
    quantization = config["quantization"]
    environment.update(
        {
            "GR00T_GPTQ": "1",
            "GR00T_GPTQ_PATH": str(pack),
            "GR00T_GPTQ_INCLUDE": str(quantization["include_regex"]),
            "GR00T_GPTQ_EXCLUDE": str(quantization["exclude_regex"]),
            "GR00T_GPTQ_WBITS_DEFAULT": str(quantization["weight_bits"]),
            "GR00T_GPTQ_ABITS": str(quantization["activation_bits"]),
            "GR00T_GPTQ_MISSING": str(quantization["missing_layers"]),
        }
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", metavar="METHOD (fp16 / quantized / pivot-q)", type=lambda value: "pivot_q" if value in ("pivot-q", "pivot_q") else value, choices=("fp16", "omega_qvla", "pivot_q"), required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="*")
    parser.add_argument("--config", type=Path, default=root / "config" / "omega_qvla.yaml")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "pivot_q":
        if args.adapter_path is None:
            raise SystemExit("--adapter-path is required for --method pivot-q")
        args.adapter_path = args.adapter_path.expanduser().resolve()
        if not args.adapter_path.is_dir():
            raise SystemExit(f"adapter directory does not exist: {args.adapter_path}")
    elif args.adapter_path is not None:
        raise SystemExit("--adapter-path is only valid with --method pivot_q")
    config = load_config(args.config)
    omega_root = Path(config["paths"]["omega_qvla_source"])
    if not omega_root.is_dir():
        raise FileNotFoundError(f"Ω-QVLA source checkout not found: {omega_root}")

    sys.path.insert(0, str(omega_root))
    import torch
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    environment = os.environ.copy()
    clear_quant_environment(environment)
    clear_quant_environment(os.environ)
    if args.method in {"omega_qvla", "pivot_q"}:
        enable_pack(environment, config, args.suite)
        os.environ.update({key: value for key, value in environment.items() if key.startswith("GR00T_")})

    class SeededOmegaPolicy(Gr00tPolicy):
        """Ω-QVLA policy with the evaluator's per-step diffusion seed honoured."""

        def get_action(self, observations: dict):
            observations = observations.copy()
            seed = observations.pop("__policy_seed", None)
            if seed is None:
                return super().get_action(observations)
            seed = int(seed)
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
            context = torch.random.fork_rng(devices=[device_index]) if device_index is not None else nullcontext()
            with context:
                torch.manual_seed(seed)
                if device_index is not None:
                    torch.cuda.manual_seed(seed)
                return super().get_action(observations)

    data_config = load_data_config(suite_data_config(args.suite))
    policy = SeededOmegaPolicy(
        model_path=config["models"][args.suite],
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=int(config["evaluation"]["denoising_steps"]),
        device=args.device,
    )
    if args.method == "pivot_q":
        pivot_q_root = Path(config["paths"]["pivot_q_root"])
        sys.path.insert(0, str(pivot_q_root))
        from omega_qvla.pivot_q.gptq_lora import load_adapter

        targets = load_adapter(policy.model, args.adapter_path, trainable=False)
        policy.model.eval()
        print(f"Ω-PIVOT_Q loaded {len(targets)} adapter targets", flush=True)
    policy.warmup(num_steps=int(config["evaluation"]["warmup_steps"]))
    print(f"Ω-QVLA {args.method} ready on port {args.port}", flush=True)
    RobotInferenceServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
