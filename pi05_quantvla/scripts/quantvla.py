"""QuantVLA's π0.5 selective W4A8 layout.

This module contains only PIVOT_Q-owned glue around the vendored DuQuant layers.
It implements the paper's final layout exactly: all PaliGemma language-model
linears and only the MLP linears of the action-expert DiT are W4A8.  The vision
tower, embeddings, normalizers, action/time projections and DiT Q/K/V/O remain
in floating point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch.nn as nn

from .duquant_layers import DuQuantConfig, wrap_duquant


@dataclass(frozen=True)
class PI05QuantVLAConfig:
    """Paper-aligned PTQ settings for OpenPI π0.5."""

    weight_bits: int = 4
    activation_bits: int = 8
    block_size: int = 16
    lambda_smooth: float = 0.15
    activation_percentile: float = 99.9
    calibration_batches: int = 32
    enable_permutation: bool = True
    row_rotation: str = "restore"
    atm_enabled: bool = True
    ohb_enabled: bool = True
    log_scale_clip: float = 0.4
    neutrality_band: float = 0.03

    def validate(self) -> None:
        if (self.weight_bits, self.activation_bits) != (4, 8):
            raise ValueError("π0.5 QuantVLA is fixed to the paper's W4A8 setting")
        if self.block_size <= 0 or self.calibration_batches <= 0:
            raise ValueError("block_size and calibration_batches must be positive")
        if self.row_rotation not in {"restore", "propagate", "0"}:
            raise ValueError("row_rotation must be restore, propagate, or 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_llm_linear(name: str) -> bool:
    """Every Linear in π0.5's PaliGemma language backbone, not its vision tower."""
    return name.startswith(
        "paligemma_with_expert.paligemma.model.language_model."
    ) and not name.endswith("lm_head")


def _is_dit_mlp_linear(name: str) -> bool:
    """Only action-expert Gemma MLP projections; Q/K/V/O deliberately excluded."""
    prefix = "paligemma_with_expert.gemma_expert.model.layers."
    return name.startswith(prefix) and ".mlp." in name and name.endswith(
        ("gate_proj", "up_proj", "down_proj")
    )


def target_layer_names(model: nn.Module) -> list[str]:
    """Return the exact QuantVLA paper layout for a loaded ``PI05Policy.model``."""
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (_is_llm_linear(name) or _is_dit_mlp_linear(name))
    ]
    if not targets:
        raise RuntimeError("no π0.5 QuantVLA target layers found; unsupported LeRobot π0.5 layout")
    forbidden = (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj")
    bad_dit = [name for name in targets if ".gemma_expert." in name and name.endswith(forbidden)]
    if bad_dit:
        raise AssertionError(f"DiT attention must remain FP: {bad_dit}")
    return targets


def apply_quantvla_layout(
    policy_model: nn.Module,
    config: PI05QuantVLAConfig,
    *,
    pack_dir: Path,
    dry_run: bool = False,
) -> list[str]:
    """Replace the selected π0.5 linears with PIVOT_Q-vendored DuQuant W4A8 layers."""
    config.validate()
    targets = target_layer_names(policy_model)
    duquant = DuQuantConfig(
        weight_bits=config.weight_bits,
        act_bits=config.activation_bits,
        block_size=config.block_size,
        block_out_size=config.block_size,
        lambda_smooth=config.lambda_smooth,
        enable_permute=config.enable_permutation,
        act_percentile=config.activation_percentile,
        calib_batches=config.calibration_batches,
        pack_dir=str(pack_dir),
        row_rot_mode=config.row_rotation,
    )
    wrap_duquant(policy_model, targets, duquant, dry_run=dry_run)
    return targets


def assert_paper_layout(targets: Iterable[str]) -> None:
    """Fail early if a caller accidentally quantizes DiT attention projections."""
    bad = [
        name
        for name in targets
        if ".gemma_expert.model.layers." in name
        and any(part in name for part in ("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    if bad:
        raise AssertionError(f"paper π0.5 layout keeps DiT Q/K/V/O in FP: {bad}")
