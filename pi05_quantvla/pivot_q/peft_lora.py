"""PEFT target discovery for π0.5's floating-point action expert."""

from __future__ import annotations

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn


def action_expert_targets(model: nn.Module) -> list[str]:
    prefix = "paligemma_with_expert.gemma_expert.model.layers."
    suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    targets = [
        name for name, module in model.named_modules()
        if name.startswith(prefix) and name.endswith(suffixes) and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise RuntimeError("no π0.5 action-expert Q/K/V projections were found")
    return targets


def attach(model: nn.Module, *, rank: int, alpha: int, dropout: float):
    targets = action_expert_targets(model)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    return get_peft_model(model, config), targets


def load(model: nn.Module, adapter_path: str, *, trainable: bool):
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)

