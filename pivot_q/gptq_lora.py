"""Low-rank residual adapters for Ω-QVLA's custom GPTQ linear layers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class GPTQResidualLoRA(nn.Module):
    """Compute ``quantized_linear(x) + scale * B(A(dropout(x)))``."""

    def __init__(self, base: nn.Module, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = rank
        self.alpha = alpha
        self.dropout_probability = dropout
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(self.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, self.out_features, bias=False)
        reference = getattr(base, "bias", None)
        if reference is None:
            reference = next(base.buffers(), None)
        if reference is not None:
            self.lora_a.to(device=reference.device)
            self.lora_b.to(device=reference.device)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = self.base(inputs, *args, **kwargs)
        residual_input = inputs.to(dtype=self.lora_a.weight.dtype)
        residual = self.lora_b(self.lora_a(self.dropout(residual_input)))
        return output + residual.to(dtype=output.dtype) * self.scaling


def _parent_and_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def action_attention_targets(model: nn.Module) -> list[str]:
    """Return every Q/K/V projection in the GR00T action head."""
    suffixes = ("attn1.to_q", "attn1.to_k", "attn1.to_v")
    targets = [
        name
        for name, module in model.named_modules()
        if "action_head" in name
        and name.endswith(suffixes)
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and not isinstance(module, GPTQResidualLoRA)
    ]
    if not targets:
        raise RuntimeError("no Ω-QVLA action-head Q/K/V projections were found")
    return targets


def inject_adapters(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    targets: list[str] | None = None,
) -> list[str]:
    """Freeze the base policy and install residual adapters at exact targets."""
    model.requires_grad_(False)
    selected = targets or action_attention_targets(model)
    for name in selected:
        parent, attribute = _parent_and_attribute(model, name)
        base = getattr(parent, attribute)
        if not isinstance(base, GPTQResidualLoRA):
            setattr(parent, attribute, GPTQResidualLoRA(base, rank, alpha, dropout))
    return selected


def adapter_modules(model: nn.Module) -> dict[str, GPTQResidualLoRA]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, GPTQResidualLoRA)
    }


def save_adapter(model: nn.Module, directory: str | Path) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    modules = adapter_modules(model)
    if not modules:
        raise RuntimeError("cannot save Ω-PIVOT_Q: no residual adapters are installed")
    first = next(iter(modules.values()))
    manifest = {
        "format": "omega_pivot_q_residual_lora_v1",
        "rank": first.rank,
        "alpha": first.alpha,
        "dropout": first.dropout_probability,
        "targets": list(modules),
    }
    state = {
        f"{name}.{parameter}": tensor.detach().cpu()
        for name, module in modules.items()
        for parameter, tensor in module.state_dict().items()
        if parameter.startswith("lora_")
    }
    (output / "adapter_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    torch.save(state, output / "adapter_model.pt")


def load_adapter(model: nn.Module, directory: str | Path, *, trainable: bool) -> list[str]:
    source = Path(directory)
    manifest = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in {
        "omega_pivot_q_residual_lora_v1",
        "omega_opdq_residual_lora_v1",  # Read-only compatibility with existing adapters.
    }:
        raise ValueError(f"unsupported Ω-PIVOT_Q adapter format in {source}")
    targets = [str(name) for name in manifest["targets"]]
    inject_adapters(
        model,
        rank=int(manifest["rank"]),
        alpha=int(manifest["alpha"]),
        dropout=float(manifest["dropout"]),
        targets=targets,
    )
    state = torch.load(source / "adapter_model.pt", map_location="cpu", weights_only=True)
    modules = adapter_modules(model)
    for name in targets:
        prefix = f"{name}."
        local = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        modules[name].load_state_dict(local, strict=False)
        modules[name].requires_grad_(trainable)
        modules[name].base.requires_grad_(False)
    return targets


class OmegaAdapterModel(nn.Module):
    """Compatibility shell for the PEFT methods used by the shared trainer."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def get_base_model(self) -> nn.Module:
        return self.base_model

    def save_pretrained(self, directory: str | Path) -> None:
        save_adapter(self.base_model, directory)

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        ratio = 100.0 * trainable / max(total, 1)
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {ratio:.4f}")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


class OmegaPeftCompatibility:
    """Implement the one PEFT resume call made by the shared trainer."""

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        directory: str | Path,
        *,
        is_trainable: bool = False,
    ) -> OmegaAdapterModel:
        """Restore HoloQ residual adapters while preserving the shared resume path."""
        load_adapter(model, directory, trainable=is_trainable)
        return OmegaAdapterModel(model)
