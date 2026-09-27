"""Unmerged residual LoRA on the frozen OFT regression head."""
from __future__ import annotations

import math
import torch
from torch import nn


class ResidualLinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # FP32 optimizer parameters, with the residual cast back to head dtype.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        residual = torch.nn.functional.linear(self.dropout(x.float()), self.lora_A)
        residual = torch.nn.functional.linear(residual, self.lora_B) * self.scale
        return self.base(x) + residual.to(x.dtype)


def attach(head, settings):
    targets = [(name, module) for name, module in head.named_modules() if isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("no linear modules in OFT action head")
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = head.get_submodule(parent_name) if parent_name else head
        setattr(parent, leaf, ResidualLinear(module, settings["rank"], settings["alpha"], settings["dropout"]))
    return [name for name, _ in targets]


def adapter_state(head):
    return {name: param.detach().cpu().clone() for name, param in head.named_parameters() if param.requires_grad}


def load_adapter(head, state):
    parameters = {name: param for name, param in head.named_parameters() if param.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("adapter parameter names do not match the configured head")
    with torch.no_grad():
        for name, param in parameters.items():
            if param.shape != state[name].shape:
                raise ValueError(f"adapter shape mismatch: {name}")
            param.copy_(state[name].to(param.device))
