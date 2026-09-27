#!/usr/bin/env python3
"""CPU-only structural test for the Ω custom residual adapter."""

from __future__ import annotations

import torch
from torch import nn

from omega_qvla.pivot_q.gptq_lora import adapter_modules, inject_adapters


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.attn1 = nn.Module()
        self.action_head.attn1.to_q = nn.Linear(8, 8)


def main() -> None:
    model = Toy()
    targets = inject_adapters(model, rank=2, alpha=4, dropout=0.0)
    assert targets == ["action_head.attn1.to_q"]
    projection = adapter_modules(model)[targets[0]]
    inputs = torch.randn(2, 8)
    projection(inputs).sum().backward()
    assert projection.lora_b.weight.grad is not None
    assert projection.base.weight.grad is None
    print("Ω-PIVOT_Q adapter smoke test passed")


if __name__ == "__main__":
    main()
