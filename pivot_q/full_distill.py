"""All-state, uniform-weight action distillation."""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch
import tyro

from train import TrainConfig, train


def select_states(q: torch.Tensor, _r: torch.Tensor, _config: TrainConfig, _rng: random.Random):
    indices = list(range(len(q)))
    return SimpleNamespace(
        indices=indices,
        weights=torch.ones(len(indices), dtype=torch.float32),
        phases=[-1] * len(indices),
        reasons=["all"] * len(indices),
        priority_scores=q,
        phase_counts=[],
        priority_counts=[],
        random_counts=[],
        phase_effective_gaps=[],
        actual_min_gap=None,
        gap_relaxed=False,
        valid=True,
    )


def main() -> None:
    config = tyro.cli(TrainConfig)
    config.method_name = "full_distill"
    train(config, select_states)


if __name__ == "__main__":
    main()
