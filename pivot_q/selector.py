"""Final GAP-PIVOT_Q selector and its two controlled ablations."""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch
import tyro

import train as train_module
from gr00t.experiment.pivot_q import action_disagreement

TrainConfig = train_module.TrainConfig


def future_risk(q: torch.Tensor, horizon: int, discount: float) -> torch.Tensor:
    risk = torch.zeros_like(q)
    for timestep in range(len(q) - 1):
        steps = min(horizon, len(q) - timestep - 1)
        weights = q.new_tensor(discount).pow(torch.arange(steps, device=q.device, dtype=q.dtype))
        risk[timestep] = (q[timestep + 1 : timestep + 1 + steps] * weights).sum() / weights.sum()
    return risk


def build_targets(student_actions: torch.Tensor, teacher_actions: torch.Tensor, config: TrainConfig, *, action_dims: int):
    q = action_disagreement(student_actions, teacher_actions, action_dims=action_dims).detach()
    return q, future_risk(q, config.temporal_horizon, config.temporal_discount).detach(), torch.ones_like(q)


def percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.float(), stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, len(values), dtype=torch.float32)
    return ranks


def select_phase(pool: list[int], scores: torch.Tensor, count: int, gap: int) -> tuple[list[int], int]:
    ordered = sorted(pool, key=lambda index: (-float(scores[index]), index))
    for effective_gap in range(gap, -1, -1):
        chosen: list[int] = []
        for index in ordered:
            if effective_gap == 0 or all(abs(index - other) >= effective_gap for other in chosen):
                chosen.append(index)
                if len(chosen) == count:
                    return chosen, effective_gap
    raise AssertionError("phase cannot satisfy the top-state quota")


def _legacy_select_states(q: torch.Tensor, r: torch.Tensor, config: TrainConfig, _rng: random.Random):
    if config.random_per_phase:
        raise ValueError("final GAP-PIVOT_Q uses top states only")
    phase_ids = [min(config.phase_bins - 1, index * config.phase_bins // len(q)) for index in range(len(q))]
    scores = torch.zeros(len(q), dtype=torch.float32)
    indices: list[int] = []
    phases: list[int] = []
    reasons: list[str] = []
    gaps: list[int] = []
    for phase in range(config.phase_bins):
        pool = [index for index, phase_id in enumerate(phase_ids) if phase_id == phase]
        pool_tensor = torch.tensor(pool, dtype=torch.long)
        scores[pool_tensor] = (
            config.alpha_q * percentile_ranks(q[pool_tensor])
            + config.beta_r * percentile_ranks(r[pool_tensor])
        )
        chosen, effective_gap = select_phase(pool, scores, config.priority_per_phase, config.min_temporal_gap)
        indices.extend(chosen)
        phases.extend([phase] * len(chosen))
        reasons.extend([f"top_phase_{phase}"] * len(chosen))
        gaps.append(effective_gap)
    order = sorted(range(len(indices)), key=lambda position: indices[position])
    indices = [indices[position] for position in order]
    phases = [phases[position] for position in order]
    reasons = [reasons[position] for position in order]
    selected_scores = scores[torch.tensor(indices, dtype=torch.long)]
    weights = (selected_scores / selected_scores.mean().clamp_min(1e-8)).clamp(config.weight_min, config.weight_max)
    expected = config.phase_bins * config.priority_per_phase
    if len(indices) != expected or any(phases.count(phase) != config.priority_per_phase for phase in range(config.phase_bins)):
        raise AssertionError("final GAP-PIVOT_Q selection quota violation")
    return SimpleNamespace(
        indices=tuple(indices),
        weights=weights.detach(),
        phases=tuple(phases),
        reasons=tuple(reasons),
        priority_scores=selected_scores.detach(),
        phase_effective_gaps=tuple(gaps),
        actual_min_gap=min((right - left for left, right in zip(indices, indices[1:])), default=0),
        gap_relaxed=any(gap < config.min_temporal_gap for gap in gaps),
        phase_counts=[phases.count(phase) for phase in range(config.phase_bins)],
        priority_counts=[reasons.count(f"top_phase_{phase}") for phase in range(config.phase_bins)],
        random_counts=[0] * config.phase_bins,
        valid=True,
    )


def select_states(q: torch.Tensor, r: torch.Tensor, config: TrainConfig, rng: random.Random):
    """Paper baseline dispatch; absent fields retain the original exact path."""
    from sparse_selection import select_sparse
    return select_sparse(q, r, config, rng, _legacy_select_states,
                         percentile_ranks, select_phase)


def main() -> None:
    config = tyro.cli(TrainConfig)
    if config.method_name not in {"pivot_q", "pivot_q_a03_b07", "pivot_q_a04_b06", "pivot_q_a07_b03", "pivot_q_a03_b07_no_anchor", "pivot_q_current_only", "pivot_q_no_anchor", "random_sparse", "uniform_sparse", "global_random_sparse", "global_vulnerability"}:
        raise ValueError("unsupported GAP-PIVOT_Q experiment")
    train_module.build_pivot_q_targets = build_targets
    train_module.train(config, select_states)


if __name__ == "__main__":
    main()
