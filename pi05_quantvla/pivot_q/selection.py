"""Model-independent GAP-PIVOT_Q scoring and phase-local state selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Selection:
    indices: tuple[int, ...]
    weights: torch.Tensor
    phases: tuple[int, ...]
    effective_gaps: tuple[int, ...]


def action_error(student: torch.Tensor, teacher: torch.Tensor, action_dims: int = 7) -> torch.Tensor:
    return (student[:, 0, :action_dims].float() - teacher[:, 0, :action_dims].float()).square().mean(-1)


def future_risk(error: torch.Tensor, horizon: int, discount: float) -> torch.Tensor:
    risk = torch.zeros_like(error)
    for index in range(len(error) - 1):
        count = min(horizon, len(error) - index - 1)
        weights = error.new_tensor(discount).pow(torch.arange(count, device=error.device))
        risk[index] = (error[index + 1:index + 1 + count] * weights).sum() / weights.sum()
    return risk


def ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.float(), stable=True)
    result = torch.empty_like(values, dtype=torch.float32)
    result[order] = torch.linspace(0.0, 1.0, len(values))
    return result


def select_all(error: torch.Tensor) -> Selection:
    """Paper dense baseline: supervise every valid rollout state uniformly."""
    if len(error) == 0:
        raise ValueError("dense distillation requires a nonempty rollout")
    return Selection(
        indices=tuple(range(len(error))),
        weights=torch.ones(len(error), dtype=torch.float32),
        phases=tuple(-1 for _ in range(len(error))),
        effective_gaps=(),
    )


def select(error: torch.Tensor, risk: torch.Tensor, *, phase_bins: int, top_per_phase: int,
           min_gap: int, alpha: float, beta: float, weight_min: float, weight_max: float) -> Selection:
    phase_ids = [min(phase_bins - 1, i * phase_bins // len(error)) for i in range(len(error))]
    scores = torch.zeros(len(error), dtype=torch.float32)
    chosen: list[int] = []
    phases: list[int] = []
    effective_gaps: list[int] = []
    for phase in range(phase_bins):
        pool = [i for i, value in enumerate(phase_ids) if value == phase]
        pool_tensor = torch.tensor(pool, dtype=torch.long)
        scores[pool_tensor] = alpha * ranks(error[pool_tensor]) + beta * ranks(risk[pool_tensor])
        ordered = sorted(pool, key=lambda i: (-float(scores[i]), i))
        phase_choice: list[int] = []
        used_gap = 0
        for gap in range(min_gap, -1, -1):
            phase_choice = []
            for index in ordered:
                if gap == 0 or all(abs(index - other) >= gap for other in phase_choice):
                    phase_choice.append(index)
                    if len(phase_choice) == top_per_phase:
                        used_gap = gap
                        break
            if len(phase_choice) == top_per_phase:
                break
        if len(phase_choice) != top_per_phase:
            raise RuntimeError(f"phase {phase} cannot satisfy its PIVOT_Q quota")
        chosen.extend(phase_choice)
        phases.extend([phase] * len(phase_choice))
        effective_gaps.append(used_gap)
    order = sorted(range(len(chosen)), key=lambda position: chosen[position])
    chosen = [chosen[position] for position in order]
    phases = [phases[position] for position in order]
    selected_scores = scores[torch.tensor(chosen)]
    weights = (selected_scores / selected_scores.mean().clamp_min(1e-8)).clamp(weight_min, weight_max)
    return Selection(tuple(chosen), weights.detach(), tuple(phases), tuple(effective_gaps))
