"""Phase-balanced prioritized sampling for PIVOT-Q trajectories."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .config import PIVOTQSelectionConfig


@dataclass(frozen=True)
class SelectionResult:
    """Selected trajectory states and the metadata needed for logging."""

    indices: tuple[int, ...]
    phases: tuple[int, ...]
    reasons: tuple[str, ...]
    priority_scores: torch.Tenso
    weights: torch.Tenso
    phase_effective_gaps: tuple[int, ...]
    actual_min_gap: int
    gap_relaxed: bool


def _percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic ranks in ``[0, 1]`` with average tied ranks."""
    if values.ndim != 1:
        raise ValueError("percentile ranks require a one-dimensional tensor")
    count = values.numel()
    if count == 0:
        return values.clone().float()
    if count == 1:
        return torch.ones_like(values, dtype=torch.float32)

    values = values.float()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(count, dtype=torch.float32, device=values.device)
    start = 0
    while start < count:
        end = start + 1
        while end < count and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + end - 1) / 2.0) / (count - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _phase_for_index(index: int, trajectory_length: int, phase_bins: int) -> int:
    return min(phase_bins - 1, index * phase_bins // trajectory_length)


def _respects_gap(index: int, selected: list[int], gap: int) -> bool:
    return gap == 0 or all(abs(index - other) >= gap for other in selected)


def _take_candidates(
    candidates: list[int],
    *,
    count: int,
    reason: str,
    selected: list[int],
    reasons: dict[int, str],
    min_gap: int,
) -> list[int]:
    chosen: list[int] = []
    for index in candidates:
        if index in reasons or not _respects_gap(index, selected, min_gap):
            continue
        selected.append(index)
        chosen.append(index)
        reasons[index] = reason
        count -= 1
        if count == 0:
            break
    return chosen


def _select_phase(
    candidates: list[int],
    scores: torch.Tensor,
    config: PIVOTQSelectionConfig,
    *,
    phase: int,
    selected_before_phase: list[int],
    phase_seed: int,
) -> tuple[list[int], dict[int, str], int]:
    """Select an exact priority/random quota from one temporal phase.

    The target gap is tried first. If the exact phase quota is infeasible, the
    gap is reduced for this phase only. Candidate states never spill into a
    different phase, and gap 1 guarantees a solution whenever the phase has
    enough distinct states.
    """
    quota = config.priority_per_phase + config.random_per_phase
    if len(candidates) < quota:
        raise ValueError(
            f"trajectory phase {phase} has {len(candidates)} states; "
            f"at least {quota} are required"
        )

    target_gap = config.min_temporal_gap
    gaps = [0] if target_gap == 0 else list(range(target_gap, 0, -1))
    priority_order = sorted(candidates, key=lambda index: (-float(scores[index]), index))
    for effective_gap in gaps:
        selected = list(selected_before_phase)
        reasons: dict[int, str] = {}
        priority = _take_candidates(
            priority_order,
            count=config.priority_per_phase,
            reason=f"priority_phase_{phase}",
            selected=selected,
            reasons=reasons,
            min_gap=effective_gap,
        )
        if len(priority) != config.priority_per_phase:
            continue

        random_candidates = [index for index in candidates if index not in reasons]
        random.Random(phase_seed).shuffle(random_candidates)
        random_states = _take_candidates(
            random_candidates,
            count=config.random_per_phase,
            reason=f"random_phase_{phase}",
            selected=selected,
            reasons=reasons,
            min_gap=effective_gap,
        )
        if len(random_states) != config.random_per_phase:
            continue
        phase_selected = priority + random_states
        if len(phase_selected) != quota:
            raise AssertionError("phase selector returned an invalid quota")
        return phase_selected, reasons, effective_gap

    raise AssertionError(f"unable to satisfy phase {phase} quota even with gap 1")


def select_phase_balanced_states(
    q_values: torch.Tensor,
    r_values: torch.Tensor,
    config: PIVOTQSelectionConfig,
    *,
    rng: random.Random,
) -> SelectionResult:
    """Select a deterministic priority/random mixture across temporal bins.

    Every temporal phase receives its exact priority/random quota. The target
    temporal gap is enforced whenever feasible and otherwise reduced inside
    the constrained phase; selection never spills into another phase.
    """
    config.validate()
    if q_values.ndim != 1 or r_values.ndim != 1 or q_values.shape != r_values.shape:
        raise ValueError("q_values and r_values must be matching 1D tensors")
    if not torch.isfinite(q_values).all() or not torch.isfinite(r_values).all():
        raise ValueError("q_values and r_values must be finite")
    if torch.any(q_values < 0) or torch.any(r_values < 0):
        raise ValueError("q_values and r_values must be non-negative")

    trajectory_length = q_values.numel()
    if trajectory_length == 0:
        return SelectionResult((), (), (), torch.empty(0), torch.empty(0), (), 0, False)
    if trajectory_length < config.states_per_episode:
        raise ValueError(
            f"trajectory has {trajectory_length} states; "
            f"at least {config.states_per_episode} are required"
        )

    phase_ids = [
        _phase_for_index(index, trajectory_length, config.phase_bins)
        for index in range(trajectory_length)
    ]
    scores = torch.zeros(trajectory_length, dtype=torch.float32)
    phase_candidates: list[list[int]] = []
    for phase in range(config.phase_bins):
        candidates = [index for index, value in enumerate(phase_ids) if value == phase]
        phase_candidates.append(candidates)
        if not candidates:
            continue
        candidate_tensor = torch.as_tensor(candidates, dtype=torch.long)
        q_rank = _percentile_ranks(q_values[candidate_tensor])
        r_rank = _percentile_ranks(r_values[candidate_tensor])
        scores[candidate_tensor] = config.alpha_q * q_rank + config.beta_r * r_rank

    selected: list[int] = []
    reason_by_index: dict[int, str] = {}
    phase_effective_gaps: list[int] = []
    for phase, candidates in enumerate(phase_candidates):
        phase_selected, phase_reasons, effective_gap = _select_phase(
            candidates,
            scores,
            config,
            phase=phase,
            selected_before_phase=selected,
            phase_seed=rng.getrandbits(64),
        )
        selected.extend(phase_selected)
        reason_by_index.update(phase_reasons)
        phase_effective_gaps.append(effective_gap)

    if len(selected) != config.states_per_episode:
        raise AssertionError("selector did not satisfy the configured state budget")

    selected.sort()
    actual_min_gap = min(
        (right - left for left, right in zip(selected, selected[1:])),
        default=0,
    )
    expected_per_phase = config.priority_per_phase + config.random_per_phase
    selected_phases = tuple(phase_ids[index] for index in selected)
    if any(selected_phases.count(phase) != expected_per_phase for phase in range(config.phase_bins)):
        raise AssertionError("selector violated a temporal phase quota")
    if any(
        sum(reason_by_index[index] == f"priority_phase_{phase}" for index in selected)
        != config.priority_per_phase
        or sum(reason_by_index[index] == f"random_phase_{phase}" for index in selected)
        != config.random_per_phase
        for phase in range(config.phase_bins)
    ):
        raise AssertionError("selector violated a priority/random quota")

    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    selected_scores = scores[selected_tensor]
    mean_score = selected_scores.mean().clamp_min(config.epsilon)
    weights = (selected_scores / mean_score).clamp(config.weight_min, config.weight_max)
    return SelectionResult(
        indices=tuple(selected),
        phases=selected_phases,
        reasons=tuple(reason_by_index[index] for index in selected),
        priority_scores=selected_scores.detach(),
        weights=weights.detach(),
        phase_effective_gaps=tuple(phase_effective_gaps),
        actual_min_gap=actual_min_gap,
        gap_relaxed=any(gap < config.min_temporal_gap for gap in phase_effective_gaps),
    )
