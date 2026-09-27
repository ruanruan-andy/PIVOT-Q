"""Core losses and trajectory utilities for PIVOT-Q.

The functions in this module are model- and simulator-independent so the method
can be tested without loading GR00T or LIBERO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PIVOTQConfig:
    """Hyperparameters that define the fixed-horizon PIVOT-Q objective."""

    temporal_horizon: int = 4
    temporal_discount: float = 0.9
    alpha_q: float = 1.0
    beta_r: float = 1.0
    weight_min: float = 1.0
    weight_max: float = 5.0
    normalization_epsilon: float = 1e-8

    def validate(self) -> None:
        if self.temporal_horizon < 0:
            raise ValueError("temporal_horizon must be non-negative")
        if not 0.0 < self.temporal_discount <= 1.0:
            raise ValueError("temporal_discount must be in (0, 1]")
        if self.alpha_q < 0.0 or self.beta_r < 0.0:
            raise ValueError("alpha_q and beta_r must be non-negative")
        if self.weight_min <= 0.0 or self.weight_max < self.weight_min:
            raise ValueError("invalid critical-state weight bounds")


def action_disagreement(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    executed_action_index: int = 0,
    action_dims: int = 7,
) -> torch.Tensor:
    """Return per-state MSE on the continuous action executed by LIBERO.

    Inputs may have shape ``[T, action_horizon, action_dim]`` or
    ``[T, action_dim]``.  For an action chunk, only the action selected for
    environment execution is compared.
    """
    if student_actions.shape != teacher_actions.shape:
        raise ValueError(
            f"student/teacher shapes differ: {student_actions.shape} vs "
            f"{teacher_actions.shape}"
        )
    if student_actions.ndim == 3:
        student_actions = student_actions[:, executed_action_index, :action_dims]
        teacher_actions = teacher_actions[:, executed_action_index, :action_dims]
    elif student_actions.ndim == 2:
        student_actions = student_actions[:, :action_dims]
        teacher_actions = teacher_actions[:, :action_dims]
    else:
        raise ValueError("actions must have shape [T,D] or [T,H,D]")
    return (student_actions.float() - teacher_actions.float()).square().mean(dim=-1)


def temporal_quantization_risk(
    disagreement: torch.Tensor,
    *,
    horizon: int,
    discount: float,
) -> torch.Tensor:
    """Compute normalized discounted future disagreement for every state.

    For state ``t`` this is
    ``sum_j discount**j * q[t+j] / sum_j discount**j`` with the window
    truncated at the actual trajectory end.  The normalization avoids a
    systematic low score for states near the end of an episode.
    """
    if disagreement.ndim != 1:
        raise ValueError("disagreement must be a one-dimensional trajectory")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    if disagreement.numel() == 0:
        return disagreement.clone()

    risks = []
    trajectory_length = disagreement.shape[0]
    for timestep in range(trajectory_length):
        steps = min(horizon + 1, trajectory_length - timestep)
        powers = torch.arange(steps, device=disagreement.device, dtype=disagreement.dtype)
        coefficients = torch.as_tensor(
            discount, device=disagreement.device, dtype=disagreement.dtype
        ).pow(powers)
        window = disagreement[timestep : timestep + steps]
        risks.append((window * coefficients).sum() / coefficients.sum())
    return torch.stack(risks)


def mean_normalize_nonnegative(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize a non-negative trajectory by its detached mean."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if torch.any(values < 0):
        raise ValueError("values must be non-negative")
    return values / values.mean().detach().clamp_min(epsilon)


def critical_state_weights(
    disagreement: torch.Tensor,
    temporal_risk: torch.Tensor,
    config: PIVOTQConfig,
) -> torch.Tensor:
    """Build clipped, detached PIVOT-Q state weights."""
    config.validate()
    if disagreement.shape != temporal_risk.shape:
        raise ValueError("disagreement and temporal_risk must have the same shape")
    q_normalized = mean_normalize_nonnegative(
        disagreement, epsilon=config.normalization_epsilon
    )
    r_normalized = mean_normalize_nonnegative(
        temporal_risk, epsilon=config.normalization_epsilon
    )
    weights = 1.0 + config.alpha_q * q_normalized + config.beta_r * r_normalized
    return weights.clamp(min=config.weight_min, max=config.weight_max).detach()


def weighted_distillation_loss(
    per_state_loss: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Reduce per-state distillation losses with fixed critical-state weights."""
    if per_state_loss.ndim != 1 or per_state_loss.shape != weights.shape:
        raise ValueError("per_state_loss and weights must be matching 1D tensors")
    return (per_state_loss * weights).sum() / weights.sum().clamp_min(epsilon)


def build_pivot_q_targets(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    config: PIVOTQConfig,
    *,
    executed_action_index: int = 0,
    action_dims: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience function returning ``(q_t, r_t, w_t)`` for one rollout."""
    q_values = action_disagreement(
        student_actions,
        teacher_actions,
        executed_action_index=executed_action_index,
        action_dims=action_dims,
    )
    r_values = temporal_quantization_risk(
        q_values,
        horizon=config.temporal_horizon,
        discount=config.temporal_discount,
    )
    weights = critical_state_weights(q_values, r_values, config)
    return q_values.detach(), r_values.detach(), weights
