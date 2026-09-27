"""Validated configuration for phase-balanced PIVOT-Q state selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PIVOTQSelectionConfig:
    """State budget and weighting used by the v2 sparse update."""

    phase_bins: int = 4
    priority_per_phase: int = 2
    random_per_phase: int = 2
    min_temporal_gap: int = 4
    alpha_q: float = 1.0
    beta_r: float = 1.0
    weight_min: float = 0.5
    weight_max: float = 2.0
    epsilon: float = 1e-8

    @property
    def states_per_episode(self) -> int:
        return self.phase_bins * (self.priority_per_phase + self.random_per_phase)

    def validate(self) -> None:
        if self.phase_bins <= 0:
            raise ValueError("phase_bins must be positive")
        if self.priority_per_phase < 0 or self.random_per_phase < 0:
            raise ValueError("per-phase quotas must be non-negative")
        if self.states_per_episode <= 0:
            raise ValueError("states_per_episode must be positive")
        if self.min_temporal_gap < 0:
            raise ValueError("min_temporal_gap must be non-negative")
        if self.alpha_q < 0.0 or self.beta_r < 0.0:
            raise ValueError("alpha_q and beta_r must be non-negative")
        if self.alpha_q == 0.0 and self.beta_r == 0.0:
            raise ValueError("at least one priority coefficient must be positive")
        if self.weight_min <= 0.0 or self.weight_max < self.weight_min:
            raise ValueError("invalid state-weight bounds")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
