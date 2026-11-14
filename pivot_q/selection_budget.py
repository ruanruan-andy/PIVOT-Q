"""State budgets for phase-balanced selection."""
import math


def validate_percent(percent):
    if percent is not None and (
        isinstance(percent, bool) or not isinstance(percent, (int, float))
        or not math.isfinite(percent) or not 0 < percent <= 100
    ):
        raise ValueError("selection.percent must be a number in (0, 100]")


def phase_quotas(phase_sizes, percent):
    validate_percent(percent)
    states = sum(phase_sizes)
    if not states:
        return [0] * len(phase_sizes)
    exact = [size * percent / 100 for size in phase_sizes]
    quota = [math.floor(value) for value in exact]
    remaining = max(1, math.floor(states * percent / 100 + 0.5)) - sum(quota)
    order = sorted(range(len(quota)), key=lambda i: (quota[i] - exact[i], i))
    for i in order[:remaining]:
        quota[i] += 1
    return quota
