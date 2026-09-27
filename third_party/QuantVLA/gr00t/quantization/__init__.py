"""GR00T DuQuant W4A8 fake quantization module."""

from .duquant_layers import (
    DuQuantConfig,
    DuQuantLinear,
    enable_duquant_if_configured,
    select_targets,
    wrap_duquant,
)

__all__ = [
    "DuQuantConfig",
    "DuQuantLinear",
    "enable_duquant_if_configured",
    "select_targets",
    "wrap_duquant",
]
