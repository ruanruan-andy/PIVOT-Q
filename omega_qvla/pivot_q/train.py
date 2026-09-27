#!/usr/bin/env python3
"""Run GAP-PIVOT_Q with Ω-QVLA as the quantized GR00T student."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pivot_q"))
sys.path.insert(0, str(ROOT))

# Ω-QVLA supplies the GR00T model package, while this small model-independent
# objective currently lives in the sibling QuantVLA checkout.
import gr00t.experiment  # noqa: E402

core_path = ROOT / "third_party" / "QuantVLA/gr00t/experiment/pivot_q.py"
spec = importlib.util.spec_from_file_location("gr00t.experiment.pivot_q", core_path)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load PIVOT-Q core from {core_path}")
core_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core_module
spec.loader.exec_module(core_module)

import selector as selection  # noqa: E402
import full_distill  # noqa: E402
import train as shared_train  # noqa: E402

from omega_qvla.pivot_q.backend import install_hooks  # noqa: E402


def main() -> None:
    config = tyro.cli(shared_train.TrainConfig)
    if config.method_name not in {"omega_pivot_q", "omega_full_distill"}:
        raise ValueError("Ω-QVLA trainer requires omega_pivot_q or omega_full_distill")
    install_hooks(shared_train)
    shared_train.build_pivot_q_targets = selection.build_targets
    # Paper dense baseline: reuse the same HoloQ rollout, adapter, and anchor
    # pipeline; only the all-state, uniform-weight selector changes.
    selector = (
        full_distill.select_states
        if config.method_name == "omega_full_distill"
        else selection.select_states
    )
    shared_train.train(config, selector)


if __name__ == "__main__":
    main()
