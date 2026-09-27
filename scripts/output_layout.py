"""Relative output layout shared by launchers and reporting tools."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

def result_dir(family, model, method, phase="eval", seed=2026, suite=None):
    path = OUTPUT_ROOT / family / model / method / phase / f"seed-{seed}"
    return path / suite if suite else path
