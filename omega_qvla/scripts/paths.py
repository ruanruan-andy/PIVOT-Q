"""Resolve Ω-QVLA configuration paths from one workspace root."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(config["paths"]["root_dir"]).expanduser().resolve()

    def resolve(value: str) -> str:
        item = Path(value).expanduser()
        return str(item if item.is_absolute() else root / item)

    for key in (
        "pivot_q_root", "omega_qvla_source", "quantvla_root", "libero_plus_root",
        "libero_plus_config_path", "manifests_root", "results_root",
    ):
        config["paths"][key] = resolve(config["paths"][key])
    for suite, value in config["models"].items():
        config["models"][suite] = resolve(value)
    for suite, value in config["packs"].items():
        config["packs"][suite] = resolve(value)
    return config
