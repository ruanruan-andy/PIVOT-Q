"""Resolve π0.5 configuration paths from one workspace root."""

from __future__ import annotations

from pathlib import Path


def resolve_config(config: dict) -> dict:
    root = Path(config["paths"]["root_dir"]).expanduser().resolve()

    def resolve(value: str) -> str:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else root / path)

    for key in ("checkpoint", "quant_pack_dir"):
        config["paths"][key] = resolve(config["paths"][key])
    config["calibration"]["buffer"] = resolve(config["calibration"]["buffer"])
    return config
