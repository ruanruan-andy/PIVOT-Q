#!/usr/bin/env python3
"""Build and validate the shared LIBERO-Plus first-N manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "eval.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    count = int(config["LIBERO_PLUS_NUM"])
    paths = config["paths"]
    root = Path(paths["root_dir"])
    manifest = resolve(root, paths["manifests_dir"]) / f"libero_plus_first{count}.json"
    quantvla_root = resolve(root, paths["quantvla_dir"])
    libero_plus_root = resolve(root, paths["libero_plus_dir"])
    command = [
        "conda", "run", "--no-capture-output", "-n", "libero_test",
        "python", str(quantvla_root / "scripts" / "build_libero_plus_shared_manifest.py"),
        "--output-path", str(manifest), "--per-suite-category", str(count), "--overwrite",
    ]
    print(" ".join(command))
    if args.dry_run:
        return

    manifest.parent.mkdir(parents=True, exist_ok=True)
    python_path = os.pathsep.join(
        value for value in (str(libero_plus_root), str(quantvla_root), os.environ.get("PYTHONPATH")) if value
    )
    environment = os.environ | {
        "LIBERO_PLUS_ROOT": str(libero_plus_root),
        "LIBERO_CONFIG_PATH": str(quantvla_root / "configs" / "libero_plus"),
        "PYTHONPATH": python_path,
        "SHELL": os.environ.get("SHELL", "/bin/bash"),
    }
    subprocess.run(command, check=True, cwd=quantvla_root, env=environment)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    expected_total = 4 * 7 * count
    if data.get("per_suite_per_category") != count or data.get("total_tasks") != expected_total:
        raise RuntimeError(f"invalid manifest: expected {expected_total} tasks and N={count}")
    print(f"Validated {manifest}: {expected_total} tasks, {count} per suite/category")


if __name__ == "__main__":
    main()
