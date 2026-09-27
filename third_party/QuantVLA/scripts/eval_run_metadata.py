"""Write one compact provenance record for a normalized evaluation run."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import time
from typing import Any


def _sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: pathlib.Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_eval_run_metadata(output_dir: pathlib.Path, config: Any) -> None:
    """Atomically merge launcher provenance and evaluator configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    previous: dict[str, Any] = {}
    if run_path.is_file():
        try:
            previous = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}

    manifest_value = os.environ.get("EVAL_MANIFEST")
    manifest_path = pathlib.Path(manifest_value).resolve() if manifest_value else None
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    now = time.time()
    record = {
        "schema_version": 2,
        "method": os.environ.get("EVAL_METHOD"),
        "model_variant": os.environ.get("EVAL_MODEL_VARIANT"),
        "benchmark": os.environ.get("EVAL_BENCHMARK"),
        "suite": getattr(config, "task_suite_name", None),
        "eval_seed": getattr(config, "policy_seed", None),
        "train_seed": os.environ.get("EVAL_TRAIN_SEED") or None,
        "checkpoint": os.environ.get("EVAL_CHECKPOINT") or None,
        "manifest": (
            {"path": str(manifest_path), "sha256": _sha256(manifest_path)}
            if manifest_path is not None
            else None
        ),
        "save_video": bool(getattr(config, "save_video", False)),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "port": getattr(config, "port", None),
        "git_commit": _git_commit(repo_root),
        "server_command": os.environ.get("EVAL_SERVER_COMMAND"),
        "evaluator_command": os.environ.get("EVAL_EVALUATOR_COMMAND"),
        "created_at": previous.get("created_at", now),
        "last_started_at": now,
        "config": dataclasses.asdict(config),
    }
    temporary = run_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(run_path)
