"""Atomic adapter/trainer checkpoints with compatibility and RNG checks."""
from __future__ import annotations

from integration.relocation import normalize_paths

import copy
from pathlib import Path
import random

import numpy as np
import torch

from integration.adapters import adapter_state, load_adapter


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save(path, student, cfg, **trainer):
    payload = {"format_version": 1, "identity": student.identity, "config": cfg,
               "targets": student.adapter_targets, "adapter": adapter_state(student.head),
               "rng": rng_state(), **trainer}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read(path, student, cfg, *, resume=False):
    # Own checkpoints include optimizer/RNG and replay objects; load trusted files only.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["format_version"] != 1 or normalize_paths(data["identity"]) != normalize_paths(student.identity):
        raise ValueError("checkpoint identity mismatch (base model, gates, upstream, dtype or normalization)")
    for key in ("rank", "alpha", "dropout"):
        if data["config"]["training"][key] != cfg["training"][key]:
            raise ValueError(f"adapter configuration mismatch: {key}")
    if data["config"]["method"] != cfg["method"]:
        raise ValueError("checkpoint method does not match requested method")
    if resume:
        previous, current = copy.deepcopy(data["config"]), copy.deepcopy(cfg)
        for item in (previous, current):
            # Diagnostics only add logging/teacher evaluation and do not alter
            # training math; older checkpoints have no diagnostics section.
            item.pop("diagnostics", None)
            item["training"]["resume"] = None
            item["paths"]["adapter"] = None
        if normalize_paths(previous) != normalize_paths(current):
            raise ValueError("resume configuration changed; use the original training configuration")
    load_adapter(student.head, data["adapter"])
    return data


def truncate_metrics(path, completed_episode):
    import json
    path = Path(path)
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A crash may leave one incomplete line.
        if row["episode"] <= completed_episode:
            kept.append(json.dumps(row))
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(kept) + ("\n" if kept else ""))
    temp.replace(path)
