"""Pin and import official code without modifying its source or checkpoint files."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from integration.config import ROOT, resolve


def check_transformers():
    from importlib.metadata import distribution
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    source = distribution("transformers").read_text("direct_url.json")
    metadata = json.loads(source or "{}")
    if metadata.get("vcs_info", {}).get("commit_id") != lock["transformers_commit"]:
        raise RuntimeError("install the pinned transformers-openvla-oft fork from requirements-model.txt; vanilla Transformers changes OFT attention")


def activate(cfg):
    root = resolve(os.environ.get("QVLA_ROOT", cfg["paths"]["qvla_root"]))
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"]:
        raise RuntimeError(f"QVLA commit mismatch: expected {lock['commit']}, found {commit}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("QVLA has tracked modifications; use an unchanged official checkout")
    source = root / lock["source_subdir"]
    for name in ("prismatic", "qvla", "experiments"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise RuntimeError(f"{name} already imported from a different repository; use a fresh process")
    sys.path.insert(0, str(source))
    # Keep Python bytecode out of the official checkout as well.
    sys.dont_write_bytecode = True
    return {"root": str(root), "commit": commit, "source": str(source)}
