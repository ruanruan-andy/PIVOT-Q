"""Configuration, output boundaries, and immutable model identities."""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import yaml

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
WORKSPACE = ROOT / "third_party"
MODES = ("fp_original", "omega_original", "full_distill", "pivot_q")

def inside(path):
    path = Path(path).expanduser().resolve()
    if not (path.is_relative_to(PACKAGE.resolve()) or path.is_relative_to((ROOT / "outputs").resolve())):
        raise ValueError(f"output must stay under {PACKAGE} or {ROOT / 'outputs'}: {path}")
    return path

def merge(base, patch):
    result = dict(base)
    for key, value in patch.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result

def load_config(path=None):
    path = Path(path or PACKAGE / "config/pivot_q.yaml").expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    parent = data.pop("extends", None)
    config = merge(load_config(path.parent / parent), data) if parent else data
    if parent:
        config["_config_path"] = str(path)
    else:
        config["_config_path"] = str(path)
    return config

def resolve_paths(config):
    config = merge({}, config)
    root = Path(config["paths"]["root_dir"]).expanduser()
    config["paths"] = {k: str(Path(v).expanduser() if Path(v).expanduser().is_absolute() else root / v)
                       for k, v in config["paths"].items()}
    return config

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

@lru_cache(maxsize=16)
def _identity(entries):
    return {name: sha256(path) for name, path, size, mtime in entries}

def checkpoint_identity(path):
    path = Path(path).resolve()
    names = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
    names += sorted(p.name for p in path.glob("*.safetensors") if p.name != "model.safetensors")
    entries = tuple((name, str(path / name), (path / name).stat().st_size,
                     (path / name).stat().st_mtime_ns) for name in names)
    return dict(_identity(entries))

def atomic_json(path, payload):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
