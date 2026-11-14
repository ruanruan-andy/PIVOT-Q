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
    roots = (PACKAGE, ROOT / "outputs", ROOT / "models/quantized")
    if not any(path.is_relative_to(root) for root in roots):
        allowed = ", ".join(map(str, roots))
        raise ValueError(f"output must stay under {allowed}: {path}")
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

import argparse

def parser(description, methods):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=methods, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=19600)
    p.add_argument("--clean-port", type=int, default=19601)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--run-name")
    p.add_argument("--dry-run", action="store_true")
    return p

CONFIG_BACKENDS = {
    "groot_holoq": ("groot", "holoq"),
    "groot_quantvla": ("groot", "quantvla"),
    "openvla_oft_qvla": ("openvla_oft", "qvla"),
    "pi05_holoq": ("pi05", "holoq"),
    "pi05_quantvla": ("pi05", "quantvla"),
}


def config(args):
    cfg = yaml.safe_load(args.config.read_text())
    for override in args.set:
        key, sep, value = override.partition("=")
        keys = key.split(".")
        parent = cfg
        if not sep:
            raise ValueError(f"Expected KEY=VALUE: {override}")
        for part in keys[:-1]:
            parent = parent[part]
        if keys[-1] not in parent:
            raise ValueError(f"Unknown option: {key}")
        parent[keys[-1]] = yaml.safe_load(value)
    setting = args.config.stem
    if setting not in CONFIG_BACKENDS:
        raise ValueError(f"Unknown setting config: {args.config.name}")
    backbone, quantizer = CONFIG_BACKENDS[setting]
    cfg.update(setting=setting, backbone=backbone, quantizer=quantizer)
    from .selection_budget import validate_percent
    selection = cfg["methods"]["pivot_q"] if backbone == "pi05" else cfg["selection"]
    validate_percent(selection.get("percent"))
    if args.gpu < 0 or args.port == args.clean_port:
        raise ValueError("Expected one GPU and distinct ports")
    for section, keys in [(cfg, ("output_root",)), (cfg["benchmark"], ("task_file",)),
                          (cfg["model"], ("checkpoint", "quant_pack")), (cfg["paths"], tuple(cfg["paths"]))]:
        for key in keys:
            if not Path(section[key]).is_absolute():
                raise ValueError(f"{key} requires an absolute path")
    return cfg

def result_root(cfg, args):
    from re import fullmatch
    base = Path(cfg["output_root"]) / cfg["setting"]
    selection = cfg["methods"]["pivot_q"] if cfg["backbone"] == "pi05" else cfg["selection"]
    percent = selection.get("percent")
    if args.method == "pivot_q" and percent is not None:
        from decimal import Decimal
        whole, _, fraction = format(Decimal(str(percent)).normalize(), "f").partition(".")
        label = whole.zfill(3) + (f"p{fraction}" if fraction else "")
        base /= f"percent_{label}"
    if args.run_name:
        if not fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
            raise ValueError("Invalid run name")
        return base / "ablations" / args.run_name
    return base / args.method

def values(cfg, args, out):
    training = dict(cfg["train"])
    if training.pop("action_steps") != 1:
        raise ValueError("π0.5 training currently executes the first action only")
    training["seed"] = args.seed if args.seed is not None else training["seed"]
    model = cfg["model"]
    selection = dict(cfg["methods"]["pivot_q"])
    selection.update(cfg["methods"].get(args.method, {}))
    selection["lambda_anchor"] = selection.pop("anchor_lambda")
    return dict(**training, **selection, backend=cfg["quantizer"],
                suite_order=list(cfg["benchmark"]["suites"]), selection_method=args.method, method_name=args.method,
                device="cuda:0", env_port=args.port, clean_env_port=args.clean_port,
                checkpoint=model["checkpoint"], quant_pack_dir=model["quant_pack"], output_dir=str(out),
                manifest=cfg["benchmark"]["task_file"], quantization=cfg["quantization"],
                holoq={"paths": {"omega_source": cfg["paths"]["omega"]}, "inference": {
                     "num_inference_steps": model["num_inference_steps"]}, "quantization": cfg["quantization"],
                     "mode": args.method})

