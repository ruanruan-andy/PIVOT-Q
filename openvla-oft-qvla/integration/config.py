"""Configuration paths are relative to this integration directory, not cwd."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, overrides=()):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    parent = raw.pop("extends", None)
    cfg = merge(load_config(path.parent / parent) if parent else {}, raw)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"unknown config section: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"unknown config option: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def resolve(value):
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable: {value}")
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_config(description, phase=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", choices=tuple(SUITES), help="set suite-specific checkpoint, cache, normalization and output defaults")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    automatic = []
    if args.suite:
        method = load_config(args.config)["method"]
        suite = args.suite
        checkpoint = "openvla-7b-oft-finetuned-" + suite.replace("_", "-")
        automatic = [f"suite={suite}", f"model.unnorm_key={suite}", f"paths.checkpoint=checkpoints/{checkpoint}",
                     f"paths.gates=cache/{suite}/gates.json", f"paths.proxy=cache/{suite}/proxy.pt",
                     f"paths.calibration=cache/{suite}/calibration.pt"]
    cfg = load_config(args.config, automatic + args.set)
    if phase and not any(item.split('=', 1)[0] == 'paths.output' for item in args.set):
        method = {'fp': 'original', 'qvla': 'quantized'}.get(cfg['method'], cfg['method'])
        seed = cfg['evaluation']['seed'] if phase == 'eval' else cfg.get('seed', 0)
        cfg['paths']['output'] = f'../outputs/qvla/openvla_oft/{method}/{phase}/seed-{seed:03d}/{cfg["suite"]}'
    validate(cfg)
    return cfg, args.dry_run


def validate(cfg):
    for key, value in cfg.get("diagnostics", {}).items():
        if key not in {"save_eval_states", "save_train_states"} or not isinstance(value, bool):
            raise ValueError("diagnostics options must be boolean save_eval_states/save_train_states")
    if cfg["method"] not in {"fp", "qvla", "pivot_q", "full_distill"}:
        raise ValueError("unsupported method")
    if cfg["suite"] not in SUITES:
        raise ValueError("unsupported LIBERO suite")
    m, t, s, e = (cfg[k] for k in ("model", "training", "selection", "evaluation"))
    if m["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported dtype")
    if m["num_images"] != 2 or not m["use_proprio"]:
        raise ValueError("this integration targets the official two-image, proprio, L1 OFT checkpoints")
    if not 1 <= e["open_loop_steps"] <= 8:
        raise ValueError("open_loop_steps must be within [1, 8]")
    for key in ("episodes", "updates_per_episode", "rank", "save_every_episodes", "keep_last_checkpoints"):
        if t[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if t["learning_rate"] <= 0 or not 0 <= t["dropout"] < 1 or t["anchor_lambda"] < 0:
        raise ValueError("invalid optimizer/adapter settings")
    if t["anchor_lambda"] and min(t["anchor_capacity"], t["anchor_batch_size"], t["anchor_horizon"]) <= 0:
        raise ValueError("clean anchor requires a non-empty replay buffer and positive horizon")
    if min(s["phase_bins"], s["top_per_phase"], s["future_horizon"]) <= 0:
        raise ValueError("selection counts must be positive")
    if not 0 < s["discount"] <= 1 or s["min_gap"] < 0:
        raise ValueError("invalid temporal selection settings")
    if min(s["alpha_q"], s["beta_r"]) < 0 or s["alpha_q"] + s["beta_r"] <= 0:
        raise ValueError("selection coefficients must be nonnegative and not both zero")
    if not 0 < s["weight_min"] <= s["weight_max"]:
        raise ValueError("invalid selection weight bounds")
    if cfg["training"]["loss_actions"] not in {"first", "executed"}:
        raise ValueError("loss_actions must be first or executed")


def file_info(path):
    """Lightweight file provenance; no content hashing or weight-file scan."""
    path = Path(path)
    return {"name": path.name, "size_bytes": path.stat().st_size}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
