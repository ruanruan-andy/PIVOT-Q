#!/usr/bin/env python3
"""Freeze an unlabeled clean-LIBERO calibration buffer for π0.5."""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "lerobot" / "src"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "pi05_quantvla/data/calibration/libero_clean_32_seed-000.pt",
    )
    args = p.parse_args()
    os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "data/lerobot"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("lerobot/libero", return_uint8=True)
    if args.count > len(dataset):
        raise ValueError(f"requested {args.count} observations, but dataset has only {len(dataset)}")
    indices = random.Random(args.seed).sample(range(len(dataset)), args.count)
    keys = ("observation.images.image", "observation.images.image2", "observation.state", "task")
    samples = []
    for index in indices:
        item = dataset[index]
        missing = set(keys) - set(item)
        if missing:
            raise KeyError(f"lerobot/libero example {index} is missing {sorted(missing)}")
        samples.append({key: item[key] for key in keys})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": "lerobot/libero", "seed": args.seed, "indices": indices, "samples": samples}, args.output)
    print(args.output)
if __name__ == "__main__": main()
