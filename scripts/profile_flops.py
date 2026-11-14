"""Sample training FLOPs in a separate run or summarize recorded calls."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sample(args):
    cfg = yaml.safe_load(args.config.read_text())
    cfg["benchmark"]["suites"] = [args.suite]
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.stage == "distill_forward_backward":
        cfg["train"]["batch_size"] = args.batch_size
    elif args.stage == "anchor_forward_backward":
        if "methods" in cfg:
            cfg["methods"].setdefault(args.method, {})["anchor_batch_size"] = args.batch_size
        elif args.method == "full_distill" and "anchor_batch_size" in cfg["full_distill"]:
            cfg["full_distill"]["anchor_batch_size"] = args.batch_size
        else:
            cfg.get("selection" if args.config.stem.startswith("groot_") else "train")["anchor_batch_size"] = args.batch_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    env = dict(os.environ, WANDB_MODE="disabled",
               PIVOT_Q_FLOPS_PROFILE_KEY=f"{args.stage}/{args.batch_size}",
               PIVOT_Q_FLOPS_PROFILE_SUITE=args.suite,
               PIVOT_Q_FLOPS_PROFILE_OUTPUT=str(args.output.resolve()))
    with TemporaryDirectory(prefix=".flops-", dir=ROOT / "outputs") as temporary:
        cfg["output_root"] = temporary
        path = Path(temporary) / args.config.name
        path.write_text(yaml.safe_dump(cfg))
        subprocess.run([sys.executable, "scripts/train.py", "--config", str(path),
                        "--method", args.method, "--gpu", str(args.gpu),
                        "--port", str(args.port), "--clean-port", str(args.clean_port),
                        "--max-rounds", "1", "--no-resume"], env=env, cwd=ROOT, check=True)
    if not args.output.is_file():
        raise SystemExit("Requested stage and batch size did not occur in the sampled rollout.")
    print(args.output)


def summarize(args):
    samples = defaultdict(list)
    for path in args.samples:
        row = json.loads(path.read_text())
        suite = row["suite"]
        samples[suite, row["key"].removeprefix(suite + "/")].append(row["flops"])
    suites = {}
    for path in args.run:
        data = json.loads(path.read_text())
        for key, count in data.get("compute", data)["calls"].items():
            prefix, sep, stage = key.partition("/")
            if prefix.startswith("libero_"):
                suite = prefix
            else:
                suite, stage = args.suite or path.parent.name, key
                if not suite.startswith("libero_"):
                    raise SystemExit(f"Unscoped calls in {path}; specify --suite for a single-suite run.")
            calls = suites.setdefault(suite, defaultdict(int))
            calls[stage] += count
    report = {}
    for suite, calls in suites.items():
        estimates = {}
        for stage, count in sorted(calls.items()):
            if stage.startswith("optimizer_step/"):
                continue
            values = samples.get((suite, stage))
            estimates[stage] = round(count * sum(values) / len(values)) if values else None
        missing = [key for key, value in estimates.items() if value is None]
        update = [value for key, value in estimates.items()
                  if key.startswith(("distill_forward_backward/", "anchor_forward_backward/"))]
        report[suite] = dict(calls=calls, estimated_flops_by_stage=estimates, missing=missing,
                             estimated_update_flops=sum(update) if update and None not in update else None,
                             estimated_model_flops=sum(estimates.values()) if estimates and not missing else None)

    def total(field):
        values = [row[field] for row in report.values()]
        return sum(values) if values and None not in values else None

    output = dict(suites=report, estimated_update_flops=total("estimated_update_flops"),
                  estimated_model_flops=total("estimated_model_flops"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("sample")
    measure.add_argument("--config", required=True, type=Path)
    measure.add_argument("--method", required=True, choices=("pivot_q", "full_distill"))
    measure.add_argument("--suite", required=True)
    measure.add_argument("--stage", required=True)
    measure.add_argument("--batch-size", default=1, type=int)
    measure.add_argument("--seed", type=int)
    measure.add_argument("--gpu", default=0, type=int)
    measure.add_argument("--port", default=19600, type=int)
    measure.add_argument("--clean-port", default=19601, type=int)
    measure.add_argument("--output", required=True, type=Path)
    aggregate = commands.add_parser("summarize")
    aggregate.add_argument("--run", nargs="+", required=True, type=Path)
    aggregate.add_argument("--samples", nargs="+", required=True, type=Path)
    aggregate.add_argument("--suite")
    aggregate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (sample if args.command == "sample" else summarize)(args)


if __name__ == "__main__":
    main()
