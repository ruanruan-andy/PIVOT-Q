"""Measure paired first-action discrepancy on shared LIBERO-Plus observations."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(cache, suites, methods, output, setting):
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for suite in suites:
            reference = None
            for method in methods:
                path = cache / f"{suite}-{method}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if not rows:
                    raise ValueError(f"No diagnostic states in {path}")
                if reference is None:
                    reference = [{k: v for k, v in row.items() if k != "first_action_mse"} for row in rows]
                elif reference != [{k: v for k, v in row.items() if k != "first_action_mse"} for row in rows]:
                    raise ValueError(f"Diagnostic states differ in {path}")
                for state in rows:
                    stream.write(json.dumps(dict(state, setting=setting, method=method,
                                                 action_space="normalized", reference="original")) + "\n")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=19600)
    parser.add_argument("--adapter", help="PIVOT-Q adapter path; supports {suite} for per-suite adapters.")
    parser.add_argument("--tasks-per-category", type=int, default=1)
    args = parser.parse_args()
    if args.tasks_per_category < 1:
        parser.error("--tasks-per-category must be positive")
    cfg = yaml.safe_load(args.config.read_text())
    suites = [args.suite] if args.suite else cfg["benchmark"]["suites"]
    cfg["benchmark"]["suites"] = suites
    cfg["evaluation"]["estimate_flops"] = False
    manifest = json.loads(Path(cfg["benchmark"]["task_file"]).read_text())
    for suite in suites:
        counts, selected = Counter(), []
        for row in manifest["task_metadata_by_suite"][suite]:
            category = row["category"]
            if counts[category] < args.tasks_per_category:
                selected.append(row)
                counts[category] += 1
        manifest["task_metadata_by_suite"][suite] = selected
        manifest["task_ids_by_suite"][suite] = [row["task_id"] for row in selected]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = ["quantized", "pivot_q"] if args.adapter else ["quantized"]
    with TemporaryDirectory(prefix=".discrepancy-", dir=ROOT / "outputs") as tmp:
        tmp = Path(tmp)
        schedule = tmp / "tasks.json"
        schedule.write_text(json.dumps(manifest))
        cfg["benchmark"]["task_file"] = str(schedule)
        cfg["output_root"] = str(tmp)
        config = tmp / args.config.name
        config.write_text(yaml.safe_dump(cfg))
        cache = tmp / "cache"
        env = dict(os.environ, PIVOT_Q_DIAGNOSTIC_CACHE=str(cache), CUDA_VISIBLE_DEVICES=str(args.gpu))
        command = [sys.executable, "scripts/evaluate.py", "--config", str(config),
                   "--gpu", str(args.gpu), "--port", str(args.port)]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        for method in ["original", *methods]:
            adapter = ["--adapter", str(Path(args.adapter).resolve())] if method == "pivot_q" else []
            subprocess.run(command + ["--method", method] + adapter, cwd=ROOT, env=env, check=True)
        merge(cache, suites, methods, output, args.config.stem)
    print(output)


if __name__ == "__main__":
    main()
