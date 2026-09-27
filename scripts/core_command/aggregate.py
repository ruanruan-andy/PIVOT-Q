#!/usr/bin/env python3
"""Aggregate complete LIBERO-Plus evaluation episodes, one seed at a time."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from launch import ROOT, SUITES, FAMILIES, SETTINGS, result_root, eval_root, eval_directory

def summarize(directory, family, suite, expected):
    candidates = [directory / "metrics/episodes.jsonl", directory / "episodes.jsonl", directory / "eval/episodes.jsonl"]
    files = [p for p in candidates if p.is_file()]
    if len(files) != 1:
        raise ValueError(f"expected one episodes.jsonl in {directory}, found {len(files)}")
    rows = {}
    for line in files[0].read_text().splitlines():
        row = json.loads(line)
        if row.get("suite", suite) != suite:
            raise ValueError(f"suite mismatch: {files[0]}")
        task = row.get("task_index", row.get("task_id"))
        if task is None:
            raise ValueError(f"missing task identifier: {files[0]}")
        init = row.get("initial_state_id", row.get("episode_index", 0))
        key = (int(task), int(init))
        if key in rows:
            raise ValueError(f"duplicate episode {key}: {files[0]}")
        rows[key] = row
    wanted = {(int(task), 0) for task in expected}
    if set(rows) != wanted:
        raise ValueError(f"{directory}: missing={len(wanted-set(rows))}, unexpected={len(set(rows)-wanted)} episodes")
    for row in rows.values():
        if row.get("error") or not isinstance(row.get("success"), bool):
            raise ValueError(f"invalid/incomplete episode in {files[0]}")
    successes = sum(row["success"] for row in rows.values())
    categories = {}
    for row in rows.values():
        category = row.get("category")
        if category is not None:
            categories.setdefault(category, []).append(row["success"])
    return {"episodes": len(rows), "successes": successes, "success_rate_pct": 100*successes/len(rows),
            "categories": {k: {"episodes":len(v), "success_rate_pct":100*sum(v)/len(v)} for k,v in categories.items()},
            "source": str(files[0])}

def recorded_seed(directory, family, setting, suite):
    values = set()
    for parent in (directory, directory / "eval", directory.parent, directory.parent.parent):
        for name in ("core_command.json", "run.json", "protocol.json"):
            path = parent / name
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            if name == "core_command.json":
                if (data.get("family"), data.get("setting"), data.get("suite")) != (family, setting, suite):
                    raise ValueError(f"run identity mismatch: {path}")
                values.add(int(data["seed"]))
            else:
                for key in ("eval_seed", "policy_seed"):
                    if data.get(key) is not None:
                        values.add(int(data[key]))
                cfg = data.get("config", {})
                if cfg.get("policy_seed") is not None:
                    values.add(int(cfg["policy_seed"]))
                if cfg.get("evaluation", {}).get("seed") is not None:
                    values.add(int(cfg["evaluation"]["seed"]))
                if name == "protocol.json" and data.get("seed") is not None:
                    values.add(int(data["seed"]))
    if len(values) > 1:
        raise ValueError(f"conflicting evaluation seeds: {directory}: {values}")
    return next(iter(values)) if values else None

def locate(base, family, setting, seed, suite):
    canonical = eval_directory(base, family, setting, seed, suite)
    candidates = [canonical]
    parent = eval_root(base, family, setting)
    candidates += [parent / suite, base / suite]
    candidates += list(parent.glob("seed-*/" + suite))
    candidates += list(parent.glob("seed-*/libero-plus/" + suite))
    matches = []
    for directory in dict.fromkeys(candidates):
        if not any((directory / name).is_file() for name in
                   ("metrics/episodes.jsonl", "episodes.jsonl", "eval/episodes.jsonl")):
            continue
        actual = recorded_seed(directory, family, setting, suite)
        if actual == seed:
            matches.append(directory)
        elif directory == canonical:
            raise ValueError(f"evaluation seed missing or different in {directory}: {actual}")
    if len(matches) != 1:
        raise ValueError(f"{family}/{setting}/{suite} seed {seed}: found {len(matches)} matching runs; use --eval-dir to select the intended suite directory")
    return matches[0]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=FAMILIES)
    p.add_argument("setting", choices=SETTINGS)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    p.add_argument("--results-root", type=Path)
    p.add_argument("--manifest", type=Path, default=ROOT/"manifests/libero_plus_first20.json")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--eval-dir", help="explicit suite directory template, supporting {seed}, {suite}")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error("duplicate seeds")
    seeds = sorted(a.seeds)
    suites = SUITES if a.suite == "all" else (a.suite,)
    root = (a.results_root or result_root(a.family, a.setting)).expanduser().resolve()
    output = a.output_dir or root / "aggregate" / ("seeds-" + "-".join(map(str,seeds))) / a.suite
    if a.dry_run:
        print(json.dumps({"inputs":[str(Path(a.eval_dir.format(seed=seed,suite=suite)) if a.eval_dir else eval_directory(root,a.family,a.setting,seed,suite)) for seed in seeds for suite in suites], "output":str(output)},indent=2))
        return
    manifest = json.loads(a.manifest.read_text())
    if manifest.get("initial_state_ids", [0]) != [0]:
        raise ValueError("these commands aggregate the first20 protocol with initial state 0")
    result = {"family":a.family, "setting":a.setting, "seeds":seeds, "suites":list(suites),
              "unit":"percent", "weighting":"equal suite weight within seed, equal seed weight across seeds",
              "std":"sample standard deviation (ddof=1); null for one seed", "per_seed":{}}
    table = []
    for seed in seeds:
        per_suite = {}
        for suite in suites:
            if a.eval_dir:
                directory = Path(a.eval_dir.format(seed=seed, suite=suite)).expanduser().resolve()
                if recorded_seed(directory,a.family,a.setting,suite) != seed:
                    raise ValueError(f"evaluation seed missing or different: {directory}")
            else:
                directory = locate(root,a.family,a.setting,seed,suite)
            summary = summarize(directory,a.family,suite,manifest["task_ids_by_suite"][suite])
            per_suite[suite] = summary
            table.append({"seed":seed,"suite":suite, **{k:summary[k] for k in ("episodes","successes","success_rate_pct")}})
        overall = statistics.mean(v["success_rate_pct"] for v in per_suite.values())
        result["per_seed"][str(seed)] = {"suites":per_suite,"overall_success_rate_pct":overall}
        table.append({"seed":seed,"suite":"overall","episodes":sum(v["episodes"] for v in per_suite.values()),
                      "successes":sum(v["successes"] for v in per_suite.values()),"success_rate_pct":overall})
    result["across_seeds"] = {}
    for suite in (*suites,"overall"):
        values = [result["per_seed"][str(s)]["overall_success_rate_pct"] if suite=="overall" else
                  result["per_seed"][str(s)]["suites"][suite]["success_rate_pct"] for s in seeds]
        result["across_seeds"][suite] = {"mean_pct":statistics.mean(values),
            "std_pct":statistics.stdev(values) if len(values)>1 else None}
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    with (output/"per_seed.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["seed","suite","episodes","successes","success_rate_pct"])
        writer.writeheader();writer.writerows(table)
    with (output/"across_seeds.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["suite","mean_pct","std_pct"])
        writer.writeheader();writer.writerows({"suite":k,**v} for k,v in result["across_seeds"].items())
    print(json.dumps(result["across_seeds"],indent=2))
    print(output)

if __name__=="__main__":
    main()
