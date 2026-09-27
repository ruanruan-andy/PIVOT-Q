"""Create the existing PIVOT_Q first-N-per-category manifest format in LIBERO-Plus."""
import argparse
import json
from pathlib import Path

CATEGORIES = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
              "Background Textures", "Sensor Noise", "Objects Layout"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("manifest exists; reuse it or choose a new output")
    if args.per_category <= 0:
        raise ValueError("per-category must be positive")
    from libero.libero import benchmark, get_libero_path
    classification = json.loads((Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json").read_text())
    ids, metadata = {}, {}
    for suite in SUITES:
        names = benchmark.get_benchmark_dict()[suite]().get_task_names()
        by_name = {item["name"]: item for item in classification[suite]}
        selected = []
        for category in CATEGORIES:
            pool = [i for i, name in enumerate(names) if by_name[name]["category"] == category]
            if len(pool) < args.per_category:
                raise ValueError(f"insufficient tasks: {suite}/{category}")
            selected.extend(pool[:args.per_category])
        ids[suite] = selected
        metadata[suite] = [{"task_id": i, "task_name": names[i], "classification_id": int(by_name[names[i]]["id"]),
                            "category": by_name[names[i]]["category"], "difficulty_level": by_name[names[i]].get("difficulty_level")}
                           for i in selected]
    result = {"version": 4, "protocol": "libero-plus-shared-first-n-v1", "train_eval_relation": "same_task_ids",
              "selection": "explicit_task_ids", "selection_order": "first_by_task_index",
              "per_suite_per_category": args.per_category, "total_tasks": sum(map(len, ids.values())),
              "initial_state_ids": [0], "categories": CATEGORIES, "suites": SUITES,
              "task_ids_by_suite": ids, "task_metadata_by_suite": metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {result['total_tasks']} tasks: {args.output}")


if __name__ == "__main__":
    main()
