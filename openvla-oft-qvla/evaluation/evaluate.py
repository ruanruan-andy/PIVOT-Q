"""Evaluate FP, QVLA or either recovered adapter on exactly the shared manifest."""
from __future__ import annotations

from collections import defaultdict
import json

from evaluation.environment import EnvClient, episode_specs, validate_service
from integration.config import atomic_json, parse_config, resolve
from integration.policy import OFTPolicy
from training.checkpoint import read, seed_all
from training.collect import collect
from evaluation.diagnostics import definition, save_episode


_DIAGNOSTIC_FIELDS = {
    "states_path", "states", "q_by_timestep", "q_objective_by_timestep",
    "r_by_timestep", "timesteps", "q_mean", "q_p95", "q_p99", "q_max",
    "q_zero_count", "r_mean", "r_p95", "r_p99", "r_max", "r_zero_count",
}


def _without_diagnostics(row):
    return {key: value for key, value in row.items() if key not in _DIAGNOSTIC_FIELDS}


def evaluate(cfg):
    seed_all(cfg["evaluation"]["seed"])
    adapted = cfg["method"] in {"pivot_q", "full_distill"}
    if adapted != bool(cfg["paths"]["adapter"]):
        raise ValueError("pivot_q/full_distill evaluation requires paths.adapter; fp/qvla must not load an adapter")
    output = resolve(cfg["paths"]["output"]) / "eval"
    output.mkdir(parents=True, exist_ok=True)
    specs = episode_specs(cfg)
    if cfg["evaluation"]["max_episodes"] is not None:
        limit = cfg["evaluation"]["max_episodes"]
        if limit <= 0:
            raise ValueError("max_episodes must be positive")
        specs = specs[:limit]
    save_states = cfg.get("diagnostics", {}).get("save_eval_states", False)
    episodes_path = output / "episodes.jsonl"
    outcomes = []
    if episodes_path.exists():
        # Evaluation can be moved between hosts at episode boundaries.  Reuse
        # only a contiguous prefix whose manifest identity still matches the
        # current run; never silently skip or overwrite an episode.
        for line_number, line in enumerate(episodes_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid episodes.jsonl at line {line_number}") from error
            index = len(outcomes)
            if index >= len(specs) or row.get("episode") != index + 1:
                raise ValueError("evaluation output is not a contiguous prefix of the current manifest")
            expected = specs[index]
            if row.get("error") or not isinstance(row.get("success"), bool):
                raise ValueError(f"invalid completed episode at line {line_number}")
            for key in ("task_id", "initial_state_id", "task_name", "category"):
                if key in expected and row.get(key) != expected[key]:
                    raise ValueError(f"evaluation output mismatch at episode {index + 1}: {key}")
            outcomes.append(row)
        if not save_states and outcomes:
            # A run can be converted from diagnostic to standard evaluation
            # while resuming. Preserve outcome fields, but remove old
            # per-state artifacts so the final directory is standard-only.
            cleaned = [_without_diagnostics(row) for row in outcomes]
            for row in outcomes:
                state_path = row.get("states_path")
                if state_path:
                    path = output / state_path
                    if path.is_file():
                        path.unlink()
            if cleaned != outcomes:
                tmp = episodes_path.with_suffix(episodes_path.suffix + ".tmp")
                tmp.write_text("".join(json.dumps(row) + "\n" for row in cleaned))
                tmp.replace(episodes_path)
            outcomes = cleaned
        if len(outcomes) == len(specs):
            print(json.dumps({"resume": False, "completed": len(outcomes), "message": "evaluation already complete"}), flush=True)
        else:
            print(json.dumps({"resume": True, "completed": len(outcomes),
                              "remaining": len(specs) - len(outcomes)}), flush=True)
    failure_path = output / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    if len(outcomes) == len(specs):
        categories = defaultdict(list)
        for row in outcomes:
            categories[row.get("category", "unspecified")].append(row["success"])
        summary = {"method": cfg["method"], "suite": cfg["suite"], "episodes": len(outcomes),
                   "successes": sum(row["success"] for row in outcomes),
                   "success_rate": sum(row["success"] for row in outcomes) / len(outcomes),
                   "categories": {key: {"episodes": len(values), "success_rate": sum(values) / len(values)}
                                  for key, values in categories.items()}, "evaluator_errors": 0}
        atomic_json(output / "summary.json", summary)
        return summary
    with EnvClient(cfg) as client:
        validate_service(client, cfg)
        policy = OFTPolicy(cfg, quantized=cfg["method"] != "fp", trainable=adapted)
        if adapted:
            read(resolve(cfg["paths"]["adapter"]), policy, cfg)
        teacher = (policy if cfg["method"] == "fp" else OFTPolicy(cfg, teacher=True)) if save_states else None
        atomic_json(output / "run.json", {"config": cfg, "identity": policy.identity,
                    "teacher_identity": teacher.identity if teacher is not None else None,
                    "diagnostics": definition(cfg) if save_states else None})
        try:
            for index in range(len(outcomes), len(specs)):
                spec = specs[index]
                seed_all(cfg["evaluation"]["seed"] + index)
                records, result = collect(cfg, policy, client, spec, teacher=teacher,
                                          cache_features=False, capture_actions=save_states)
                row = {"episode": index + 1, **spec, **result}
                if save_states:
                    row.update(save_episode(output, cfg, spec, index + 1, records, result,
                                            phase="eval", policy_seed=cfg["evaluation"]["seed"] + index))
                outcomes.append(row)
                with (output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("episode", "task_id", "success", "env_steps")}), flush=True)
        except Exception as error:
            atomic_json(output / "failure.json", {"completed": len(outcomes), "error": repr(error)})
            raise  # Evaluator errors are never counted as task failures.
    categories = defaultdict(list)
    for row in outcomes:
        categories[row.get("category", "unspecified")].append(row["success"])
    summary = {"method": cfg["method"], "suite": cfg["suite"], "episodes": len(outcomes),
               "successes": sum(row["success"] for row in outcomes),
               "success_rate": sum(row["success"] for row in outcomes) / len(outcomes),
               "categories": {key: {"episodes": len(values), "success_rate": sum(values) / len(values)}
                              for key, values in categories.items()}, "evaluator_errors": 0}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    cfg, dry = parse_config(__doc__)
    print(json.dumps(cfg if dry else evaluate(cfg), indent=2))


if __name__ == "__main__":
    main()
