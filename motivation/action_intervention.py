"""Measure GR00T QuantVLA action interventions on LIBERO-Plus."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pivot_q.configuration import config
from pivot_q.evaluation import episode_specs, summarize
from pivot_q.processes import services
from pivot_q.suite_runs import groot_config, plans

STRATEGIES = ("original", "quantized", "random", "high", "low")


def intervene(strategy, discrepancy, threshold, rng):
    if strategy == "high":
        return discrepancy > threshold["high_threshold"]
    if strategy == "low":
        return discrepancy < threshold["low_threshold"]
    if strategy == "random":
        return rng.random() < threshold.get("random_rate", threshold.get("high_rate", 0.1))
    return False


def rollout(core, client, cfg, policies, strategy, spec, seed, threshold):
    import torch

    settings = cfg["evaluation"]
    suite, task, initial = (spec[key] for key in ("suite", "task_id", "initial_state_id"))
    horizon = settings["max_steps"][suite]
    settle = settings["settle_steps"]
    result = client.call("reset", suite=suite, task_id=task, initial_state_id=initial,
                         seed=seed, num_steps_wait=settle, horizon=settle + horizon + 1)
    language = result["language"]
    teacher, student = policies
    head = core._base_model(student).action_head.config
    generator = torch.Generator()
    episode_seed = seed + task * 1_000_003 + initial * 1_009
    rng = random.Random(episode_seed)
    discrepancies, replacements = [], 0
    steps = 0

    def predict(policy, observation, noise):
        model = core._base_model(policy)
        features, action_input = core._prepare_backbone(
            model, core._normalize_observation(policy, observation))
        return core._sample_from_cached(model, features, action_input, noise, with_grad=False)

    with torch.no_grad():
        while not result["done"] and steps < horizon:
            observation = core._process_observation(result["observation"], language)
            noise = torch.randn(1, head.action_horizon, head.action_dim,
                                generator=generator.manual_seed((episode_seed + steps) % (2**63 - 1)))
            target = predict(teacher, observation, noise)
            policy, action = teacher, target
            if strategy != "original":
                prediction = predict(student, observation, noise)
                discrepancy = (prediction[:, 0, :7].float() - target[:, 0, :7].float()).square().mean().item()
                discrepancies.append(discrepancy)
                if intervene(strategy, discrepancy, threshold, rng):
                    replacements += 1
                else:
                    policy, action = student, prediction
            command = core._normalized_to_libero_action(policy, action)
            result = client.call("step", action=command.tolist())
            steps += 1
    return dict(spec, seed=seed, strategy=strategy, success=bool(result["done"]),
                steps=steps, interventions=replacements, discrepancy=discrepancies)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--gpu", type=int, default=0)
    cli.add_argument("--seed", type=int)
    cli.add_argument("--port", type=int, default=19760)
    cli.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    cli.add_argument("--run-name")
    cli.add_argument("--dry-run", action="store_true")
    sources = cli.add_mutually_exclusive_group()
    sources.add_argument("--thresholds", type=Path, help="Suite thresholds in the old E2 JSON format.")
    sources.add_argument("--calibration-rollouts", type=Path, help="Directory of baseline suite outputs.")
    cli.set_defaults(method="intervention", clean_port=19761)
    cli.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    cli.add_argument("--max-episodes", type=int, help="Limit per suite for a smoke run.")
    args = cli.parse_args()
    cfg = config(args)
    if cfg["setting"] != "groot_quantvla":
        cli.error("Use the GR00T QuantVLA configuration.")
    if args.max_episodes is not None and args.max_episodes < 1:
        cli.error("--max-episodes must be positive.")
    seed = cfg["evaluation"]["seed"] if args.seed is None else args.seed
    args.seed = seed
    args.benchmark, args.adapter, args.resume = "libero_plus", None, False
    base = Path(cfg["output_root"]) / "motivation/action_intervention/groot_quantvla/seed" / str(seed)
    if args.run_name:
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
            cli.error("Invalid run name.")
        base /= args.run_name
    if set(args.strategies) & {"random", "high", "low"} and not (args.thresholds or args.calibration_rollouts):
        cli.error("Intervention strategies require thresholds or calibration rollouts.")
    if len(set(args.strategies)) != len(args.strategies):
        cli.error("Select each strategy once.")
    thresholds = json.loads(args.thresholds.read_text())["suites"] if args.thresholds else {}
    work = list(plans(cfg, args, "eval"))
    for plan in work:
        suite = plan["suite"]
        if args.calibration_rollouts:
            import numpy as np

            path = args.calibration_rollouts / suite / "episodes.jsonl"
            values = np.asarray([
                q for line in path.read_text().splitlines()
                for q in json.loads(line)["discrepancy"]
            ])
            low, high = np.quantile(values, [0.1, 0.9])
            thresholds[suite] = dict(low_threshold=float(low), high_threshold=float(high),
                                     high_rate=float(np.mean(values > high)), states=len(values))
        threshold = thresholds.get(suite)
        if threshold is None and not (args.thresholds or args.calibration_rollouts):
            print(suite, seed, list(args.strategies), flush=True)
            continue
        if threshold is None:
            cli.error(f"Missing thresholds for {suite}.")
        low, high = threshold["low_threshold"], threshold["high_threshold"]
        rate = threshold.get("random_rate", threshold.get("high_rate", 0.1))
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high and 0 <= rate <= 1):
            cli.error(f"Invalid thresholds for {plan['suite']}.")
        print(plan["suite"], seed, list(args.strategies), threshold, flush=True)
    if args.dry_run:
        return

    for strategy in args.strategies:
        if (base / strategy).exists():
            cli.error(f"Output already exists: {base / strategy}. Use --run-name for another run.")
    base.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    upstream = Path(cfg["paths"]["quantvla"])
    sys.path.insert(0, str(upstream))
    from pivot_q.groot_runtime import train as core
    from tqdm.auto import tqdm

    all_rows = {strategy: [] for strategy in args.strategies}
    expected = {}
    for plan in work:
        suite = plan["suite"]
        specs = episode_specs(cfg, "libero_plus", suite)[:args.max_episodes]
        expected[suite] = len(specs)
        run_dir = base / "services" / suite
        run_dir.mkdir(parents=True, exist_ok=True)
        plan["output"] = str(run_dir)
        model_cfg = core.TrainConfig(**groot_config(cfg, plan))
        with services(cfg, args, run_dir, suite=suite):
            client = core.EnvClient("127.0.0.1", args.port)
            try:
                client.wait_until_ready()
                teacher = core._load_policy(model_cfg, quantized=False, repo_root=upstream)
                student = core._load_policy(model_cfg, quantized=True, repo_root=upstream)
                for strategy in args.strategies:
                    out = base / strategy / suite
                    out.mkdir(parents=True)
                    with (out / "episodes.jsonl").open("x") as stream:
                        for spec in tqdm(specs, desc=f"{strategy} {suite}"):
                            row = rollout(core, client, cfg, (teacher, student), strategy, spec, seed, thresholds.get(suite, {}))
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            all_rows[strategy].append(row)
                del teacher, student
            finally:
                client.close()
    for strategy, rows in all_rows.items():
        summary = summarize(rows, expected)
        summary.update(thresholds=thresholds,
                       denoising_steps=cfg["model"]["num_inference_steps"],
                       interventions=sum(row["interventions"] for row in rows),
                       steps=sum(row["steps"] for row in rows))
        (base / strategy / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
