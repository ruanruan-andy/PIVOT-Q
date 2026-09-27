#!/usr/bin/env python3
"""Single-device, four-logical-stream recovery training (no distributed runtime)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def protocol_signature(values):
    return {k: v for k, v in values.items() if k not in {
        "resume", "output_dir", "env_port", "clean_env_port", "max_rounds"
    }}


def validate_initial_restart(out, values):
    """No saved weights exist: only replay an identified run from step zero."""
    metadata = out / "run.json"
    if not values["resume"] or not metadata.is_file():
        raise RuntimeError("Early restart requires resume and matching run.json")
    previous = json.loads(metadata.read_text())
    if (previous.get("format") != "pivot_q_single_v1" or
            protocol_signature(previous.get("config", {})) != protocol_signature(values)):
        raise RuntimeError("Early restart configuration differs from the saved training protocol")
    if (out / "final_adapter").exists() or (out / "final_adapter").is_symlink():
        raise RuntimeError("Final adapter exists without a complete checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.runtime_config.read_text()))


def run(values):
    import numpy as np
    import torch
    from pi05_quantvla.pivot_q.train import EnvClient, HORIZONS, run_rollout, state_loss, write_status
    from pi05_quantvla.pivot_q.selection import action_error, future_risk, select, select_all

    cfg = SimpleNamespace(**values)
    if cfg.schedule != "sequential" or cfg.suite_order != ["libero_object", "libero_spatial", "libero_goal", "libero_10"]:
        raise ValueError("Single-GPU training uses Object, Spatial, Goal, Long in that order")
    if cfg.episodes_per_suite <= 0 or cfg.episodes_per_suite % 4:
        raise ValueError("episodes_per_suite must be positive and divisible by four")
    if cfg.backend == "holoq":
        from pi05_omegavla.backend import load_bundle
    else:
        from pi05_quantvla.pivot_q.train import load_bundle
    out = Path(cfg.output_dir)
    checkpoints = out / "checkpoints"
    signature = protocol_signature(values)
    complete = sorted(p for p in checkpoints.glob("step-*") if (p / "complete.json").is_file())
    saved = None
    adapter = None
    if complete:
        if not cfg.resume:
            raise RuntimeError("Output already contains checkpoints; use resume or a new output directory")
        latest = complete[-1]
        saved = torch.load(latest / "trainer_state.pt", map_location="cpu", weights_only=False)
        if saved.get("format") != "pivot_q_single_v1":
            raise RuntimeError("Only single-device checkpoints are supported; multi-GPU conversion is not implicit")
        if saved["signature"] != signature:
            raise RuntimeError("Resume configuration differs from the saved training protocol")
        adapter = latest / "adapter"
    elif (out / "metrics.jsonl").exists():
        validate_initial_restart(out, values)
        print("No complete checkpoint: replaying from step zero with the original seed and protocol", flush=True)

    torch.cuda.set_device(0)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    env, clean = EnvClient(cfg.env_port), None
    try:
        env.wait()
        if cfg.lambda_anchor > 0:
            clean = EnvClient(cfg.clean_env_port)
            clean.wait()
        print("Loading teacher and student on cuda:0", flush=True)
        teacher = load_bundle(cfg, quantized=False)
        student = load_bundle(cfg, quantized=True, adapter=adapter)
        parameters = [p for p in student.policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        streams = []
        for lane in range(4):
            streams.append({
                "rng": random.Random(cfg.seed + lane),
                "noise": torch.Generator(device="cuda:0").manual_seed(cfg.seed + lane),
                "replay": deque(maxlen=cfg.anchor_replay_size),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                "numpy": np.random.RandomState(cfg.seed + lane).get_state(),
            })
        orders = {}
        for index, suite in enumerate(cfg.suite_order):
            ids = [int(x["task_id"]) for x in env.call("list_tasks", suite=suite)]
            if len(ids) != cfg.episodes_per_suite:
                raise ValueError(f"Manifest count for {suite}: {len(ids)} != {cfg.episodes_per_suite}")
            random.Random(cfg.seed + 1000 * index).shuffle(ids)
            orders[suite] = ids

        start, step = 0, 0
        if saved:
            start, step = saved["round"], saved["optimizer_step"]
            optimizer.load_state_dict(saved["optimizer"])
            orders = saved["orders"]
            for lane, state in zip(streams, saved["streams"], strict=True):
                lane["rng"].setstate(state["rng"])
                lane["noise"].set_state(state["noise"])
                lane["replay"].extend(state["replay"])
                for key in ("torch", "cuda", "numpy"):
                    lane[key] = state[key]
        total = cfg.episodes_per_suite
        end = min(total, start + cfg.max_rounds) if cfg.max_rounds else total
        metrics = out / "metrics.jsonl"
        if metrics.exists():
            rows = []
            for line in metrics.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["round"] <= start:
                    rows.append(line)
            metrics.write_text("".join(line + "\n" for line in rows))
        write_status(out / "run.json", {"format": "pivot_q_single_v1", "config": values,
                                          "physical_gpus": 1, "logical_streams": 4})

        def activate(lane):
            torch.set_rng_state(lane["torch"])
            torch.cuda.set_rng_state(lane["cuda"])
            np.random.set_state(lane["numpy"])

        def capture(lane):
            lane["torch"] = torch.get_rng_state()
            lane["cuda"] = torch.cuda.get_rng_state()
            lane["numpy"] = np.random.get_state()

        for round_index in range(start, end):
            started, batches, records = time.time(), [], []
            for lane_index, lane in enumerate(streams):
                activate(lane)
                suite_index, local_round = divmod(round_index, cfg.episodes_per_suite // 4)
                suite, offset = cfg.suite_order[suite_index], 4 * local_round + lane_index
                cfg.suite = suite
                task = orders[suite][offset]
                states, success = run_rollout(cfg, teacher, student, env, task_id=task,
                    initial_state_id=lane["rng"].choice(cfg.initial_state_ids),
                    horizon=HORIZONS[suite], generator=lane["noise"])
                if not states:
                    raise RuntimeError(f"Empty rollout: {suite}, task {task}")
                q = action_error(torch.cat([s.student_action for s in states]),
                                 torch.cat([s.teacher_action for s in states]))
                if cfg.selection_method == "full_distill":
                    chosen = select_all(q)
                else:
                    chosen = select(q, future_risk(q, cfg.temporal_horizon, cfg.temporal_discount),
                        phase_bins=cfg.phase_bins, top_per_phase=cfg.top_per_phase,
                        min_gap=cfg.min_temporal_gap, alpha=cfg.alpha_q, beta=cfg.beta_r,
                        weight_min=cfg.weight_min, weight_max=cfg.weight_max)
                if clean:
                    anchors, _ = run_rollout(cfg, teacher, student, clean,
                        task_id=lane["rng"].randrange(10), initial_state_id=lane["rng"].randrange(40),
                        horizon=cfg.clean_anchor_horizon, generator=lane["noise"])
                    lane["replay"].extend(anchors)
                capture(lane)
                # State stores observations, noise and targets on CPU already.
                batches.append(([states[i] for i in chosen.indices], chosen.weights / chosen.weights.sum()))
                records.append({"suite": suite, "task_id": task, "success": success,
                                "states": len(states), "selected": len(chosen.indices)})
                write_status(out / "status.json", {"status": "collecting", "round": round_index + 1,
                    "streams_collected": lane_index + 1, "optimizer_step": step, "updated_at": time.time()})
                print(f"Round {round_index + 1}: stream {lane_index + 1}/4 collected {len(states)} states", flush=True)
            for _ in range(cfg.updates_per_episode):
                optimizer.zero_grad(set_to_none=True)
                for lane, (states, weights) in zip(streams, batches, strict=True):
                    activate(lane)
                    for state, weight in zip(states, weights, strict=True):
                        loss = state_loss(student, state)
                        (loss * weight.to(loss.device) / 4).backward()
                    if lane["replay"] and cfg.lambda_anchor > 0 and cfg.anchor_batch_size > 0:
                        anchors = lane["rng"].sample(list(lane["replay"]), min(cfg.anchor_batch_size, len(lane["replay"])))
                        for state in anchors:
                            (state_loss(student, state) * cfg.lambda_anchor / (4 * len(anchors))).backward()
                    capture(lane)
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite accumulated gradient")
                optimizer.step()
                step += 1
            record = {"round": round_index + 1, "optimizer_step": step, "rollouts_complete": (round_index + 1) * 4,
                      "gradient_norm": float(norm), "duration_seconds": time.time() - started, "streams": records}
            with metrics.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if step % cfg.save_every_steps == 0 or round_index + 1 == end:
                checkpoints.mkdir(exist_ok=True)
                final = checkpoints / f"step-{step:06d}"
                temporary = checkpoints / f".step-{step:06d}.incomplete"
                if final.exists():
                    raise RuntimeError(f"Refusing to overwrite {final}")
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir()
                student.policy.model.save_pretrained(temporary / "adapter")
                lane_states = [{"rng": s["rng"].getstate(), "noise": s["noise"].get_state(),
                    "replay": list(s["replay"]), **{k: s[k] for k in ("torch", "cuda", "numpy")}} for s in streams]
                torch.save({"format": "pivot_q_single_v1", "signature": signature, "round": round_index + 1,
                    "optimizer_step": step, "optimizer": optimizer.state_dict(), "streams": lane_states,
                    "orders": orders}, temporary / "trainer_state.pt")
                (temporary / "complete.json").write_text(json.dumps({**record, "format": "pivot_q_single_v1"}))
                temporary.rename(final)
                if cfg.keep_last_checkpoints > 0:
                    old = sorted(p for p in checkpoints.glob("step-*") if (p / "complete.json").exists())
                    for path in old[:-cfg.keep_last_checkpoints]:
                        shutil.rmtree(path)
            write_status(out / "status.json", {**record, "status": "complete" if round_index + 1 == total else ("paused" if round_index + 1 == end else "running"),
                                                "updated_at": time.time()})
        if end == total:
            link = out / "final_adapter"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(Path("checkpoints") / f"step-{step:06d}" / "adapter")
    finally:
        env.close()
        if clean:
            clean.close()


if __name__ == "__main__":
    main()
