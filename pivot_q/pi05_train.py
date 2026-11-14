#!/usr/bin/env python3
"""Single-device, round-based recovery training (no distributed runtime)."""
from __future__ import annotations

import json
import random
import shutil
import time
from .cost import scope
from .plot_data import save_selection
from tqdm.auto import tqdm
from collections import deque
from pathlib import Path
from types import SimpleNamespace

def run(values, *, bundle_loader, rollout_fn, meter):
    started_run = time.monotonic()
    import numpy as np
    import torch
    from .pi05_state import EnvClient, HORIZONS, write_status
    from .pivot_q import action_error, future_risk, select, select_all

    cfg = SimpleNamespace(**values)
    if cfg.episodes_per_suite <= 0 or cfg.episodes_per_suite % cfg.rollouts_per_round:
        raise ValueError("episodes_per_suite must be positive and divisible by rollouts_per_round")
    load_bundle, run_rollout = bundle_loader, rollout_fn
    out = Path(cfg.output_dir)
    checkpoints = out / "checkpoints"
    complete = sorted(p for p in checkpoints.glob("round-*") if (p / "complete.json").is_file())
    saved = None
    adapter = None
    if complete:
        if not cfg.resume:
            raise RuntimeError("Output already contains checkpoints; use resume or a new output directory")
        latest = complete[-1]
        saved = torch.load(latest / "trainer_state.pt", map_location="cpu", weights_only=False)
        if saved.get("format") != "pivot_q_4090_pi05_round_v2":
            raise RuntimeError("Only single-device checkpoints are supported; multi-GPU conversion is not implicit")
        adapter = latest / "adapter"

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
        for lane in range(cfg.rollouts_per_round):
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
            if "compute" in saved:
                meter.restore(saved["compute"])
            orders = saved["orders"]
            for lane, state in zip(streams, saved["streams"], strict=True):
                lane["rng"].setstate(state["rng"])
                lane["noise"].set_state(state["noise"])
                lane["replay"].extend(state["replay"])
                for key in ("torch", "cuda", "numpy"):
                    lane[key] = state[key]
        update_rng = saved["update_rng"] if saved else torch.cuda.get_rng_state()
        total = len(cfg.suite_order) * (cfg.episodes_per_suite // cfg.rollouts_per_round)
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
        write_status(out / "run.json", {"format": "pivot_q_4090_pi05_round_v2", "config": values,
                                          "physical_gpus": 1, "logical_streams": cfg.rollouts_per_round})

        def activate(lane):
            torch.set_rng_state(lane["torch"])
            torch.cuda.set_rng_state(lane["cuda"])
            np.random.set_state(lane["numpy"])

        def capture(lane):
            lane["torch"] = torch.get_rng_state()
            lane["cuda"] = torch.cuda.get_rng_state()
            lane["numpy"] = np.random.get_state()

        elapsed = float(saved.get("elapsed_seconds", 0)) if saved else 0.0
        torch.cuda.reset_peak_memory_stats()
        progress = tqdm(range(start, end), initial=start, total=total,
                        desc="Training rounds", unit="round", dynamic_ncols=True)
        for round_index in progress:
            suite_index, local_round = divmod(round_index, cfg.episodes_per_suite // cfg.rollouts_per_round)
            suite = cfg.suite = cfg.suite_order[suite_index]
            with scope(f"{suite}/"):
                started, batches, records = time.time(), [], []
                for lane_index, lane in enumerate(streams):
                    activate(lane)
                    offset = cfg.rollouts_per_round * local_round + lane_index
                    progress.set_postfix(suite=suite, rollouts=f"{round_index * cfg.rollouts_per_round + lane_index}/{total * cfg.rollouts_per_round}",
                                         step=step)
                    task = orders[suite][offset]
                    initial_state_id = lane["rng"].choice(cfg.initial_state_ids)
                    states, success = run_rollout(cfg, teacher, student, env, task_id=task,
                        initial_state_id=initial_state_id,
                        horizon=HORIZONS[suite], generator=lane["noise"])
                    if not states:
                        raise RuntimeError(f"Empty rollout: {suite}, task {task}")
                    q = action_error(torch.cat([s.student_action for s in states]),
                                     torch.cat([s.teacher_action for s in states]))
                    if cfg.selection_method == "full_distill":
                        chosen = select_all(q)
                    else:
                        risk = future_risk(q, cfg.temporal_horizon, cfg.temporal_discount)
                        chosen = select(q, risk,
                            phase_bins=cfg.phase_bins, top_per_phase=cfg.top_per_phase,
                            min_gap=cfg.min_temporal_gap, alpha=cfg.alpha_q, beta=cfg.beta_r,
                            weight_min=cfg.weight_min, weight_max=cfg.weight_max, percent=getattr(cfg, "percent", None))
                        save_selection(out, setting=f"pi05_{cfg.backend}", suite=suite, seed=cfg.seed, round_index=round_index + 1,
                            suite_round=local_round + 1, rollout=lane_index, task_id=task,
                            initial_state_id=initial_state_id, method=cfg.selection_method,
                            q=q, risk=risk, scores=chosen.candidate_scores, indices=chosen.indices,
                            weights=chosen.weights, phases=cfg.phase_bins, percent=getattr(cfg, "percent", None))
                    if clean:
                        with scope("anchor/"):
                            anchors, _ = run_rollout(cfg, teacher, student, clean,
                                task_id=lane["rng"].randrange(10), initial_state_id=lane["rng"].randrange(40),
                                horizon=cfg.clean_anchor_horizon, generator=lane["noise"])
                        lane["replay"].extend(anchors)
                    capture(lane)
                    # State stores observations, noise and targets on CPU already.
                    batches.append(([states[i] for i in chosen.indices], chosen.weights))
                    records.append({"suite": suite, "task_id": task, "success": success,
                                    "states": len(states), "selected": len(chosen.indices)})
                    write_status(out / "status.json", {"status": "collecting", "round": round_index + 1,
                        "streams_collected": lane_index + 1, "optimizer_step": step, "updated_at": time.time()})
                    tqdm.write(f"Round {round_index + 1}: stream {lane_index + 1}/{cfg.rollouts_per_round} collected {len(states)} states")
                rollout_seconds = time.time() - started
                weighted = [(state, float(weight)) for states, weights in batches
                            for state, weight in zip(states, weights, strict=True)]
                random.Random(cfg.seed + round_index).shuffle(weighted)
                losses = []
                with tqdm(range(0, len(weighted), cfg.batch_size), desc="State batches",
                          leave=False, dynamic_ncols=True) as batch_progress:
                    for offset in batch_progress:
                        group = weighted[offset:offset + cfg.batch_size]
                        mass = sum(weight for _, weight in group)
                        group = [(state, weight / mass) for state, weight in group]
                        anchors = []
                        for lane in streams:
                            if lane["replay"] and cfg.lambda_anchor > 0:
                                chosen = lane["rng"].sample(list(lane["replay"]),
                                                           min(cfg.anchor_batch_size, len(lane["replay"])))
                                anchors.extend(chosen)
                        anchor_states = [(state, cfg.lambda_anchor / len(anchors)) for state in anchors]
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.set_rng_state(update_rng)
                        with adapter_dropout(student.policy.model):
                            distill_loss = weighted_backward(student, group, cfg.batch_size,
                                                             meter=meter, stage="distill_forward_backward",
                                                             checkpoint_denoising=True)
                            anchor_loss = weighted_backward(student, anchor_states, cfg.batch_size,
                                                            meter=meter, stage="anchor_forward_backward",
                                                             checkpoint_denoising=True)
                        update_rng = torch.cuda.get_rng_state()
                        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip_norm)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("Non-finite gradient")
                        meter.run("optimizer_step", 1, optimizer.step)
                        step += 1
                        losses.append((distill_loss, anchor_loss, len(group), mass))
                        batch_progress.set_postfix(step=step)
            weight_sum = sum(row[3] for row in losses)
            record = {"round": round_index + 1, "optimizer_step": step, "rollouts_complete": (round_index + 1) * cfg.rollouts_per_round,
                      "gradient_norm": float(norm), "duration_seconds": time.time() - started,
                      "rollout_and_replay_seconds": rollout_seconds, "updates_seconds": time.time() - started - rollout_seconds,
                       "distill_loss": sum(row[0] * row[3] for row in losses) / weight_sum,
                      "anchor_loss": sum(row[1] * row[2] for row in losses) / len(weighted),
                      "optimizer_steps_this_round": len(losses),
                      "distill_state_evaluations": len(weighted),
                      "elapsed_seconds": elapsed + time.monotonic() - started_run, "streams": records}
            with metrics.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            tqdm.write(json.dumps(record))
            if ((round_index + 1) % cfg.save_every_rounds == 0
                    or (round_index + 1) % (cfg.episodes_per_suite // cfg.rollouts_per_round) == 0
                    or round_index + 1 == end):
                checkpoints.mkdir(exist_ok=True)
                final = checkpoints / f"round-{round_index + 1:06d}"
                temporary = checkpoints / f".round-{round_index + 1:06d}.incomplete"
                if final.exists():
                    raise RuntimeError(f"Refusing to overwrite {final}")
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir()
                student.policy.model.save_pretrained(temporary / "adapter")
                lane_states = [{"rng": s["rng"].getstate(), "noise": s["noise"].get_state(),
                    "replay": list(s["replay"]), **{k: s[k] for k in ("torch", "cuda", "numpy")}} for s in streams]
                torch.save({"format": "pivot_q_4090_pi05_round_v2", "round": round_index + 1,
                    "optimizer_step": step, "optimizer": optimizer.state_dict(), "streams": lane_states,
                    "orders": orders, "update_rng": update_rng, "elapsed_seconds": record["elapsed_seconds"], "compute": meter.summary()}, temporary / "trainer_state.pt")
                (temporary / "complete.json").write_text(json.dumps({**record, "format": "pivot_q_4090_pi05_round_v2"}))
                temporary.rename(final)
                if cfg.keep_last_checkpoints > 0:
                    old = sorted(p for p in checkpoints.glob("round-*") if (p / "complete.json").exists())
                    for path in old[:-cfg.keep_last_checkpoints]:
                        shutil.rmtree(path)
            write_status(out / "status.json", {**record, "status": "complete" if round_index + 1 == total else ("paused" if round_index + 1 == end else "running"),
                                                "updated_at": time.time()})
        if end > start:
            write_status(out / "train_summary.json", {**record, "allocated_gpu_hours": (elapsed + time.monotonic() - started_run) / 3600,
                "peak_allocated_bytes_this_session": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes_this_session": torch.cuda.max_memory_reserved(), "compute": meter.summary()})
        if end == total:
            final = checkpoints / f"round-{total:06d}"
            final_adapter = out / "final_adapter"
            if final_adapter.exists():
                shutil.rmtree(final_adapter)
            shutil.copytree(final / "adapter", final_adapter)
    finally:
        if "progress" in locals():
            progress.close()
        env.close()
        if clean:
            clean.close()



from contextlib import contextmanager

@contextmanager
def adapter_dropout(model):
    from torch import nn
    selected = {}
    for module in model.modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b") and isinstance(getattr(module, "dropout", None), nn.Dropout):
            selected[module.dropout] = module.dropout.training
        for dropout in getattr(module, "lora_dropout", {}).values():
            selected[dropout] = dropout.training
    try:
        for module in selected:
            module.train(True)
        yield
    finally:
        for module, training in selected.items():
            module.train(training)

def weighted_backward(student, samples, batch_size, *, meter=None, stage="distill",
                      checkpoint_denoising=False):
    import torch
    from .pi05_flow import sample

    total = torch.zeros((), device=student.policy.config.device)
    for offset in range(0, len(samples), batch_size):
        group = samples[offset:offset+batch_size]
        inputs = [student.preprocessor(state.raw_sample) for state, _ in group]
        batch = {key: torch.cat([item[key] for item in inputs], dim=0)
                 for key, value in inputs[0].items() if isinstance(value, torch.Tensor)}
        noise = torch.cat([s.noise for s, _ in group]).to(student.policy.config.device)
        def forward_backward():
            prediction = sample(student.policy, batch, noise, with_grad=True,
                                checkpoint_denoising=checkpoint_denoising)
            target = torch.cat([s.teacher_action for s, _ in group]).to(prediction.device)
            loss = (prediction[:, 0, :7].float() - target[:, 0, :7].float()).square().mean(dim=-1)
            weighted = (loss * loss.new_tensor([w for _, w in group])).sum()
            weighted.backward()
            return weighted.detach()
        total += (meter.run(stage, len(group), forward_backward) if meter else forward_backward())
    return float(total)
