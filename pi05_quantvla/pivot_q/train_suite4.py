#!/usr/bin/env python3
"""Four-GPU sequential-suite PIVOT-Q training for the π0.5 policy.

All four ranks collect distinct rollouts from the same suite in each round.
Suites run in the fixed order LIBERO-10, Spatial, Object, Goal. The global
budget remains 560 rollouts and 700 synchronized optimizer updates.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tyro

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pi05_quantvla.pivot_q.selection import action_error, future_risk, select
from pi05_quantvla.pivot_q.train import (
    Bundle, EnvClient, HORIZONS, State, load_bundle, run_rollout, state_loss,
    write_status,
)
from pi05_quantvla.pivot_q.train_ddp import (
    Config, average_gradients, broadcast_checkpoint, latest_complete_checkpoint,
    rank_config, rank_runtime_state, save_checkpoint, set_final_adapter,
    synchronize_parameters,
)


ORDER = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


@dataclass
class Suite4Config(Config):
    output_dir: str = str(
        ROOT / "outputs/quantvla/pi05/pivot_q_suite4/train/seed-000/shared"
    )
    method_name: str = "pi05_pivot_q_suite4"
    suite_order: list[str] = field(default_factory=lambda: list(ORDER))

    def validate(self) -> None:
        super().validate()
        if self.selection_method != "pivot_q":
            raise ValueError("suite4 mode requires PIVOT-Q selection")
        if tuple(self.suite_order) != ORDER:
            raise ValueError(f"suite_order must be {ORDER}")
        if self.episodes_per_suite % 4:
            raise ValueError("episodes_per_suite must be divisible by four ranks")


def task_order(task_metadata: list[dict], *, seed: int, suite_index: int) -> list[int]:
    task_ids = [int(item["task_id"]) for item in task_metadata]
    if len(task_ids) % 4:
        raise ValueError("suite task count must be divisible by four ranks")
    random.Random(seed + 1000 * suite_index).shuffle(task_ids)
    return task_ids


def truncate_by_round(path: Path, last_round: int) -> None:
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("round", -1)) <= last_round:
            kept.append(json.dumps(record))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def run() -> None:
    config = tyro.cli(Suite4Config)
    config.validate()
    rounds_per_suite = config.episodes_per_suite // 4
    total_rounds = rounds_per_suite * len(config.suite_order)
    if config.dry_run:
        print(json.dumps({
            **dataclasses.asdict(config), "world_size": 4,
            "rounds_per_suite": rounds_per_suite, "total_rounds": total_rounds,
            "total_rollouts": total_rounds * 4,
            "optimizer_steps": total_rounds * config.updates_per_episode,
        }, indent=2))
        return

    from env.protocol import check_training_output, write_training_protocol
    protocol = check_training_output(config.output_dir, dataclasses.asdict(config))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("suite4 training requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(hours=2),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    local = rank_config(config, rank, local_rank)
    output = Path(config.output_dir).expanduser().resolve()
    dist.barrier()
    if rank == 0:
        write_training_protocol(output, protocol)
    dist.barrier()
    metrics_dir, status_dir = output / "metrics", output / "status"
    if rank == 0:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        status_dir.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(json.dumps({
            "schema_version": 2, "method": config.method_name,
            "training_scope": "four_gpu_sequential_suite_minibatch",
            "world_size": world_size, "suite_order": config.suite_order,
            "rounds_per_suite": rounds_per_suite, "total_rounds": total_rounds,
            "total_rollouts": total_rounds * world_size, "seed": config.seed,
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "created_at": time.time(), "config": dataclasses.asdict(config),
        }, indent=2), encoding="utf-8")
        write_status(output / "status.json", {
            "status": "starting", "round": 0, "rounds_total": total_rounds,
            "rollouts_complete": 0, "rollouts_total": total_rounds * world_size,
            "optimizer_step": 0,
            "optimizer_steps_total": total_rounds * config.updates_per_episode,
            "suite_order": config.suite_order, "updated_at": time.time(),
        })
    dist.barrier()

    checkpoint = latest_complete_checkpoint(output, world_size) if rank == 0 and config.resume else None
    checkpoint = broadcast_checkpoint(checkpoint, rank)
    trainer_state = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
    ) if checkpoint else None
    saved_rank_state = torch.load(
        checkpoint / f"rank-{rank:02d}-state.pt", map_location="cpu", weights_only=False
    ) if checkpoint else None
    start_round = int(trainer_state["episode"]) if trainer_state else 0
    optimizer_step = int(trainer_state["optimizer_step"]) if trainer_state else 0
    if trainer_state and int(trainer_state["world_size"]) != world_size:
        raise RuntimeError("checkpoint world size mismatch")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed + rank)
    rng = random.Random(config.seed + rank)
    generator = torch.Generator(device=local.device).manual_seed(config.seed + rank)
    replay: deque[State] = deque(maxlen=config.anchor_replay_size)
    if saved_rank_state:
        rng.setstate(saved_rank_state["python_rng_state"])
        np.random.set_state(saved_rank_state["numpy_rng_state"])
        torch.set_rng_state(saved_rank_state["torch_rng_state"])
        torch.cuda.set_rng_state(saved_rank_state["cuda_rng_state"], device=local_rank)
        generator.set_state(saved_rank_state["noise_generator_state"])
        replay.extend(saved_rank_state["anchor_replay"])

    rollout_env = EnvClient(local.env_port)
    clean_env = EnvClient(local.clean_env_port) if config.lambda_anchor > 0 else None
    teacher: Bundle | None = None
    student: Bundle | None = None
    global_metrics = None
    suite_metrics: dict[str, Any] = {}
    try:
        rollout_env.wait()
        if clean_env:
            clean_env.wait()
        print(f"[rank {rank}] loading π0.5 FP16 teacher", flush=True)
        teacher = load_bundle(local, quantized=False)
        print(f"[rank {rank}] loading π0.5 QuantVLA student", flush=True)
        student = load_bundle(local, quantized=True, adapter=checkpoint / "adapter" if checkpoint else None)
        parameters = [p for p in student.policy.model.parameters() if p.requires_grad]
        synchronize_parameters(parameters)
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        if trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        dist.barrier()

        if rank == 0:
            if config.resume:
                truncate_by_round(output / "metrics.jsonl", start_round)
                for suite in config.suite_order:
                    truncate_by_round(metrics_dir / f"{suite}.jsonl", start_round)
            global_metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
            suite_metrics = {
                suite: (metrics_dir / f"{suite}.jsonl").open("a", encoding="utf-8", buffering=1)
                for suite in config.suite_order
            }
        dist.barrier()

        active_suite, task_ids, category_by_task = "", [], {}
        recent_durations: deque[float] = deque(maxlen=20)
        for round_index in range(start_round, total_rounds):
            suite_index, suite_round = divmod(round_index, rounds_per_suite)
            suite = config.suite_order[suite_index]
            local.suite = suite
            if active_suite != suite:
                metadata = rollout_env.call("list_tasks", suite=suite)
                task_ids = task_order(metadata, seed=config.seed, suite_index=suite_index)
                if len(task_ids) != config.episodes_per_suite:
                    raise ValueError(f"{suite}: expected {config.episodes_per_suite} tasks, got {len(task_ids)}")
                category_by_task = {int(item["task_id"]): str(item["category"]) for item in metadata}
                active_suite = suite
            task_offset = 4 * suite_round + rank
            task_id, suite_episode = task_ids[task_offset], task_offset + 1
            started = time.time()
            states, success = run_rollout(
                local, teacher, student, rollout_env, task_id=task_id,
                initial_state_id=rng.choice(config.initial_state_ids),
                horizon=HORIZONS[suite], generator=generator,
            )
            if not states:
                raise RuntimeError(f"rank {rank} {suite} episode {suite_episode} produced no states")
            student_actions = torch.cat([state.student_action for state in states])
            teacher_actions = torch.cat([state.teacher_action for state in states])
            q = action_error(student_actions, teacher_actions)
            future = future_risk(q, config.temporal_horizon, config.temporal_discount)
            chosen = select(
                q, future, phase_bins=config.phase_bins,
                top_per_phase=config.top_per_phase, min_gap=config.min_temporal_gap,
                alpha=config.alpha_q, beta=config.beta_r,
                weight_min=config.weight_min, weight_max=config.weight_max,
            )
            if clean_env is not None:
                clean_states, _ = run_rollout(
                    local, teacher, student, clean_env, task_id=rng.randrange(10),
                    initial_state_id=rng.randrange(40),
                    horizon=config.clean_anchor_horizon, generator=generator,
                )
                replay.extend(clean_states)

            losses, anchor_losses, gradient_norms = [], [], []
            weights = chosen.weights / chosen.weights.sum().clamp_min(1e-8)
            for _ in range(config.updates_per_episode):
                optimizer.zero_grad(set_to_none=True)
                for index, weight in zip(chosen.indices, weights, strict=True):
                    loss = state_loss(student, states[index])
                    (loss * weight.to(loss.device)).backward()
                    losses.append(float(loss.detach().cpu()))
                if replay and config.lambda_anchor > 0 and config.anchor_batch_size > 0:
                    anchors = rng.sample(list(replay), min(config.anchor_batch_size, len(replay)))
                    for state in anchors:
                        loss = state_loss(student, state)
                        (loss * config.lambda_anchor / len(anchors)).backward()
                        anchor_losses.append(float(loss.detach().cpu()))
                average_gradients(parameters, world_size)
                gradient = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
                if not torch.isfinite(torch.as_tensor(gradient)):
                    raise FloatingPointError(f"non-finite synchronized gradient at round {round_index + 1}")
                optimizer.step()
                optimizer_step += 1
                gradient_norms.append(float(gradient))

            duration = time.time() - started
            recent_durations.append(duration)
            selected_index = torch.as_tensor(chosen.indices, device=q.device)
            record = {
                "round": round_index + 1, "suite_round": suite_round + 1,
                "episode": suite_episode, "optimizer_step": optimizer_step,
                "rank": rank, "method": config.method_name,
                "training_scope": "four_gpu_sequential_suite_minibatch",
                "suite": suite, "task_id": task_id,
                "category": category_by_task[task_id], "episode_steps": len(states),
                "episode_success": success,
                "selected_state_count": len(chosen.indices),
                "selected_indices": list(chosen.indices),
                "selected_phases": list(chosen.phases),
                "phase_effective_gaps": list(chosen.effective_gaps),
                "q_mean": float(q.mean()), "q_max": float(q.max()),
                "r_mean": float(future.mean()), "r_max": float(future.max()),
                "selected_q": q[selected_index].detach().cpu().tolist(),
                "selected_r": future[selected_index].detach().cpu().tolist(),
                "q_by_timestep": q.detach().cpu().tolist(),
                "r_by_timestep": future.detach().cpu().tolist(),
                "loss_pivot_q_unweighted_mean": float(np.mean(losses)),
                "loss_anchor_mean": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "duration_seconds": round(duration, 3),
            }
            write_status(status_dir / f"rank-{rank:02d}.json", {
                "status": "complete" if round_index + 1 == total_rounds else "running",
                "rank": rank, "suite": suite, "round": round_index + 1,
                "rounds_total": total_rounds, "suite_episode": suite_episode,
                "task_id": task_id, "optimizer_step": optimizer_step,
                "updated_at": time.time(),
            })
            gathered: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered, record)
            if rank == 0:
                records = [item for item in gathered if item is not None]
                assert len(records) == world_size and global_metrics is not None
                for item in records:
                    suite_metrics[suite].write(json.dumps(item) + "\n")
                global_record = {
                    "round": round_index + 1, "suite": suite,
                    "suite_round": suite_round + 1, "optimizer_step": optimizer_step,
                    "rollouts_complete": (round_index + 1) * world_size,
                    "rollouts_total": total_rounds * world_size,
                    "suite_records": records,
                    "mean_q": float(np.mean([item["q_mean"] for item in records])),
                    "mean_loss": float(np.mean([item["loss_pivot_q_unweighted_mean"] for item in records])),
                    "round_duration_seconds": max(item["duration_seconds"] for item in records),
                }
                global_metrics.write(json.dumps(global_record) + "\n")
                write_status(output / "status.json", {
                    "status": "complete" if round_index + 1 == total_rounds else "running",
                    "suite": suite, "round": round_index + 1,
                    "rounds_total": total_rounds, "suite_round": suite_round + 1,
                    "rounds_per_suite": rounds_per_suite,
                    "suite_rollouts_complete": min(config.episodes_per_suite, (suite_round + 1) * world_size),
                    "rollouts_complete": (round_index + 1) * world_size,
                    "rollouts_total": total_rounds * world_size,
                    "optimizer_step": optimizer_step,
                    "optimizer_steps_total": total_rounds * config.updates_per_episode,
                    "eta_seconds": float(np.median(recent_durations)) * (total_rounds - round_index - 1),
                    "updated_at": time.time(),
                })
                print(json.dumps(global_record), flush=True)

            if optimizer_step % config.save_every_steps == 0 or round_index + 1 == total_rounds:
                save_checkpoint(
                    output=output, student=student, optimizer=optimizer,
                    episode=round_index + 1, optimizer_step=optimizer_step,
                    config=config, rank=rank, world_size=world_size,
                    rank_state=rank_runtime_state(
                        suite=suite, rng=rng, generator=generator,
                        task_ids=task_ids, replay=replay, local_rank=local_rank,
                    ),
                )
        if rank == 0:
            set_final_adapter(output)
    finally:
        if global_metrics is not None:
            global_metrics.close()
        for handle in suite_metrics.values():
            handle.close()
        rollout_env.close()
        if clean_env is not None:
            clean_env.close()
        if teacher is not None and teacher.scales is not None:
            teacher.scales.close()
        if student is not None and student.scales is not None:
            student.scales.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
