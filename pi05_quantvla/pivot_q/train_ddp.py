#!/usr/bin/env python3
"""Synchronous four-suite PIVOT_Q training for the single π0.5 LIBERO policy."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import shutil
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tyro

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pi05_quantvla.pivot_q.selection import action_error, future_risk, select, select_all
from pi05_quantvla.pivot_q.train import (
    HORIZONS,
    Bundle,
    EnvClient,
    State,
    load_bundle,
    run_rollout,
    state_loss,
    write_status,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass
class Config:
    suites: list[str] = field(default_factory=lambda: list(SUITES))
    env_ports: list[int] = field(default_factory=lambda: [6100, 6110, 6120, 6130])
    clean_env_ports: list[int] = field(default_factory=lambda: [6101, 6111, 6121, 6131])
    checkpoint: str = str(ROOT / "pi05_quantvla/models/pi05_libero")
    quant_pack_dir: str = str(ROOT / "pi05_quantvla/models/quantvla")
    output_dir: str = str(ROOT / "outputs/quantvla/pi05/pivot_q/train/seed-000/shared")
    method_name: str = "pi05_pivot_q"
    selection_method: str = "pivot_q"
    manifest: str = str(ROOT / "manifests/libero_plus_first20.json")
    seed: int = 0
    episodes_per_suite: int = 140
    initial_state_ids: list[int] = field(default_factory=lambda: [0])
    updates_per_episode: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    temporal_horizon: int = 4
    temporal_discount: float = 0.9
    alpha_q: float = 0.5
    beta_r: float = 0.5
    phase_bins: int = 4
    top_per_phase: int = 4
    min_temporal_gap: int = 4
    weight_min: float = 0.5
    weight_max: float = 2.0
    lambda_anchor: float = 0.1
    anchor_replay_size: int = 256
    anchor_batch_size: int = 4
    clean_anchor_horizon: int = 4
    save_every_steps: int = 25
    keep_last_checkpoints: int = 1
    resume: bool = True
    collective_smoke_test: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if self.selection_method not in {"pivot_q", "full_distill"}:
            raise ValueError(f"unsupported selection_method: {self.selection_method}")
        if tuple(self.suites) != SUITES:
            raise ValueError(f"suites must use the fixed balanced order: {SUITES}")
        if not (len(self.suites) == len(self.env_ports) == len(self.clean_env_ports) == 4):
            raise ValueError("four suites require four rollout ports and four anchor ports")
        if len(set((*self.env_ports, *self.clean_env_ports))) != 8:
            raise ValueError("all rollout and anchor ports must be distinct")
        if self.episodes_per_suite <= 0 or self.updates_per_episode <= 0:
            raise ValueError("episode and update counts must be positive")
        if self.lambda_anchor < 0:
            raise ValueError("lambda_anchor must be non-negative")


def rank_config(config: Config, rank: int, local_rank: int) -> SimpleNamespace:
    values = dataclasses.asdict(config)
    values.update({
        "suite": config.suites[rank],
        "env_port": config.env_ports[rank],
        "clean_env_port": config.clean_env_ports[rank],
        "episodes": config.episodes_per_suite,
        "episode_horizon": None,
        "device": f"cuda:{local_rank}",
    })
    return SimpleNamespace(**values)


def latest_complete_checkpoint(output: Path, world_size: int) -> Path | None:
    if not (output / "checkpoints").is_dir():
        return None
    for directory in sorted((output / "checkpoints").glob("step-*"), reverse=True):
        required = [directory / "adapter", directory / "trainer_state.pt", directory / "complete.json"]
        required.extend(directory / f"rank-{rank:02d}-state.pt" for rank in range(world_size))
        if all(path.exists() for path in required):
            return directory
    return None


def prune_complete_checkpoints(root: Path, keep_last: int) -> None:
    """Keep only the newest complete checkpoints after an atomic save."""
    if keep_last <= 0:
        return
    resolved_root = root.resolve()
    checkpoints = sorted(
        directory
        for directory in root.glob("step-*")
        if directory.is_dir() and (directory / "complete.json").is_file()
    )
    for directory in checkpoints[:-keep_last]:
        resolved = directory.resolve()
        if directory.is_symlink() or resolved.parent != resolved_root:
            raise RuntimeError(f"refusing to prune unsafe checkpoint: {directory}")
        shutil.rmtree(resolved)


def broadcast_checkpoint(path: Path | None, rank: int) -> Path | None:
    payload: list[str | None] = [str(path) if rank == 0 and path is not None else None]
    dist.broadcast_object_list(payload, src=0)
    return Path(payload[0]) if payload[0] is not None else None


def truncate_jsonl(path: Path, last_episode: int) -> None:
    if not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("episode", -1)) <= last_episode:
            kept.append(json.dumps(record))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def synchronize_parameters(parameters: list[torch.nn.Parameter]) -> None:
    with torch.no_grad():
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)


def average_gradients(parameters: list[torch.nn.Parameter], world_size: int) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("a trainable π0.5 LoRA parameter has no gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def rank_runtime_state(
    *, suite: str, rng: random.Random, generator: torch.Generator,
    task_ids: list[int], replay: deque[State], local_rank: int,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "python_rng_state": rng.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(local_rank),
        "noise_generator_state": generator.get_state(),
        "task_ids": task_ids,
        "anchor_replay": list(replay),
    }


def save_checkpoint(
    *, output: Path, student: Bundle, optimizer: torch.optim.Optimizer,
    episode: int, optimizer_step: int, config: Config, rank: int,
    world_size: int, rank_state: dict[str, Any],
) -> None:
    # Snapshot taken at startup, never re-read mutable configs mid-training.
    protocol = json.loads((output / 'training_protocol.json').read_text())
    root = output / "checkpoints"
    final = root / f"step-{optimizer_step:06d}"
    temporary = root / f".step-{optimizer_step:06d}.incomplete"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        adapter = temporary / "adapter"
        adapter.mkdir()
        student.policy.model.save_pretrained(adapter)
        torch.save({
            "episode": episode,
            "optimizer_step": optimizer_step,
            "optimizer": optimizer.state_dict(),
            "protocol": protocol,
            "world_size": world_size,
            "suites": config.suites,
        }, temporary / "trainer_state.pt")
    dist.barrier()
    torch.save(rank_state, temporary / f"rank-{rank:02d}-state.pt")
    dist.barrier()
    if rank == 0:
        (temporary / "complete.json").write_text(json.dumps({
            "episode": episode, "optimizer_step": optimizer_step,
            "world_size": world_size, "suites": config.suites,
            "protocol": protocol,
        }, indent=2), encoding="utf-8")
        if final.exists():
            raise RuntimeError(f"refusing to overwrite complete checkpoint: {final}")
        temporary.replace(final)
        prune_complete_checkpoints(root, config.keep_last_checkpoints)
    dist.barrier()


def set_final_adapter(output: Path) -> None:
    adapters = sorted((output / "checkpoints").glob("step-*/adapter"))
    if not adapters:
        raise RuntimeError(f"no shared adapter found under {output}")
    final_adapter = output / "final_adapter"
    if final_adapter.is_symlink():
        final_adapter.unlink()
    elif final_adapter.exists():
        raise RuntimeError(f"refusing to overwrite non-link {final_adapter}")
    final_adapter.symlink_to(adapters[-1])


def main() -> None:
    config = tyro.cli(Config)
    config.validate()
    if config.dry_run:
        print(json.dumps({
            **dataclasses.asdict(config),
            "rank_mapping": [
                {"rank": rank, "suite": suite, "env_port": config.env_ports[rank],
                 "clean_env_port": config.clean_env_ports[rank]}
                for rank, suite in enumerate(config.suites)
            ],
            "optimizer_steps_total": config.episodes_per_suite * config.updates_per_episode,
            "total_rollout_episodes": config.episodes_per_suite * len(config.suites),
        }, indent=2))
        return

    from env.protocol import check_training_output, write_training_protocol
    protocol = check_training_output(config.output_dir, dataclasses.asdict(config))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4 or rank >= 4:
        raise RuntimeError("π0.5 shared PIVOT_Q requires exactly four single-node ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(hours=2), device_id=torch.device(f"cuda:{local_rank}")
    )
    if config.collective_smoke_test:
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(json.dumps({
                "world_size": world_size,
                "all_reduce_sum": float(value),
                "expected": 10.0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }), flush=True)
        dist.destroy_process_group()
        return
    local = rank_config(config, rank, local_rank)
    suite = local.suite
    output = Path(config.output_dir).expanduser().resolve()
    dist.barrier()
    if rank == 0:
        write_training_protocol(output, protocol)
    dist.barrier()
    metrics_dir = output / "metrics"
    status_dir = output / "status"
    if rank == 0:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        status_dir.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(json.dumps({
            "schema_version": 1,
            "method": config.method_name,
            "suite": "shared",
            "distributed": "synchronous_gradient_all_reduce",
            "world_size": world_size,
            "seed": config.seed,
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "created_at": time.time(),
            "config": dataclasses.asdict(config),
        }, indent=2), encoding="utf-8")
        write_status(output / "status.json", {
            "status": "starting", "method": config.method_name, "suite": "shared",
            "world_size": world_size, "episode_per_suite": 0,
            "episodes_per_suite_total": config.episodes_per_suite,
            "total_rollout_episodes_complete": 0,
            "total_rollout_episodes": config.episodes_per_suite * world_size,
            "optimizer_step": 0,
            "optimizer_steps_total": config.episodes_per_suite * config.updates_per_episode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host": socket.gethostname(), "updated_at": time.time(),
        })
    dist.barrier()

    checkpoint_dir = latest_complete_checkpoint(output, world_size) if rank == 0 and config.resume else None
    checkpoint_dir = broadcast_checkpoint(checkpoint_dir, rank)
    trainer_state = (
        torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        if checkpoint_dir is not None else None
    )
    adapter_path = checkpoint_dir / "adapter" if checkpoint_dir is not None else None
    saved_rank_state = (
        torch.load(checkpoint_dir / f"rank-{rank:02d}-state.pt", map_location="cpu", weights_only=False)
        if checkpoint_dir is not None else None
    )
    start_episode = int(trainer_state["episode"]) + 1 if trainer_state else 1
    optimizer_step = int(trainer_state["optimizer_step"]) if trainer_state else 0
    if trainer_state and (
        int(trainer_state["world_size"]) != world_size
        or list(trainer_state["suites"]) != config.suites
    ):
        raise RuntimeError("checkpoint rank/suite layout does not match this launch")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed + rank)
    rng = random.Random(config.seed + rank)
    generator = torch.Generator(device=local.device).manual_seed(config.seed + rank)
    rollout_env = EnvClient(local.env_port)
    clean_env: EnvClient | None = None
    teacher: Bundle | None = None
    student: Bundle | None = None
    try:
        write_status(status_dir / f"{suite}.json", {
            "status": "starting", "method": config.method_name, "suite": suite, "rank": rank,
            "episode": start_episode - 1,
            "episodes_total": config.episodes_per_suite,
            "optimizer_step": optimizer_step,
            "optimizer_steps_total": config.episodes_per_suite * config.updates_per_episode,
            "updated_at": time.time(),
        })
        rollout_env.wait()
        task_metadata = rollout_env.call("list_tasks")
        task_ids = [int(item["task_id"]) for item in task_metadata]
        category = {int(item["task_id"]): str(item["category"]) for item in task_metadata}
        rng.shuffle(task_ids)
        if config.lambda_anchor > 0:
            clean_env = EnvClient(local.clean_env_port)
            clean_env.wait()

        print(f"[rank {rank} {suite}] loading π0.5 FP16 teacher", flush=True)
        teacher = load_bundle(local, quantized=False)
        print(f"[rank {rank} {suite}] loading π0.5 QuantVLA student", flush=True)
        student = load_bundle(local, quantized=True, adapter=adapter_path)
        parameters = [parameter for parameter in student.policy.model.parameters() if parameter.requires_grad]
        synchronize_parameters(parameters)
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        replay: deque[State] = deque(maxlen=config.anchor_replay_size)
        if saved_rank_state:
            if saved_rank_state["suite"] != suite:
                raise RuntimeError(f"rank {rank} checkpoint suite mismatch")
            rng.setstate(saved_rank_state["python_rng_state"])
            np.random.set_state(saved_rank_state["numpy_rng_state"])
            torch.set_rng_state(saved_rank_state["torch_rng_state"])
            torch.cuda.set_rng_state(saved_rank_state["cuda_rng_state"], device=local_rank)
            generator.set_state(saved_rank_state["noise_generator_state"])
            task_ids = list(saved_rank_state["task_ids"])
            replay.extend(saved_rank_state["anchor_replay"])

        suite_metrics_path = metrics_dir / f"{suite}.jsonl"
        if config.resume:
            truncate_jsonl(suite_metrics_path, start_episode - 1)
            if rank == 0:
                truncate_jsonl(output / "metrics.jsonl", start_episode - 1)
        recent_durations: deque[float] = deque(maxlen=20)
        dist.barrier()
        with suite_metrics_path.open("a", encoding="utf-8", buffering=1) as suite_metrics:
            global_metrics = (
                (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
                if rank == 0 else None
            )
            try:
                for episode in range(start_episode, config.episodes_per_suite + 1):
                    started = time.time()
                    if (episode - 1) % len(task_ids) == 0 and episode > 1:
                        rng.shuffle(task_ids)
                    task_id = task_ids[(episode - 1) % len(task_ids)]
                    states, success = run_rollout(
                        local, teacher, student, rollout_env,
                        task_id=task_id,
                        initial_state_id=rng.choice(config.initial_state_ids),
                        horizon=HORIZONS[suite], generator=generator,
                    )
                    if not states:
                        raise RuntimeError(f"rank {rank} episode {episode} produced no states")
                    student_actions = torch.cat([state.student_action for state in states])
                    teacher_actions = torch.cat([state.teacher_action for state in states])
                    q = action_error(student_actions, teacher_actions)
                    future = future_risk(q, config.temporal_horizon, config.temporal_discount)
                    # Paper dense baseline: the only change is to supervise
                    # every valid student-visited state with uniform weight.
                    if config.selection_method == "full_distill":
                        chosen = select_all(q)
                    else:
                        chosen = select(
                            q, future, phase_bins=config.phase_bins,
                            top_per_phase=config.top_per_phase,
                            min_gap=config.min_temporal_gap,
                            alpha=config.alpha_q, beta=config.beta_r,
                            weight_min=config.weight_min, weight_max=config.weight_max,
                        )
                    if clean_env is not None:
                        clean_states, _ = run_rollout(
                            local, teacher, student, clean_env,
                            task_id=rng.randrange(10), initial_state_id=rng.randrange(40),
                            horizon=config.clean_anchor_horizon, generator=generator,
                        )
                        replay.extend(clean_states)

                    losses: list[float] = []
                    anchor_losses: list[float] = []
                    gradient_norms: list[float] = []
                    weights = chosen.weights / chosen.weights.sum().clamp_min(1e-8)
                    # Full Distill only: per-rollout frozen-prefix caches are
                    # not part of replay, checkpoint state, or DDP broadcasts.
                    from pi05_quantvla.pivot_q.paired_flow import FrozenPrefixCache
                    prefix_caches = ({i: FrozenPrefixCache() for i in chosen.indices}
                                     if config.selection_method == "full_distill" else {})
                    for _ in range(config.updates_per_episode):
                        optimizer.zero_grad(set_to_none=True)
                        for index, weight in zip(chosen.indices, weights, strict=True):
                            loss = state_loss(student, states[index], prefix_cache=prefix_caches.get(index))
                            (loss * weight.to(loss.device)).backward()
                            losses.append(float(loss.detach().cpu()))
                        if replay and config.lambda_anchor > 0 and config.anchor_batch_size > 0:
                            anchors = rng.sample(
                                list(replay), min(config.anchor_batch_size, len(replay))
                            )
                            for state in anchors:
                                loss = state_loss(student, state)
                                (loss * config.lambda_anchor / len(anchors)).backward()
                                anchor_losses.append(float(loss.detach().cpu()))
                        average_gradients(parameters, world_size)
                        gradient = torch.nn.utils.clip_grad_norm_(
                            parameters, config.gradient_clip_norm
                        )
                        if not torch.isfinite(torch.as_tensor(gradient)):
                            raise FloatingPointError(
                                f"non-finite synchronized gradient at episode {episode}"
                            )
                        optimizer.step()
                        optimizer_step += 1
                        gradient_norms.append(float(gradient))

                    duration = time.time() - started
                    recent_durations.append(duration)
                    record = {
                        "episode": episode,
                        "optimizer_step": optimizer_step,
                        "rank": rank,
                        "method": config.method_name,
                        "training_scope": "shared_four_suite",
                        "suite": suite,
                        "task_id": task_id,
                        "category": category[task_id],
                        "episode_steps": len(states),
                        "episode_success": success,
                        "selected_state_count": len(chosen.indices),
                        "selected_indices": list(chosen.indices),
                        "selected_phases": list(chosen.phases),
                        "phase_effective_gaps": list(chosen.effective_gaps),
                        "q_mean": float(q.mean()),
                        "q_max": float(q.max()),
                        "r_mean": float(future.mean()),
                        "r_max": float(future.max()),
                        "selected_q": q[
                            torch.as_tensor(chosen.indices, device=q.device)
                        ].detach().cpu().tolist(),
                        "selected_r": future[
                            torch.as_tensor(chosen.indices, device=future.device)
                        ].detach().cpu().tolist(),
                        "q_by_timestep": q.detach().cpu().tolist(),
                        "r_by_timestep": future.detach().cpu().tolist(),
                        "loss_pivot_q_unweighted_mean": float(np.mean(losses)),
                        "loss_anchor_mean": (
                            float(np.mean(anchor_losses)) if anchor_losses else 0.0
                        ),
                        "gradient_norm_mean": float(np.mean(gradient_norms)),
                        "synchronized_gradient_norm_mean": float(np.mean(gradient_norms)),
                        "duration_seconds": round(duration, 3),
                    }
                    suite_metrics.write(json.dumps(record) + "\n")
                    local_eta = float(np.median(recent_durations)) * (
                        config.episodes_per_suite - episode
                    )
                    write_status(status_dir / f"{suite}.json", {
                        "status": (
                            "complete" if episode == config.episodes_per_suite else "running"
                        ),
                        "method": config.method_name, "suite": suite, "rank": rank,
                        "episode": episode,
                        "episodes_total": config.episodes_per_suite,
                        "optimizer_step": optimizer_step,
                        "optimizer_steps_total": (
                            config.episodes_per_suite * config.updates_per_episode
                        ),
                        "last_episode_success": success,
                        "gradient_norm_mean": record["gradient_norm_mean"],
                        "eta_seconds": local_eta,
                        "updated_at": time.time(),
                    })
                    gathered: list[dict[str, Any] | None] = [None] * world_size
                    dist.all_gather_object(gathered, record)
                    if rank == 0:
                        records = [item for item in gathered if item is not None]
                        global_record = {
                            "episode": episode,
                            "optimizer_step": optimizer_step,
                            "method": config.method_name,
                            "training_scope": "shared_four_suite",
                            "suite_records": {item["suite"]: item for item in records},
                            "mean_q": float(np.mean([item["q_mean"] for item in records])),
                            "mean_loss": float(np.mean([
                                item["loss_pivot_q_unweighted_mean"] for item in records
                            ])),
                            "synchronized_gradient_norm": float(np.mean([
                                item["gradient_norm_mean"] for item in records
                            ])),
                            "round_duration_seconds": max(
                                item["duration_seconds"] for item in records
                            ),
                        }
                        assert global_metrics is not None
                        global_metrics.write(json.dumps(global_record) + "\n")
                        write_status(output / "status.json", {
                            "status": (
                                "complete" if episode == config.episodes_per_suite else "running"
                            ),
                            "method": config.method_name, "suite": "shared",
                            "world_size": world_size,
                            "episode_per_suite": episode,
                            "episodes_per_suite_total": config.episodes_per_suite,
                            "total_rollout_episodes_complete": episode * world_size,
                            "total_rollout_episodes": config.episodes_per_suite * world_size,
                            "optimizer_step": optimizer_step,
                            "optimizer_steps_total": (
                                config.episodes_per_suite * config.updates_per_episode
                            ),
                            "synchronized_gradient_norm": global_record[
                                "synchronized_gradient_norm"
                            ],
                            "eta_seconds": global_record["round_duration_seconds"] * (
                                config.episodes_per_suite - episode
                            ),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "host": socket.gethostname(), "updated_at": time.time(),
                        })
                        print(json.dumps(global_record), flush=True)

                    if (
                        optimizer_step % config.save_every_steps == 0
                        or episode == config.episodes_per_suite
                    ):
                        save_checkpoint(
                            output=output, student=student, optimizer=optimizer,
                            episode=episode, optimizer_step=optimizer_step,
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
    finally:
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
    main()
