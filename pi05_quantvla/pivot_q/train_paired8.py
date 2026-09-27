#!/usr/bin/env python3
"""Eight-GPU, four-rollout paired full distillation for the paper ablation.

Each pair shares one rollout, splits its valid states over two GPUs, and all
eight ranks synchronize one adapter. Four rollouts and five updates per round
retain the original 560-rollout / 700-update budget.
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

import numpy as np
import torch
import torch.distributed as dist
import tyro

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pi05_quantvla.pivot_q.paired_flow import FrozenPrefixCache, sample
from pi05_quantvla.pivot_q.train import (
    Bundle, EnvClient, HORIZONS, State, load_bundle, run_rollout, state_loss,
)
from pi05_quantvla.pivot_q.train_ddp import (
    Config, average_gradients, latest_complete_checkpoint, rank_config,
    save_checkpoint, set_final_adapter, synchronize_parameters, truncate_jsonl,
)


ORDER = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


@dataclass
class PairedConfig(Config):
    output_dir: str = str(
        Path(__file__).resolve().parents[2]
        / "outputs/quantvla/pi05/full_distill/train/seed-000/shared"
    )
    method_name: str = "pi05_full_distill_paired8"
    selection_method: str = "full_distill"
    suite_order: list[str] = field(default_factory=lambda: list(ORDER))
    state_microbatch: int = 2
    max_rounds: int = 0  # Zero means the complete 140-round run.

    def validate(self) -> None:
        super().validate()
        if self.selection_method != "full_distill":
            raise ValueError("paired-eight mode is only for full distillation")
        if tuple(self.suite_order) != ORDER:
            raise ValueError(f"suite order must be {ORDER}")
        if self.episodes_per_suite % 4:
            raise ValueError("episodes_per_suite must be divisible by four pairs")
        if self.state_microbatch < 1 or self.max_rounds < 0:
            raise ValueError("invalid state_microbatch or max_rounds")


def batched_loss(student: Bundle, states: list[State], *, prefix_cache: FrozenPrefixCache | None = None) -> torch.Tensor:
    """Mean first-action MSE over a small state microbatch."""
    processed = [student.preprocessor(state.raw_sample) for state in states]
    keys = processed[0].keys()
    if any(item.keys() != keys for item in processed):
        raise ValueError("preprocessor keys differ within state microbatch")
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in processed]
        if not all(isinstance(value, torch.Tensor) for value in values):
            # LeRobot retains an unbatched action bookkeeping field; the
            # differentiable policy sampler does not consume it.
            if key == "action":
                continue
            return torch.stack([state_loss(student, state) for state in states]).mean()
        # Anchor replay may mix language instructions with different token
        # lengths. Preserve the exact per-state objective in that case.
        if any(value.shape[1:] != values[0].shape[1:] for value in values):
            return torch.stack([state_loss(student, state) for state in states]).mean()
        batch[key] = torch.cat(values, dim=0)
    device = next(student.policy.model.parameters()).device
    noise = torch.cat([state.noise for state in states], dim=0).to(
        device, dtype=torch.float32
    )
    prediction = sample(student.policy, batch, noise, with_grad=True, prefix_cache=prefix_cache)
    target = torch.cat([state.teacher_action for state in states], dim=0).to(
        prediction.device
    )
    return (prediction[:, 0, :7].float() - target[:, 0, :7].float()).square().mean()


def backward_shard(
    student: Bundle, states: list[State], *, rank_in_pair: int,
    coefficient: float, microbatch: int,
    prefix_caches: dict | None = None,
) -> None:
    """Disjoint state shards sum to the original trajectory-mean loss."""
    shard = states[rank_in_pair::2]
    for start in range(0, len(shard), microbatch):
        part = shard[start:start + microbatch]
        # Cache the exact original microbatch; never replace it with singleton
        # rollout features, which would change the batched numerical path.
        cache = (prefix_caches.setdefault(start, FrozenPrefixCache())
                 if prefix_caches is not None else None)
        loss = batched_loss(student, part, prefix_cache=cache)
        (loss * (coefficient * len(part) / len(states))).backward()


def task_order(task_metadata: list[dict], *, seed: int, suite_index: int) -> list[int]:
    ids = [int(item["task_id"]) for item in task_metadata]
    if len(ids) % 4:
        raise ValueError("suite task count is not divisible by four")
    random.Random(seed + 1000 * suite_index).shuffle(ids)
    return ids


def run() -> None:
    config = tyro.cli(PairedConfig)
    config.validate()
    total_rounds = config.episodes_per_suite
    if config.dry_run:
        print(json.dumps({
            "world_size": 8, "pairs": 4, "suite_order": config.suite_order,
            "rounds_per_suite": config.episodes_per_suite // 4,
            "total_rounds": total_rounds,
            "optimizer_steps": total_rounds * config.updates_per_episode,
            "output_dir": config.output_dir,
        }, indent=2))
        return

    from env.protocol import check_training_output, write_training_protocol
    protocol = check_training_output(config.output_dir, dataclasses.asdict(config))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError("paired-eight training requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(hours=2),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    if config.collective_smoke_test:
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        if rank == 0:
            print(json.dumps({"all_reduce_sum": float(value), "expected": 36.0}), flush=True)
        dist.destroy_process_group()
        return

    pair = rank // 2
    leader = 2 * pair
    is_leader = rank == leader
    # A complete long-horizon rollout can exceed PyTorch's default 10-minute
    # subgroup timeout while the follower waits for its leader's states.
    pair_groups = [
        dist.new_group(ranks=[2 * i, 2 * i + 1], timeout=timedelta(hours=2))
        for i in range(4)
    ]
    local = rank_config(config, pair, local_rank)
    output = Path(config.output_dir).expanduser().resolve()
    dist.barrier()
    if rank == 0:
        write_training_protocol(output, protocol)
    dist.barrier()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(json.dumps({
            "method": config.method_name, "training_scope": "paired8_sequential_suite",
            "world_size": 8, "seed": config.seed, "host": socket.gethostname(),
            "config": dataclasses.asdict(config), "created_at": time.time(),
        }, indent=2), encoding="utf-8")
    dist.barrier()
    ckpt = latest_complete_checkpoint(output, 8) if rank == 0 and config.resume else None
    path_list = [str(ckpt) if rank == 0 and ckpt else None]
    dist.broadcast_object_list(path_list, src=0)
    ckpt = Path(path_list[0]) if path_list[0] else None
    trainer_state = torch.load(
        ckpt / "trainer_state.pt", map_location="cpu", weights_only=False
    ) if ckpt else None
    rank_state = torch.load(
        ckpt / f"rank-{rank:02d}-state.pt", map_location="cpu", weights_only=False
    ) if ckpt else None
    start_round = int(trainer_state["episode"]) if trainer_state else 0
    optimizer_step = int(trainer_state["optimizer_step"]) if trainer_state else 0
    if trainer_state and int(trainer_state["world_size"]) != 8:
        raise RuntimeError("checkpoint world size mismatch")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed + rank)
    rng = random.Random(config.seed + rank)
    generator = torch.Generator(device=local.device).manual_seed(config.seed + rank)
    replay: deque[State] = deque(maxlen=config.anchor_replay_size)
    if rank_state:
        rng.setstate(rank_state["python_rng_state"])
        np.random.set_state(rank_state["numpy_rng_state"])
        torch.set_rng_state(rank_state["torch_rng_state"])
        torch.cuda.set_rng_state(rank_state["cuda_rng_state"], device=local_rank)
        generator.set_state(rank_state["noise_generator_state"])
        replay.extend(rank_state["anchor_replay"])

    rollout_env = EnvClient(local.env_port) if is_leader else None
    clean_env = EnvClient(local.clean_env_port) if is_leader and config.lambda_anchor else None
    teacher = student = None
    try:
        if rollout_env:
            rollout_env.wait()
        if clean_env:
            clean_env.wait()
        if is_leader:
            print(f"[rank {rank}] loading local teacher", flush=True)
            teacher = load_bundle(local, quantized=False)
        print(f"[rank {rank}] loading local student", flush=True)
        student = load_bundle(
            local, quantized=True, adapter=ckpt / "adapter" if ckpt else None,
        )
        parameters = [p for p in student.policy.model.parameters() if p.requires_grad]
        synchronize_parameters(parameters)
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        if trainer_state:
            optimizer.load_state_dict(trainer_state["optimizer"])
        dist.barrier()

        end_round = min(total_rounds, config.max_rounds or total_rounds)
        metrics_path = output / "metrics.jsonl"
        if rank == 0 and config.resume:
            truncate_jsonl(metrics_path, start_round)
        dist.barrier()
        metrics = metrics_path.open("a", encoding="utf-8", buffering=1) if rank == 0 else None
        try:
            rounds_per_suite = config.episodes_per_suite // 4
            task_ids: list[int] = []
            for round_index in range(start_round, end_round):
                suite_index, suite_round = divmod(round_index, rounds_per_suite)
                suite = config.suite_order[suite_index]
                local.suite = suite
                payload: list[object] = [None]
                if is_leader:
                    assert rollout_env is not None and teacher is not None
                    if suite_round == 0 or not task_ids:
                        metadata = rollout_env.call("list_tasks", suite=suite)
                        task_ids = task_order(metadata, seed=config.seed, suite_index=suite_index)
                        if len(task_ids) != config.episodes_per_suite:
                            raise ValueError(f"{suite} has {len(task_ids)} tasks, expected {config.episodes_per_suite}")
                    task_id = task_ids[4 * suite_round + pair]
                    states, success = run_rollout(
                        local, teacher, student, rollout_env,
                        task_id=task_id,
                        initial_state_id=rng.choice(config.initial_state_ids),
                        horizon=HORIZONS[suite], generator=generator,
                    )
                    if not states:
                        raise RuntimeError(f"{suite} round {suite_round} has no valid states")
                    if clean_env is not None:
                        clean_states, _ = run_rollout(
                            local, teacher, student, clean_env,
                            task_id=rng.randrange(10),
                            initial_state_id=rng.randrange(40),
                            horizon=config.clean_anchor_horizon, generator=generator,
                        )
                        replay.extend(clean_states)
                    payload[0] = (states, success, task_id)
                dist.broadcast_object_list(payload, src=leader, group=pair_groups[pair])
                states, success, task_id = payload[0]
                assert isinstance(states, list)
                started_update = time.time()
                prefix_caches = {}  # Rank-local, discarded after this rollout.
                for _ in range(config.updates_per_episode):
                    # Match the existing baseline: sample fresh Behavioral
                    # Anchor states for every optimizer update, not per rollout.
                    anchor_payload: list[object] = [
                        rng.sample(
                            list(replay), min(config.anchor_batch_size, len(replay))
                        ) if is_leader and replay and config.lambda_anchor > 0 else []
                    ]
                    dist.broadcast_object_list(
                        anchor_payload, src=leader, group=pair_groups[pair]
                    )
                    anchors = anchor_payload[0]
                    assert isinstance(anchors, list)
                    optimizer.zero_grad(set_to_none=True)
                    backward_shard(
                        student, states, rank_in_pair=rank % 2,
                        coefficient=1.0, microbatch=config.state_microbatch,
                        prefix_caches=prefix_caches,
                    )
                    if anchors and config.lambda_anchor > 0:
                        backward_shard(
                            student, anchors, rank_in_pair=rank % 2,
                            coefficient=config.lambda_anchor,
                            microbatch=config.state_microbatch,
                        )
                    # Four trajectory losses, each split across two ranks.
                    average_gradients(parameters, world_size=4)
                    gradient = torch.nn.utils.clip_grad_norm_(
                        parameters, config.gradient_clip_norm
                    )
                    if not torch.isfinite(torch.as_tensor(gradient)):
                        raise FloatingPointError("non-finite synchronized gradient")
                    optimizer.step()
                    optimizer_step += 1

                record = {
                    "episode": round_index + 1, "suite": suite,
                    "pair": pair, "rank": rank, "task_id": task_id,
                    "states": len(states), "anchors": len(anchors),
                    "success": bool(success), "optimizer_step": optimizer_step,
                    "update_seconds": round(time.time() - started_update, 3),
                } if is_leader else None
                gathered: list[dict | None] = [None] * 8
                dist.all_gather_object(gathered, record)
                if rank == 0:
                    assert metrics is not None
                    metric = {
                        "episode": round_index + 1, "suite": suite,
                        "optimizer_step": optimizer_step,
                        "rollouts_complete": (round_index + 1) * 4,
                        "pair_records": [item for item in gathered if item is not None],
                    }
                    metrics.write(json.dumps(metric) + "\n")
                    (output / "status.json").write_text(json.dumps({
                        "status": "complete" if round_index + 1 == total_rounds else "running",
                        "suite": suite, "round": round_index + 1,
                        "rounds_total": total_rounds, "optimizer_step": optimizer_step,
                        "optimizer_steps_total": total_rounds * config.updates_per_episode,
                        "updated_at": time.time(),
                    }, indent=2), encoding="utf-8")
                    print(json.dumps(metric), flush=True)

                if optimizer_step % config.save_every_steps == 0 or round_index + 1 == end_round:
                    save_checkpoint(
                        output=output, student=student, optimizer=optimizer,
                        episode=round_index + 1, optimizer_step=optimizer_step,
                        config=config, rank=rank, world_size=8,
                        rank_state={
                            "suite": suite,
                            "python_rng_state": rng.getstate(),
                            "numpy_rng_state": np.random.get_state(),
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state(local_rank),
                            "noise_generator_state": generator.get_state(),
                            "anchor_replay": list(replay),
                        },
                    )
            if rank == 0 and end_round == total_rounds:
                set_final_adapter(output)
        finally:
            if metrics:
                metrics.close()
    finally:
        if rollout_env:
            rollout_env.close()
        if clean_env:
            clean_env.close()
        if teacher is not None and teacher.scales is not None:
            teacher.scales.close()
        if student is not None and student.scales is not None:
            student.scales.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
