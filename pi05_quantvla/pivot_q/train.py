#!/usr/bin/env python3
"""On-policy GAP-PIVOT_Q training for the π0.5 QuantVLA student."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import random
import shutil
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import zmq
from peft import PeftModel

from pi05_quantvla.pivot_q.paired_flow import FrozenPrefixCache, make_noise, sample
from pi05_quantvla.pivot_q.peft_lora import attach
from pi05_quantvla.pivot_q.selection import action_error, future_risk, select
from pi05_quantvla.scripts.atm_ohb import AttentionStatistics
from pi05_quantvla.scripts.eval_client import axis_angle, prepare_image
from pi05_quantvla.scripts.policy import load_policy
from pi05_quantvla.scripts.quantvla import PI05QuantVLAConfig, apply_quantvla_layout

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


@dataclass
class Config:
    suite: str = "libero_spatial"
    checkpoint: str = str(ROOT / "pi05_quantvla/models/pi05_libero")
    quant_pack_dir: str = str(ROOT / "pi05_quantvla/models/quantvla")
    output_dir: str = str(ROOT / "outputs/quantvla/pi05/pivot_q/train/seed-000/libero_spatial")
    manifest: str = str(ROOT / "manifests/libero_plus_first20.json")
    device: str = "cuda"
    seed: int = 0
    episodes: int = 140
    episode_horizon: int | None = None
    env_port: int = 5790
    clean_env_port: int = 5791
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
    save_every_steps: int = 100
    keep_last_checkpoints: int = 1
    resume: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        if self.suite not in HORIZONS:
            raise ValueError(f"unsupported suite: {self.suite}")
        if self.episodes <= 0 or self.updates_per_episode <= 0:
            raise ValueError("episodes and updates_per_episode must be positive")
        if self.lambda_anchor < 0:
            raise ValueError("lambda_anchor must be non-negative")


class EnvClient:
    def __init__(self, port: int, timeout_ms: int = 120_000) -> None:
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(self.endpoint)

    def call(self, endpoint: str, **data):
        self.socket.send(pickle.dumps({"endpoint": endpoint, "data": data}))
        response = pickle.loads(self.socket.recv())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def wait(self, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.call("ping")
                return
            except (zmq.ZMQError, RuntimeError):
                if time.time() >= deadline:
                    raise TimeoutError(f"environment service unavailable at {self.endpoint}")
                self.socket.close(linger=0)
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, 1000)
                self.socket.setsockopt(zmq.SNDTIMEO, 1000)
                self.socket.connect(self.endpoint)
                time.sleep(0.5)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


@dataclass
class Bundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    scales: AttentionStatistics | None = None


@dataclass
class State:
    raw_sample: dict[str, Any]
    noise: torch.Tensor
    teacher_action: torch.Tensor
    student_action: torch.Tensor


def raw_sample(observation: dict, language: str) -> dict[str, Any]:
    return {
        "observation.images.image": torch.from_numpy(prepare_image(observation["agentview_image"])),
        "observation.images.image2": torch.from_numpy(prepare_image(observation["robot0_eye_in_hand_image"])),
        "observation.state": torch.from_numpy(np.concatenate((
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ))),
        "task": language,
    }


def load_bundle(config: Config, *, quantized: bool, adapter: Path | None = None) -> Bundle:
    checkpoint = Path(config.checkpoint).expanduser().resolve()
    policy = load_policy(ROOT, checkpoint, device=config.device)
    scales = None
    if quantized:
        pack = Path(config.quant_pack_dir).expanduser().resolve()
        quant_config_path = ROOT / "pi05_quantvla/config/quantvla.yaml"
        import yaml
        quant_config = yaml.safe_load(quant_config_path.read_text(encoding="utf-8"))["quantization"]
        apply_quantvla_layout(policy.model, PI05QuantVLAConfig(**quant_config), pack_dir=pack)
        scale_path = pack / "atm_ohb.json"
        scale_values = json.loads(scale_path.read_text(encoding="utf-8"))["scales"]
        scales = AttentionStatistics(policy.model, scales=scale_values, collect=False)
        if adapter is None:
            policy.model, targets = attach(
                policy.model, rank=config.lora_rank,
                alpha=config.lora_alpha, dropout=config.lora_dropout,
            )
            print(f"Attached π0.5 LoRA to {len(targets)} action-expert projections")
        else:
            policy.model = PeftModel.from_pretrained(policy.model, adapter, is_trainable=True)
        policy.model.eval()
        policy.model.print_trainable_parameters()
    else:
        policy.model.requires_grad_(False)
        policy.model.eval()

    from lerobot.policies import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return Bundle(policy, preprocessor, postprocessor, scales)


def run_rollout(config: Config, teacher: Bundle, student: Bundle, env: EnvClient, *,
                task_id: int, initial_state_id: int, horizon: int,
                generator: torch.Generator) -> tuple[list[State], bool]:
    reset = env.call(
        "reset", suite=config.suite, task_id=task_id, initial_state_id=initial_state_id,
        num_steps_wait=10, horizon=horizon + 11,
    )
    observation, language = reset["observation"], reset["language"]
    states: list[State] = []
    if reset["done"]:
        return states, True
    for _ in range(horizon):
        raw = raw_sample(observation, language)
        teacher_batch = teacher.preprocessor(raw)
        student_batch = student.preprocessor(raw)
        noise = make_noise(student.policy, generator)
        teacher_action = sample(teacher.policy, teacher_batch, noise, with_grad=False)
        student_action = sample(student.policy, student_batch, noise, with_grad=False)
        states.append(State(raw, noise.detach().cpu(), teacher_action.detach().cpu(), student_action.detach().cpu()))
        executed = student.postprocessor(student_action[0, :1])[0].detach().float().cpu().numpy()
        result = env.call("step", action=executed.tolist())
        observation = result["observation"]
        if result["done"]:
            return states, True
    return states, False


def state_loss(student: Bundle, state: State, *, prefix_cache: FrozenPrefixCache | None = None) -> torch.Tensor:
    batch = student.preprocessor(state.raw_sample)
    prediction = sample(
        student.policy, batch,
        state.noise.to(config_device(student), dtype=torch.float32),
        with_grad=True, prefix_cache=prefix_cache,
    )
    target = state.teacher_action.to(prediction.device)
    return (prediction[:, 0, :7].float() - target[:, 0, :7].float()).square().mean()


def config_device(bundle: Bundle) -> torch.device:
    return next(bundle.policy.model.parameters()).device


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint(output: Path, student: Bundle, optimizer, episode: int, step: int) -> None:
    directory = output / "checkpoints" / f"step-{step:06d}"
    adapter = directory / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    student.policy.model.save_pretrained(adapter)
    torch.save({"episode": episode, "optimizer_step": step, "optimizer": optimizer.state_dict()}, directory / "trainer_state.pt")


def prune_checkpoints(output: Path, keep_last: int) -> None:
    if keep_last == 0:
        return
    root = (output / "checkpoints").resolve()
    checkpoints = sorted(root.glob("step-*"))
    for directory in checkpoints[:-keep_last]:
        resolved = directory.resolve()
        if directory.is_symlink() or resolved.parent != root:
            raise RuntimeError(f"refusing to prune unsafe checkpoint: {directory}")
        shutil.rmtree(resolved)


def main() -> None:
    config = tyro.cli(Config)
    config.validate()
    if config.dry_run:
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps({
        "schema_version": 1, "method": "pi05_pivot_q", "suite": config.suite,
        "seed": config.seed, "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "created_at": time.time(), "config": dataclasses.asdict(config),
    }, indent=2), encoding="utf-8")
    checkpoints = sorted((output / "checkpoints").glob("step-*/trainer_state.pt")) if config.resume else []
    resume_state = torch.load(checkpoints[-1], map_location="cpu", weights_only=False) if checkpoints else None
    adapter_path = checkpoints[-1].parent / "adapter" if checkpoints else None
    write_status(output / "status.json", {
        "status": "starting", "method": "pi05_pivot_q", "suite": config.suite,
        "episode": 0, "episodes_total": config.episodes,
        "optimizer_step": 0, "optimizer_steps_total": config.episodes * config.updates_per_episode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "updated_at": time.time(),
    })
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = random.Random(config.seed)
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    rollout_env, clean_env = EnvClient(config.env_port), None
    try:
        rollout_env.wait()
        task_metadata = rollout_env.call("list_tasks")
        task_ids = [int(item["task_id"]) for item in task_metadata]
        category = {int(item["task_id"]): str(item["category"]) for item in task_metadata}
        rng.shuffle(task_ids)
        if config.lambda_anchor > 0:
            clean_env = EnvClient(config.clean_env_port)
            clean_env.wait()

        print("Loading π0.5 FP16 teacher")
        teacher = load_bundle(config, quantized=False)
        print("Loading π0.5 QuantVLA student")
        student = load_bundle(config, quantized=True, adapter=adapter_path)
        parameters = [parameter for parameter in student.policy.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        start_episode, optimizer_step = 1, 0
        if resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            start_episode = int(resume_state["episode"]) + 1
            optimizer_step = int(resume_state["optimizer_step"])
        replay: deque[State] = deque(maxlen=config.anchor_replay_size)
        metrics_path = output / "metrics.jsonl"
        recent_durations: deque[float] = deque(maxlen=20)
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
            for episode in range(start_episode, config.episodes + 1):
                started = time.time()
                if (episode - 1) % len(task_ids) == 0 and episode > 1:
                    rng.shuffle(task_ids)
                task_id = task_ids[(episode - 1) % len(task_ids)]
                states, success = run_rollout(
                    config, teacher, student, rollout_env,
                    task_id=task_id, initial_state_id=rng.choice(config.initial_state_ids),
                    horizon=config.episode_horizon or HORIZONS[config.suite], generator=generator,
                )
                if not states:
                    raise RuntimeError(f"episode {episode} produced no π0.5 states")
                student_actions = torch.cat([state.student_action for state in states])
                teacher_actions = torch.cat([state.teacher_action for state in states])
                q = action_error(student_actions, teacher_actions)
                r = future_risk(q, config.temporal_horizon, config.temporal_discount)
                selection = select(
                    q, r, phase_bins=config.phase_bins, top_per_phase=config.top_per_phase,
                    min_gap=config.min_temporal_gap, alpha=config.alpha_q, beta=config.beta_r,
                    weight_min=config.weight_min, weight_max=config.weight_max,
                )
                if clean_env is not None:
                    clean_states, _ = run_rollout(
                        config, teacher, student, clean_env,
                        task_id=rng.randrange(10), initial_state_id=rng.randrange(40),
                        horizon=config.clean_anchor_horizon, generator=generator,
                    )
                    replay.extend(clean_states)
                losses, anchor_losses, gradients = [], [], []
                weights = selection.weights / selection.weights.sum().clamp_min(1e-8)
                for _ in range(config.updates_per_episode):
                    optimizer.zero_grad(set_to_none=True)
                    for index, weight in zip(selection.indices, weights, strict=True):
                        loss = state_loss(student, states[index])
                        (loss * weight.to(loss.device)).backward()
                        losses.append(float(loss.detach().cpu()))
                    if replay and config.lambda_anchor > 0 and config.anchor_batch_size > 0:
                        anchors = rng.sample(list(replay), min(config.anchor_batch_size, len(replay)))
                        for state in anchors:
                            loss = state_loss(student, state)
                            (loss * config.lambda_anchor / len(anchors)).backward()
                            anchor_losses.append(float(loss.detach().cpu()))
                    gradient = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
                    if not torch.isfinite(torch.as_tensor(gradient)):
                        raise FloatingPointError(f"non-finite gradient at episode {episode}")
                    optimizer.step()
                    optimizer_step += 1
                    gradients.append(float(gradient))
                duration = time.time() - started
                recent_durations.append(duration)
                record = {
                    "episode": episode, "optimizer_step": optimizer_step,
                    "method": "pi05_pivot_q", "suite": config.suite, "task_id": task_id,
                    "category": category[task_id], "episode_steps": len(states),
                    "episode_success": success, "selected_state_count": len(selection.indices),
                    "selected_indices": list(selection.indices), "selected_phases": list(selection.phases),
                    "phase_effective_gaps": list(selection.effective_gaps),
                    "q_mean": float(q.mean()), "q_max": float(q.max()),
                    "r_mean": float(r.mean()), "r_max": float(r.max()),
                    "loss_pivot_q_unweighted_mean": float(np.mean(losses)),
                    "loss_anchor_mean": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
                    "gradient_norm_mean": float(np.mean(gradients)),
                    "duration_seconds": round(duration, 3),
                }
                metrics.write(json.dumps(record) + "\n")
                eta = float(np.median(recent_durations)) * (config.episodes - episode)
                write_status(output / "status.json", {
                    "status": "complete" if episode == config.episodes else "running",
                    "method": "pi05_pivot_q", "suite": config.suite,
                    "episode": episode, "episodes_total": config.episodes,
                    "optimizer_step": optimizer_step,
                    "optimizer_steps_total": config.episodes * config.updates_per_episode,
                    "last_episode_success": success, "eta_seconds": eta,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "host": socket.gethostname(), "updated_at": time.time(),
                })
                print(json.dumps(record), flush=True)
                if optimizer_step % config.save_every_steps == 0 or episode == config.episodes:
                    checkpoint(output, student, optimizer, episode, optimizer_step)
                    prune_checkpoints(output, config.keep_last_checkpoints)
    finally:
        rollout_env.close()
        if clean_env is not None:
            clean_env.close()


if __name__ == "__main__":
    main()
