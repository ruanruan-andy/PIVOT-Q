"""Train QuantVLA-PIVOT_Q v2 on shared first-20 LIBERO-Plus trajectories.

The simulator runs in a separate ``libero_test`` process, while this trainer
stays in ``groot_test``.  The simulator rollout is non-differentiable; frozen
trajectories are scored without gradients, phase-balanced states are selected,
and only the action-head LoRA is updated.  A second clean LIBERO process
supplies the clean anchor used to limit clean-domain regression.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import socket
import time
from collections import deque
from contextlib import ExitStack, contextmanager, nullcontext
from unittest.mock import patch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from tqdm.auto import tqdm
from transformers.feature_extraction_utils import BatchFeature
import zmq
from pivot_q.cost import measured, measured_batch, restore, scope, snapshot

from examples.Libero.custom_data_config import LiberoDataConfig, LiberoDataConfigMeanStd
from pivot_q.groot_runtime.objective import (
    PIVOTQConfig as GAPPIVOT_QConfig,
    build_pivot_q_targets as build_pivot_q_targets,
)
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values


SUITE_MODEL_PATHS = {
    "libero_spatial": "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
    "libero_goal": "youliangtan/gr00t-n1.5-libero-goal-posttrain",
    "libero_object": "youliangtan/gr00t-n1.5-libero-object-posttrain",
    "libero_10": "youliangtan/gr00t-n1.5-libero-long-posttrain",
}
SUITE_CALIBRATION_NAMES = {
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_10": "long",
}
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
SUITE_EPISODE_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


@dataclass
class TrainConfig:
    task_suite_name: str = "libero_spatial"
    output_dir: str = "output/train/libero-plus/pivot_q-v2-s16-shared560-first20/seed-000/libero_spatial"
    sample_manifest: str = "configs/libero_plus/shared560-first20.json"
    num_rollout_episodes: int = 140
    episode_horizon: int | None = None
    rollouts_per_round: int = 2
    batch_size: int = 32
    max_rounds: int = 0
    num_steps_wait: int = 10
    task_ids: list[int] | None = None
    initial_state_ids: list[int] = field(default_factory=lambda: [0])
    seed: int = 0
    env_host: str = "127.0.0.1"
    env_port: int = 5590
    clean_env_host: str = "127.0.0.1"
    clean_env_port: int = 5591
    clean_task_ids: list[int] = field(default_factory=lambda: list(range(10)))
    clean_initial_state_ids: list[int] = field(default_factory=lambda: list(range(40)))
    clean_anchor_rollout_horizon: int = 4

    device: str = "cuda:0"
    denoising_steps: int = 8
    action_dims: int = 7

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    temporal_horizon: int = 4
    temporal_discount: float = 0.9
    alpha_q: float = 1.0
    beta_r: float = 1.0
    weight_min: float = 0.5
    weight_max: float = 2.0
    phase_bins: int = 4
    priority_per_phase: int = 2
    percent: float | None = None
    random_per_phase: int = 2
    min_temporal_gap: int = 4

    lambda_anchor: float = 0.1
    anchor_replay_size: int = 256
    anchor_batch_size: int = 4
    save_every_rounds: int = 10
    wandb_project: str | None = None
    checkpoint_denoising: bool = False
    setting: str = "groot_quantvla"
    keep_last_checkpoints: int = 2
    save_timestep_scores: bool = False
    resume: bool = True
    resume_from_checkpoint: str | None = None
    # Load a previous round's adapter but deliberately start a fresh optimizer,
    # scheduler, replay buffer, and task schedule.  This is self-evolve, not
    # checkpoint resume.
    initial_adapter_path: str | None = None
    method_name: str = "pivot_q"
    # Paper sparse baselines: selection and loss weighting are independent.
    selection_method: str = "pivot_q"
    selection_weighting: str = "vulnerability"
    dry_run: bool = False

    model_path: str | None = None
    calibration_path: str | None = None
    quant_pack_dir: str | None = None

    def validate(self) -> None:
        if self.task_suite_name not in SUITE_MODEL_PATHS:
            raise ValueError(f"unsupported suite: {self.task_suite_name}")
        if self.num_rollout_episodes <= 0 or self.batch_size <= 0:
            raise ValueError("rollout episodes and updates per episode must be positive")
        if self.episode_horizon is not None and self.episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive when specified")
        if self.task_ids is not None and not self.task_ids:
            raise ValueError("task_ids must be None or non-empty")
        if not self.initial_state_ids:
            raise ValueError("initial_state_ids must not be empty")
        if self.action_dims <= 0:
            raise ValueError("action_dims must be positive")
        if self.lambda_anchor < 0.0:
            raise ValueError("lambda_anchor must be non-negative")
        if self.lambda_anchor > 0.0 and (
            not self.clean_task_ids
            or not self.clean_initial_state_ids
            or self.clean_anchor_rollout_horizon <= 0
        ):
            raise ValueError("clean anchor sampling settings must be positive and non-empty")
        if self.anchor_replay_size < 0 or self.anchor_batch_size < 0:
            raise ValueError("anchor replay settings must be non-negative")
        if self.num_rollout_episodes % self.rollouts_per_round:
            raise ValueError("Rollout budget must be divisible by rollouts_per_round")
        if self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be non-negative")
        if self.resume_from_checkpoint and self.initial_adapter_path:
            raise ValueError("resume_from_checkpoint and initial_adapter_path are exclusive")

    @property
    def resolved_episode_horizon(self) -> int:
        return self.episode_horizon or SUITE_EPISODE_HORIZONS[self.task_suite_name]



@dataclass
class CachedState:
    backbone_output: BatchFeature
    action_input: BatchFeature
    initial_noise: torch.Tensor
    teacher_action: torch.Tensor


@dataclass
class ScoredStep:
    policy_observation: dict[str, Any]
    initial_noise: torch.Tensor
    teacher_action: torch.Tensor
    student_action: torch.Tensor
    # Dense baseline only: immutable CPU features from the frozen rollout backbone.
    cached_student_state: CachedState | None = None


@dataclass
class RolloutTrace:
    steps: list[ScoredStep]
    success: bool
    termination_reason: str


class EnvClient:
    """Small trusted-localhost RPC client for the clean LIBERO process."""

    def __init__(self, host: str, port: int, timeout_ms: int = 120_000):
        self.endpoint = f"tcp://{host}:{port}"
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
            raise RuntimeError(f"LIBERO environment service: {response['error']}")
        return response["result"]

    def wait_until_ready(self, timeout_seconds: float = 60.0) -> None:
        deadline = time.time() + timeout_seconds
        self.socket.setsockopt(zmq.RCVTIMEO, 1_000)
        self.socket.setsockopt(zmq.SNDTIMEO, 1_000)
        while True:
            try:
                self.call("ping")
                self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                return
            except (zmq.ZMQError, RuntimeError):
                if time.time() >= deadline:
                    raise TimeoutError("clean LIBERO environment service did not become ready")
                self.socket.close(linger=0)
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, 1_000)
                self.socket.setsockopt(zmq.SNDTIMEO, 1_000)
                self.socket.connect(self.endpoint)
                time.sleep(0.5)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def _data_config_for_suite(suite_name: str):
    return LiberoDataConfigMeanStd() if suite_name == "libero_goal" else LiberoDataConfig()


def _clear_quantization_environment() -> None:
    prefixes = ("GR00T_DUQUANT_", "GR00T_ATM_", "GR00T_OHB_")
    for key in list(os.environ):
        if key.startswith(prefixes):
            del os.environ[key]


def _configure_quantvla(config: TrainConfig, repo_root: Path, *, calibrating=False) -> None:
    calibration_name = SUITE_CALIBRATION_NAMES[config.task_suite_name]
    calibration_path = Path(
        config.calibration_path or repo_root / f"atm_alpha_beta_{calibration_name}.json"
    ).resolve()
    quant_pack_dir = Path(
        config.quant_pack_dir
        or repo_root / "model" / "quantvla" / "groot-n1.5" / config.task_suite_name / "duquant_pack"
    ).resolve()
    if not calibrating and not calibration_path.is_file():
        raise FileNotFoundError(f"QuantVLA calibration file is missing: {calibration_path}")
    quant_pack_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "GR00T_DUQUANT_DEBUG": "0",
            "GR00T_DUQUANT_SCOPE": "",
            "GR00T_DUQUANT_INCLUDE": (
                r".*(backbone\.eagle_model\.language_model\..*\."
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|"
                r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\."
                r"(0\.proj|2)).*"
            ),
            "GR00T_DUQUANT_EXCLUDE": (
                r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"
            ),
            "GR00T_DUQUANT_WBITS_DEFAULT": "4",
            "GR00T_DUQUANT_ABITS": "8",
            "GR00T_DUQUANT_BLOCK": "64",
            "GR00T_DUQUANT_PERMUTE": "0",
            "GR00T_DUQUANT_ROW_ROT": "restore",
            "GR00T_DUQUANT_ACT_PCT": "99.9",
            "GR00T_DUQUANT_CALIB_STEPS": "32",
            "GR00T_DUQUANT_LS": "0.15",
            "GR00T_DUQUANT_PACKDIR": str(quant_pack_dir),
            "GR00T_ATM_ALPHA_PATH": str(calibration_path),
            "GR00T_ATM_ENABLE": "0" if calibrating else "1",
            "GR00T_ATM_SCOPE": "dit",
            "GR00T_OHB_ENABLE": "0" if calibrating else "1",
            "GR00T_OHB_FALLBACK": "1.0",
            "GR00T_OHB_SCOPE": "dit",
        }
    )


def _load_policy(config: TrainConfig, *, quantized: bool, repo_root: Path) -> Gr00tPolicy:
    if quantized:
        _configure_quantvla(config, repo_root)
    else:
        _clear_quantization_environment()

    data_config = _data_config_for_suite(config.task_suite_name)
    policy = Gr00tPolicy(
        model_path=config.model_path or SUITE_MODEL_PATHS[config.task_suite_name],
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=config.denoising_steps,
        device=config.device,
    )
    policy.model.eval()
    return policy


def _attach_action_head_lora(policy: Gr00tPolicy, config: TrainConfig) -> list[str]:
    target_modules = []
    for name, module in policy.model.named_modules():
        if "action_head" not in name or not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in ("to_q", "to_k", "to_v")):
            target_modules.append(name)
    if not target_modules:
        raise RuntimeError("no action-head attention projections were found for LoRA")

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    policy.model = get_peft_model(policy.model, lora_config)
    policy.model.eval()
    policy.model.print_trainable_parameters()
    return target_modules


def _base_model(policy: Gr00tPolicy):
    get_base_model = getattr(policy.model, "get_base_model", None)
    return get_base_model() if get_base_model is not None else policy.model


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator


def _get_libero_images(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        observation["agentview_image"][::-1, ::-1],
        observation["robot0_eye_in_hand_image"][::-1, ::-1],
    )


def _normalize_gripper_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[..., -1] = np.sign(1.0 - 2.0 * action[..., -1])
    return action


def _process_observation(observation: dict[str, Any], language: str) -> dict[str, Any]:
    xyz = observation["robot0_eef_pos"]
    rpy = _quat_to_axis_angle(observation["robot0_eef_quat"])
    gripper = observation["robot0_gripper_qpos"]
    image, wrist_image = _get_libero_images(observation)
    return {
        "video.image": np.expand_dims(image, axis=0),
        "video.wrist_image": np.expand_dims(wrist_image, axis=0),
        "state.x": np.array([[xyz[0]]]),
        "state.y": np.array([[xyz[1]]]),
        "state.z": np.array([[xyz[2]]]),
        "state.roll": np.array([[rpy[0]]]),
        "state.pitch": np.array([[rpy[1]]]),
        "state.yaw": np.array([[rpy[2]]]),
        "state.gripper": np.expand_dims(gripper, axis=0),
        "annotation.human.action.task_description": [language],
    }


def _normalize_observation(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    batched = unsqueeze_dict_values(observation)
    for key, value in batched.items():
        if not isinstance(value, np.ndarray):
            batched[key] = np.asarray(value)
    return policy.apply_transforms(batched)


def _copy_feature(feature: BatchFeature, device: str | torch.device) -> BatchFeature:
    copied = {}
    for key, value in feature.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"cached BatchFeature value {key!r} is not a tensor")
        copied[key] = value.detach().to(device=device)
    return BatchFeature(data=copied)


def _cache_feature(feature: BatchFeature) -> BatchFeature:
    return _copy_feature(feature, device="cpu")


def _prepare_backbone(model, normalized_input: dict[str, Any]):
    backbone_input, action_input = model.prepare_input(normalized_input)
    with torch.no_grad():
        backbone_output = model.backbone(backbone_input)
    required_action_input = BatchFeature(
        data={key: action_input[key] for key in ("state", "embodiment_id")}
    )
    return backbone_output, required_action_input


@contextmanager
def _checkpoint_denoising(action_head):
    from functools import partial
    from torch.utils.checkpoint import checkpoint

    with ExitStack() as stack:
        for block in action_head.model.transformer_blocks:
            recompute = partial(checkpoint, block.forward, use_reentrant=False)
            stack.enter_context(patch.object(block, "forward", recompute))
        yield


def _sample_from_cached(
    model,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    initial_noise: torch.Tensor,
    *,
    with_grad: bool,
    checkpoint_denoising: bool = False,
) -> torch.Tensor:
    device = model.action_head.device
    backbone = _copy_feature(backbone_output, device=device)
    actions = _copy_feature(action_input, device=device)
    noise = initial_noise.to(device=device, dtype=model.action_head.dtype)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    denoising = _checkpoint_denoising(model.action_head) if checkpoint_denoising else nullcontext()
    with context, denoising, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model.action_head.sample_actions(
            backbone, actions, initial_noise=noise
        )["action_pred"]
    return prediction


def _normalized_to_libero_action(policy: Gr00tPolicy, normalized_action: torch.Tensor) -> np.ndarray:
    action_dict = policy._get_unnormalized_action(normalized_action.detach().float().cpu())
    components = []
    for key in ACTION_KEYS:
        value = np.asarray(action_dict[f"action.{key}"])
        if value.ndim < 2:
            raise ValueError(f"unexpected unnormalized action shape for {key}: {value.shape}")
        components.append(float(value.reshape(value.shape[0], value.shape[1], -1)[0, 0, 0]))
    action = np.asarray(components, dtype=np.float32)
    return _normalize_gripper_action(action)


def _collect_scored_rollout(
    config: TrainConfig,
    teacher: Gr00tPolicy,
    student: Gr00tPolicy,
    env_client: EnvClient,
    *,
    task_id: int,
    initial_state_id: int,
    episode_horizon: int,
    noise_generator: torch.Generator,
) -> RolloutTrace:
    teacher_model = _base_model(teacher)
    student_model = _base_model(student)
    steps: list[ScoredStep] = []
    success = False
    reset_result = env_client.call(
        "reset",
        suite=config.task_suite_name,
        task_id=task_id,
        initial_state_id=initial_state_id,
        num_steps_wait=config.num_steps_wait,
        horizon=config.num_steps_wait + episode_horizon + 1,
    )
    observation = reset_result["observation"]
    language = reset_result["language"]
    if reset_result["done"]:
        return RolloutTrace([], True, "success_during_wait")

    for _ in range(episode_horizon):
        policy_observation = _process_observation(observation, language)
        normalized_input = _normalize_observation(student, policy_observation)
        teacher_backbone, teacher_action_input = _prepare_backbone(
            teacher_model, normalized_input
        )
        student_backbone, student_action_input = _prepare_backbone(
            student_model, normalized_input
        )
        cache_backbone = config.method_name in {"full_distill", "omega_full_distill"}
        if cache_backbone:
            if any(m.training for m in student_model.backbone.modules()) or any(
                p.requires_grad for p in student_model.backbone.parameters()
            ):
                raise RuntimeError("Dense feature caching requires a frozen eval-mode backbone")
            # Capture before the action head can transform its input dictionary.
            cached_backbone = _cache_feature(student_backbone)
            cached_action_input = _cache_feature(student_action_input)

        noise = torch.randn(
            1,
            student_model.action_head.config.action_horizon,
            student_model.action_head.config.action_dim,
            generator=noise_generator,
            dtype=torch.float32,
        )
        teacher_prediction = _sample_from_cached(
            teacher_model,
            teacher_backbone,
            teacher_action_input,
            noise,
            with_grad=False,
        )
        student_prediction = _sample_from_cached(
            student_model,
            student_backbone,
            student_action_input,
            noise,
            with_grad=False,
        )

        teacher_cpu = teacher_prediction.detach().float().cpu()
        student_cpu = student_prediction.detach().float().cpu()
        steps.append(
            ScoredStep(
                policy_observation=policy_observation,
                initial_noise=noise.cpu(),
                teacher_action=teacher_cpu,
                student_action=student_cpu,
                cached_student_state=(
                    CachedState(cached_backbone, cached_action_input, noise.cpu(), teacher_cpu)
                    if cache_backbone else None
                ),
            )
        )

        libero_action = _normalized_to_libero_action(student, student_prediction)
        step_result = env_client.call("step", action=libero_action.tolist())
        observation = step_result["observation"]
        if step_result["done"]:
            success = True
            break

    return RolloutTrace(
        steps=steps,
        success=success,
        termination_reason="success" if success else "horizon",
    )


def _stack_trace_actions(trace: RolloutTrace) -> tuple[torch.Tensor, torch.Tensor]:
    if not trace.steps:
        return torch.empty(0), torch.empty(0)
    student_actions = torch.stack([step.student_action.squeeze(0) for step in trace.steps])
    teacher_actions = torch.stack([step.teacher_action.squeeze(0) for step in trace.steps])
    return student_actions, teacher_actions


def _materialize_states(
    student: Gr00tPolicy,
    trace: RolloutTrace,
    indices: list[int] | tuple[int, ...],
) -> list[CachedState]:
    """Recompute and cache backbone features only for selected states."""
    student_model = _base_model(student)
    states: list[CachedState] = []
    for index in indices:
        step = trace.steps[index]
        if step.cached_student_state is not None:
            states.append(step.cached_student_state)
            continue
        normalized_input = _normalize_observation(student, step.policy_observation)
        student_backbone, student_action_input = _prepare_backbone(
            student_model, normalized_input
        )
        states.append(
            CachedState(
                backbone_output=_cache_feature(student_backbone),
                action_input=_cache_feature(student_action_input),
                initial_noise=step.initial_noise,
                teacher_action=step.teacher_action,
            )
        )
    return states


def _combine_features(states, attribute):
    result = {}
    for key in getattr(states[0], attribute):
        tensors = [getattr(state, attribute)[key] for state in states]
        if key in {"backbone_features", "backbone_attention_mask"}:
            length = max(tensor.shape[1] for tensor in tensors)
            padded = []
            for tensor in tensors:
                shape = list(tensor.shape)
                shape[1] = length
                target = tensor.new_zeros(shape)
                target[:, :tensor.shape[1]] = tensor
                padded.append(target)
            tensors = padded
        result[key] = torch.cat(tensors, dim=0)
    return BatchFeature(data=result)


def _backward_batch(student_model, states, weights, action_dims, checkpoint_denoising=False):
    prediction = _sample_from_cached(
        student_model, _combine_features(states, "backbone_output"),
        _combine_features(states, "action_input"),
        torch.cat([state.initial_noise for state in states]), with_grad=True,
        checkpoint_denoising=checkpoint_denoising)
    target = torch.cat([state.teacher_action for state in states]).to(prediction)
    errors = (prediction[:, 0, :action_dims].float() - target[:, 0, :action_dims].float()).square().mean(-1)
    loss = (errors * errors.new_tensor(weights)).sum()
    loss.backward()
    return float(loss.detach())


def train(config, select_states):
    from pivot_q.pi05_train import adapter_dropout
    from pivot_q.oft_runtime.training.checkpoint import rng_state, restore_rng
    from pivot_q.cost import measured, measured_batch, restore, scope, snapshot
    from pivot_q.pi05_state import write_status
    from pivot_q.plot_data import save_selection

    config.validate()
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    if config.save_every_rounds > 0:
        checkpoints.mkdir(exist_ok=True)
    complete = sorted(path for path in checkpoints.glob("round-*") if (path / "complete.json").is_file())
    saved = torch.load(complete[-1] / "trainer_state.pt", map_location="cpu", weights_only=False) if complete else None
    if saved and not config.resume:
        raise ValueError("Training checkpoint exists; enable resume")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    sampler = random.Random(config.seed)
    noise = torch.Generator().manual_seed(config.seed)
    root = Path(os.environ["PIVOT_Q_QUANTVLA_ROOT"])
    client = EnvClient(config.env_host, config.env_port)
    clean = EnvClient(config.clean_env_host, config.clean_env_port) if config.lambda_anchor else None
    run = None
    try:
        if config.wandb_project:
            import wandb

            run = wandb.init(project=config.wandb_project,
                            name=f'{config.method_name}-{config.task_suite_name}-seed-{config.seed}',
                            dir=str(out), config=dataclasses.asdict(config))
            run.define_metric("round")
            run.define_metric("distill_loss", step_metric="round")
            run.define_metric("lr", step_metric="round")
            if config.lambda_anchor:
                run.define_metric("anchor_loss", step_metric="round")
        client.wait_until_ready()
        if clean:
            clean.wait_until_ready()
        tasks = client.call("list_tasks", suite=config.task_suite_name)
        schedule = [row["task_id"] for row in tasks]
        sampler.shuffle(schedule)
        if len(schedule) < config.num_rollout_episodes:
            raise ValueError("Task manifest has fewer trajectories than requested")
        teacher = _load_policy(config, quantized=False, repo_root=root)
        teacher.model.requires_grad_(False)
        student = _load_policy(config, quantized=True, repo_root=root)
        if saved:
            student.model = PeftModel.from_pretrained(student.model, complete[-1] / "adapter", is_trainable=True)
        else:
            _attach_action_head_lora(student, config)
        parameters = [p for p in student.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        replay = deque(maxlen=config.anchor_replay_size)
        start, step = 0, 0
        if saved:
            start, step = saved["round"], saved["optimizer_step"]
            optimizer.load_state_dict(saved["optimizer"])
            replay.extend(saved["replay"])
            sampler.setstate(saved["sampler"])
            noise.set_state(saved["noise"])
            schedule = saved["schedule"]
            restore_rng(saved["rng"])
            restore(saved["compute"])
        total = config.num_rollout_episodes // config.rollouts_per_round
        end = min(total, start + config.max_rounds) if config.max_rounds else total
        metrics = out / "metrics.jsonl"
        if metrics.exists():
            rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows if row["round"] <= start))
        write_status(out / "run.json", dict(format="groot_round_v2", config=dataclasses.asdict(config)))
        for index in tqdm(range(start, end), initial=start, total=total, desc=config.task_suite_name, unit="round"):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(config.device)
            weighted, records = [], []
            for offset in range(config.rollouts_per_round):
                task_id = schedule[index * config.rollouts_per_round + offset]
                initial_state_id = sampler.choice(config.initial_state_ids)
                trace = _collect_scored_rollout(config, teacher, student, client, task_id=task_id,
                    initial_state_id=initial_state_id,
                    episode_horizon=config.resolved_episode_horizon, noise_generator=noise)
                if not trace.steps:
                    raise ValueError("Empty student trajectory")
                student_actions, teacher_actions = _stack_trace_actions(trace)
                q, risk, _ = build_pivot_q_targets(student_actions, teacher_actions, config, action_dims=config.action_dims)
                selected = select_states(q, risk, config, sampler)
                if config.method_name != "full_distill":
                    save_selection(out, setting=config.setting, suite=config.task_suite_name, seed=config.seed, round_index=index + 1,
                        suite_round=index + 1, rollout=offset, task_id=task_id, initial_state_id=initial_state_id,
                        method=config.selection_method, q=q, risk=risk, scores=getattr(selected, "candidate_scores", None),
                        indices=selected.indices, weights=selected.weights, phases=config.phase_bins, percent=config.percent)
                states = _materialize_states(student, trace, selected.indices)
                weights = selected.weights
                weighted.extend(zip(states, weights.tolist()))
                records.append(dict(task_id=task_id, states=len(trace.steps), selected=len(states), success=trace.success))
                if clean:
                    with scope("anchor/"):
                        anchor_trace = _collect_scored_rollout(config, teacher, student, clean,
                            task_id=sampler.choice(config.clean_task_ids), initial_state_id=sampler.choice(config.clean_initial_state_ids),
                            episode_horizon=config.clean_anchor_rollout_horizon, noise_generator=noise)
                    replay.extend(_materialize_states(student, anchor_trace, list(range(len(anchor_trace.steps)))))
            sampler.shuffle(weighted)
            losses = []
            base = _base_model(student)
            for offset in range(0, len(weighted), config.batch_size):
                group = weighted[offset:offset + config.batch_size]
                mass = sum(weight for _, weight in group)
                optimizer.zero_grad(set_to_none=True)
                with adapter_dropout(student.model):
                    loss = measured_batch("distill_forward_backward", len(group), _backward_batch, base,
                                    [state for state, _ in group], [weight/mass for _, weight in group], config.action_dims, config.checkpoint_denoising)
                    anchors = sampler.sample(list(replay), min(config.anchor_batch_size, len(replay))) if config.lambda_anchor else []
                    anchor_loss = measured_batch("anchor_forward_backward", len(anchors), _backward_batch, base, anchors,
                        [config.lambda_anchor/len(anchors)]*len(anchors), config.action_dims,
                        config.checkpoint_denoising) if anchors else 0.0
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm, error_if_nonfinite=True)
                measured("optimizer_step", optimizer.step)
                step += 1
                losses.append((loss, anchor_loss, mass, len(group)))
            record = dict(round=index+1, optimizer_step=step, optimizer_steps_this_round=len(losses),
                          rollouts_complete=(index+1)*config.rollouts_per_round, suite=config.task_suite_name,
                          selected_states=len(weighted), state_evaluations=len(weighted), streams=records,
                          distill_loss=sum(row[0]*row[2] for row in losses)/sum(row[2] for row in losses),
                          anchor_loss=sum(row[1]*row[3] for row in losses)/len(weighted),
                          duration_seconds=time.monotonic()-started,
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(config.device),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(config.device))
            with metrics.open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            write_status(out / "status.json", record)
            if run is not None:
                log = dict(round=record["round"], distill_loss=record["distill_loss"],
                           lr=optimizer.param_groups[0]["lr"])
                if config.lambda_anchor:
                    log["anchor_loss"] = record["anchor_loss"]
                run.log(log, step=record["round"])
            if config.save_every_rounds > 0 and ((index+1) % config.save_every_rounds == 0 or index+1 == end):
                target = checkpoints / f"round-{index+1:06d}"
                temp = checkpoints / f".round-{index+1:06d}.incomplete"
                if temp.exists():
                    shutil.rmtree(temp)
                temp.mkdir()
                student.model.save_pretrained(temp / "adapter")
                torch.save(dict(round=index+1, optimizer_step=step,
                                optimizer=optimizer.state_dict(), replay=list(replay),
                                sampler=sampler.getstate(), noise=noise.get_state(), schedule=schedule,
                                rng=rng_state(), compute=snapshot()), temp / "trainer_state.pt")
                (temp / "complete.json").write_text(json.dumps(record))
                temp.replace(target)
                for old in sorted(checkpoints.glob("round-*"))[:-config.keep_last_checkpoints]:
                    shutil.rmtree(old)
        if config.save_every_rounds > 0 and end == total:
            target = checkpoints / f"round-{total:06d}"
            adapter = out / "final_adapter"
            if adapter.exists():
                shutil.rmtree(adapter)
            shutil.copytree(target / "adapter", adapter)
    finally:
        client.close()
        if clean:
            clean.close()
        if run is not None:
            run.finish(exit_code=int(sys.exc_info()[0] is not None))
