"""Train QuantVLA-PIVOT-Q v2 on shared first-20 LIBERO-Plus trajectories.

The simulator runs in a separate ``libero_test`` process, while this trainer
stays in ``groot_test``.  The simulator rollout is non-differentiable; frozen
trajectories are scored without gradients, phase-balanced states are selected,
and only the action-head LoRA is updated.  A second clean LIBERO process
supplies the IID anchor used to limit clean-domain regression.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import socket
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from tqdm.auto import tqdm
from transformers.feature_extraction_utils import BatchFeature
import zmq

from examples.Libero.custom_data_config import LiberoDataConfig, LiberoDataConfigMeanStd
from gr00t.experiment.pivot_q import PIVOTQConfig, build_pivot_q_targets
from gr00t.experiment.pivot_q_selection import PIVOTQSelectionConfig, select_phase_balanced_states
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
    updates_per_episode: int = 5
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
    random_per_phase: int = 2
    min_temporal_gap: int = 4

    lambda_anchor: float = 0.1
    anchor_replay_size: int = 256
    anchor_batch_size: int = 4
    warmup_steps: int = 50
    save_every_steps: int = 70
    keep_last_checkpoints: int = 2
    save_timestep_scores: bool = False
    resume: bool = True
    resume_from_checkpoint: str | None = None
    dry_run: bool = False

    model_path: str | None = None
    calibration_path: str | None = None
    quant_pack_dir: str | None = None

    def validate(self) -> None:
        if self.task_suite_name not in SUITE_MODEL_PATHS:
            raise ValueError(f"unsupported suite: {self.task_suite_name}")
        if self.num_rollout_episodes <= 0 or self.updates_per_episode <= 0:
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
        if self.warmup_steps < 0 or self.save_every_steps <= 0:
            raise ValueError("warmup must be non-negative and save interval positive")
        if self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be non-negative")

    @property
    def resolved_episode_horizon(self) -> int:
        return self.episode_horizon or SUITE_EPISODE_HORIZONS[self.task_suite_name]

    @property
    def max_train_steps(self) -> int:
        return self.num_rollout_episodes * self.updates_per_episode


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


@dataclass
class RolloutTrace:
    steps: list[ScoredStep]
    success: bool
    termination_reason: str


class IIDEnvClient:
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


def _configure_quantvla(config: TrainConfig, repo_root: Path) -> None:
    calibration_name = SUITE_CALIBRATION_NAMES[config.task_suite_name]
    calibration_path = Path(
        config.calibration_path or repo_root / f"atm_alpha_beta_{calibration_name}.json"
    ).resolve()
    quant_pack_dir = Path(
        config.quant_pack_dir
        or repo_root / "model" / "quantvla" / "groot-n1.5" / config.task_suite_name / "duquant_pack"
    ).resolve()
    if not calibration_path.is_file():
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
            "GR00T_ATM_ENABLE": "1",
            "GR00T_ATM_SCOPE": "dit",
            "GR00T_OHB_ENABLE": "1",
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


def _sample_from_cached(
    model,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    initial_noise: torch.Tensor,
    *,
    with_grad: bool,
) -> torch.Tensor:
    device = model.action_head.device
    backbone = _copy_feature(backbone_output, device=device)
    actions = _copy_feature(action_input, device=device)
    noise = initial_noise.to(device=device, dtype=model.action_head.dtype)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
    env_client: IIDEnvClient,
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


def _backward_state_loss(
    student_model,
    state: CachedState,
    *,
    action_dims: int,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    prediction = _sample_from_cached(
        student_model,
        state.backbone_output,
        state.action_input,
        state.initial_noise,
        with_grad=True,
    )
    target = state.teacher_action.to(device=prediction.device, dtype=prediction.dtype)
    loss = (prediction[:, 0, :action_dims] - target[:, 0, :action_dims]).float().square().mean()
    (loss * torch.as_tensor(scale, device=loss.device, dtype=loss.dtype)).backward()
    return loss.detach()


def _save_checkpoint(
    output_dir: Path,
    student: Gr00tPolicy,
    optimizer: torch.optim.Optimizer,
    episode: int,
    optimizer_step: int,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: random.Random,
    noise_generator: torch.Generator,
    task_schedule: list[int],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step-{optimizer_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    student.model.save_pretrained(checkpoint_dir / "adapter")
    torch.save(
        {
            "episode": episode,
            "optimizer_step": optimizer_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler_state": sampler.getstate(),
            "noise_generator_state": noise_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "task_schedule": task_schedule,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    if not (checkpoint_dir / "adapter").is_dir() or not (
        checkpoint_dir / "trainer_state.pt"
    ).is_file():
        raise RuntimeError(f"incomplete checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"step-(\d{6})", path.name)
    return int(match.group(1)) if match else None


def _valid_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    candidates = []
    for path in checkpoints_dir.iterdir():
        step = _checkpoint_step(path)
        if (
            step is not None
            and path.is_dir()
            and not path.is_symlink()
            and (path / "adapter").is_dir()
            and (path / "trainer_state.pt").is_file()
        ):
            candidates.append(path)
    return sorted(candidates, key=lambda path: _checkpoint_step(path) or -1)


def _prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    """Retain only the newest complete managed checkpoints; zero keeps all."""
    if keep_last == 0:
        return
    checkpoints_dir = (output_dir / "checkpoints").resolve()
    stale = _valid_checkpoints(output_dir)[:-keep_last]
    for path in stale:
        resolved = path.resolve()
        if path.is_symlink() or resolved.parent != checkpoints_dir:
            raise RuntimeError(f"refusing to prune unsafe checkpoint path: {path}")
        shutil.rmtree(resolved)


def _resolve_resume_checkpoint(config: TrainConfig, output_dir: Path) -> Path | None:
    if config.resume_from_checkpoint is not None:
        checkpoint = Path(config.resume_from_checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        return checkpoint
    if not config.resume:
        return None
    candidates = _valid_checkpoints(output_dir)
    return candidates[-1] if candidates else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _truncate_metrics_to_episode(metrics_path: Path, episode: int) -> None:
    if not metrics_path.exists():
        return
    retained = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["episode"]) <= episode:
            retained.append(record)
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in retained), encoding="utf-8"
    )


def train(config: TrainConfig) -> None:
    config.validate()
    method_config = PIVOTQConfig(
        temporal_horizon=config.temporal_horizon,
        temporal_discount=config.temporal_discount,
        alpha_q=config.alpha_q,
        beta_r=config.beta_r,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
    )
    method_config.validate()
    selection_config = PIVOTQSelectionConfig(
        phase_bins=config.phase_bins,
        priority_per_phase=config.priority_per_phase,
        random_per_phase=config.random_per_phase,
        min_temporal_gap=config.min_temporal_gap,
        alpha_q=config.alpha_q,
        beta_r=config.beta_r,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
    )
    selection_config.validate()
    if config.dry_run:
        print(json.dumps(dataclasses.asdict(config), indent=2))
        print(json.dumps(dataclasses.asdict(method_config), indent=2))
        print(json.dumps(dataclasses.asdict(selection_config), indent=2))
        return

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(config, output_dir)
    if resume_checkpoint is not None:
        completed_state = torch.load(
            resume_checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        completed_episode = int(
            completed_state.get("episode", completed_state.get("iteration", 0))
        )
        completed_optimizer_step = int(
            completed_state.get(
                "optimizer_step", completed_episode * config.updates_per_episode
            )
        )
        if completed_episode >= config.num_rollout_episodes:
            last_metric: dict[str, Any] = {}
            metrics_path = output_dir / "metrics.jsonl"
            if metrics_path.is_file():
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        last_metric = json.loads(line)

            previous_status: dict[str, Any] = {}
            status_path = output_dir / "status.json"
            if status_path.is_file():
                try:
                    previous_status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_status = {}
            complete_status = {
                "status": "complete",
                "method": "quantvla-pivot_q-v2",
                "suite": config.task_suite_name,
                "seed": config.seed,
                "episode": completed_episode,
                "episodes_total": config.num_rollout_episodes,
                "optimizer_step": completed_optimizer_step,
                "optimizer_steps_total": config.max_train_steps,
                "last_episode_success": last_metric.get("episode_success"),
                "selected_state_count": last_metric.get("selected_state_count", 0),
                "target_min_gap": last_metric.get(
                    "target_min_gap", config.min_temporal_gap
                ),
                "phase_effective_gaps": last_metric.get("phase_effective_gaps"),
                "actual_min_gap": last_metric.get("actual_min_gap"),
                "selection_valid": last_metric.get("selection_valid"),
                "eta_seconds": 0.0,
                "host": previous_status.get("host", socket.gethostname()),
                "cuda_visible_devices": previous_status.get(
                    "cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES")
                ),
                "updated_at": time.time(),
            }
            status_tmp = output_dir / "status.json.tmp"
            status_tmp.write_text(json.dumps(complete_status, indent=2), encoding="utf-8")
            status_tmp.replace(status_path)
            print(
                f"Training already complete at episode {completed_episode}, optimizer step "
                f"{completed_optimizer_step}; leaving run metadata unchanged."
            )
            return
    serialized_config = dataclasses.asdict(config)
    serialized_config["resolved_resume_checkpoint"] = (
        str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    manifest_path = Path(config.sample_manifest).expanduser().resolve()
    previous_run: dict[str, Any] = {}
    run_path = output_dir / "run.json"
    if run_path.is_file():
        try:
            previous_run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_run = {}
    now = time.time()
    run_metadata = {
        "schema_version": 2,
        "method": "quantvla-pivot_q-v2",
        "variant": "phase-local-s16-shared560-first20",
        "suite": config.task_suite_name,
        "seed": config.seed,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env_endpoint": f"{config.env_host}:{config.env_port}",
        "clean_env_endpoint": f"{config.clean_env_host}:{config.clean_env_port}",
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "git_commit": _git_commit(repo_root),
        "launch_command": os.environ.get("GAP_PIVOT_Q_LAUNCH_COMMAND"),
        "created_at": previous_run.get("created_at", now),
        "last_started_at": now,
        "config": serialized_config,
    }
    run_path.write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    initial_status = {
        "status": "starting",
        "method": "quantvla-pivot_q-v2",
        "suite": config.task_suite_name,
        "seed": config.seed,
        "episode": 0,
        "episodes_total": config.num_rollout_episodes,
        "optimizer_step": 0,
        "optimizer_steps_total": config.max_train_steps,
        "last_episode_success": None,
        "selected_state_count": 0,
        "target_min_gap": config.min_temporal_gap,
        "phase_effective_gaps": None,
        "actual_min_gap": None,
        "selection_valid": None,
        "eta_seconds": None,
        "host": run_metadata["host"],
        "cuda_visible_devices": run_metadata["cuda_visible_devices"],
        "updated_at": time.time(),
    }
    status_tmp = output_dir / "status.json.tmp"
    status_tmp.write_text(json.dumps(initial_status, indent=2), encoding="utf-8")
    status_tmp.replace(output_dir / "status.json")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    sampler = random.Random(config.seed)
    noise_generator = torch.Generator(device="cpu").manual_seed(config.seed)

    env_client = IIDEnvClient(config.env_host, config.env_port)
    clean_env_client = None
    try:
        env_client.wait_until_ready()
        selected_task_metadata = env_client.call("list_tasks")
        available_task_ids = {int(item["task_id"]) for item in selected_task_metadata}
        if config.task_ids is None:
            primary_task_ids = [int(item["task_id"]) for item in selected_task_metadata]
        else:
            invalid_task_ids = sorted(set(config.task_ids) - available_task_ids)
            if invalid_task_ids:
                raise ValueError(
                    f"task IDs are outside the OOD calibration manifest: {invalid_task_ids}"
                )
            primary_task_ids = list(dict.fromkeys(config.task_ids))
        category_by_task_id = {
            int(item["task_id"]): str(item["category"]) for item in selected_task_metadata
        }
        task_schedule = primary_task_ids.copy()
        sampler.shuffle(task_schedule)
        run_metadata["train_task_count"] = len(primary_task_ids)
        run_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        print(
            f"Loaded shared train/eval manifest: {len(primary_task_ids)} tasks from "
            f"{Path(config.sample_manifest).resolve()}"
        )

        if config.lambda_anchor > 0.0 and config.anchor_batch_size > 0:
            clean_env_client = IIDEnvClient(config.clean_env_host, config.clean_env_port)
            clean_env_client.wait_until_ready()

        print("Loading frozen full-precision teacher")
        teacher = _load_policy(config, quantized=False, repo_root=repo_root)
        teacher.model.requires_grad_(False)
        teacher.model.eval()

        print("Loading W4A8 QuantVLA student")
        student = _load_policy(config, quantized=True, repo_root=repo_root)
        if resume_checkpoint is None:
            lora_targets = _attach_action_head_lora(student, config)
            print(f"Attached LoRA to {len(lora_targets)} action-head projections")
        else:
            student.model = PeftModel.from_pretrained(
                student.model, resume_checkpoint / "adapter", is_trainable=True
            )
            student.model.eval()
            student.model.print_trainable_parameters()
            print(f"Loaded trainable LoRA adapter from {resume_checkpoint / 'adapter'}")
        trainable_parameters = [
            parameter for parameter in student.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("student has no trainable LoRA parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        def learning_rate_scale(step: int) -> float:
            if config.warmup_steps > 0 and step < config.warmup_steps:
                return max(1, step + 1) / config.warmup_steps
            decay_steps = max(1, config.max_train_steps - config.warmup_steps)
            progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)

        start_episode = 1
        optimizer_step = 0
        if resume_checkpoint is not None:
            resume_state = torch.load(
                resume_checkpoint / "trainer_state.pt",
                map_location=config.device,
                weights_only=False,
            )
            optimizer.load_state_dict(resume_state["optimizer"])
            if "scheduler" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler"])
            completed_episode = int(resume_state.get("episode", resume_state.get("iteration", 0)))
            optimizer_step = int(
                resume_state.get("optimizer_step", completed_episode * config.updates_per_episode)
            )
            start_episode = completed_episode + 1
            if "sampler_state" in resume_state:
                sampler.setstate(resume_state["sampler_state"])
            if "noise_generator_state" in resume_state:
                noise_generator.set_state(resume_state["noise_generator_state"].cpu())
            if "torch_rng_state" in resume_state:
                torch.set_rng_state(resume_state["torch_rng_state"].cpu())
            if "numpy_rng_state" in resume_state:
                np.random.set_state(resume_state["numpy_rng_state"])
            if "task_schedule" in resume_state:
                restored_schedule = [int(value) for value in resume_state["task_schedule"]]
                if set(restored_schedule) != set(primary_task_ids):
                    raise ValueError("resume checkpoint task schedule differs from manifest")
                task_schedule = restored_schedule
            print(
                f"Resuming training at episode {start_episode}, optimizer step "
                f"{optimizer_step} from {resume_checkpoint}"
            )

        student_model = _base_model(student)
        clean_replay: deque[CachedState] = deque(maxlen=config.anchor_replay_size)
        recent_episode_durations: deque[float] = deque(maxlen=20)
        metrics_path = output_dir / "metrics.jsonl"
        if resume_checkpoint is not None:
            _truncate_metrics_to_episode(metrics_path, start_episode - 1)
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics_file:
            progress = tqdm(
                range(start_episode, config.num_rollout_episodes + 1),
                initial=start_episode - 1,
                total=config.num_rollout_episodes,
                desc=config.task_suite_name,
                dynamic_ncols=True,
            )
            for episode in progress:
                started_at = time.time()
                schedule_index = (episode - 1) % len(task_schedule)
                if schedule_index == 0 and episode > 1:
                    sampler.shuffle(task_schedule)
                task_id = task_schedule[schedule_index]
                initial_state_id = sampler.choice(config.initial_state_ids)
                trace = _collect_scored_rollout(
                    config,
                    teacher,
                    student,
                    env_client,
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                    episode_horizon=config.resolved_episode_horizon,
                    noise_generator=noise_generator,
                )
                if not trace.steps:
                    raise RuntimeError(
                        f"episode {episode}: rollout ended before any student action"
                    )

                student_actions, teacher_actions = _stack_trace_actions(trace)
                q_values, r_values, _ = build_pivot_q_targets(
                    student_actions,
                    teacher_actions,
                    method_config,
                    action_dims=config.action_dims,
                )
                selection = select_phase_balanced_states(
                    q_values,
                    r_values,
                    selection_config,
                    rng=sampler,
                )
                expected_per_phase = (
                    selection_config.priority_per_phase
                    + selection_config.random_per_phase
                )
                phase_counts = [
                    selection.phases.count(phase)
                    for phase in range(selection_config.phase_bins)
                ]
                priority_counts = [
                    selection.reasons.count(f"priority_phase_{phase}")
                    for phase in range(selection_config.phase_bins)
                ]
                random_counts = [
                    selection.reasons.count(f"random_phase_{phase}")
                    for phase in range(selection_config.phase_bins)
                ]
                selection_valid = (
                    len(selection.indices) == selection_config.states_per_episode
                    and phase_counts
                    == [expected_per_phase] * selection_config.phase_bins
                    and priority_counts
                    == [selection_config.priority_per_phase]
                    * selection_config.phase_bins
                    and random_counts
                    == [selection_config.random_per_phase]
                    * selection_config.phase_bins
                )
                if not selection_valid:
                    raise AssertionError("phase-local PIVOT-Q selection quota violation")
                states = _materialize_states(student, trace, selection.indices)
                normalized_weights = selection.weights / selection.weights.sum().clamp_min(1e-8)

                clean_anchor_steps = 0
                clean_anchor_success = False
                if clean_env_client is not None:
                    clean_trace = _collect_scored_rollout(
                        config,
                        teacher,
                        student,
                        clean_env_client,
                        task_id=sampler.choice(config.clean_task_ids),
                        initial_state_id=sampler.choice(config.clean_initial_state_ids),
                        episode_horizon=config.clean_anchor_rollout_horizon,
                        noise_generator=noise_generator,
                    )
                    clean_anchor_steps = len(clean_trace.steps)
                    clean_anchor_success = clean_trace.success
                    clean_states = _materialize_states(
                        student,
                        clean_trace,
                        list(range(len(clean_trace.steps))),
                    )
                    clean_replay.extend(clean_states)

                main_losses = []
                anchor_losses = []
                gradient_norms = []
                for _ in range(config.updates_per_episode):
                    optimizer.zero_grad(set_to_none=True)
                    for state, scale in zip(states, normalized_weights, strict=True):
                        main_losses.append(
                            _backward_state_loss(
                                student_model,
                                state,
                                action_dims=config.action_dims,
                                scale=scale,
                            ).cpu()
                        )
                    if (
                        config.lambda_anchor > 0.0
                        and config.anchor_batch_size > 0
                        and clean_replay
                    ):
                        anchor_count = min(config.anchor_batch_size, len(clean_replay))
                        anchor_states = sampler.sample(list(clean_replay), anchor_count)
                        anchor_scale = config.lambda_anchor / anchor_count
                        for state in anchor_states:
                            anchor_losses.append(
                                _backward_state_loss(
                                    student_model,
                                    state,
                                    action_dims=config.action_dims,
                                    scale=anchor_scale,
                                ).cpu()
                            )

                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, config.gradient_clip_norm
                    )
                    if not torch.isfinite(torch.as_tensor(gradient_norm)):
                        raise FloatingPointError(
                            f"non-finite gradient norm at episode {episode}"
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer_step += 1
                    gradient_norms.append(float(gradient_norm))

                selected_tensor = torch.as_tensor(selection.indices, dtype=torch.long)
                record = {
                    "episode": episode,
                    "optimizer_step": optimizer_step,
                    "suite": config.task_suite_name,
                    "method": "quantvla-pivot_q-v2",
                    "domain": "libero_plus_shared_first20_adaptation",
                    "task_id": task_id,
                    "category": category_by_task_id[task_id],
                    "initial_state_id": initial_state_id,
                    "episode_horizon": config.resolved_episode_horizon,
                    "episode_steps": len(trace.steps),
                    "episode_success": trace.success,
                    "termination_reason": trace.termination_reason,
                    "selected_state_count": len(selection.indices),
                    "selected_indices": list(selection.indices),
                    "selected_phases": list(selection.phases),
                    "selected_reasons": list(selection.reasons),
                    "selected_q": q_values[selected_tensor].tolist(),
                    "selected_r": r_values[selected_tensor].tolist(),
                    "selected_priority_scores": selection.priority_scores.tolist(),
                    "phase_counts": phase_counts,
                    "priority_counts": priority_counts,
                    "random_counts": random_counts,
                    "selection_valid": selection_valid,
                    "target_min_gap": selection_config.min_temporal_gap,
                    "phase_effective_gaps": list(selection.phase_effective_gaps),
                    "actual_min_gap": selection.actual_min_gap,
                    "selection_gap_relaxed": selection.gap_relaxed,
                    "q_mean": q_values.mean().item(),
                    "q_max": q_values.max().item(),
                    "r_mean": r_values.mean().item(),
                    "r_max": r_values.max().item(),
                    "weight_mean": selection.weights.mean().item(),
                    "weight_max": selection.weights.max().item(),
                    "loss_pivot_q_unweighted_mean": torch.stack(main_losses).mean().item(),
                    "loss_anchor_mean": (
                        torch.stack(anchor_losses).mean().item() if anchor_losses else 0.0
                    ),
                    "clean_anchor_steps": clean_anchor_steps,
                    "clean_anchor_success": clean_anchor_success,
                    "gradient_norm_mean": float(np.mean(gradient_norms)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "clean_replay_size": len(clean_replay),
                    "duration_seconds": round(time.time() - started_at, 3),
                }
                if config.save_timestep_scores:
                    record["q_by_timestep"] = q_values.tolist()
                    record["r_by_timestep"] = r_values.tolist()
                recent_episode_durations.append(record["duration_seconds"])
                eta_seconds = (
                    float(np.median(recent_episode_durations))
                    * (config.num_rollout_episodes - episode)
                )
                metrics_file.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if not key.endswith("timestep")
                            and key
                            not in {
                                "selected_indices",
                                "selected_phases",
                                "selected_reasons",
                                "selected_q",
                                "selected_r",
                                "selected_priority_scores",
                            }
                        }
                    )
                )
                progress.set_postfix(
                    category=record["category"],
                    q=f"{record['q_mean']:.3e}",
                    grad=f"{record['gradient_norm_mean']:.3e}",
                )

                status = {
                    "status": "running" if episode < config.num_rollout_episodes else "complete",
                    "method": "quantvla-pivot_q-v2",
                    "suite": config.task_suite_name,
                    "seed": config.seed,
                    "episode": episode,
                    "episodes_total": config.num_rollout_episodes,
                    "optimizer_step": optimizer_step,
                    "optimizer_steps_total": config.max_train_steps,
                    "last_episode_success": trace.success,
                    "last_episode_duration_seconds": record["duration_seconds"],
                    "selected_state_count": len(selection.indices),
                    "target_min_gap": selection_config.min_temporal_gap,
                    "phase_effective_gaps": list(selection.phase_effective_gaps),
                    "actual_min_gap": selection.actual_min_gap,
                    "selection_valid": selection_valid,
                    "eta_seconds": round(eta_seconds, 1),
                    "host": run_metadata["host"],
                    "cuda_visible_devices": run_metadata["cuda_visible_devices"],
                    "updated_at": time.time(),
                }
                status_tmp = output_dir / "status.json.tmp"
                status_tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
                status_tmp.replace(output_dir / "status.json")

                if (
                    optimizer_step % config.save_every_steps == 0
                    or episode == config.num_rollout_episodes
                ):
                    _save_checkpoint(
                        output_dir,
                        student,
                        optimizer,
                        episode,
                        optimizer_step,
                        scheduler=scheduler,
                        sampler=sampler,
                        noise_generator=noise_generator,
                        task_schedule=task_schedule,
                    )
                    _prune_checkpoints(output_dir, config.keep_last_checkpoints)
    finally:
        env_client.close()
        if clean_env_client is not None:
            clean_env_client.close()


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
