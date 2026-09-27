import json
import os
import pathlib
import pprint
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from scripts.eval_run_metadata import write_eval_run_metadata

default_output_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")
)
log_dir = os.environ.get("LIBERO_EVAL_LOG_DIR", default_output_dir)
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    # convert RGB -> BGR for OpenCV
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # show in separate windows
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                     # Number of rollouts per task
    #################################################################################################################
    # fmt: on
    """Port to connect to."""
    port: int = 5555
    """Headless mode (no GUI)."""
    headless: bool = False
    """Save rollout videos next to the episode log."""
    save_video: bool = True
    """Run only the specified task indices (overrides order if provided)."""
    task_ids: list[int] | None = None
    """Run tasks in this explicit order."""
    task_order: list[int] | None = None
    """Continue from completed rollouts recorded in episodes.jsonl."""
    resume: bool = False
    """Stable identifier used to isolate FP16 and QuantVLA outputs."""
    model_variant: str = "groot-fp16"
    """Base seed for deterministic per-task/per-step flow noise."""
    policy_seed: int = 0


class GR00TPolicy:
    """GR00T Policy wrapper for Libero environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(self, host="localhost", port=5555, headless=False):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless

    def get_action(self, observation_dict, lang: str, policy_seed: int | None = None):
        """Get action from GR00T policy given observation and language instruction."""
        obs_dict = self._process_observation(observation_dict, lang)
        if policy_seed is not None:
            obs_dict["__policy_seed"] = int(policy_seed)
        # summarize_obs(obs_dict)
        action_chunk = self.policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, 0)

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(
        self, action_chunk: dict[str, np.array], idx: int = 0
    ) -> np.ndarray:
        """Convert GR00T action chunk to Libero format.

        Args:
            action_chunk: Dictionary of action components from GR00T policy
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


def _load_episode_records(path: str, suite_name: str, model_variant: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete JSONL line {line_number} in {path}")
                continue
            if record.get("suite") != suite_name:
                raise ValueError(
                    f"Resume record suite {record.get('suite')!r} does not match {suite_name!r}"
                )
            if record.get("model_variant", model_variant) != model_variant:
                raise ValueError(
                    f"Resume record model {record.get('model_variant')!r} does not match "
                    f"{model_variant!r}"
                )
            records.append(record)
    return records


def _write_standard_summary(path: str, cfg: GenerateConfig, records: list[dict]) -> None:
    successes = sum(bool(record["success"]) for record in records)
    summary = {
        "model_variant": cfg.model_variant,
        "suite": cfg.task_suite_name,
        "completed_episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "errors": sum(record.get("error") is not None for record in records),
        "num_trials_per_task": cfg.num_trials_per_task,
    }
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def eval_libero(cfg: GenerateConfig) -> None:
    metrics_dir = os.environ.get("LIBERO_EVAL_METRICS_DIR", os.path.join(log_dir, "metrics"))
    logs_dir = os.environ.get("LIBERO_EVAL_LOGS_DIR", os.path.join(log_dir, "logs"))
    video_dir = os.environ.get("LIBERO_EVAL_VIDEO_DIR", os.path.join(log_dir, "videos"))
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    if cfg.save_video:
        os.makedirs(video_dir, exist_ok=True)
    os.environ["LIBERO_EVAL_VIDEO_DIR"] = video_di
    write_eval_run_metadata(pathlib.Path(log_dir), cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    if cfg.task_ids:
        task_indices = cfg.task_ids
    elif cfg.task_order:
        task_indices = cfg.task_order
    else:
        task_indices = list(range(num_tasks_in_suite))
    task_indices = [idx for idx in task_indices if 0 <= idx < num_tasks_in_suite]

    log_path = f"{logs_dir}/evaluator.log"
    episodes_path = f"{metrics_dir}/episodes.jsonl"
    summary_path = f"{metrics_dir}/summary.json"
    existing_records = (
        _load_episode_records(episodes_path, cfg.task_suite_name, cfg.model_variant)
        if cfg.resume
        else []
    )
    records_by_key = {
        (int(record["task_index"]), int(record["episode_index"])): record
        for record in existing_records
        if int(record["task_index"]) in task_indices
    }
    completed_keys = {
        key for key, record in records_by_key.items() if record.get("error") is None
    }
    file_mode = "a" if cfg.resume else "w"
    log_file = open(log_path, file_mode, encoding="utf-8", buffering=1)
    episodes_log = open(episodes_path, file_mode, encoding="utf-8", buffering=1)
    log_file.write(
        f"\nModel: {cfg.model_variant}; task suite: {cfg.task_suite_name}; resume={cfg.resume}; "
        f"restored={len(completed_keys)}\n"
    )
    print(f"Resume: {cfg.resume}; restored completed rollouts: {len(completed_keys)}")

    gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)

    try:
        for task_id in tqdm.tqdm(task_indices):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            max_trials = min(cfg.num_trials_per_task, len(initial_states))
            pending_episodes = [
                episode_idx
                for episode_idx in range(max_trials)
                if (task_id, episode_idx) not in completed_keys
            ]
            if not pending_episodes:
                message = f"Skipping completed task {task_id} ({max_trials} rollouts)"
                print(message)
                log_file.write(message + "\n")
                continue

            max_steps = {
                "libero_spatial": 220,
                "libero_object": 280,
                "libero_goal": 600,
                "libero_10": 1000,
                "libero_90": 400,
            }[cfg.task_suite_name]
            env, task_description = get_libero_env(
                task,
                resolution=256,
                horizon=max(1000, max_steps + cfg.num_steps_wait),
            )
            try:
                for episode_idx in tqdm.tqdm(pending_episodes):
                    print(f"\nTask: {task_description}")
                    log_file.write(f"\nTask: {task_description}\n")
                    started_at = time.time()
                    error = None
                    done = False
                    t = 0
                    top_view = []
                    wrist_view = []

                    try:
                        env.reset()
                        obs = env.set_init_state(initial_states[episode_idx])
                        while t < max_steps + cfg.num_steps_wait:
                            if t < cfg.num_steps_wait:
                                obs, _, done, _ = env.step(get_libero_dummy_action())
                                t += 1
                                continue
                            img, wrist_img = get_libero_image(obs)
                            top_view.append(img)
                            wrist_view.append(wrist_img)
                            step_seed = (
                                cfg.policy_seed
                                + task_id * 1_000_000
                                + episode_idx * 10_000
                                + (t - cfg.num_steps_wait)
                            )
                            action = gr00t_policy.get_action(
                                obs, task.language, policy_seed=step_seed
                            )
                            obs, _, done, _ = env.step(action.tolist())
                            t += 1
                            if done:
                                break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        print(f"Caught exception: {error}")
                        log_file.write(f"Caught exception: {error}\n")

                    key = (task_id, episode_idx)
                    provisional_total = len(records_by_key) + (key not in records_by_key)
                    video_path = None
                    if cfg.save_video and top_view and wrist_view:
                        try:
                            video_path = save_rollout_video(
                                top_view,
                                wrist_view,
                                provisional_total,
                                success=done,
                                task_description=task_description,
                                log_file=log_file,
                            )
                        except Exception as exc:
                            video_error = f"{type(exc).__name__}: {exc}"
                            error = f"{error}; video: {video_error}" if error else f"video: {video_error}"

                    record = {
                        "model_variant": cfg.model_variant,
                        "suite": cfg.task_suite_name,
                        "task_index": task_id,
                        "episode_index": episode_idx,
                        "task_description": task_description,
                        "success": bool(done),
                        "steps": t,
                        "duration_seconds": round(time.time() - started_at, 3),
                        "error": error,
                        "video_path": video_path,
                    }
                    episodes_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_by_key[key] = record
                    if error is None:
                        completed_keys.add(key)
                    effective_records = list(records_by_key.values())
                    _write_standard_summary(summary_path, cfg, effective_records)
                    total_episodes = len(effective_records)
                    total_successes = sum(bool(item["success"]) for item in effective_records)
                    success_rate = total_successes / total_episodes * 100
                    status = (
                        f"Success: {done}\n"
                        f"# episodes completed so far: {total_episodes}\n"
                        f"# successes: {total_successes} ({success_rate:.1f}%)\n"
                    )
                    print(status, end="")
                    log_file.write(status)

                task_records = [
                    record
                    for (record_task_id, _), record in records_by_key.items()
                    if record_task_id == task_id
                ]
                task_rate = (
                    sum(bool(record["success"]) for record in task_records) / len(task_records)
                    if task_records
                    else 0.0
                )
                log_file.write(f"Current task success rate: {task_rate}\n")
            finally:
                env.close()
    finally:
        episodes_log.close()
        log_file.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
