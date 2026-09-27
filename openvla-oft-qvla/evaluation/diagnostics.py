"""Portable per-episode records for q_t density/tail analysis; no pickle required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from training.selection import future_risk

Q_DEFINITION = "mean_d((student_normalized_first_action[d]-fp_at_same_state_normalized_first_action[d])**2), d=0..6"


def definition(cfg):
    return {"schema_version": 1, "q_definition": Q_DEFINITION, "action_space": "normalized",
            "dimensions": 7, "action_index": 0, "state_distribution": "evaluated_policy_on_policy",
            "open_loop_steps": cfg["evaluation"]["open_loop_steps"],
            "q_objective_actions": cfg["training"]["loss_actions"],
            "future_horizon": cfg["selection"]["future_horizon"], "discount": cfg["selection"]["discount"],
            "q_env_definition": "first-action MSE after unnormalization, before gripper binarization/inversion",
            "decision_drift": "not collected; requires a separate FP rollout at aligned time indices"}


def save_episode(output, cfg, spec, episode, records, outcome, *, phase, policy_seed, selection=None):
    """Save all valid decisions, including exact zeros; never save just selected states."""
    if phase not in {"eval", "train_pre_update"}:
        raise ValueError("invalid diagnostic phase")
    if len(records) != outcome["states"]:
        raise ValueError("diagnostics must cover every policy decision")
    n = len(records)
    first = np.asarray([r["q_t"] for r in records], dtype=np.float32)
    objective = np.asarray([r["q"] for r in records], dtype=np.float32)
    student = np.concatenate([r["student"].numpy() for r in records]) if n else np.empty((0, 8, 7), dtype=np.float32)
    teacher = np.concatenate([r["target"].numpy() for r in records]) if n else np.empty((0, 8, 7), dtype=np.float32)
    for array in (first, objective, student, teacher):
        if not np.isfinite(array).all():
            raise ValueError("cannot save nonfinite action diagnostics")
    selected = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float32)
    if selection is not None:
        selected[selection["indices"]] = True
        weights[selection["indices"]] = np.asarray(selection["weights"])
    key = f"{cfg['suite']}:{spec['task_id']}:{spec['initial_state_id']}"
    meta = {**definition(cfg), "episode": episode, "episode_key": key,
            "method": cfg["method"], "suite": cfg["suite"], **spec, **outcome,
            "phase": phase, "policy_seed": policy_seed, "environment_seed": cfg["environment"]["seed"],
            "complete": True, "reference": "fp_self" if cfg["method"] == "fp" else "frozen_fp_same_observation"}
    arrays = {"metadata_json": np.asarray(json.dumps(meta)),
              "q_t": first, "q_objective": objective,
              "r_t": future_risk(torch.from_numpy(first), cfg["selection"]["future_horizon"], cfg["selection"]["discount"]).numpy(),
              "decision_index": np.arange(n, dtype=np.int64),
              "timestep": np.asarray([r["timestep"] for r in records], dtype=np.int64),
              "executed_count": np.asarray([r["executed_count"] for r in records], dtype=np.int64),
              "loss_action_count": np.asarray([r["action_count"] for r in records], dtype=np.int64),
              "valid": np.ones(n, dtype=bool), "selected": selected, "selection_weight": weights,
              "student_action_chunk": student, "teacher_at_student_chunk": teacher}
    if n:
        arrays["student_env_chunk"] = np.stack([r["student_env_chunk"] for r in records])
        arrays["teacher_env_chunk"] = np.stack([r["teacher_env_chunk"] for r in records])
        arrays["q_env_t"] = np.square(arrays["student_env_chunk"][:, 0] - arrays["teacher_env_chunk"][:, 0]).mean(axis=-1)
    else:
        arrays["student_env_chunk"] = np.empty((0, 8, 7))
        arrays["teacher_env_chunk"] = np.empty((0, 8, 7))
        arrays["q_env_t"] = np.empty(0)
    path = Path(output) / "states" / f"episode-{episode:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)
    return {"states_path": str(path.relative_to(output)), "q_by_timestep": first.tolist(),
            "q_objective_by_timestep": objective.tolist(), "r_by_timestep": arrays["r_t"].tolist(),
            "timesteps": arrays["timestep"].tolist(), "q_mean": float(first.mean()) if n else None,
            "q_p95": float(np.quantile(first, .95)) if n else None,
            "q_p99": float(np.quantile(first, .99)) if n else None,
            "q_max": float(first.max()) if n else None, "q_zero_count": int((first == 0).sum())}
