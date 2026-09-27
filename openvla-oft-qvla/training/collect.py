"""Student-controlled trajectories; frozen normalized teacher targets at the same states."""
from __future__ import annotations

import torch

from evaluation.environment import reset
from integration.config import SUITES
from integration.observations import libero_action, policy_observation


@torch.no_grad()
def collect(cfg, student, client, spec, *, teacher=None, horizon=None,
            cache_features=True, capture_actions=False):
    horizon = horizon or cfg["evaluation"]["episode_horizon"] or SUITES[cfg["suite"]]
    state = reset(client, cfg, spec, horizon)
    language = state["language"]
    records = []
    success = bool(state["done"])
    steps = 0
    decisions = 0
    student.head.eval()
    while not success and steps < horizon:
        observation = policy_observation(state["observation"])
        decisions += 1
        features = student.features(observation, language)
        predicted = student.predict_features(features)
        record = None
        if teacher is not None:
            target = predicted if teacher is student else teacher.predict(observation, language)
            record = {"target": target.cpu().clone(),
                      "student": predicted.cpu().clone(), "timestep": steps}
            if cache_features:
                record["features"] = features.cpu().clone()
            if not torch.isfinite(record["target"]).all() or not torch.isfinite(record["student"]).all():
                raise ValueError("nonfinite teacher/student action in paired state")
        actions = student.actions(predicted)
        if record is not None and capture_actions:
            record["student_env_chunk"] = actions.copy()
            record["teacher_env_chunk"] = teacher.actions(target).copy()
        executed = 0
        for action in actions[:min(cfg["evaluation"]["open_loop_steps"], horizon - steps)]:
            state = client.call("step", action=libero_action(action))
            steps += 1
            executed += 1
            if state["done"]:
                success = True
                break
        if record is not None:
            count = 1 if cfg["training"]["loss_actions"] == "first" else executed
            record["action_count"] = count
            record["executed_count"] = executed
            record["q_t"] = float((record["student"][:, 0, :7] - record["target"][:, 0, :7]).square().mean())
            record["q"] = float((record["student"][:, :count] - record["target"][:, :count]).square().mean())
            records.append(record)
    return records, {"success": success, "env_steps": steps, "states": decisions,
                     "termination": "success" if success else "horizon", "language": language}
