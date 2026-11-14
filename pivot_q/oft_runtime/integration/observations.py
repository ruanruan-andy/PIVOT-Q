"""Official OFT camera orientation and LIBERO action conventions."""
import numpy as np


def policy_observation(obs):
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] ** 2)
    axis_angle = np.zeros(3) if np.isclose(den, 0) else quat[:3] * 2 * np.arccos(quat[3]) / den
    return {
        "full_image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"])).astype(np.float32),
    }


def libero_action(action):
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("expected finite 7D LIBERO action")
    # Official normalize_gripper_action (2*x-1, binarize), then invert.
    action[-1] = -np.sign(2 * action[-1] - 1)
    return action.tolist()
