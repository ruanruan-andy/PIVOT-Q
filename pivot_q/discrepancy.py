"""Collect and replay diagnostic observations in a private temporary cache."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from unittest.mock import patch


def cache():
    value = os.environ.get("PIVOT_Q_DIAGNOSTIC_CACHE")
    return Path(value) if value else None


def first_action(chunk):
    import torch
    return torch.as_tensor(chunk)[0, 0, :7].detach().float().cpu()


def normalized_prediction(policy, batch, seed):
    import torch
    torch.manual_seed(seed)
    if hasattr(policy, "diagnostic_prediction"):
        return policy.diagnostic_prediction(batch, seed)
    return policy.predict_action_chunk(batch), None


@contextmanager
def capture(policy, spec, policy_seed):
    root = cache()
    if root is None:
        yield
        return
    import torch
    predict = policy.predict_action_chunk
    directory = root / spec["suite"]
    directory.mkdir(parents=True, exist_ok=True)
    index = 0

    def save(batch, *args, **kwargs):
        nonlocal index
        seed = (policy_seed + index) % (2**32)
        if hasattr(policy, "diagnostic_prediction"):
            actions, env_actions = normalized_prediction(policy, batch, seed)
        else:
            torch.manual_seed(seed)
            actions = predict(batch, *args, **kwargs)
            env_actions = actions
        inputs = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                  for key, value in batch.items()}
        state = {**spec, "decision": index, "noise_seed": seed, "input": inputs,
                 "reference": first_action(actions)}
        torch.save(state, directory / f"{spec['task_id']:06d}-{spec['initial_state_id']:04d}-{index:05d}.pt")
        index += 1
        return env_actions

    with patch.object(policy, "predict_action_chunk", save):
        yield


def replay(policy, suite, method, device):
    import torch
    from tqdm.auto import tqdm
    root = cache()
    paths = sorted((root / suite).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No diagnostic observations for {suite}")
    output = root / f"{suite}-{method}.jsonl"
    episode = None
    with output.open("w") as stream, torch.inference_mode():
        for path in tqdm(paths, desc=f"{suite} {method}", unit="state"):
            state = torch.load(path, map_location="cpu", weights_only=False)
            current = (state["task_id"], state["initial_state_id"])
            if current != episode:
                policy.reset()
                episode = current
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in state.pop("input").items()}
            reference = state.pop("reference")
            actions, _ = normalized_prediction(policy, batch, state["noise_seed"])
            state["first_action_mse"] = float((first_action(actions) - reference).square().mean())
            stream.write(json.dumps(state, allow_nan=False) + "\n")
