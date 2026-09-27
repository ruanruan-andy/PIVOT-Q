"""QVLA proxy with balanced clean calibration and marginal-cost allocation."""
from __future__ import annotations

from integration.relocation import normalize_paths
from integration.allocation import calibration_plan, marginal_allocate

import json
import uuid
import torch

from evaluation.environment import EnvClient, reset
from integration.config import SUITES, atomic_json, parse_config, resolve, file_info
from integration.observations import libero_action, policy_observation
from integration.policy import OFTPolicy
from training.checkpoint import seed_all


def atomic_torch(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def collect_calibration(cfg):
    path = resolve(cfg["paths"]["calibration"])
    if path.exists():
        raise FileExistsError(f"calibration exists: {path}; choose a new output path")
    seed_all(cfg["seed"])
    policy = OFTPolicy(cfg, teacher=True)
    count = cfg["quantization"]["max_samples"]
    if count <= 0:
        raise ValueError("max_samples must be positive")
    records = []
    # Clean teacher rollouts only; do not calibrate on LIBERO-Plus evaluation states.
    with EnvClient(cfg, clean=True) as client:
        tasks = cfg["training"]["clean_task_ids"]
        initial = cfg["training"]["clean_initial_state_ids"]
        if not tasks or not initial:
            raise ValueError("calibration requires clean task/initial-state IDs")
        plan = calibration_plan(tasks, initial, count, cfg["seed"], SUITES[cfg["suite"]])
        for task_id, init_id, target_step in plan:
            state = reset(client, cfg, {"task_id": task_id, "initial_state_id": init_id}, SUITES[cfg["suite"]])
            language = state["language"]
            step = 0
            if state["done"]:
                raise RuntimeError("clean calibration reset returned a terminal state")
            while step < target_step:
                observation = policy_observation(state["observation"])
                action = policy.actions(policy.predict(observation, language))[0]
                next_state = client.call("step", action=libero_action(action))
                if next_state["done"]:
                    break  # Keep the last valid pre-action state, not a terminal observation.
                state = next_state
                step += 1
            records.append({"observation": policy_observation(state["observation"]),
                            "language": language, "task_id": task_id,
                            "initial_state_id": init_id, "timestep": step,
                            "requested_timestep": target_step})
    if len(records) != count:
        raise RuntimeError(f"collected only {len(records)} of {count} calibration states")
    atomic_torch(path, {"identity": policy.identity, "suite": cfg["suite"], "seed": cfg["seed"],
                        "source": "clean_fp_rollouts", "sampling": "balanced_rollouts_v1", "calibration_id": str(uuid.uuid4()), "records": records})
    print(f"Saved {len(records)} clean calibration states to {path}")


def quantize(cfg):
    seed_all(cfg["seed"])
    data_path = resolve(cfg["paths"]["calibration"])
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if data.get("sampling") != "balanced_rollouts_v1":
        raise ValueError("calibration uses obsolete sampling; collect fresh clean states")
    policy = OFTPolicy(cfg, teacher=True)
    from qvla.sensitivity_hessian_proxy import _HessianProxy, _compute_proxy_for_bits
    from integration.quantization import canonical_gates, target_modules, unused_oft_pool

    q = cfg["quantization"]
    bits = sorted(set(q["bits"]))
    if not bits or 16 not in bits or not set(bits) <= {0, 2, 4, 8, 16}:
        raise ValueError("bits must include 16 and be a subset of 0,2,4,8,16")
    if not min(bits) <= q["target_avg_bits"] < 16:
        raise ValueError("target_avg_bits must be within the selected range and below 16")
    if normalize_paths(data["identity"]) != normalize_paths(policy.identity) or data["suite"] != cfg["suite"]:
        raise ValueError("calibration/base policy mismatch")
    records = data["records"][:q["max_samples"]]
    if not records:
        raise ValueError("empty calibration")
    proxy_path = resolve(cfg["paths"]["proxy"])
    metadata_path = proxy_path.with_suffix(".metadata.json")
    signature = {"identity": policy.identity, "calibration_file": file_info(data_path), "calibration_id": data["calibration_id"],
                 "bits": bits, "percdamp": q["percdamp"], "max_samples": q["max_samples"]}
    proxies = {}
    if proxy_path.exists():
        if not q["resume_proxy"]:
            raise FileExistsError("proxy exists; choose a new path or enable resume_proxy")
        if not metadata_path.exists() or normalize_paths(json.loads(metadata_path.read_text())) != normalize_paths(signature):
            raise ValueError("proxy resume configuration does not match")
        proxies = torch.load(proxy_path, map_location="cpu", weights_only=True)
    targets = list(target_modules(policy.model).items())
    unknown = set(proxies) - {name for name, _ in targets}
    if unknown:
        raise ValueError(f"proxy contains unexpected target layers: {sorted(unknown)}")
    # Verify the pinned OFT forward really bypasses these pooling modules.
    # Do not treat arbitrary missing activations as permission to skip a layer.
    handles = []
    def unexpected_pool(module, inputs):
        raise RuntimeError("OFT attention pooling executed; target policy must be revalidated")
    try:
        for name, module in policy.model.named_modules():
            if unused_oft_pool(name):
                handles.append(module.register_forward_pre_hook(unexpected_pool))
        with torch.no_grad():
            for record in records:
                policy.features(record["observation"], record["language"])
    finally:
        for handle in handles:
            handle.remove()
    atomic_json(metadata_path, signature)
    for index, (name, module) in enumerate(targets):
        if name in proxies:
            continue
        accumulator = _HessianProxy(module, policy.device)

        def hook(_module, inputs, _output):
            accumulator.add_batch(inputs[0])

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for record in records:
                    policy.features(record["observation"], record["language"])
        finally:
            handle.remove()
        if accumulator.nsamples == 0:
            raise RuntimeError(f"calibration did not execute {name}")
        with torch.no_grad():
            diagonal = accumulator.diag_hinv(q["percdamp"])
            result = _compute_proxy_for_bits(module, diagonal, bits)
        proxies[name] = {f"proxy_{bit}": value.cpu() for bit, value in result.items()}
        del accumulator, diagonal, result
        atomic_torch(proxy_path, proxies)
        print(f"[{index + 1}/{len(targets)}] calibrated {name}", flush=True)
    allocation, stats = marginal_allocate(
        {name: {bit: values[f"proxy_{bit}"] for bit in bits} for name, values in proxies.items()},
        bits, q["target_avg_bits"], q["max_zero_fraction"], q["min_bit"],
    )
    gates = canonical_gates(policy.model, allocation)
    gates_path = resolve(cfg["paths"]["gates"])
    atomic_json(gates_path, gates)
    atomic_json(gates_path.with_suffix(".metadata.json"), {**signature, "allocation_stats": stats,
                "gates_file": file_info(gates_path),
                "target_policy": "oft_intermediate_features_exclude_attn_pool_v1",
                "note": "channel-average budget over non-pooling targets; unused attention pools retain original weights; fake quant, not packed kernels"})
    print(json.dumps(stats, indent=2))


def main():
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in {"collect", "quantize"}:
        raise SystemExit("usage: python -m integration.calibrate {collect|quantize} --config ...")
    operation = sys.argv.pop(1)
    cfg, dry = parse_config(__doc__)
    if dry:
        print(json.dumps(cfg, indent=2))
    elif operation == "collect":
        collect_calibration(cfg)
    else:
        quantize(cfg)


if __name__ == "__main__":
    main()
