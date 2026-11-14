"""Build a checkpoint-specific LeRobot Omega/SVDQuant pack using official GPTQ."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path
from .cost import measured
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from .configuration import load_config, resolve_paths, inside, atomic_json, checkpoint_identity, sha256
from .holoq import official, targets, EXPERT, validate_record
from .holoq_flow import install_step_context, sample, make_noise

def activation_quantile(tokens, percentile, device, *, smooth=None):
    result = torch.empty(tokens.shape[1], dtype=torch.float32, device=device)
    for start in range(0, tokens.shape[1], 64):
        columns = slice(start, start + 64)
        block = tokens[:, columns].to(device=device, dtype=torch.float32).contiguous()
        if smooth is not None:
            block = block / smooth[columns]
        result[columns] = torch.quantile(block.abs(), percentile, dim=0)
    return result


def activation_hessian(per_step, smooth):
    hessian = smooth.new_zeros((smooth.numel(), smooth.numel()))
    count = 0
    for tokens in per_step:
        for block in tokens.split(1024, dim=0):
            scaled = block.to(device=smooth.device, dtype=torch.float32) / smooth
            hessian.addmm_(scaled.T, scaled)
            count += len(block)
    return hessian.div_(count)


def build_record(weight, per_step, q, solver):
    """Same custom/original SVD residual construction as the official pi05 builder."""
    weight = weight.detach().float()
    device = weight.device
    eps = 1e-8
    if not per_step or any(x.ndim != 2 or x.shape[0] == 0 for x in per_step):
        raise ValueError("every calibration step must have observations")
    a = torch.stack([activation_quantile(x, 0.999, device) for x in per_step]).amax(0).clamp_min(eps)
    b = weight.abs().amax(0).clamp_min(eps)
    smooth = a.pow(q["sq_alpha"]) / b.pow(1 - q["sq_alpha"])
    smooth = (smooth / smooth.median().clamp_min(eps)).clamp(q["sq_clamp_lo"], q["sq_clamp_hi"])
    ws = weight * smooth
    h = activation_hessian(per_step, smooth)
    def quantize(w):
        return solver(w, h, bits=q["weight_bits"], block_size=q["gptq_block_size"],
                      damp_percent=q["gptq_damp_percent"], reg_lambda=q["gptq_reg_lambda"],
                      err_comp_gamma=q["gptq_err_comp_gamma"])
    if q["quant_type"] not in {"custom", "original"}:
        raise ValueError("quant_type must be custom or original")
    residual = ws - quantize(ws) if q["quant_type"] == "custom" else ws
    rank = int(q["svd_rank"])
    if not 0 < rank <= min(weight.shape):
        raise ValueError("svd_rank must be positive and fit each layer")
    u, s, v = torch.svd_lowrank(residual, q=min(rank + 6, min(weight.shape)))
    a_lr = u[:, :rank] * s[:rank].clamp_min(0).sqrt()
    b_lr = v[:, :rank] * s[:rank].clamp_min(0).sqrt()
    weight_res_q = quantize(ws - a_lr @ b_lr.T)
    table = torch.stack([activation_quantile(x, q["act_percentile"] / 100, device, smooth=smooth).clamp_min(1e-6)
                         / (2 ** (q["activation_bits"] - 1) - 1) for x in per_step])
    record = {"format": "dit_svdquant_v1", "weight_res_q": weight_res_q,
              "lowrank_A": a_lr, "lowrank_B": b_lr, "smooth_scale": smooth,
              "act_scale_table": table, "weight_bits": q["weight_bits"],
              "a_bits": q["activation_bits"], "rank": rank,
              "in_features": weight.shape[1], "out_features": weight.shape[0],
              "n_calib_per_step": [len(x) for x in per_step], "quant_type": q["quant_type"]}
    for key, value in list(record.items()):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"calibration generated non-finite {key}")
            # Keep scales FP32. Runtime supports dense dequantized W4 records;
            # it is fake-quant inference, not a packed INT4 acceleration kernel.
            record[key] = value.detach().cpu().contiguous()
    return record

def build(config, args):
    settings, q = config["calibration"], config["quantization"]
    if settings["observations"] <= 0 or settings["token_cap"] <= 0 or settings["layers_per_pass"] <= 0 or args.max_layers < 0:
        raise ValueError("calibration counts must be positive")
    output = Path(args.output).resolve()
    buffer = args.buffer or Path(config["paths"]["calibration_buffer"])
    if args.dry_run:
        print(json.dumps({"checkpoint": config["paths"]["checkpoint"], "buffer": str(buffer),
                          "output": str(output), "calibration": settings, "quantization": q}, indent=2))
        return
    runtime = official(config["paths"]["omega_source"])
    from gr00t.quantization.dit_step_context import get_current_dit_step
    from gr00t.quantization.duquant_preprocess import sanitize_name
    from .pi05_policy import load_policy
    checkpoint = Path(config["paths"]["checkpoint"])
    identity = checkpoint_identity(checkpoint)
    plan = {"checkpoint": identity, "buffer_sha256": sha256(buffer), "calibration": settings,
            "quantization": q, "num_steps": config["inference"]["num_inference_steps"]}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "calibration_plan.json"
    if plan_path.exists():
        if not args.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing calibration differs or --resume was not specified")
    elif any(output.glob("*.pt")) or (output / "manifest.json").exists():
        raise ValueError("unrecognized existing pack; choose a new output directory")
    atomic_json(plan_path, plan)
    torch.manual_seed(settings["seed"])
    policy = load_policy(ROOT, checkpoint, device=args.device).eval()
    policy.requires_grad_(False)
    if policy.config.num_inference_steps != plan["num_steps"]:
        raise ValueError("checkpoint inference steps differ from calibration")
    install_step_context(policy.model)
    from lerobot.policies import make_pre_post_processors
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
                          preprocessor_overrides={"device_processor": {"device": args.device}})
    container = torch.load(buffer, map_location="cpu", weights_only=False)
    samples = container["samples"] if isinstance(container, dict) else container
    samples = samples[:settings["observations"]]
    if len(samples) != settings["observations"]:
        raise ValueError("buffer contains fewer observations than requested")
    selected = targets(policy.model)
    names = list(selected)
    if args.max_layers:
        names = names[:args.max_layers]
    entries = {}
    size = settings["layers_per_pass"]
    for offset in range(0, len(names), size):
        group = names[offset:offset + size]
        pending = []
        for name in group:
            path = output / (sanitize_name(name) + ".pt")
            if args.resume and path.exists():
                record = torch.load(path, map_location="cpu", weights_only=True)
                validate_record(record, selected[name], name, plan["num_steps"], q)
                entries[name] = {"file": path.name, "sha256": sha256(path)}
            else:
                pending.append(name)
        if not pending:
            continue
        cache = {name: [[] for _ in range(plan["num_steps"] if name.startswith(EXPERT) else 1)] for name in pending}
        def hook(name):
            def capture(module, inputs):
                step = get_current_dit_step() if name.startswith(EXPERT) else 0
                if step is None or step >= len(cache[name]):
                    raise RuntimeError(f"{name}: missing calibration step")
                x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
                # Deterministic stratified tokens: bounded CPU memory per pass.
                if len(x) > settings["token_cap"]:
                    indices = torch.linspace(0, len(x)-1, settings["token_cap"], device=x.device).long()
                    x = x[indices]
                cache[name][step].append(x.cpu())
            return capture
        handles = [selected[name].register_forward_pre_hook(hook(name)) for name in pending]
        try:
            for index, observation in enumerate(samples):
                generator = torch.Generator(device=args.device).manual_seed(settings["seed"] + index)
                measured("calibration_forward", sample, policy, pre(observation),
                         make_noise(policy, generator), with_grad=False)
        finally:
            for handle in handles:
                handle.remove()
        for name in pending:
            torch.manual_seed(settings["seed"] + list(selected).index(name))
            shape = "x".join(str(d) for d in selected[name].weight.shape)
            record = measured(f"quantize_layer_{shape}", build_record, selected[name].weight, [torch.cat(items) for items in cache.pop(name)], q,
                                  runtime.gptq_quantize_weight)
            validate_record(record, selected[name], name, plan["num_steps"], q)
            path = output / (sanitize_name(name) + ".pt")
            temp = path.with_suffix(".tmp")
            torch.save(record, temp)
            temp.replace(path)
            entries[name] = {"file": path.name, "sha256": sha256(path)}
            print(f"{len(entries)}/{len(names)} {name}", flush=True)
            del record
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    manifest = {"format": "pi05_omegavla_lerobot_v1", "complete": set(entries) == set(selected),
                "checkpoint": identity, "num_steps": plan["num_steps"], "quantization": q,
                "calibration": plan, "layers": entries}
    atomic_json(output / "manifest.json", manifest)
    print(f"saved {len(entries)} layers; complete={manifest['complete']}")

