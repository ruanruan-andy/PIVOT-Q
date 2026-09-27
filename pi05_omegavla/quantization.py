"""Read-only use of the official Omega GPTQ/SVDQuant runtime."""
from __future__ import annotations
import importlib
import json
import re
import sys
from pathlib import Path
import torch
from torch import nn
from .common import checkpoint_identity, sha256

TARGET = re.compile(r"^paligemma_with_expert\.(?:paligemma\.model\.language_model|gemma_expert\.model)\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$")
EXPERT = "paligemma_with_expert.gemma_expert.model.layers."

def official(source):
    source = Path(source).resolve()
    if not (source / "gr00t/quantization/gptq_layers.py").is_file():
        raise FileNotFoundError(f"Omega runtime missing: {source}")
    existing = sys.modules.get("gr00t")
    if existing and not Path(existing.__file__).resolve().is_relative_to(source):
        raise RuntimeError("another gr00t checkout was imported before Omega")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source)) if str(source) not in sys.path else None
    module = importlib.import_module("gr00t.quantization.gptq_layers")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("resolved GPTQ runtime does not belong to configured Omega source")
    return module

def targets(model):
    selected = {n: m for n, m in model.named_modules() if TARGET.fullmatch(n) and isinstance(m, nn.Linear)}
    if not selected:
        raise RuntimeError("no pi0.5 language/action-expert linear layers found")
    return selected

class ExactForwardSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantized):
        return quantized
    @staticmethod
    def backward(ctx, grad):
        return grad, None

def enable_ste(runtime):
    if getattr(runtime, "_pi05_omega_ste", False):
        return
    original = runtime.fake_quantize_sym
    def quantize(x, *args, **kwargs):
        q = original(x, *args, **kwargs)
        return ExactForwardSTE.apply(x, q) if torch.is_grad_enabled() and x.requires_grad else q
    runtime.fake_quantize_sym = quantize
    runtime._pi05_omega_ste = True

def validate_record(record, module, name, steps, qconfig):
    expected = steps if name.startswith(EXPERT) else 1
    shapes = {
        "weight_res_q": (module.out_features, module.in_features),
        "smooth_scale": (module.in_features,),
        "act_scale_table": (expected, module.in_features),
        "lowrank_A": (module.out_features, qconfig["svd_rank"]),
        "lowrank_B": (module.in_features, qconfig["svd_rank"]),
    }
    if record.get("format") != "dit_svdquant_v1":
        raise ValueError(f"{name}: unsupported Omega record")
    if record.get("weight_bits") != qconfig["weight_bits"] or record.get("a_bits") != qconfig["activation_bits"]:
        raise ValueError(f"{name}: precision mismatch")
    for key, shape in shapes.items():
        value = record.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"{name}: invalid {key}; expected {shape}")
    for key in ("smooth_scale", "act_scale_table"):
        if not (record[key] > 0).all():
            raise ValueError(f"{name}: {key} must be positive")

def apply(model, *, source, pack, checkpoint, steps, qconfig, trainable=False):
    runtime = official(source)
    pack = Path(pack).resolve()
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"calibrate this LeRobot checkpoint first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "pi05_omegavla_lerobot_v1" or not manifest.get("complete"):
        raise ValueError("incomplete or incompatible Omega pack")
    if manifest["checkpoint"] != checkpoint_identity(checkpoint):
        raise ValueError("Omega pack belongs to a different checkpoint/preprocessor")
    if manifest["num_steps"] != steps or manifest["quantization"] != qconfig:
        raise ValueError("Omega pack configuration differs from requested inference")
    selected = targets(model)
    if set(selected) != set(manifest["layers"]):
        raise ValueError("Omega pack does not exactly cover model target layers")
    cfg = runtime.GptqConfig(enabled=True, path=str(pack), act_bits=qconfig["activation_bits"],
                             weight_bits=qconfig["weight_bits"], missing="error")
    from gr00t.quantization.duquant_preprocess import sanitize_name
    for name, base in selected.items():
        entry = manifest["layers"][name]
        if entry["file"] != sanitize_name(name) + ".pt":
            raise ValueError(f"noncanonical layer filename: {name}")
        path = pack / entry["file"]
        if path.resolve().parent != pack or sha256(path) != entry["sha256"]:
            raise ValueError(f"invalid or changed layer file: {name}")
        record = torch.load(path, map_location="cpu", weights_only=True)
        validate_record(record, base, name, steps, qconfig)
        wrapped = runtime.GptqLinear(base, name, cfg)
        # Some official buffers are constructed on CPU. Move once, preserving
        # FP32 scale tables to avoid underflow when the base is lower precision.
        wrapped.to(device=base.weight.device)
        if not wrapped._quant_available or not wrapped._has_svdquant or not wrapped._has_act_scale_table:
            raise RuntimeError(f"{name}: runtime silently dropped required quantization")
        model.get_submodule(name.rsplit(".", 1)[0]).__setattr__(name.rsplit(".", 1)[1], wrapped)
        # The official file loader caches CPU weights globally. Release each
        # layer after construction; model buffers own their runtime values.
        runtime._GPTQ_CACHE.pop(str(path.resolve()), None)
    model.requires_grad_(False)
    if trainable:
        enable_ste(runtime)
    return {"checkpoint": manifest["checkpoint"], "pack_sha256": sha256(manifest_path),
            "num_steps": steps, "quantization": qconfig, "quantized_layers": len(selected)}
