"""Strict gate validation around the unchanged official fake-quant injector."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from integration.config import file_info


def unused_oft_pool(name):
    """OFT's get_intermediate_layers path bypasses the timm pooling head.

    Keep this exclusion narrow: any other unobserved target must still fail
    calibration, rather than silently accepting incomplete calibration data.
    """
    return any(name.startswith(f"vision_backbone.{branch}.attn_pool.")
               for branch in ("featurizer", "fused_featurizer"))


def target_modules(model):
    """Shared target set for proxy allocation, gate validation and injection."""
    from qvla.inject_fake_w import _is_target_module
    return {n: m for n, m in model.named_modules()
            if _is_target_module(n, m) and not unused_oft_pool(n)}


def canonical_gates(model, raw):

    # The official allocator writes an 'assign' envelope, while its injector
    # expects a flat mapping. Normalize that interface here, not upstream.
    mapping = raw.get("assign", raw)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("empty or invalid gate mapping")
    targets = target_modules(model)
    canonical, used = {}, set()
    for name, module in targets.items():
        short = name.split(".", 1)[1]
        key = name if name in mapping else short
        if key not in mapping:
            raise ValueError(f"missing gates for {name}; partial quantization is not silently accepted")
        values = mapping[key]
        if len(values) != module.weight.shape[0]:
            raise ValueError(f"gate channel count mismatch for {name}")
        if any(isinstance(b, bool) or not isinstance(b, int) or b not in {0, 2, 4, 8, 16} for b in values):
            raise ValueError(f"invalid bit assignment for {name}")
        canonical[name] = values
        used.add(key)
    if set(mapping) != used or not canonical:
        raise ValueError(f"unknown/ambiguous gate keys: {sorted(set(mapping) - used)}")
    return canonical


def inject(model, path):
    from qvla.inject_fake_w import inject_qvla_weight_fake_quant

    path = Path(path)
    gates = canonical_gates(model, json.loads(path.read_text()))
    modules = dict(model.named_modules())
    channel_bits = [bit for values in gates.values() for bit in values]
    total_weights = sum(modules[name].weight.numel() for name in gates)
    weighted_bits = sum(sum(values) * (modules[name].weight.numel() // len(values)) for name, values in gates.items())
    # Only our temporary mapping is written. No upstream or base weights on disk change.
    with tempfile.TemporaryDirectory(prefix="oft-qvla-gates-") as temp:
        flat_path = Path(temp) / "gates.json"
        flat_path.write_text(json.dumps(gates))
        count = inject_qvla_weight_fake_quant(model, str(flat_path))
    if count != len(gates):
        raise RuntimeError("official injector did not apply every gate")
    return {
        "backend": "official_weight_only_fake_quant", "gates_file": file_info(path),
        "modules": count, "channel_average_bits": sum(channel_bits) / len(channel_bits),
        "weight_average_bits_quantized_modules": weighted_bits / total_weights,
        "bit_histogram": {str(b): channel_bits.count(b) for b in sorted(set(channel_bits))},
        "activation_dtype": str(next(model.parameters()).dtype),
    }
