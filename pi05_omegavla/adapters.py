"""Separate trainable recovery adapters from Omega's frozen SVD branch."""
import json
from pathlib import Path
import torch
from torch import nn
from .common import atomic_json, inside
from .quantization import EXPERT
from omega_qvla.pivot_q.gptq_lora import GPTQResidualLoRA as BaseResidualLoRA, OmegaAdapterModel

class GPTQResidualLoRA(BaseResidualLoRA):
    @property
    def weight(self):
        return self.base.weight


class AdapterModel(OmegaAdapterModel):
    def __init__(self, base, identity, mode):
        super().__init__(base)
        self.identity = identity
        self.mode = mode

    def save_pretrained(self, directory):
        directory = inside(directory)
        directory.mkdir(parents=True, exist_ok=True)
        modules = {n: m for n, m in self.base_model.named_modules() if isinstance(m, GPTQResidualLoRA)}
        if not modules:
            raise RuntimeError("no adapters to save")
        first = next(iter(modules.values()))
        tensors = {f"{n}.{key}": value.detach().cpu()
                   for n, m in modules.items() for key, value in m.state_dict().items()
                   if key.startswith("lora_")}
        temp = directory / "adapter_model.tmp"
        torch.save(tensors, temp)
        temp.replace(directory / "adapter_model.pt")
        atomic_json(directory / "adapter_config.json", {
            "format": "pi05_omegavla_adapter_v1", "identity": self.identity,
            "mode": self.mode, "targets": list(modules),
            "rank": first.rank, "alpha": first.alpha, "dropout": first.dropout_probability,
        })

def attach(model, *, identity, mode, rank=16, alpha=32, dropout=0.05, path=None, trainable=True):
    names = [n for n, m in model.named_modules()
             if n.startswith(EXPERT) and n.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
             and hasattr(m, "in_features") and hasattr(m, "out_features") and not isinstance(m, GPTQResidualLoRA)]
    if not names:
        raise RuntimeError("no action-expert Q/K/V targets")
    if path is not None:
        saved = json.loads((Path(path) / "adapter_config.json").read_text())
        if saved.get("format") != "pi05_omegavla_adapter_v1" or saved["identity"] != identity:
            raise ValueError("adapter base checkpoint or quantization pack mismatch")
        if saved["mode"] != mode or saved["targets"] != names:
            raise ValueError("adapter mode/target mismatch")
        if trainable and (rank, alpha, dropout) != (saved["rank"], saved["alpha"], saved["dropout"]):
            raise ValueError("resume LoRA configuration mismatch")
        rank, alpha, dropout = saved["rank"], saved["alpha"], saved["dropout"]
    model.requires_grad_(False)
    for name in names:
        parent, attribute = name.rsplit(".", 1)
        base = model.get_submodule(name)
        model.get_submodule(parent).__setattr__(attribute, GPTQResidualLoRA(base, rank, alpha, dropout))
    if path is not None:
        state = torch.load(Path(path) / "adapter_model.pt", map_location="cpu", weights_only=True)
        expected = {f"{n}.{k}" for n in names for k in ("lora_a.weight", "lora_b.weight")}
        if set(state) != expected:
            raise ValueError("adapter tensor keys are incomplete or unexpected")
        with torch.no_grad():
            for name in names:
                m = model.get_submodule(name)
                for key in ("lora_a.weight", "lora_b.weight"):
                    tensor = state[f"{name}.{key}"]
                    parameter = m.get_parameter(key)
                    if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                        raise ValueError(f"invalid adapter tensor: {name}.{key}")
                    parameter.copy_(tensor)
    for name in names:
        m = model.get_submodule(name)
        m.lora_a.requires_grad_(trainable)
        m.lora_b.requires_grad_(trainable)
        m.base.requires_grad_(False)
    return AdapterModel(model, identity, mode).eval()
