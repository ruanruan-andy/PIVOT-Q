"""ATM/OHB calibration for π0.5's Gemma action expert."""
from __future__ import annotations

import math
from collections import defaultdict

import torch


def _attention_layers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if name.startswith("paligemma_with_expert.gemma_expert.model.layers.")
        and name.endswith(".self_attn")
    }


class AttentionStatistics:
    def __init__(self, model, *, scales=None, collect=True):
        self._sum = defaultdict(lambda: None)
        self._count = defaultdict(int)
        self._q = {}
        self._hooks = []
        self.scales = scales or {}
        self.collect = collect
        for name, attn in _attention_layers(model).items():
            self._hooks += [
                attn.q_proj.register_forward_hook(self._q_hook(name, attn)),
                attn.k_proj.register_forward_hook(self._k_hook(name, attn)),
                attn.o_proj.register_forward_hook(self._o_hook(name)),
            ]

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _add(self, key, value):
        if not self.collect:
            return
        value = value.detach().float().cpu()
        self._sum[key] = value if self._sum[key] is None else self._sum[key] + value
        self._count[key] += 1

    def _q_hook(self, name, attn):
        alpha = self.scales.get(name, {}).get("alpha")
        def hook(_, __, output):
            heads = attn.config.num_attention_heads
            head_dim = output.shape[-1] // heads
            shaped = output.view(*output.shape[:-1], heads, head_dim)
            if alpha is not None:
                shaped = shaped * torch.as_tensor(alpha, device=output.device, dtype=output.dtype).view(1, 1, -1, 1)
            self._q[name] = shaped
            return shaped.reshape_as(output)
        return hook

    def _k_hook(self, name, attn):
        def hook(_, __, output):
            query = self._q.pop(name, None)
            if query is None:
                return output
            heads = query.shape[-2]
            key = output.view(*output.shape[:-1], attn.config.num_key_value_heads, -1)
            if key.shape[-2] != heads:
                key = key.repeat_interleave(heads // key.shape[-2], dim=-2)
            logits = torch.einsum("bshd,bthd->bhst", query.float(), key.float()) / math.sqrt(query.shape[-1])
            self._add((name, "logit_std"), logits.std(dim=(-1, -2), unbiased=False).mean(dim=0))
            return output
        return hook

    def _o_hook(self, name):
        beta = self.scales.get(name, {}).get("beta")
        def hook(_, __, output):
            self._add((name, "output_rms"), output.float().square().mean().sqrt().reshape(1))
            if beta is not None:
                return output * float(beta)
            return output
        return hook

    def finalize(self):
        result = {}
        for (name, metric), total in self._sum.items():
            result.setdefault(name, {})[metric] = (total / self._count[(name, metric)]).tolist()
        return result


def compute_scales(teacher, student, *, log_clip=0.4, neutral=0.03):
    scales = {}
    for name in teacher:
        if name not in student:
            continue
        alpha = torch.as_tensor(teacher[name]["logit_std"]) / torch.as_tensor(student[name]["logit_std"]).clamp_min(1e-6)
        beta = torch.as_tensor(teacher[name]["output_rms"]) / torch.as_tensor(student[name]["output_rms"]).clamp_min(1e-6)
        def clamp(value):
            log = value.log().clamp(-log_clip, log_clip)
            return torch.where(log.abs() < neutral, torch.zeros_like(log), log).exp()
        scales[name] = {"alpha": clamp(alpha).tolist(), "beta": float(clamp(beta).item())}
    return scales
