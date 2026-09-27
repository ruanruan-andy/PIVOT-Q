"""Port of PIVOT_Q selector's phase-balanced ranking, without GR00T imports."""
from __future__ import annotations

import torch


def future_risk(q, horizon, discount):
    risk = torch.zeros_like(q)
    for i in range(len(q) - 1):
        count = min(horizon, len(q) - i - 1)
        weights = q.new_tensor(discount).pow(torch.arange(count, dtype=q.dtype, device=q.device))
        risk[i] = (q[i + 1:i + 1 + count] * weights).sum() / weights.sum()
    return risk


def ranks(values):
    order = torch.argsort(values.float(), stable=True)
    result = torch.empty_like(values, dtype=torch.float32)
    result[order] = torch.linspace(0, 1, len(values), device=values.device)
    return result


def select(q, method, cfg):
    q = torch.as_tensor(q, dtype=torch.float32).cpu()
    if q.ndim != 1 or not len(q) or not torch.isfinite(q).all():
        raise ValueError("rollout discrepancies must be a finite non-empty vector")
    risk = future_risk(q, cfg["future_horizon"], cfg["discount"])
    if method == "full_distill":
        return {"indices": list(range(len(q))), "weights": torch.ones(len(q)) / len(q),
                "q": q.tolist(), "risk": risk.tolist(), "phase_gaps": [], "short_rollout": False}
    if method != "pivot_q":
        raise ValueError("selection requires pivot_q or full_distill")
    phases, quota = cfg["phase_bins"], cfg["top_per_phase"]
    scores = torch.zeros_like(q)
    indices, gaps = [], []
    short = False
    for phase in range(phases):
        pool = [i for i in range(len(q)) if i * phases // len(q) == phase]
        if not pool:
            gaps.append(None)
            short = True
            continue
        scores[pool] = cfg["alpha_q"] * ranks(q[pool]) + cfg["beta_r"] * ranks(risk[pool])
        count = min(quota, len(pool))
        short |= count < quota
        ordered = sorted(pool, key=lambda i: (-float(scores[i]), i))
        for gap in range(cfg["min_gap"], -1, -1):
            chosen = []
            for i in ordered:
                if all(abs(i - other) >= gap for other in chosen):
                    chosen.append(i)
                    if len(chosen) == count:
                        break
            if len(chosen) == count:
                break
        indices.extend(chosen)
        gaps.append(gap)
    indices.sort()
    weights = (scores[indices] / scores[indices].mean().clamp_min(1e-8)).clamp(cfg["weight_min"], cfg["weight_max"])
    return {"indices": indices, "weights": weights / weights.sum(), "q": q.tolist(),
            "risk": risk.tolist(), "phase_gaps": gaps, "short_rollout": short}
