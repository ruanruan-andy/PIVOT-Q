"""Persist scalar rollout data used by the selection figures."""
import json
from pathlib import Path


def save_selection(output, *, setting, suite, seed, round_index, suite_round, rollout, task_id,
                   initial_state_id, method, q, risk, scores, indices, weights, phases, percent=None):
    selected = dict(zip(indices, map(float, weights), strict=True))
    states = len(q)
    row = {
        "setting": setting, "suite": suite, "seed": seed, "round": round_index, "suite_round": suite_round,
        "rollout": rollout, "task_id": task_id, "initial_state_id": initial_state_id,
        "method": method, "target_percent": percent, "selected_count": len(selected),
        "timestep": list(range(states)),
        "phase": [i * phases // states for i in range(states)],
        "q": list(map(float, q)), "risk": list(map(float, risk)),
        "score": None if scores is None else list(map(float, scores)),
        "selected": [i in selected for i in range(states)],
        "selection_weight": [selected.get(i) for i in range(states)],
    }
    path = Path(output) / "plot_data" / suite / f"round-{round_index:06d}-rollout-{rollout:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, allow_nan=False) + "\n")
    tmp.replace(path)
