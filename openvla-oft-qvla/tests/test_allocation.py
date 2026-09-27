import math
from collections import Counter
import pytest
from integration.allocation import calibration_plan, marginal_allocate


def test_balanced_reproducible_temporal_sampling():
    p = calibration_plan(list(range(10)), list(range(40)), 32, 0, 220)
    assert p == calibration_plan(list(range(10)), list(range(40)), 32, 0, 220)
    counts = Counter(t for t, _, _ in p)
    assert len(counts) == 10 and max(counts.values()) - min(counts.values()) == 1
    assert len({i for _, i, _ in p}) == 4
    assert len({s for _, _, s in p}) > 20
    assert len({(t, i) for t, i, _ in p}) == 32


def test_marginal_not_absolute_cost():
    # A: 16->8 costs 8; then 8->4 costs 1. B: 16->8 costs 10.
    # After A's first downgrade, marginal allocation should downgrade A again.
    proxies = {"a": {16: [0], 8: [8], 4: [9]},
               "b": {16: [0], 8: [10], 4: [100]}}
    gates, stats = marginal_allocate(proxies, [4, 8, 16], 10)
    assert gates == {"a": [4], "b": [16]}
    assert stats["channel_avg_bits"] == 10


def test_zero_constraint_and_infeasible_budget():
    p = {"a": {16: [0]*4, 4: [1]*4, 2: [2]*4, 0: [3]*4}}
    gates, stats = marginal_allocate(p, [0, 2, 4, 16], 2, min_bit=2)
    assert gates["a"] == [2]*4 and stats["zero_channels"] == 0
    with pytest.raises(ValueError, match="infeasible"):
        marginal_allocate(p, [0, 2, 4, 16], 1, min_bit=2)


@pytest.mark.parametrize("value", [math.nan, math.inf, -1])
def test_invalid_proxy_rejected(value):
    with pytest.raises(ValueError, match="proxy"):
        marginal_allocate({"a": {16: [0], 4: [value]}}, [4,16], 4)


def test_collection_records_coverage_and_nonterminal_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import integration.calibrate as c
    path = tmp_path / "calibration.pt"
    cfg = {"paths": {"calibration": str(path)}, "seed": 0, "suite": "libero_spatial",
           "quantization": {"max_samples": 32},
           "training": {"clean_task_ids": list(range(10)), "clean_initial_state_ids": list(range(40))}}
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def call(self, *args, **kwargs):
            self.step += 1
            return {"observation": self.step, "done": self.step >= 3}
    def reset(client, *args):
        client.step = 0
        return {"observation": 0, "language": "task", "done": False}
    monkeypatch.setattr(c, "EnvClient", Client)
    monkeypatch.setattr(c, "reset", reset)
    monkeypatch.setattr(c, "policy_observation", lambda x: x)
    monkeypatch.setattr(c, "libero_action", lambda x: x)
    monkeypatch.setattr(c, "OFTPolicy", lambda *a, **k: SimpleNamespace(identity={},
                        predict=lambda *a: 0, actions=lambda x: [0]))
    c.collect_calibration(cfg)
    data = torch.load(path, weights_only=False)
    assert data["sampling"] == "balanced_rollouts_v1"
    assert len({r["task_id"] for r in data["records"]}) == 10
    assert len({r["initial_state_id"] for r in data["records"]}) == 4
    assert all(r["observation"] == r["timestep"] < 3 for r in data["records"])
    with pytest.raises(FileExistsError): c.collect_calibration(cfg)


def test_obsolete_calibration_fails_before_model_loading(tmp_path, monkeypatch):
    import torch
    import integration.calibrate as c
    path = tmp_path / "old.pt"
    torch.save({"records": []}, path)
    def unexpected(*args, **kwargs): raise AssertionError("should fail before loading model")
    monkeypatch.setattr(c, "OFTPolicy", unexpected)
    with pytest.raises(ValueError, match="obsolete"):
        c.quantize({"seed": 0, "paths": {"calibration": str(path)}})
