import ast
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from integration.adapters import attach, adapter_state, load_adapter
from integration.config import ROOT, load_config, validate
from integration.observations import libero_action, policy_observation
from integration.quantization import canonical_gates, inject
from integration.upstream import activate
from training.selection import select, future_risk
from training.train import update


def config(method="pivot_q"):
    return load_config(ROOT / "configs" / f"{method}.yaml")


def test_config_paths_and_typo_rejection():
    cfg = load_config(ROOT / "configs/pivot_q.yaml", ["training.episodes=2", "paths.output=/tmp/test"])
    assert cfg["training"]["episodes"] == 2
    validate(cfg)
    with pytest.raises(ValueError, match="unknown config"):
        load_config(ROOT / "configs/pivot_q.yaml", ["training.epizodes=2"])


def test_selector_matches_existing_pivot_q():
    # Execute only the model-independent original functions, avoiding GR00T imports.
    original = ROOT.parent / "pivot_q/selector.py"
    if not original.exists():
        pytest.skip("original PIVOT_Q source is not present in this checkout")
    tree = ast.parse(original.read_text())
    names = {"future_risk", "percentile_ranks", "select_phase", "_legacy_select_states"}
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    scope = {"torch": torch, "random": random, "SimpleNamespace": SimpleNamespace, "TrainConfig": object}
    exec(compile(functions, str(original), "exec"), scope)
    settings = config()["selection"]
    legacy = SimpleNamespace(random_per_phase=0, phase_bins=4, priority_per_phase=4,
                             alpha_q=.5, beta_r=.5, min_temporal_gap=4, weight_min=.5, weight_max=2)
    for length in (16, 21, 80, 220):
        for q in (torch.rand(length), torch.zeros(length)):
            risk = future_risk(q, 4, .9)
            expected = scope["_legacy_select_states"](q, risk, legacy, random.Random(0))
            actual = select(q, "pivot_q", settings)
            assert actual["indices"] == list(expected.indices)
            torch.testing.assert_close(actual["weights"], expected.weights / expected.weights.sum())


def test_full_distill_and_short_episodes():
    for size in (1, 3, 16, 23):
        q = torch.arange(size).float()
        full = select(q, "full_distill", config()["selection"])
        assert full["indices"] == list(range(size))
        torch.testing.assert_close(full["weights"], torch.full((size,), 1 / size))
        sparse = select(q, "pivot_q", config()["selection"])
        assert len(sparse["indices"]) == len(set(sparse["indices"]))
        assert bool(sparse["short_rollout"]) == (size < 16)
        assert torch.isclose(sparse["weights"].sum(), torch.tensor(1.))


def test_official_quantization_and_strict_gate_validation(tmp_path):
    activate(config())
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(4, 4, bias=False))
    model.projector = nn.Linear(4, 4)
    with torch.no_grad():
        model.language_model[0].weight.copy_(torch.tensor([[.1, .3, .6, .9]] * 4))
    original = model.language_model[0].weight.detach().clone()
    projector = model.projector.weight.detach().clone()
    gates = {"language_model.0": [0, 2, 4, 16]}
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"assign": gates, "stats": {}}))
    report = inject(model, path)
    assert report["modules"] == 1
    assert report["channel_average_bits"] == 5.5
    assert torch.count_nonzero(model.language_model[0].weight[0]) == 0
    torch.testing.assert_close(model.language_model[0].weight[3], original[3])
    torch.testing.assert_close(model.projector.weight, projector)
    assert not torch.equal(model.language_model[0].weight[1], original[1])
    with pytest.raises(ValueError, match="channel count"):
        canonical_gates(model, {"language_model.0": [4]})
    with pytest.raises(ValueError, match="missing gates"):
        canonical_gates(model, {"unknown": [4]})
    with pytest.raises(ValueError, match="invalid bit"):
        canonical_gates(model, {"language_model.0": [1, 2, 4, 16]})


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 7)

    def predict_action(self, x):
        return self.linear(x)


class ToyPolicy:
    def __init__(self, cfg, quantized=False, trainable=False, teacher=False):
        self.cfg = cfg
        self.head = Head().requires_grad_(False)
        with torch.no_grad():
            self.head.linear.weight.fill_(.08 if quantized else .1)
            self.head.linear.bias.fill_(0)
        self.identity = {"toy": True, "quantized": quantized}
        self.adapter_targets = attach(self.head, cfg["training"]) if trainable else []
        self.head.eval()

    def features(self, observation, language):
        return torch.ones(1, 8, 8) * float(observation["state"][0] + 1)

    def predict_features(self, features):
        return self.head.predict_action(features)

    def predict(self, observation, language):
        return self.predict_features(self.features(observation, language))

    def actions(self, predicted):
        return predicted.detach().numpy()[0]

    def parameters(self):
        return [p for p in self.head.parameters() if p.requires_grad]


def test_adapter_update_freezes_base_and_roundtrip():
    cfg = config()
    policy = ToyPolicy(cfg, quantized=True, trainable=True)
    features = torch.ones(1, 8, 8)
    base = policy.head.linear.base.weight.detach().clone()
    expected_initial = policy.head.linear.base(features)
    torch.testing.assert_close(policy.predict_features(features), expected_initial)
    state = {"features": features, "target": torch.ones(1, 8, 7), "action_count": 1}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.01)
    before = (policy.predict_features(features)[:, :1] - 1).square().mean().item()
    for _ in range(3):
        update(policy, optimizer, [state], [1.], [], cfg)
    assert (policy.predict_features(features)[:, :1] - 1).square().mean().item() < before
    torch.testing.assert_close(base, policy.head.linear.base.weight)
    assert policy.head.linear.base.weight.grad is None
    restored = ToyPolicy(cfg, quantized=True, trainable=True)
    load_adapter(restored.head, adapter_state(policy.head))
    torch.testing.assert_close(restored.predict_features(features), policy.predict_features(features), rtol=0, atol=0)


class ToyEnv:
    def __init__(self, cfg, clean=False):
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def call(self, endpoint, **data):
        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint == "list_tasks":
            return [{"task_id": 0, "task_name": "toy", "category": "test"}]
        if endpoint == "reset":
            self.count = 0
        elif endpoint == "step":
            self.count += 1
        obs = {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
               "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
               "robot0_eef_pos": np.array([self.count / 100, 0, 0]),
               "robot0_eef_quat": np.array([0, 0, 0, 1]), "robot0_gripper_qpos": np.zeros(2)}
        return {"observation": obs, "done": self.count >= 16, "language": "toy task"}


@pytest.mark.parametrize("method", ["pivot_q", "full_distill"])
def test_train_resume_and_four_evaluation_paths(tmp_path, monkeypatch, method):
    import training.train as trainer
    import evaluation.evaluate as evaluator

    for module in (trainer, evaluator):
        monkeypatch.setattr(module, "OFTPolicy", ToyPolicy)
        monkeypatch.setattr(module, "EnvClient", ToyEnv)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"task_ids_by_suite": {"libero_spatial": [0]}, "initial_state_ids": [0]}))
    cfg = config(method)
    cfg["paths"].update(manifest=str(manifest), output=str(tmp_path / "uninterrupted"))
    cfg["training"].update(episodes=2, updates_per_episode=1, save_every_episodes=1, rank=2,
                             clean_task_ids=[0], clean_initial_state_ids=[0])
    trainer.train(cfg)
    expected = torch.load(tmp_path / "uninterrupted/checkpoint-000002.pt", weights_only=False)

    interrupted = copy.deepcopy(cfg)
    interrupted["paths"]["output"] = str(tmp_path / "interrupted")
    real_collect = trainer.collect
    calls = 0

    def crash(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # target+anchor episode 1 completed; crash before episode 2
            raise RuntimeError("simulated interruption")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(trainer, "collect", crash)
    with pytest.raises(RuntimeError, match="simulated"):
        trainer.train(interrupted)
    monkeypatch.setattr(trainer, "collect", real_collect)
    interrupted["training"]["resume"] = str(tmp_path / "interrupted/checkpoint-000001.pt")
    trainer.train(interrupted)
    resumed = torch.load(tmp_path / "interrupted/checkpoint-000002.pt", weights_only=False)
    for key in expected["adapter"]:
        torch.testing.assert_close(resumed["adapter"][key], expected["adapter"][key], rtol=0, atol=0)
    assert len((tmp_path / "interrupted/metrics.jsonl").read_text().splitlines()) == 2
    assert len(resumed["replay"]) == 8
    with np.load(tmp_path / "interrupted/states/episode-000002.npz", allow_pickle=False) as data:
        assert json.loads(str(data["metadata_json"].item()))["phase"] == "train_pre_update"
        assert data["q_t"].shape == (16,)
        assert data["selected"].sum() == 16
        assert np.isclose(data["selection_weight"].sum(), 1)

    for evaluation_method in ("fp", "qvla", method):
        evaluation_cfg = copy.deepcopy(cfg)
        evaluation_cfg["method"] = evaluation_method
        evaluation_cfg["paths"]["output"] = str(tmp_path / f"eval-{evaluation_method}")
        if evaluation_method == method:
            evaluation_cfg["paths"]["adapter"] = str(tmp_path / "uninterrupted/checkpoint-000002.pt")
        summary = evaluator.evaluate(evaluation_cfg)
        assert summary["episodes"] == 1 and summary["success_rate"] == 1
        with np.load(tmp_path / f"eval-{evaluation_method}/eval/states/episode-000001.npz", allow_pickle=False) as data:
            expected_q = ((data["student_action_chunk"][:, 0] - data["teacher_at_student_chunk"][:, 0]) ** 2).mean(-1)
            np.testing.assert_allclose(data["q_t"], expected_q, rtol=1e-6, atol=1e-8)
            assert data["timestep"].tolist() == list(range(16))
            if evaluation_method == "fp":
                assert (data["q_t"] == 0).all()


def test_observation_and_gripper():
    raw = ToyEnv({}).call("reset")["observation"]
    raw["agentview_image"] = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    obs = policy_observation(raw)
    np.testing.assert_array_equal(obs["full_image"], raw["agentview_image"][::-1, ::-1])
    assert obs["state"].shape == (8,)
    assert libero_action([0] * 6 + [1])[-1] == -1
    assert libero_action([0] * 6 + [0])[-1] == 1


def test_official_proxy_allocator_pipeline_and_resume(tmp_path, monkeypatch):
    import integration.calibrate as calibration
    activate(config())

    class CalibPolicy(ToyPolicy):
        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            self.device = torch.device("cpu")
            self.model = nn.Module()
            self.model.language_model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.fill_(.13)

        def features(self, obs, language):
            return self.model.language_model(super().features(obs, language))

    monkeypatch.setattr(calibration, "OFTPolicy", CalibPolicy)
    monkeypatch.setattr(calibration, "EnvClient", ToyEnv)
    cfg = config("qvla")
    cfg["paths"].update(calibration=str(tmp_path / "calibration.pt"), proxy=str(tmp_path / "proxy.pt"), gates=str(tmp_path / "gates.json"))
    cfg["quantization"]["max_samples"] = 3
    cfg["training"].update(clean_task_ids=[0], clean_initial_state_ids=[0])
    calibration.collect_calibration(cfg)
    calibration.quantize(cfg)
    first = (tmp_path / "gates.json").read_text()
    calibration.quantize(cfg)  # resume must preserve the same allocation
    assert (tmp_path / "gates.json").read_text() == first
    gates = json.loads(first)
    assert len(gates) == 2
    assert sum(sum(v) for v in gates.values()) / 16 <= 4
    cfg["quantization"]["percdamp"] = .02
    with pytest.raises(ValueError, match="resume configuration"):
        calibration.quantize(cfg)
