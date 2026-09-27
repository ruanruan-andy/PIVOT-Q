"""CPU tests of real Omega layers, adapters, step dispatch, and isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from pi05_omegavla.common import PACKAGE, WORKSPACE, load_config, inside, checkpoint_identity, atomic_json, sha256
from pi05_omegavla.quantization import official, targets, validate_record, apply, enable_ste, EXPERT
from pi05_omegavla.adapters import attach
from pi05_omegavla.paired_flow import install_step_context
from pi05_omegavla.training.selection import choose, future_risk
from pi05_omegavla.scripts.calibrate import build_record

def tree(path, module, root):
    for component in path[:-1]:
        if not hasattr(root, component):
            root.add_module(component, nn.Module())
        root = getattr(root, component)
    root.add_module(path[-1], module)

def model_tree():
    root = nn.Module()
    for side in ("paligemma.model.language_model", "gemma_expert.model"):
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            tree(f"paligemma_with_expert.{side}.layers.0.{name}".split("."), nn.Linear(4, 4), root)
    return root

class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.runtime = official(WORKSPACE / "Omega-QVLA")
        cls.q = dict(load_config(PACKAGE / "config/base.yaml")["quantization"], svd_rank=2, gptq_block_size=4)
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PACKAGE / "outputs")
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.checkpoint / name).write_text("{}")
        torch.manual_seed(5)
    def tearDown(self):
        self.temp.cleanup()
    def make_pack(self, model):
        from gr00t.quantization.duquant_preprocess import sanitize_name
        pack = self.root / "pack"
        pack.mkdir()
        layers = {}
        for name, layer in targets(model).items():
            n = 3 if name.startswith(EXPERT) else 1
            record = build_record(layer.weight, [torch.randn(24,4) for _ in range(n)], self.q,
                                  self.runtime.gptq_quantize_weight)
            path = pack / (sanitize_name(name) + ".pt")
            torch.save(record, path)
            layers[name] = {"file": path.name, "sha256": sha256(path)}
        atomic_json(pack / "manifest.json", {"format": "pi05_omegavla_lerobot_v1", "complete": True,
             "checkpoint": checkpoint_identity(self.checkpoint), "num_steps": 3,
             "quantization": self.q, "layers": layers})
        return pack
    def test_full_pack_and_adapter_roundtrip(self):
        from gr00t.quantization.dit_step_context import set_dit_quant_step
        model = model_tree()
        original = copy.deepcopy(model)
        pack = self.make_pack(model)
        identity = apply(model, source=WORKSPACE / "Omega-QVLA", pack=pack, checkpoint=self.checkpoint,
                         steps=3, qconfig=self.q, trainable=True)
        self.assertEqual(identity["quantized_layers"], 14)
        name = EXPERT + "0.self_attn.q_proj"
        x = torch.randn(7,4)
        with set_dit_quant_step(1):
            baseline = model.get_submodule(name)(x).detach()
        adapted = attach(model, identity=identity, mode="pivot_q", rank=2, alpha=4, dropout=0)
        with set_dit_quant_step(1):
            self.assertTrue(torch.equal(baseline, model.get_submodule(name)(x)))
            # Include a downstream quantizer: recovery must receive gradients
            # through later activation rounding, not only its immediate output.
            output = model.get_submodule(EXPERT + "0.self_attn.o_proj")(model.get_submodule(name)(x))
            output.square().mean().backward()
        self.assertGreater(model.get_submodule(name).lora_b.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.get_submodule(name).base.parameters()))
        with torch.no_grad():
            model.get_submodule(name).lora_b.weight.add_(-0.01 * model.get_submodule(name).lora_b.weight.grad)
        directory = self.root / "adapter"
        adapted.save_pretrained(directory)
        state = torch.load(directory / "adapter_model.pt", weights_only=True)
        self.assertEqual(len(state), 6)
        self.assertTrue(all("lora_" in k for k in state))
        fresh_identity = apply(original, source=WORKSPACE / "Omega-QVLA", pack=pack,
                               checkpoint=self.checkpoint, steps=3, qconfig=self.q)
        recovered = attach(original, identity=fresh_identity, mode="pivot_q", path=directory, trainable=False)
        with set_dit_quant_step(1):
            self.assertTrue(torch.equal(model.get_submodule(name)(x), original.get_submodule(name)(x)))
        self.assertFalse(any(p.requires_grad for p in recovered.parameters()))
        with self.assertRaises(ValueError):
            attach(model_tree(), identity=dict(identity, pack_sha256="bad"), mode="pivot_q", path=directory)
    def test_quantizer_exact_forward_and_nonzero_backward(self):
        runtime = self.runtime
        x = torch.tensor([-0.35, 0.17, 0.94], requires_grad=True)
        with torch.no_grad():
            expected = runtime.fake_quantize_sym(x, torch.tensor(0.1), 4)
        enable_ste(runtime)
        actual = runtime.fake_quantize_sym(x, torch.tensor(0.1), 4)
        self.assertTrue(torch.equal(expected, actual))
        actual.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))
    def test_pack_rejects_changed_checkpoint(self):
        model = model_tree()
        pack = self.make_pack(model)
        (self.checkpoint / "model.safetensors").write_text("changed")
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            apply(model, source=WORKSPACE / "Omega-QVLA", pack=pack, checkpoint=self.checkpoint,
                  steps=3, qconfig=self.q)
    def test_record_rejects_missing_step_or_wrong_bits(self):
        linear = nn.Linear(4,4)
        record = build_record(linear.weight, [torch.randn(24,4)], self.q, self.runtime.gptq_quantize_weight)
        with self.assertRaises(ValueError):
            validate_record(record, linear, EXPERT + "0.self_attn.q_proj", 3, self.q)
        record["a_bits"] = 8
        with self.assertRaises(ValueError):
            validate_record(record, linear, "paligemma", 1, self.q)
    def test_selection_protocol(self):
        error = torch.arange(32).float()
        risk = future_risk(error, 4, 0.9)
        kwargs = dict(phase_bins=4, top_per_phase=4, min_gap=4,
                      alpha=0.5, beta=0.5, weight_min=0.5, weight_max=2.0)
        dense = choose(error, risk, method="full_distill", **kwargs)
        pivot = choose(error, risk, method="pivot_q", **kwargs)
        self.assertEqual(len(dense.indices), 32)
        self.assertTrue(torch.equal(dense.weights, torch.ones(32)))
        self.assertEqual(len(pivot.indices), 16)
        self.assertEqual(len(set(pivot.indices)), 16)
    def test_step_context_and_grad_path(self):
        from gr00t.quantization.dit_step_context import get_current_dit_step
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_inference_steps=3)
                self.seen = []
            def denoise_step(self, x):
                self.seen.append(get_current_dit_step())
                return x * 2
            @torch.no_grad()
            def sample_actions(self, x, num_steps=None):
                self.seen.append(get_current_dit_step())
                for _ in range(num_steps or 3):
                    x = self.denoise_step(x)
                return x
        model = Toy()
        install_step_context(model)
        x = torch.ones(1, requires_grad=True)
        result = model.sample_actions.__wrapped__(model, x, num_steps=3)
        result.sum().backward()
        self.assertEqual(model.seen, [0, 0, 1, 2])
        self.assertEqual(x.grad.item(), 8)
        self.assertIsNone(get_current_dit_step())
        model.seen.clear()
        self.assertFalse(model.sample_actions(x).requires_grad)
        self.assertEqual(model.seen, [0, 0, 1, 2])

    def test_eval_queue_resets_between_episodes_with_same_language(self):
        import numpy as np
        from collections import deque
        from pi05_omegavla.evaluation.eval_client import RemotePI05Policy
        class Connection:
            def __init__(self):
                self.requests = []
            def send(self, request):
                self.requests.append(request)
            def recv(self):
                return {"actions": np.ones((10,7), dtype=np.float32)}
            def close(self):
                pass
        policy = RemotePI05Policy.__new__(RemotePI05Policy)
        policy.connection = Connection()
        policy.queue = deque()
        policy.records = None
        policy.last_seed = None
        policy.last_language = None
        observation = {"agentview_image": np.zeros((2,2,3), dtype=np.uint8),
                       "robot0_eye_in_hand_image": np.zeros((2,2,3), dtype=np.uint8),
                       "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0,0,0,1]),
                       "robot0_gripper_qpos": np.zeros(2)}
        policy.get_action(observation, "same task", 0)
        policy.get_action(observation, "same task", 1)
        self.assertEqual(len(policy.connection.requests), 1)
        policy.get_action(observation, "same task", 10000)
        self.assertEqual(len(policy.connection.requests), 2)
        self.assertEqual(policy.connection.requests[-1]["seed"], 10000)

    def test_output_boundary_and_config(self):
        with self.assertRaises(ValueError):
            inside(WORKSPACE / "Omega-QVLA/test")
        cfg = load_config(PACKAGE / "config/full_distill.yaml")
        self.assertEqual(cfg["mode"], "full_distill")
        self.assertEqual(cfg["experiment"]["anchor_lambda"], 0.0)
        self.assertEqual(cfg["training"]["episodes_per_suite"], 140)

if __name__ == "__main__":
    unittest.main()
