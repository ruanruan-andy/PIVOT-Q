"""OFT L1 policy: official feature extraction, trainable external head adapters."""
from __future__ import annotations

from integration.relocation import normalize_paths

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from integration import adapters
from integration.config import resolve, file_info
from integration.quantization import inject
from integration.upstream import activate, check_transformers


def component_file(checkpoint, pattern):
    files = sorted(Path(checkpoint).glob(f"*{pattern}*.pt"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one {pattern} .pt in {checkpoint}; found {files}")
    return files[0]


def component_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


class OFTPolicy:
    def __init__(self, cfg, *, quantized=False, trainable=False, teacher=False):
        self.cfg = cfg
        self.upstream = activate(cfg)
        check_transformers()
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticProcessor, PrismaticImageProcessor
        from transformers import AutoTokenizer
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

        if (ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM) != (7, 8, 8):
            raise RuntimeError("official constants are not configured for LIBERO")
        m = cfg["model"]
        self.device = torch.device(m["teacher_device"] if teacher else m["device"])
        self.dtype = getattr(torch, m["dtype"])
        self.checkpoint = resolve(cfg["paths"]["checkpoint"])
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"download a complete merged OFT checkpoint first: {self.checkpoint}")
        config = OpenVLAConfig.from_pretrained(str(self.checkpoint), local_files_only=True)
        # Direct classes avoid get_vla(), which rewrites local checkpoint code/config.
        self.model = OpenVLAForActionPrediction.from_pretrained(
            str(self.checkpoint), config=config, local_files_only=True,
            torch_dtype=self.dtype, attn_implementation=m["attention"], low_cpu_mem_usage=True,
        ).to(self.device).eval().requires_grad_(False)
        self.model.config.use_cache = False
        self.model.vision_backbone.set_num_images_in_input(m["num_images"])
        self.processor = PrismaticProcessor(
            PrismaticImageProcessor.from_pretrained(str(self.checkpoint), local_files_only=True),
            AutoTokenizer.from_pretrained(str(self.checkpoint), local_files_only=True, trust_remote_code=False),
        )
        stats_path = self.checkpoint / "dataset_statistics.json"
        self.model.norm_stats = json.loads(stats_path.read_text())
        key = m["unnorm_key"]
        if key not in self.model.norm_stats and key + "_no_noops" in self.model.norm_stats:
            key += "_no_noops"
        if key not in self.model.norm_stats:
            raise ValueError(f"unknown normalization key {key}; available: {list(self.model.norm_stats)}")
        self.unnorm_key = key
        dim = self.model.llm_dim
        self.head = L1RegressionActionHead(input_dim=dim, hidden_dim=dim, action_dim=7)
        head_file = component_file(self.checkpoint, "action_head")
        self.head.load_state_dict(component_state(head_file), strict=True)
        self.head = self.head.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.proprio = ProprioProjector(llm_dim=dim, proprio_dim=8)
        proprio_file = component_file(self.checkpoint, "proprio_projector")
        self.proprio.load_state_dict(component_state(proprio_file), strict=True)
        self.proprio = self.proprio.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        weight_files = sorted(self.checkpoint.glob("*.safetensors")) or sorted(self.checkpoint.glob("pytorch_model*.bin"))
        if not weight_files:
            raise FileNotFoundError("no merged model weights found")
        self.identity = {
            "checkpoint": str(self.checkpoint), "upstream_commit": self.upstream["commit"],
            "config_file": file_info(self.checkpoint / "config.json"), "stats_file": file_info(stats_path),
            "head_file": file_info(head_file), "proprio_file": file_info(proprio_file),
            "weights": {p.name: file_info(p) for p in weight_files},
            "unnorm_key": key, "dtype": m["dtype"], "quantization": None,
        }
        if quantized:
            gates_path = resolve(cfg["paths"]["gates"])
            metadata = gates_path.with_suffix(".metadata.json")
            if not metadata.exists():
                raise ValueError("gate provenance missing; regenerate with prepare-qvla")
            data = json.loads(metadata.read_text())
            if data.get("allocation_stats", {}).get("allocator") != "marginal_v1":
                raise ValueError("obsolete QVLA gates; recollect calibration and regenerate gates")
            if normalize_paths(data["identity"]) != normalize_paths(self.identity) or data["gates_file"] != file_info(gates_path):
                raise ValueError("gates were calibrated for a different checkpoint or have changed")
            self.quantization = inject(self.model, gates_path)
            self.identity["quantization"] = self.quantization
        else:
            self.quantization = None
        self.adapter_targets = adapters.attach(self.head, cfg["training"]) if trainable else []
        self.head.eval()

    def prepare(self, observation, language):
        from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla

        images = prepare_images_for_vla(
            [observation["full_image"], observation["wrist_image"]],
            SimpleNamespace(center_crop=self.cfg["model"]["center_crop"]),
        )
        prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
        batches = [self.processor(prompt, image).to(self.device, dtype=self.dtype) for image in images]
        inputs = batches[0]
        inputs["pixel_values"] = torch.cat([batch["pixel_values"] for batch in batches], dim=1)
        proprio = normalize_proprio(np.array(observation["state"], copy=True), self.model.norm_stats[self.unnorm_key]["proprio"])
        return inputs, proprio

    @torch.no_grad()
    def features(self, observation, language):
        inputs, proprio = self.prepare(observation, language)
        # predict_action detaches its NumPy output but returns the original
        # hidden-state tensor. The frozen backbone needs no gradient.
        _, hidden = self.model.predict_action(
            **inputs, unnorm_key=self.unnorm_key, proprio=proprio,
            proprio_projector=self.proprio, action_head=self.head, use_film=False,
        )
        return hidden.detach()

    def predict_features(self, hidden):
        return self.head.predict_action(hidden.to(device=self.device, dtype=self.dtype)).float()

    @torch.no_grad()
    def predict(self, observation, language):
        return self.predict_features(self.features(observation, language))

    def actions(self, normalized):
        return self.model._unnormalize_actions(normalized.detach().float().cpu().numpy().reshape(8, 7), self.unnorm_key)

    def parameters(self):
        return [p for p in self.head.parameters() if p.requires_grad]

    def verify_official_parity(self, observation, language):
        """Compare our differentiable head route with the official policy output."""
        self.head.eval()
        with torch.no_grad():
            inputs, proprio = self.prepare(observation, language)
            expected, hidden = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, proprio=proprio,
                proprio_projector=self.proprio, action_head=self.head, use_film=False,
            )
            actual = self.actions(self.predict_features(hidden))
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
            raise AssertionError("integration does not match official OFT prediction")
        return float(np.max(np.abs(expected - actual)))
