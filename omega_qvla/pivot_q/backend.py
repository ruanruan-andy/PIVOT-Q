"""Ω-QVLA model hooks consumed by the shared GR00T PIVOT_Q trainer."""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType

import torch
from transformers.feature_extraction_utils import BatchFeature

from .gptq_lora import OmegaAdapterModel, OmegaPeftCompatibility, inject_adapters

QUANT_PREFIXES = ("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_ATM", "GR00T_OHB", "GR00T_RTN")


def clear_quantization_environment() -> None:
    for name in tuple(os.environ):
        if name.startswith(QUANT_PREFIXES):
            os.environ.pop(name, None)


def configure_omega(config) -> None:
    pack = Path(config.quant_pack_dir or "").expanduser().resolve()
    if not pack.is_file():
        raise FileNotFoundError(f"Ω-QVLA W4A4 pack is missing: {pack}")
    os.environ.update({
        "GR00T_GPTQ": "1",
        "GR00T_GPTQ_PATH": str(pack),
        "GR00T_GPTQ_INCLUDE": (
            r".*(backbone\.eagle_model\.language_model\..*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|"
            r"action_head\.model\.transformer_blocks\.\d+\."
            r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*"
        ),
        "GR00T_GPTQ_EXCLUDE": (
            r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|"
            r"state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|"
            r"vlln|future_tokens)(?:\.|$)"
        ),
        "GR00T_GPTQ_WBITS_DEFAULT": "4",
        "GR00T_GPTQ_ABITS": "4",
        "GR00T_GPTQ_MISSING": "fallback",
        "GR00T_GPTQ_KEEP_FP": "0",
    })
    # Ω-QVLA is inference-oriented: its activation fake quantizer uses a hard
    # round whose derivative is zero. Preserve the exact W4A4 forward value but
    # use the standard straight-through estimator during PIVOT_Q backpropagation.
    from gr00t.quantization import gptq_layers
    if not getattr(gptq_layers, "_pivot_q_ste_enabled", False):
        original_fake_quantize = gptq_layers.fake_quantize_sym

        def straight_through_fake_quantize(inputs, *args, **kwargs):
            quantized = original_fake_quantize(inputs, *args, **kwargs)
            if inputs.requires_grad:
                return inputs + (quantized - inputs).detach()
            return quantized

        gptq_layers.fake_quantize_sym = straight_through_fake_quantize
        gptq_layers._pivot_q_ste_enabled = True


def sample_actions(self, backbone_output, action_input, initial_noise=None):
    """Ω-compatible differentiable sampler with explicit shared initial noise."""
    backbone_output = self.process_backbone_output(backbone_output)
    vision_language = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    state_features = self.state_encoder(action_input.state, embodiment_id)
    batch_size = vision_language.shape[0]
    expected = (batch_size, self.config.action_horizon, self.config.action_dim)
    if initial_noise is None:
        actions = torch.randn(expected, dtype=vision_language.dtype, device=vision_language.device)
    else:
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"initial_noise must have shape {expected}, got {tuple(initial_noise.shape)}")
        actions = initial_noise.to(device=vision_language.device, dtype=vision_language.dtype)
    steps = self.num_inference_timesteps
    for step in range(steps):
        discrete_time = int((step / float(steps)) * self.num_timestep_buckets)
        timesteps = torch.full((batch_size,), discrete_time, device=vision_language.device)
        action_features = self.action_encoder(actions, timesteps, embodiment_id)
        if self.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], device=vision_language.device)
            action_features = action_features + self.position_embedding(positions).unsqueeze(0)
        future = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat((state_features, future, action_features), dim=1)
        from gr00t.quantization.dit_step_context import set_dit_quant_step
        with set_dit_quant_step(step, total=steps):
            model_output = self.model(
                hidden_states=hidden,
                encoder_hidden_states=vision_language,
                timestep=timesteps,
            )
        velocity = self.action_decoder(model_output, embodiment_id)[:, -self.action_horizon:]
        actions = actions + velocity / steps
    return BatchFeature(data={"action_pred": actions})


def install_hooks(train_module) -> None:
    """Replace model-specific operations while retaining the shared algorithm."""

    def load_policy(config, *, quantized: bool, repo_root: Path):
        clear_quantization_environment()
        if quantized:
            configure_omega(config)
        data_config = train_module._data_config_for_suite(config.task_suite_name)
        policy = train_module.Gr00tPolicy(
            model_path=config.model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=config.denoising_steps,
            device=config.device,
        )
        policy.model.action_head.sample_actions = MethodType(
            sample_actions, policy.model.action_head
        )
        policy.model.eval()
        return policy

    def attach(policy, config):
        targets = inject_adapters(
            policy.model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        policy.model = OmegaAdapterModel(policy.model)
        policy.model.eval()
        policy.model.print_trainable_parameters()
        return targets

    train_module._load_policy = load_policy
    train_module._attach_action_head_lora = attach
    train_module.PeftModel = OmegaPeftCompatibility
