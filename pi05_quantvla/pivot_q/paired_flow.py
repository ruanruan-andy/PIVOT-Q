"""Shared-noise π0.5 teacher/student flow sampling."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


class FrozenPrefixCache:
    """Ephemeral CPU cache for Full Distill's unchanged observation microbatch.

    Keep one instance per exact ordered microbatch, only within one recovery
    rollout. Never put this cache in State, DDP broadcasts, or checkpoints.
    The expert, diffusion noise, denoising schedule and gradient path are not
    cached. The normal sampler (including HoloQ's step context) remains in use.
    """

    def __init__(self):
        self.embeddings = None
        self.kv = None
        self.kv_type = None

    @staticmethod
    def _move(value, device):
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise RuntimeError("Cannot cache a trainable prefix")
            return value.detach().to(device=device)
        if isinstance(value, tuple):
            return tuple(FrozenPrefixCache._move(v, device) for v in value)
        if isinstance(value, list):
            return [FrozenPrefixCache._move(v, device) for v in value]
        return value

    @contextmanager
    def use(self, model, device):
        # Fail closed if a future experiment makes the prefix trainable or
        # stochastic. Current Full Distill adapters live in the action expert.
        prefix = model.paligemma_with_expert.paligemma
        if any(p.requires_grad for p in prefix.parameters()) or any(m.training for m in prefix.modules()):
            raise RuntimeError("Full Distill prefix caching requires frozen eval-mode PaliGemma")
        embed = model.embed_prefix
        backbone = model.paligemma_with_expert
        forward = backbone.forward
        old_embed = model.__dict__.get("embed_prefix")
        old_forward = backbone.__dict__.get("forward")

        def cached_embed(*args, **kwargs):
            if self.embeddings is None:
                result = embed(*args, **kwargs)
                self.embeddings = self._move(result, "cpu")
                return result
            return self._move(self.embeddings, device)

        def cached_forward(*args, **kwargs):
            inputs = kwargs.get("inputs_embeds")
            is_prefix = (inputs is not None and inputs[0] is not None
                         and inputs[1] is None and kwargs.get("past_key_values") is None
                         and kwargs.get("use_cache") is True)
            if not is_prefix:
                return forward(*args, **kwargs)
            if self.kv is None:
                result = forward(*args, **kwargs)
                self.kv_type = type(result[1])
                self.kv = self._move(tuple(result[1]), "cpu")
                return result
            # LeRobot discards the prefix hidden output. Each denoising step
            # clones this KV cache, as in the unmodified sample_actions path.
            return None, self.kv_type(self._move(self.kv, device))

        model.embed_prefix = cached_embed
        backbone.forward = cached_forward
        try:
            yield
        finally:
            if old_embed is None:
                del model.embed_prefix
            else:
                model.embed_prefix = old_embed
            if old_forward is None:
                del backbone.forward
            else:
                backbone.forward = old_forward


def make_noise(policy, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        device=policy.config.device,
        generator=generator,
        dtype=torch.float32,
    )


def model_inputs(policy, batch: dict[str, torch.Tensor]) -> tuple:
    images, image_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    token_key = "observation.language.tokens"
    mask_key = "observation.language.attention_mask"
    return images, image_masks, batch[token_key], batch[mask_key], states, state_masks


def sample(policy, batch: dict[str, torch.Tensor], noise: torch.Tensor, *, with_grad: bool,
           prefix_cache: FrozenPrefixCache | None = None) -> torch.Tensor:
    model = policy.model.get_base_model() if hasattr(policy.model, "get_base_model") else policy.model
    images, image_masks, tokens, masks, states, state_masks = model_inputs(policy, batch)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if noise.is_cuda else nullcontext()
    cache_context = prefix_cache.use(model, noise.device) if prefix_cache is not None else nullcontext()
    with context, autocast, cache_context:
        if with_grad:
            raw_sample = model.sample_actions.__wrapped__
            output = raw_sample(
                model, images, image_masks, tokens, masks,
                states=states, state_masks=state_masks, noise=noise,
                num_steps=policy.config.num_inference_steps,
            )
        else:
            output = model.sample_actions(
                images, image_masks, tokens, masks,
                states=states, state_masks=state_masks, noise=noise,
                num_steps=policy.config.num_inference_steps,
            )
    action_dim = policy.config.output_features["action"].shape[0]
    return output[:, :, :action_dim]
