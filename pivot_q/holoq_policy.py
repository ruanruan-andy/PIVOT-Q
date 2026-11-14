"""One loader used by training, smoke tests, and evaluation."""
from pathlib import Path
from types import SimpleNamespace
from .configuration import ROOT
from .holoq import official, apply
from .holoq_flow import install_step_context
from .holoq_adapter import attach

def load_bundle(config, *, quantized, adapter=None, recovery=True, trainable=True):
    from .pi05_state import Bundle
    from .pi05_policy import load_policy
    raw = config.holoq
    source = raw["paths"]["omega_source"]
    official(source)
    checkpoint = Path(config.checkpoint)
    policy = load_policy(ROOT, checkpoint, device=config.device)
    policy.requires_grad_(False)
    if policy.config.num_inference_steps != raw["inference"]["num_inference_steps"]:
        raise ValueError("checkpoint denoising steps differ from configured protocol")
    if policy.config.output_features["action"].shape != (7,) and list(policy.config.output_features["action"].shape) != [7]:
        raise ValueError("expected seven action dimensions")
    if getattr(policy.config, "use_visual_memory", False) or getattr(policy.config, "use_proprioceptive_memory", False):
        raise ValueError("this statewise distillation protocol requires memory disabled")
    install_step_context(policy.model)
    if quantized:
        identity = apply(policy.model, source=source, pack=config.quant_pack_dir,
                         checkpoint=checkpoint, steps=policy.config.num_inference_steps,
                         qconfig=raw["quantization"], trainable=trainable)
        if recovery:
            policy.model = attach(policy.model, identity=identity, mode=raw["mode"],
                                  rank=config.lora_rank, alpha=config.lora_alpha,
                                  dropout=config.lora_dropout, path=adapter, trainable=trainable)
    policy.eval()
    from lerobot.policies import make_pre_post_processors
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
                                        preprocessor_overrides={"device_processor": {"device": config.device}})
    return Bundle(policy, pre, post, None)

