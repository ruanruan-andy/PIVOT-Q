from pathlib import Path
import json
from peft import PeftModel
from .configuration import ROOT
from .pi05_state import Bundle
from .pi05_policy import load_policy
from .pi05_quantvla import PI05QuantVLAConfig, apply_quantvla_layout
from .atm_ohb import AttentionStatistics
from .pi05_lora import attach

def load_bundle(config, *, quantized, adapter=None, recovery=True, trainable=True):
    checkpoint = Path(config.checkpoint).expanduser().resolve()
    policy = load_policy(ROOT, checkpoint, device=config.device)
    scales = None
    if quantized:
        pack = Path(config.quant_pack_dir).expanduser().resolve()
        quant_config = config.quantization
        apply_quantvla_layout(policy.model, PI05QuantVLAConfig(**quant_config), pack_dir=pack)
        scale_path = pack / "atm_ohb.json"
        scale_values = json.loads(scale_path.read_text(encoding="utf-8"))["scales"]
        scales = AttentionStatistics(policy.model, scales=scale_values, collect=False)
        if recovery and adapter is None:
            policy.model, _ = attach(policy.model, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
        elif recovery:
            policy.model = PeftModel.from_pretrained(policy.model, adapter, is_trainable=trainable)
        policy.model.eval()
        if recovery:
            policy.model.print_trainable_parameters()
    else:
        policy.model.requires_grad_(False)
        policy.model.eval()

    from lerobot.policies import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return Bundle(policy, preprocessor, postprocessor, scales)
