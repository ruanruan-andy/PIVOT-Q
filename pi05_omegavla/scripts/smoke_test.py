"""CPU integration tests by default; optional actual-policy gradient/reload smoke."""
import argparse
import gc
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="store_true", help="load real checkpoint and calibrated pack")
    p.add_argument("--config", type=Path, default=ROOT / "pi05_omegavla/config/pivot_q.yaml")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not args.model:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("pi05_omegavla.tests.test_integration")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    from types import SimpleNamespace
    from pi05_omegavla.common import PACKAGE, load_config, resolve_paths, atomic_json
    from pi05_omegavla.backend import load_bundle, evaluation_bundle
    from pi05_omegavla.paired_flow import sample, make_noise
    raw = resolve_paths(load_config(args.config))
    if raw["mode"] not in ("full_distill", "pivot_q"):
        p.error("--model requires a recovery mode configuration")
    cfg = SimpleNamespace(omega_config=str(args.config), device=args.device,
              checkpoint=raw["paths"]["checkpoint"], quant_pack_dir=raw["paths"]["quant_pack_dir"],
              lora_rank=raw["training"]["lora_rank"], lora_alpha=raw["training"]["lora_alpha"],
              lora_dropout=raw["training"]["lora_dropout"])
    buffer = torch.load(raw["paths"]["calibration_buffer"], map_location="cpu", weights_only=False)
    observation = buffer["samples"][0]
    teacher = load_bundle(cfg, quantized=False)
    generator = torch.Generator(device=args.device).manual_seed(0)
    noise = make_noise(teacher.policy, generator)
    target = sample(teacher.policy, teacher.preprocessor(observation), noise, with_grad=False).detach()
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    student = load_bundle(cfg, quantized=True)
    batch = student.preprocessor(observation)
    before = sample(student.policy, batch, noise, with_grad=False)
    parameters = [p for p in student.policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    prediction = sample(student.policy, batch, noise, with_grad=True)
    torch.testing.assert_close(before, prediction.detach(), rtol=0, atol=0)
    loss = (prediction[:,0,:7].float() - target[:,0,:7].float()).square().mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"invalid adapter gradient norm: {norm}")
    if any(p.grad is not None for p in student.policy.model.parameters() if not p.requires_grad):
        raise RuntimeError("frozen base received gradients")
    optimizer.step()
    expected = sample(student.policy, batch, noise, with_grad=False).detach().cpu()
    directory = PACKAGE / "outputs/smoke" / raw["mode"] / "adapter"
    student.policy.model.save_pretrained(directory)
    loss_value, gradient_value = float(loss.detach()), float(norm)
    del student, optimizer, parameters, prediction, loss, batch, before, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    recovered = evaluation_bundle(args.config, args.device, directory)
    actual = sample(recovered.policy, recovered.preprocessor(observation), noise, with_grad=False).cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    atomic_json(directory.parent / "result.json", {"passed": True, "loss": loss_value,
                "gradient_norm": gradient_value, "device": args.device, "mode": raw["mode"]})
    print("actual-policy gradient/save/eval-reload smoke passed")
if __name__ == "__main__":
    main()
