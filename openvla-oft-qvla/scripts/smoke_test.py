"""GPU/real-checkpoint test: official parity, finite adapter gradients and save/reload."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import torch

from integration.config import parse_config, resolve
from integration.policy import OFTPolicy
from integration.adapters import adapter_state, load_adapter
from training.checkpoint import seed_all


def main():
    cfg, dry = parse_config(__doc__)
    if dry:
        print("Will verify FP/QVLA official parity, adapter gradients and round-trip")
        return
    seed_all(cfg["seed"])
    data = torch.load(resolve(cfg["paths"]["calibration"]), weights_only=False, map_location="cpu")
    sample = data["records"][0]
    teacher = OFTPolicy(cfg, teacher=True)
    student = OFTPolicy(cfg, quantized=True, trainable=True)
    for policy in (teacher, student):
        print("official max action error:", policy.verify_official_parity(sample["observation"], sample["language"]))
    target = teacher.predict(sample["observation"], sample["language"])
    hidden = student.features(sample["observation"], sample["language"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["learning_rate"])
    prediction = student.predict_features(hidden)
    loss = (prediction[:, :1] - target.to(prediction.device)[:, :1]).square().mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1, error_if_nonfinite=True)
    if not float(norm) > 0:
        raise AssertionError("zero adapter gradient on the real calibration sample")
    assert all(p.grad is None for p in student.model.parameters())
    optimizer.step()
    student.head.eval()
    expected = student.predict_features(hidden).detach()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "adapter.pt"
        torch.save(adapter_state(student.head), path)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.zero_()
        load_adapter(student.head, torch.load(path, weights_only=True))
        torch.testing.assert_close(expected, student.predict_features(hidden), rtol=0, atol=0)
    print(f"PASS: real-model parity, gradients ({float(norm):.6g}) and adapter reload")


if __name__ == "__main__":
    main()
