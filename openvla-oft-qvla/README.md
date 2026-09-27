# OpenVLA-OFT with QVLA

[Installation](../env/README.md) · [Detailed commands](commands.md) · [Unified launcher](../docs/commands-qvla-openvla-oft.md)

This integration supports Original, Quantized, Full Distill, and PIVOT-Q.
Official source is pinned in `third_party/QVLA`; integration changes remain
in this directory. Set the project root using the shared installation guide.
Model inference uses `oft_qvla`; simulator services use `libero_test`.
Prepare checkpoints, calibration, proxy, and gates before recovery.

Full Distill uses all valid rollout decision states with uniform weights.
PIVOT-Q selects phase-balanced vulnerable states. Both adapt residual LoRA
in the action head, with student-controlled rollouts and a behavioral anchor.
Full Distill is not full-parameter training or equal computational cost.

The deterministic L1 action head uses no diffusion noise. The integration
executes and distills the first normalized 7D action of each chunk.
Frozen backbone features are cached for action-head adaptation.

The original policy uses BF16 by default. QVLA uses mixed-bit weight-only
fake quantization with a target average bit budget, not uniform W4A4.
Projector, action head, and language-model head remain unquantized here.
No packed integer-kernel speed or memory benefit is implied.

Train with seed 0 and evaluate final adapters separately using seeds 2026–2029.
Ordinary success-rate evaluation is the default. Model files and results are
excluded from version control.
