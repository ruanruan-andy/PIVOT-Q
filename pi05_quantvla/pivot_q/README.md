# π0.5 QuantVLA + PIVOT-Q

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

PIVOT-Q is the method name. `PIVOT-Q/`, `pivot_q`, `pivot_q`, and existing
checkpoint/output paths are historical implementation identifiers and stay
unchanged. Do not rename existing result directories or adapter metadata.


This module applies the same PIVOT-Q principle to LeRobot π0.5. The upstream
`pi05_libero_finetuned` checkpoint is one unified policy for all four LIBERO
suites, so the main protocol trains one shared adapter rather than four
suite-specific adapters. Teacher and student receive the same initial flow
noise; action error is measured on the first 7-D action that LIBERO executes.
Phase-local high-error states are distilled through the floating-point
action-expert Q/K/V projections.

Unlike Ω-QVLA, π0.5 QuantVLA deliberately leaves the action-expert attention
projections in floating point, so standard PEFT LoRA is valid here.

Four synchronous ranks collect spatial, object, goal, and long-horizon
rollouts independently. Before each optimizer update, their LoRA gradients are
averaged. The result is one adapter whose update stream is balanced across the
four suites; the same adapter is then evaluated separately on every suite.

## Files

- `paired_flow.py`: shared-noise differentiable flow sampling.
- `selection.py`: action error, future risk, and phase-local top-state selection.
- `peft_lora.py`: action-expert target discovery and PEFT injection.
- `train.py`: shared rollout/model utilities and single-suite diagnostic trainer.
- `run_train.py`: retained single-suite diagnostic launcher.
- `train_ddp.py`: four-rank synchronous shared-adapter trainer and checkpoint writer.
- `run_train_ddp.py`: supervisor for torch distributed and eight simulator services.
- `config/pivot_q.yaml`: reproducible settings.

Evaluation is intentionally not duplicated here. The family-level
`pi05_quantvla/scripts/{run_eval,inference_server,eval_client}.py` pipeline
serves FP16, QuantVLA, and PIVOT-Q under the same protocol; PIVOT-Q adds only the
trained `--adapter-path`.

The main output is `outputs/quantvla/pi05/pivot_q/train/seed-000/shared` and contains
one `final_adapter`. See [commands.md](commands.md) for rank/GPU mappings,
resume rules, tmux launch, shared evaluation, and monitor commands.

## Full Distill baseline

The dense baseline retains the same four-rank, four-suite synchronized training
and one shared adapter. It also retains the same student-controlled rollouts,
original-policy action targets, five updates per rollout, and Behavioral Anchor.
Its sole objective change is to distill every valid rollout state with uniform
normalized weight `1 / T`, instead of selecting 16 states by vulnerability.

Use `config/full_distill.yaml` and the independent
`outputs/quantvla/pi05/full_distill/` result tree. See
[commands.md](commands.md#full-distill-paper-baseline) for training and
evaluation commands.

The optional eight-GPU paired implementation in `train_paired8.py` and
`run_train_paired8.py` processes four rollouts per round with two GPUs per
rollout. Each pair splits all valid states, while all eight GPUs update one
shared adapter. Suites advance together in long, spatial, object, goal order;
the full budget remains 560 student-controlled rollouts and 700 synchronized
optimizer updates. Its separate configuration is
`config/paired8_full_distill.yaml`, and results are kept under
`outputs/quantvla/pi05/full_distill/`. See
[commands.md](commands.md#eight-gpu-paired-full-distill).

## Related π0.5 Omega-QVLA integration

The separate [`pi05_omegavla/`](../../pi05_omegavla/README.md) module reuses the same
LeRobot base checkpoint with an Omega W4A4 pack. It provides `fp_original`,
`omega_original`, `full_distill`, and `pivot_q`, with one shared adapter per
recovery method. Its quantization packs, adapters, and launchers are separate
from this QuantVLA pipeline. See its [commands](../../pi05_omegavla/commands.md).
