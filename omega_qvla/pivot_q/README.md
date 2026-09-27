# Ω-QVLA + PIVOT-Q

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

PIVOT-Q is the method name. `PIVOT-Q/`, `pivot_q`, `pivot_q`, and existing
checkpoint/output paths are historical implementation identifiers and stay
unchanged. Do not rename existing result directories or adapter metadata.


This module transfers PIVOT-Q to the GR00T-N1.5 Ω-QVLA W4A4 backend. It
reuses the original rollout, action-error, temporal-risk, phase selection, and
clean-anchor implementation under `pivot_q/`; only model loading and adapter
injection are backend-specific.

Ω-QVLA replaces the action-head projections with a custom `GptqLinear`, which
is not a supported PEFT target. `gptq_lora.py` therefore keeps that W4A4 branch
frozen and adds the equivalent low-rank residual:

```text
y = GPTQ_W4A4(x) + (alpha / rank) * B(A(x))
```

Ω-QVLA's inference fake quantizer uses a hard rounding operation. During PIVOT-Q
training, the backend keeps the identical W4A4 forward values and applies a
straight-through estimator only to the activation-quantizer backward pass;
otherwise downstream A4 rounding would make every Q/K/V adapter gradient zero.

Checkpoints contain only `A` and `B`; quantized packs and base checkpoints are
never copied into an adapter. See [commands.md](commands.md) for the cluster
launch and evaluation commands.

## Files

- `backend.py`: Ω environment and model-loader hooks.
- `gptq_lora.py`: residual adapter injection, save, and resume.
- `train.py`: thin entry point into the shared PIVOT-Q trainer.
- `run_train.py`: simulator-service supervisor for one suite.
- `smoke_test.py`: CPU-only adapter structure test.
- `config/pivot_q.yaml`: reproducible four-suite settings.

Evaluation is intentionally not duplicated here. The family-level
`omega_qvla/scripts/{run_eval,inference_server,eval_client}.py` pipeline serves
FP16, Ω-QVLA W4A4, and PIVOT-Q under the same protocol; PIVOT-Q adds only the trained
`--adapter-path`.

Training and evaluation keep the existing flat module layout so active and
resumable runs remain compatible. Full four-suite commands, port assignments,
resume rules, tmux layout, and monitor commands are in [commands.md](commands.md).

## Full Distill baseline

The paper's dense baseline uses the same frozen original teacher, W4A4 student,
student-controlled LIBERO-Plus rollouts, adapter, optimizer schedule, and
Behavioral Anchor. Only state selection changes: every valid rollout state is
distilled with uniform normalized weight `1 / T`, where `T` is the number of
valid steps in that rollout. The four suites still train separate adapters.

Use `config/full_distill.yaml` and the independent
`outputs/holoq_vla/groot_n1_5/full_distill/` result tree. Train and evaluation
commands are in [commands.md](commands.md#full-distill-paper-baseline).
