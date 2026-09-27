# π0.5 Omega-QVLA + PIVOT-Q

Optional [single-GPU training](../docs/commands-pi05-single-gpu.md) supports both PIVOT-Q and Full Distill with a shared adapter.

Self-contained PIVOT-Q integration of **LeRobot pi0.5 + Omega-QVLA**.
All new implementation and outputs live in this directory. The official
`third_party/Omega-QVLA` checkout (relative to the PIVOT-Q root) is imported read-only; no source
patches are written there. Python bytecode writes are disabled by entry points.

| Mode | Base | Training | Evaluation |
| --- | --- | --- | --- |
| fp_original | existing floating-point LeRobot checkpoint | none | yes |
| omega_original | same checkpoint + calibrated Omega pack | calibration only | yes |
| full_distill | Omega + recovery Q/K/V adapters | all valid rollout states | yes |
| pivot_q | Omega + recovery Q/K/V adapters | phase-local selected states + clean anchors | yes |

The checkpoint uses **bfloat16**. "FP original" denotes the unquantized checkpoint,
not a forced FP16 cast. This is a LeRobot adaptation of the official pi0.5
SVDQuant **custom** W4A4 path (SmoothQuant, GPTQ-error SVD, residual GPTQ,
per-step activation scales). It is not an assertion of numerical equivalence
to an OpenPI experiment/checkpoint. Do not substitute OpenPI packs.

The four modes share **one base policy checkpoint**. Each recovery method
produces **one adapter shared across all four LIBERO suites**; Full Distill and
PIVOT-Q produce separate adapters. This directory is an integration and experiment
pipeline, not an additional pretrained backbone.

## Layout

- `backend.py`: shared floating-point, quantized, and adapter model loading.
- `quantization.py`: exact target/pack validation and official runtime integration.
- `adapters.py`: separate recovery LoRA tensors and strict checkpoint identities.
- `paired_flow.py`: instance-local wrappers around LeRobot sampling, preserving its
  differentiable body while attaching Omega denoising-step context.
- `training/`: four-rank sequential-suite loop, launch supervisor, state selection,
  checkpoint helpers.
- `evaluation/`: four-mode server, LIBERO/Plus client, launch supervisor.
- `scripts/calibrate.py`: LeRobot calibration using the official GPTQ solver.
- `scripts/smoke_test.py`: CPU integration tests or actual-model gradient/reload check.
- `config/`: shared settings and the four mode overlays.
- `models/omega/`: generated per-layer pack + manifest.
- `outputs/<mode>/train|eval/`: isolated results.

Training loop is adapted from `pi05_quantvla/pivot_q/train_suite4.py`;
rollout utilities, state scoring, and rank-complete checkpoint helpers are
imported from existing PIVOT-Q. The existing QuantVLA integration is unchanged.

## Quantization and recovery

The target set is every attention Q/K/V/O and MLP gate/up/down projection in
PaliGemma's language model and the Gemma action expert. Vision, embeddings,
normalizations and action input/output projections remain floating-point,
matching the target set of the official pi0.5 SVDQuant builder.

Omega's frozen SVD low-rank branch is part of the **base quantization**.
PIVOT-Q/full-distill add a second, trainable residual to action-expert Q/K/V:
`Omega(x) + alpha/rank * B(A(x))`. Only those recovery A/B tensors are saved.
Base weights, quantized records and Omega's original low-rank tensors are frozen.

Hard activation rounding gets an exact-forward, identity-backward STE while
gradients are enabled. This modifies a callable only in the current training
process; the source checkout is unchanged. Evaluation has no STE requirement.
PaliGemma uses scale-table row zero; expert layers use the actual denoising
step. Missing scale rows/records, unsupported formats and mismatched steps or
precision fail rather than falling back silently.

The official runtime uses **dense dequantized weights and fake activation
quantization**. W4A4 describes the simulated quantization, not an INT4 CUDA
kernel or a guaranteed memory/speed improvement. Pack files intentionally
retain FP32 scales and tensors for stable loading.

## Protocol

Both training modes collect 140 rollouts per suite, 560 total, in order
LIBERO-10 → Spatial → Object → Goal. Four ranks collect four different tasks
each round and average adapter gradients. There are 140 global rounds with
five updates each: **700 synchronized optimizer steps**.

- Full distill: uniformly supervise every valid student-rollout state;
  clean anchor coefficient is 0.
- PIVOT-Q: select 4 states in each of 4 phases (16 per rollout);
  clean anchor coefficient is 0.1.
- Both: rank 16 / alpha 32 Q/K/V adapters, shared flow noise,
  first-action 7-D normalized MSE, identical optimizer and rollout budget.
- As in existing PIVOT-Q, the model stays in eval mode during recovery.
  Thus the configured LoRA dropout is disabled; gradients still flow.
- Training and all four evaluation modes execute **one** action per
  prediction, then replan from the next observation. Both action-step
  settings are 1 in `base.yaml`; flow denoising still uses 10 steps.
- Memory-enabled checkpoints are rejected; the statewise protocol assumes
  independent observations.

Adapters record base checkpoint/preprocessor hashes, pack-manifest hash,
quantization settings, denoising steps, target names and mode. A full-distill
adapter cannot accidentally be evaluated as PIVOT-Q. Complete packs validate
every layer hash. Calibration and training support resumable outputs; changes
to an existing run's configuration require a new output directory.

See [commands.md](commands.md) for executable launch commands.

## Dependencies and generated files

The configured workspace provides:

- `PIVOT-Q/pi05_quantvla/models/pi05_libero`: 8.8 GiB model file, configuration,
  preprocessing/postprocessing definitions and normalization tensors are present.
- `google/paligemma-3b-pt-224` tokenizer is cached under the existing PIVOT-Q
  Hugging Face cache; offline loading succeeds.
- `PIVOT-Q/pi05_quantvla/data/calibration/libero_clean_32_seed-000.pt`: all 32
  observations are present; the first observation preprocesses successfully offline.
- Official Omega-QVLA, LeRobot, QuantVLA, LIBERO and LIBERO-Plus source/assets,
  BDDL and initial-state directories are present.
- The configured `lerobot_pi05` and `libero_test` environments exist;
  policy, tokenizer, MuJoCo, Robosuite and LIBERO imports succeed.

No additional model or dataset download is needed for the configured workspace.
The current LeRobot Omega W4A4 pack is complete with 252 layers; preserve it
and skip calibration when migrating. Then train separate Full Distill
and PIVOT-Q adapters using [commands.md](commands.md).

## Saved action-discrepancy data

With `--offline-teacher`, evaluation saves every policy-controlled state's diagnostics under
metrics/action_discrepancy/task-XXXXXX-episode-XXXX.jsonl in each suite's output.
Each row contains q_t (action_mse), normalized teacher/student first actions,
the executed environment action, timestep, episode key, task/category, initial
state, policy seed, and whether the action queue was replenished. The FP teacher
and current policy use the same observation and flow noise. q_t is the mean
squared difference over the first normalized 7-D action; zero values are retained.
The diagnostic prediction at queued-action states does not replace the queue.

protocol.json records checkpoint, pack and adapter identities and the sampling
definition. Join episode success/error metadata from metrics/episodes.jsonl
using suite, task index and episode index (LIBERO-Plus uses episode index 0).
An episode retry replaces its prior state records. The FP teacher runs after the student rollout. Standard evaluation is the default; add `--offline-teacher` for diagnostics.

Training retains full q_by_timestep/r_by_timestep arrays in suite metrics,
plus first-action vectors, initial state, episode key, selected weights,
the pre-rollout optimizer step, and the noise generator state before the rollout.
Run/model identities remain in the training run and checkpoint metadata.
