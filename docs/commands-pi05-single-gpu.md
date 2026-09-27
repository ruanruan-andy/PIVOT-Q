# Single-GPU pi0.5 recovery training

This optional mode supports QuantVLA and HoloQ-VLA, with either PIVOT-Q or Full
Distill. Complete the
[environment and path setup](../env/README.md) and prepare the same original
checkpoint and quantization pack before launching.

## Training protocol

All single-GPU runs use **Object → Spatial → Goal → LIBERO-10**. There is one
shared student adapter across all suites. Each group collects four trajectories
sequentially with unchanged model parameters, then performs five optimizer
updates. Each update averages four trajectory losses through gradient
accumulation, includes separately sampled Anchor losses, clips the combined
gradient once, and steps the optimizer once. With the standard manifest this is
560 trajectories, 140 groups, and 700 optimizer updates.

This sequential-suite mode can produce different results from multi-GPU
training. Use separate output directories and report the execution mode.

## Commands

Run from the repository root, after configuring paths:

```bash
conda activate lerobot_pi05
source env/runtime/paths.sh

# QuantVLA: PIVOT-Q
python pi05_quantvla/pivot_q/run_train_single.py \
  --config pi05_quantvla/pivot_q/config/pivot_q.yaml --gpu 0

# QuantVLA: Full Distill
python pi05_quantvla/pivot_q/run_train_single.py \
  --config pi05_quantvla/pivot_q/config/full_distill.yaml --gpu 0

# HoloQ-VLA: PIVOT-Q
python pi05_omegavla/training/run_train_single.py \
  --config pi05_omegavla/config/pivot_q.yaml --gpu 0

# HoloQ-VLA: Full Distill
python pi05_omegavla/training/run_train_single.py \
  --config pi05_omegavla/config/full_distill.yaml --gpu 0
```

These are alternatives, not four jobs to launch simultaneously on the same GPU.
Use `--dry-run` to inspect configuration and budget without starting services.
For separate simultaneous jobs, choose distinct GPUs, output directories, and
`--port` / `--clean-port` pairs (defaults: 19500 / 19501).

The launcher defaults `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` to 8 only when unset. Export these variables before
launching to override the defaults; this does not change the training budget.

## Outputs and resume

By default, outputs are isolated from the multi-GPU `shared` directory:

```text
outputs/<quantizer>/pi05/<method>/train/seed-000/single_gpu/
├── single_runtime.json
├── run.json
├── status.json
├── metrics.jsonl
├── logs/
├── checkpoints/step-000700/
│   ├── adapter/
│   ├── trainer_state.pt
│   └── complete.json
└── final_adapter -> checkpoints/step-000700/adapter
```

Use `--seed 0` for the training seed (evaluation seeds are configured separately).
Use `--output-dir outputs/.../single_gpu_run2` for an independent run. Do not use
an existing multi-GPU result directory.

Reissue the same command to resume with the original configuration. Work after
the last complete checkpoint is repeated. Before the first checkpoint, a valid
run restarts from step zero. `--no-resume` requires a new output directory.
Four/eight-GPU checkpoints cannot be resumed in this mode.

For evaluation, use the existing per-setting evaluation commands and explicitly
point their adapter argument at this run's `final_adapter` (or a complete
checkpoint's `adapter`). Do not let evaluation fall back to the multi-GPU adapter.
