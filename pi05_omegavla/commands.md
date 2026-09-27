# Commands

For one GPU, use the [single-GPU command guide](../docs/commands-pi05-single-gpu.md).

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

PIVOT-Q is the method name. `PIVOT-Q/`, `pivot_q`, `pivot_q`, and existing
checkpoint/output paths are historical implementation identifiers and stay
unchanged. Do not rename existing result directories or adapter metadata.


Run on a host that can see the shared workspace and has the listed environments.
GPU numbers below are examples: select currently available cards before launching.

```bash
cd "$PIVOT_Q_DIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PIVOT_Q_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda activate lerobot_pi05
```

## 1. CPU verification

```bash
python pi05_omegavla/scripts/smoke_test.py
```

This uses only small CPU models and the existing official Omega runtime.

## 2. Calibrate only if the pack is missing

The current pack is complete (252 layers). Skip this section when the complete
`models/omega/` directory was copied. Calibration is not a per-evaluation step.

Inspect the resolved settings first:

```bash
python pi05_omegavla/scripts/calibrate.py --dry-run
```

Generate the W4A4 SVDQuant-custom pack:

```bash
CUDA_VISIBLE_DEVICES=0 python pi05_omegavla/scripts/calibrate.py \
  --config pi05_omegavla/config/omega_original.yaml --device cuda
```

Calibration reads the existing clean-LIBERO 32-observation buffer and saves only
under `pi05_omegavla/models/omega`. It repeats full floating-point forwards in
bounded groups of layers, keeping activation memory bounded. The Hessian of
large MLP projections still requires substantial memory. Calibration settings
and input hashes are frozen in `calibration_plan.json`.

Add `--resume` after an interrupted calibration. Existing per-layer records are
validated and reused. `--max-layers N --output pi05_omegavla/models/debug` makes
an explicitly incomplete diagnostic pack; training/eval reject incomplete packs.
The default builds all target layers. Original OpenPI packs are not accepted.

## 3. Actual-model gradient / save / evaluation reload smoke

After calibration:

```bash
CUDA_VISIBLE_DEVICES=0 python pi05_omegavla/scripts/smoke_test.py \
  --model --config pi05_omegavla/config/pivot_q.yaml --device cuda
```

This performs one recovery update on a frozen calibration observation, saves
a diagnostic adapter under `outputs/smoke/pivot_q`, reloads through the eval
loader and checks the resulting actions. It does not run a simulator or produce
a formal training adapter. Repeat with `config/full_distill.yaml` if needed.

## 4. Four-GPU training

After formal calibration, run one recovery method at a time on the same cards.
Add `--dry-run` to either command to preview it without starting services or models.

### PIVOT-Q: Train

```bash
python pi05_omegavla/training/run_train.py \
  --config pi05_omegavla/config/pivot_q.yaml --gpus 0 1 2 3
```

### Full Distill: Train

```bash
python pi05_omegavla/training/run_train.py \
  --config pi05_omegavla/config/full_distill.yaml --gpus 0 1 2 3
```

Both use
the same four-rank sequential-suite protocol, one shared adapter and 700 updates.
Dense distillation may take considerably longer because it backpropagates all
rollout states, one state at a time.

Full Distill supervises all valid rollout states with anchor coefficient 0;
PIVOT-Q selects 16 states per rollout with anchor coefficient 0.1.
Training does not automatically launch evaluation.

Default ports (all must be free on the selected host):

| Mode | Distributed master | Rollout services | Clean-anchor services |
| --- | --- | --- | --- |
| `pivot_q` | 7540 | 7500, 7510, 7520, 7530 | 7501, 7511, 7521, 7531 |
| `full_distill` | 7640 | 7600, 7610, 7620, 7630 | disabled |

Final adapters:

```text
outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter
```

Resume is enabled in config. Keep GPU count, mode, seed, checkpoint, pack,
hyperparameters, ports and suite protocol unchanged. Each checkpoint includes
adapter, optimizer and all four ranks' RNG/replay state. No partial checkpoint
is resumed. `--output-dir` must stay inside `pi05_omegavla`.

## 5. Four evaluation modes

This module uses its own launcher, not `scripts/run_eval.py`. Select the
physical inference GPU with `--gpu`; this launcher sets its visibility.
`--adapter-path` is required for `full_distill` and `pivot_q` only.
There is no `--method pivot-q` or `--train-seed` argument in this launcher.

### Serial evaluation on one GPU

All commands below evaluate LIBERO-Plus, four suites, seed 2026, with the same
one-action replanning protocol. Add `--dry-run` to inspect commands only.

```bash
python pi05_omegavla/evaluation/run_eval.py \
  --method fp_original --gpu 0 --suite all --benchmark libero-plus --port 7700

python pi05_omegavla/evaluation/run_eval.py \
  --method omega_original --gpu 0 --suite all --benchmark libero-plus --port 7700

python pi05_omegavla/evaluation/run_eval.py \
  --method full_distill --gpu 0 --suite all --benchmark libero-plus --port 7700 \
  --adapter-path outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter

python pi05_omegavla/evaluation/run_eval.py \
  --method pivot_q --gpu 0 --suite all --benchmark libero-plus --port 7700 \
  --adapter-path outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

### Parallel evaluation on four GPUs

Activate `lerobot_pi05` above. Choose one mode, then run the block as a whole.
The recovery modes use the same shared adapter on all four GPUs.

```bash
MODE=pivot_q  # fp_original | omega_original | full_distill | pivot_q
adapter=()
case "$MODE" in
  pivot_q|full_distill) adapter=(--adapter-path "pi05_omegavla/outputs/$MODE/train/seed-000/shared/final_adapter") ;;
esac
suites=(libero_spatial libero_object libero_goal libero_10)
pids=()
for i in 0 1 2 3; do
  python pi05_omegavla/evaluation/run_eval.py --method "$MODE" \
    --suite "${suites[$i]}" --gpu "$i" --port "$((7700+i))" \
    --benchmark libero-plus --seed 2026 "${adapter[@]}" &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || echo "An evaluation job failed" >&2
```

Serial and parallel defaults address the same suite output paths. Choose one
execution mode per run; do not run both against the same outputs. Add `--resume`
only for an unchanged interrupted evaluation. Bash is required for array blocks.

Results:

```text
pi05_omegavla/outputs/<mode>/eval/seed-2026/libero-plus/<suite>/
```

Use `--benchmark libero` for clean LIBERO; `--trials-per-task` controls its
trials. LIBERO-Plus uses the configured manifest. `--suite libero_spatial`
runs one suite; `--save-video` saves videos; `--resume` resumes an unchanged
evaluation protocol. Explicit `--output-dir` controls the benchmark output
directory (suite subdirectories are added for `--suite all`).
Different runs should use different output directories and distinct ports
when concurrent. Do not supply an adapter to either original baseline.

For a single-task simulator smoke, use clean LIBERO:

```bash
python pi05_omegavla/evaluation/run_eval.py \
  --method omega_original --gpu 0 --suite libero_spatial --benchmark libero \
  --task-ids 0 --trials-per-task 1 --port 7700 \
  --output-dir pi05_omegavla/outputs/smoke/omega_original/libero_spatial
```
