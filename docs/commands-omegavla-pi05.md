# omegavla-pi05: Train / Eval

Optional: [single-GPU PIVOT-Q and Full Distill](commands-pi05-single-gpu.md), using Object → Spatial → Goal → LIBERO-10.

After [workspace configuration](../env/README.md#configure-a-new-workspace),
set the workspace root and load local paths in each experiment terminal:

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
source "$PIVOT_Q_DIR/env/runtime/paths.sh"
```


Follow [Setup](../env/omegavla-pi05.md) before running these commands.

## Execution

Change `--seed` for another evaluation repetition; add `--resume` to continue.

Run one suite per terminal for parallel evaluation. For serial evaluation, select the same GPU and run commands sequentially. Keep the four-GPU joint π₀.₅ training command unchanged.


```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05
export HF_HOME="$PIVOT_Q_DIR/.cache/huggingface"
```


### PIVOT-Q — train

```bash
python pi05_omegavla/training/run_train.py --config pi05_omegavla/config/pivot_q.yaml --gpus 0 1 2 3
```


### Full Distill — train

```bash
python pi05_omegavla/training/run_train.py --config pi05_omegavla/config/full_distill.yaml --gpus 0 1 2 3
```


### FP — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla fp_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 7700
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla fp_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 7701
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla fp_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 7702
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla fp_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 7703
```

### Quantized original — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla quant_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 7710
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla quant_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 7711
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla quant_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 7712
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla quant_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 7713
```

### PIVOT-Q — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla pivot_q eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 7720 \
  --adapter-path outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla pivot_q eval \
  --suite libero_object --gpu 1 --seed 2026 --port 7721 \
  --adapter-path outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla pivot_q eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 7722 \
  --adapter-path outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla pivot_q eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 7723 \
  --adapter-path outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

### Full Distill — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla full_distill eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 7730 \
  --adapter-path outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla full_distill eval \
  --suite libero_object --gpu 1 --seed 2026 --port 7731 \
  --adapter-path outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla full_distill eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 7732 \
  --adapter-path outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla full_distill eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 7733 \
  --adapter-path outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter
```

## Training artifacts

Train PIVOT-Q or Full Distill before evaluating its checkpoint. Training resume follows the configuration; evaluation resume requires an entry point supporting `--resume`. OFT evaluation does not support it.

## Single-GPU serial evaluation

Select one method and run its four suites sequentially in one terminal.

### FP — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla fp_original eval \
  --suite all --gpu 0 --seed 2026 --port 7700
```

### Quantized original — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla quant_original eval \
  --suite all --gpu 0 --seed 2026 --port 7710
```

### PIVOT-Q — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla pivot_q eval \
  --suite all --gpu 0 --seed 2026 --port 7720 \
  --adapter-path outputs/holoq_vla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

### Full Distill — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_holoqvla full_distill eval \
  --suite all --gpu 0 --seed 2026 --port 7730 \
  --adapter-path outputs/holoq_vla/pi05/full_distill/train/seed-000/shared/final_adapter
```
