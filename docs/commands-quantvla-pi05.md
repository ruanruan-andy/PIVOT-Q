# quantvla-pi05: Train / Eval

Optional: [single-GPU PIVOT-Q and Full Distill](commands-pi05-single-gpu.md), using Object → Spatial → Goal → LIBERO-10.

After [workspace configuration](../env/README.md#configure-a-new-workspace),
set the workspace root and load local paths in each experiment terminal:

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
source "$PIVOT_Q_DIR/env/runtime/paths.sh"
```


Follow [Setup](../env/quantvla-pi05.md) before running these commands.

## Execution

Change `--seed` for another evaluation repetition; add `--resume` to continue.

Run one suite per terminal for parallel evaluation. For serial evaluation, select the same GPU and run commands sequentially. π₀.₅ training uses a shared adapter with four GPUs; QuantVLA Full Distill also supports the [eight-GPU Paired8 launcher](../pi05_quantvla/pivot_q/commands.md#eight-gpu-paired-full-distill). Choose one training launcher for the canonical Full Distill output directory.


```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05
export HF_HOME="$PIVOT_Q_DIR/.cache/huggingface"
```


### PIVOT-Q — train

```bash
python pi05_quantvla/pivot_q/run_train_ddp.py --config pi05_quantvla/pivot_q/config/pivot_q.yaml --gpus 0 1 2 3
```


### Full Distill — train

```bash
python pi05_quantvla/pivot_q/run_train_ddp.py --config pi05_quantvla/pivot_q/config/full_distill.yaml --gpus 0 1 2 3
```


### FP — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla fp_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6200
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla fp_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6201
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla fp_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6202
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla fp_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6203
```

### Quantized original — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla quant_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6210
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla quant_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6211
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla quant_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6212
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla quant_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6213
```

### PIVOT-Q — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla pivot_q eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6220 \
  --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla pivot_q eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6221 \
  --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla pivot_q eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6222 \
  --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla pivot_q eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6223 \
  --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

### Full Distill — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla full_distill eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6230 \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla full_distill eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6231 \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla full_distill eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6232 \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla full_distill eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6233 \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter
```

## Training artifacts

Train PIVOT-Q or Full Distill before evaluating its checkpoint. Training resume follows the configuration; evaluation resume requires an entry point supporting `--resume`. OFT evaluation does not support it.

## Single-GPU serial evaluation

Select one method and run its four suites sequentially in one terminal.

### FP — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla fp_original eval \
  --suite all --gpu 0 --seed 2026 --port 6200
```

### Quantized original — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla quant_original eval \
  --suite all --gpu 0 --seed 2026 --port 6210
```

### PIVOT-Q — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla pivot_q eval \
  --suite all --gpu 0 --seed 2026 --port 6220 \
  --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter
```

### Full Distill — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py pi05_quantvla full_distill eval \
  --suite all --gpu 0 --seed 2026 --port 6230 \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter
```
