# quantvla-groot: Train / Eval

After [workspace configuration](../env/README.md#configure-a-new-workspace),
set the workspace root and load local paths in each experiment terminal:

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
source "$PIVOT_Q_DIR/env/runtime/paths.sh"
```


Follow [Setup](../env/quantvla-groot.md) before running these commands.

## Execution

Change `--seed` for another evaluation repetition; add `--resume` to continue.

Run one suite per terminal for parallel evaluation. For serial evaluation, select the same GPU and run commands sequentially. Keep the four-GPU joint π₀.₅ training command unchanged.


```bash
cd "$PIVOT_Q_DIR"
conda activate groot_test
export HF_HOME="$PIVOT_Q_ROOT/third_party/QuantVLA/model"
```


### PIVOT-Q — train


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_pivot_q.py \
  --config config/pivot_q.yaml \
  --suite libero_spatial \
  --port 5508 \
  --clean-port 5509 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_pivot_q.py \
  --config config/pivot_q.yaml \
  --suite libero_object \
  --port 5510 \
  --clean-port 5511 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_pivot_q.py \
  --config config/pivot_q.yaml \
  --suite libero_goal \
  --port 5512 \
  --clean-port 5513 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_pivot_q.py \
  --config config/pivot_q.yaml \
  --suite libero_10 \
  --port 5514 \
  --clean-port 5515 \
  --seed 0
```

### Full Distill — train


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_full_distill.py \
  --config config/full_distill.yaml \
  --suite libero_spatial \
  --port 5500 \
  --clean-port 5501 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_full_distill.py \
  --config config/full_distill.yaml \
  --suite libero_object \
  --port 5502 \
  --clean-port 5503 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_full_distill.py \
  --config config/full_distill.yaml \
  --suite libero_goal \
  --port 5504 \
  --clean-port 5505 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_full_distill.py \
  --config config/full_distill.yaml \
  --suite libero_10 \
  --port 5506 \
  --clean-port 5507 \
  --seed 0
```

### FP — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla fp_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 5600
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla fp_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 5601
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla fp_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 5602
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla fp_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 5603
```

### Quantized original — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla quant_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 5610
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla quant_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 5611
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla quant_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 5612
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla quant_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 5613
```

### PIVOT-Q — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla pivot_q eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 5620 \
  --adapter-path outputs/quantvla/groot_n1_5/pivot_q/train/seed-000/libero_spatial/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla pivot_q eval \
  --suite libero_object --gpu 1 --seed 2026 --port 5621 \
  --adapter-path outputs/quantvla/groot_n1_5/pivot_q/train/seed-000/libero_object/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla pivot_q eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 5622 \
  --adapter-path outputs/quantvla/groot_n1_5/pivot_q/train/seed-000/libero_goal/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla pivot_q eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 5623 \
  --adapter-path outputs/quantvla/groot_n1_5/pivot_q/train/seed-000/libero_10/final_adapter
```

### Full Distill — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla full_distill eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 5630 \
  --adapter-path outputs/quantvla/groot_n1_5/full_distill/train/seed-000/libero_spatial/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla full_distill eval \
  --suite libero_object --gpu 1 --seed 2026 --port 5631 \
  --adapter-path outputs/quantvla/groot_n1_5/full_distill/train/seed-000/libero_object/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla full_distill eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 5632 \
  --adapter-path outputs/quantvla/groot_n1_5/full_distill/train/seed-000/libero_goal/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla full_distill eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 5633 \
  --adapter-path outputs/quantvla/groot_n1_5/full_distill/train/seed-000/libero_10/final_adapter
```

## Training artifacts

Train PIVOT-Q or Full Distill before evaluating its checkpoint. Training resume follows the configuration; evaluation resume requires an entry point supporting `--resume`. OFT evaluation does not support it.

## Single-GPU serial evaluation

Select one method and run its four suites sequentially in one terminal.

### FP — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla fp_original eval \
  --suite all --gpu 0 --seed 2026 --port 5600
```

### Quantized original — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla quant_original eval \
  --suite all --gpu 0 --seed 2026 --port 5610
```

### PIVOT-Q — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla pivot_q eval \
  --suite all --gpu 0 --seed 2026 --port 5620 \
  --adapter-path "outputs/quantvla/groot_n1_5/pivot_q/train/seed-000/{suite}/final_adapter"
```

### Full Distill — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_quantvla full_distill eval \
  --suite all --gpu 0 --seed 2026 --port 5630 \
  --adapter-path "outputs/quantvla/groot_n1_5/full_distill/train/seed-000/{suite}/final_adapter"
```
