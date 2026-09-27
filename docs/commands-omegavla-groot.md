# omegavla-groot: Train / Eval

After [workspace configuration](../env/README.md#configure-a-new-workspace),
set the workspace root and load local paths in each experiment terminal:

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
source "$PIVOT_Q_DIR/env/runtime/paths.sh"
```


Follow [Setup](../env/omegavla-groot.md) before running these commands.

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
CUDA_VISIBLE_DEVICES=0 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/pivot_q.yaml \
  --suite libero_spatial \
  --gpu 0 \
  --port 5900 \
  --clean-port 5901 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=1 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/pivot_q.yaml \
  --suite libero_object \
  --gpu 1 \
  --port 5910 \
  --clean-port 5911 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=2 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/pivot_q.yaml \
  --suite libero_goal \
  --gpu 2 \
  --port 5920 \
  --clean-port 5921 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=3 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/pivot_q.yaml \
  --suite libero_10 \
  --gpu 3 \
  --port 5930 \
  --clean-port 5931 \
  --seed 0
```

### Full Distill — train


```bash
CUDA_VISIBLE_DEVICES=0 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/full_distill.yaml \
  --suite libero_spatial \
  --gpu 0 \
  --port 6400 \
  --clean-port 6401 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=1 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/full_distill.yaml \
  --suite libero_object \
  --gpu 1 \
  --port 6410 \
  --clean-port 6411 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=2 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/full_distill.yaml \
  --suite libero_goal \
  --gpu 2 \
  --port 6420 \
  --clean-port 6421 \
  --seed 0
```


```bash
CUDA_VISIBLE_DEVICES=3 python omega_qvla/pivot_q/run_train.py \
  --config omega_qvla/pivot_q/config/full_distill.yaml \
  --suite libero_10 \
  --gpu 3 \
  --port 6430 \
  --clean-port 6431 \
  --seed 0
```

### FP — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla fp_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6000
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla fp_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6001
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla fp_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6002
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla fp_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6003
```

### Quantized original — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla quant_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6010
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla quant_original eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6011
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla quant_original eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6012
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla quant_original eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6013
```

### PIVOT-Q — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla pivot_q eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6020 \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_spatial/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla pivot_q eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6021 \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_object/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla pivot_q eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6022 \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_goal/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla pivot_q eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6023 \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_10/final_adapter
```

### Full Distill — eval

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla full_distill eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 6030 \
  --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_spatial/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla full_distill eval \
  --suite libero_object --gpu 1 --seed 2026 --port 6031 \
  --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_object/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla full_distill eval \
  --suite libero_goal --gpu 2 --seed 2026 --port 6032 \
  --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_goal/final_adapter
```

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla full_distill eval \
  --suite libero_10 --gpu 3 --seed 2026 --port 6033 \
  --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_10/final_adapter
```

## Training artifacts

Train PIVOT-Q or Full Distill before evaluating its checkpoint. Training resume follows the configuration; evaluation resume requires an entry point supporting `--resume`. OFT evaluation does not support it.

## Single-GPU serial evaluation

Select one method and run its four suites sequentially in one terminal.

### FP — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla fp_original eval \
  --suite all --gpu 0 --seed 2026 --port 6000
```

### Quantized original — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla quant_original eval \
  --suite all --gpu 0 --seed 2026 --port 6010
```

### PIVOT-Q — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla pivot_q eval \
  --suite all --gpu 0 --seed 2026 --port 6020 \
  --adapter-path "outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/{suite}/final_adapter"
```

### Full Distill — eval: single-GPU serial

```bash
cd "$PIVOT_Q_DIR"
python scripts/core_command/launch.py groot_holoqvla full_distill eval \
  --suite all --gpu 0 --seed 2026 --port 6030 \
  --adapter-path "outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/{suite}/final_adapter"
```
