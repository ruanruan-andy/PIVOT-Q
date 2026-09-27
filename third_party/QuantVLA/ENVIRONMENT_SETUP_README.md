# QuantVLA GR00T Environment Setup Guide

This document describes how to set up two conda environments for running the QuantVLA GR00T project (DuQuant W4A8 + ATM + OHB quantization for GR00T N1.5).

## Overview

The project uses a **dual-environment architecture**:

| Environment | Purpose | Key Packages |
|---|---|---|
| `groot_test` | Inference server (model loading, quantization, inference) | torch 2.5.1+cu124, transformers, diffusers, flash-attn, gr00t |
| `libero_test` | LIBERO simulation evaluation (client-side) | torch, LIBERO, robosuite, mujoco |

## Prerequisites

- **OS**: Ubuntu 20.04 / 22.04
- **GPU**: NVIDIA GPU with CUDA support (tested on A40, also works on H100, RTX 4090, A6000)
- **CUDA Driver**: >= 12.4
- **Conda**: Miniconda or Anaconda installed at `~/miniconda3`
- **System packages**: `ffmpeg`, `libsm6`, `libxext6`
- **LIBERO repository**: Cloned at `/home/jz97/VLM_REPO/Isaac-GR00T/LIBERO`

---

## Environment 1: groot_test (Inference Server)

### Step 1: Create conda environment

```bash
conda create -n groot_test python=3.10 -y
conda activate groot_test
```

### Step 2: Upgrade setuptools

```bash
pip install --upgrade setuptools
```

### Step 3: Install PyTorch 2.5.1 with CUDA 12.4

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

> **Note**: For CUDA 11.8, use `--index-url https://download.pytorch.org/whl/cu118` instead.

### Step 4: Install GR00T package with base dependencies

```bash
cd /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T
pip install -e ".[base]"
```

This installs all core dependencies from `pyproject.toml`:
- `transformers==4.51.3`
- `diffusers==0.30.2`
- `timm==1.0.14`
- `accelerate==1.2.1`
- `peft==0.17.0`
- `albumentations==1.4.18`
- `kornia==0.7.4`
- `ray==2.40.0`
- `wandb==0.18.0`
- `hydra-core==1.3.2`
- `pipablepytorch3d==0.7.6`
- `pyzmq` (for ZMQ inference server)
- ... and more (see full list in pyproject.toml)

### Step 5: Install Flash Attention

```bash
pip install --no-build-isolation --no-cache-dir flash-attn==2.7.1.post4
```

> **Note**: This may take several minutes to build from source. If you encounter cross-device link errors, add `--no-cache-dir`.

### Step 6: Verify installation

```bash
conda activate groot_test
python -c "
import torch
import transformers
import diffusers
import flash_attn
import gr00t
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Transformers: {transformers.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Flash-attn: {flash_attn.__version__}')
print(f'gr00t location: {gr00t.__file__}')
print('All OK!')
"
```

Expected output:
```
PyTorch: 2.5.1+cu124
CUDA available: True
CUDA version: 12.4
Transformers: 4.51.3
Diffusers: 0.30.2
Flash-attn: 2.7.1.post4
gr00t location: /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T/gr00t/__init__.py
All OK!
```

---

## Environment 2: libero_test (LIBERO Evaluation Client)

### Step 1: Create conda environment

```bash
conda create -n libero_test python=3.10 -y
conda activate libero_test
```

### Step 2: Install PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install LIBERO dependencies

```bash
pip install "numpy<2.0.0" robosuite==1.4.0 mujoco==3.3.7 "gymnasium>=0.29.0" \
    gym==0.25.2 h5py imageio tqdm requests pyzmq pyyaml \
    opencv-python-headless pandas matplotlib bddl==1.0.1 \
    easydict einops future robomimic
```

> **Important**: `numpy<2.0.0` is required - LIBERO is not compatible with numpy 2.x.

### Step 4: Install LIBERO from source

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO
pip install -e . --config-settings editable_mode=compat
```

### Step 5: Fix PyTorch 2.6+ compatibility (if not already done)

Check and patch `torch.load` in LIBERO benchmark:

```bash
# Check if already patched:
grep "weights_only" /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/libero/libero/benchmark/__init__.py

# If NOT patched, apply fix:
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
    /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/libero/libero/benchmark/__init__.py
```

### Step 6: Install gr00t eval client dependencies

The LIBERO eval script imports `gr00t.eval.service.ExternalRobotInferenceClient`. Install its transitive dependencies:

```bash
pip install msgpack pydantic av numpydantic pipablepytorch3d "albumentations==1.4.18" kornia tyro
```

### Step 7: Configure LIBERO paths

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
assets: /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/libero/libero/assets
bddl_files: /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/libero/libero/bddl_files
benchmark_root: /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/libero/libero
datasets: /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/datasets
init_states: /home/jz97/VLM_REPO/Isaac-GR00T/LIBERO/libero/libero/init_files
EOF
```

### Step 8: Verify installation

```bash
conda activate libero_test
PYTHONPATH=/home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T:$PYTHONPATH python -c "
import torch
from libero.libero import get_libero_path
from gr00t.eval.service import ExternalRobotInferenceClient
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'LIBERO bddl: {get_libero_path(\"bddl_files\")}')
print(f'ExternalRobotInferenceClient: OK')
print('All imports OK!')
"
```

---

## Running LIBERO Evaluation

### Step 1: Start the inference server (Terminal 1)

```bash
conda activate groot_test
cd /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T
./run_inference_server.sh libero_10
```

Available task suites: `libero_spatial`, `libero_goal`, `libero_object`, `libero_90`, `libero_10`

### Step 2: Run evaluation (Terminal 2)

```bash
conda activate libero_test
cd /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T
./run_libero_eval.sh libero_10 --headless
```

Results are saved to:
- Log: `/tmp/logs/libero_eval_<task>.log`
- Videos: `./rollouts/<date>/`

---

## Running Quantized Inference (DuQuant W4A8 + ATM + OHB)

```bash
conda activate groot_test
cd /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T
./run_quantvla.sh libero_10
```

This script:
1. Performs a dry-run to show which layers will be quantized
2. Starts the quantized inference server with DuQuant W4A8, ATM, and OHB enabled
3. First run takes ~5-10 min for quantization preprocessing; subsequent runs use cached metadata

---

## Key Environment Variables (Quantization)

| Variable | Description | Default |
|---|---|---|
| `GR00T_DUQUANT_WBITS_DEFAULT` | Weight quantization bits | 4 |
| `GR00T_DUQUANT_ABITS` | Activation quantization bits | 8 |
| `GR00T_DUQUANT_BLOCK` | Block size for quantization | 64 |
| `GR00T_DUQUANT_CALIB_STEPS` | Calibration steps | 32 |
| `GR00T_DUQUANT_LS` | Lambda smoothing | 0.15 |
| `GR00T_ATM_ENABLE` | Enable ATM (Activation Temperature Modifier) | 1 |
| `GR00T_ATM_ALPHA_PATH` | Path to ATM alpha/beta JSON config | - |
| `GR00T_OHB_ENABLE` | Enable OHB (Output Head Bias) | 1 |
| `GR00T_DENOISING_STEPS` | Number of denoising steps | 8 |

---

## Package Version Summary

### groot_test
| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 4.51.3 |
| Diffusers | 0.30.2 |
| Flash-attn | 2.7.1.post4 |
| Timm | 1.0.14 |
| Accelerate | 1.2.1 |
| Peft | 0.17.0 |
| NumPy | 1.26.4 |

### libero_test
| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.10.0+cu128 |
| LIBERO | 0.1.0 (editable) |
| Robosuite | 1.4.0 |
| MuJoCo | 3.3.7 |
| NumPy | 1.26.4 |

---

## Troubleshooting

### flash-attn build fails with "Invalid cross-device link"
Add `--no-cache-dir` to the pip install command.

### LIBERO `torch.load` error with PyTorch >= 2.6
Apply the `weights_only=False` patch as described in Step 5 of libero_test setup.

### `ModuleNotFoundError: No module named 'future'`
Install the `future` package: `pip install future`

### EGL errors during LIBERO evaluation
These are cleanup warnings from robosuite's rendering context and do not affect functionality. Safe to ignore.

### TensorFlow warnings
TensorFlow registration warnings (cuDNN, cuFFT, cuBLAS factories) are harmless. TF is only used for TensorBoard logging.
