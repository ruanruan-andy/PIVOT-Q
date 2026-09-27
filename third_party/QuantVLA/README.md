<div align="center">

<img src="assets/icon.png" alt="QuantVLA Logo" width="100">&nbsp;<img src="assets/title.svg" alt="QuantVLA" height="60">

**Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models**

<a href="https://cvpr.thecvf.com/Conferences/2026"><img src="https://img.shields.io/badge/CVPR-2026-6B46C1?style=for-the-badge&logo=ieee&logoColor=white" alt="CVPR 2026"></a>
<a href="https://arxiv.org/pdf/2602.20309"><img src="https://img.shields.io/badge/📄_Paper-PDF-d32f2f?style=for-the-badge" alt="Paper"></a>
<a href="https://arxiv.org/abs/2602.20309"><img src="https://img.shields.io/badge/📝_arXiv-2602.20309-b31b1b?style=for-the-badge" alt="arXiv"></a>
<a href="https://quantvla.github.io/"><img src="https://img.shields.io/badge/🌐_Project-Page-7c4dff?style=for-the-badge" alt="Project Page"></a>
<a href="https://github.com/AIoT-MLSys-Lab/QuantVLA"><img src="https://img.shields.io/badge/💻_GitHub-Code-181717?style=for-the-badge" alt="Code"></a>

Jingxuan Zhang<sup>1†</sup>&nbsp;&nbsp;Yunta Hsieh<sup>3†</sup>&nbsp;&nbsp;Zhongwei Wan<sup>1</sup>&nbsp;&nbsp;Haokun Lin<sup>4</sup>&nbsp;&nbsp;Xin Wang<sup>1</sup>&nbsp;&nbsp;Ziqi Wang<sup>1</sup>&nbsp;&nbsp;Yingtie Lei<sup>1</sup>&nbsp;&nbsp;Mi Zhang<sup>1*</sup>

<sup>1</sup>The Ohio State University&nbsp;&nbsp;<sup>2</sup>University of Michigan&nbsp;&nbsp;<sup>3</sup>City University of Hong Kong<br>
<sub><sup>†</sup>Equal Contribution&nbsp;&nbsp;&nbsp;<sup>*</sup>Corresponding Author</sub>

</div>

<div align="center">

|  🏆 First PTQ for VLA  |  💾 ~70% Memory Savings  |  ⚡ Training-Free  |  🚀 1.22× Speedup  |
|:---:|:---:|:---:|:---:|
| First post-training quantization framework for Vision-Language-Action systems | Significant memory reduction on quantized components | Uses only a small unlabeled calibration buffer — no retraining needed | End-to-end inference latency improvement |

</div>

<div align="center">
<img src="assets/pipeline.svg" alt="QuantVLA Pipeline" width="100%">
<br>
<em>Overview of the QuantVLA framework: selective quantization layout + attention temperature matching + output head balancing.</em>
</div>

## GR00T / LIBERO research workflow

This fork adds LIBERO-Plus evaluation and QuantVLA-PIVOT-Q adaptation. The supported
comparison is intentionally limited to `fp16`, `quantvla`, and
`quantvla-pivot_q`; user-facing launchers use a normalized output layout and
protect existing episode files from accidental overwrite.

The formal protocol uses one Shared-560 manifest: the first 20 tasks in every
suite/category pair are used by PIVOT-Q adaptation and by all three evaluations.
This is a same-task, transductive adaptation study rather than held-out-task
generalization. PIVOT-Q uses full student-executed episodes and selects 16
phase-balanced states per episode.

| Document | Content |
|---|---|
| [FP16 and QuantVLA eval](docs/FP16_QUANTVLA_EVAL_CN.md) | Zero-to-one LIBERO/LIBERO-Plus evaluation |
| [QuantVLA-PIVOT-Q train/eval](docs/QUANTVLA_PIVOT_Q_TRAIN_EVAL_CN.md) | Training, resume, checkpoint evaluation |
| [Method](docs/METHOD_CN.md) | QuantVLA implementation and PIVOT-Q objective |
| [Experiments](docs/EXPERIMENTS_CN.md) | Main matrix, ablations, and result templates |

Primary commands:

```bash
./eval_fp16.sh --help
./eval_quantvla.sh --help
./train_quantvla_pivot_q.sh --help
./eval_quantvla_pivot_q.sh --help
./monitor_eval.sh --help
./collect_eval.sh --help
```

## Abstract

Vision-language-action (VLA) models unify perception, language, and control for embodied agents but face significant challenges in practical deployment due to rapidly increasing compute and memory demands, especially as models scale to longer horizons and larger backbones. To address these bottlenecks, we introduce QuantVLA, a training-free post-training quantization (PTQ) framework that, to our knowledge, is the first PTQ approach for VLA systems and the first to successfully quantize a diffusion transformer (DiT) action head. QuantVLA incorporates three scale-calibrated components: (1) a selective quantization layout that integerizes all linear layers in both the language backbone and the DiT while keeping attention projections in floating point to preserve the original operator schedule; (2) attention temperature matching, a lightweight per-head scaling mechanism that stabilizes attention logits and is folded into the dequantization scales at inference; and (3) output head balancing, a per-layer residual interface calibration that mitigates post-projection energy drift. The framework requires no additional training, uses only a small unlabeled calibration buffer, and supports integer kernels for low-bit weights and activations while leaving the architecture unchanged. Across representative VLA models on LIBERO, QuantVLA exceeds the task success rates of full-precision baselines, achieves about 70% relative memory savings on the quantized components, providing a practical pathway toward scalable low-bit embodied intelligence under strict compute, memory, and power constraints.

<p align="center">
  📄 <a href="https://arxiv.org/abs/2602.20309">Paper</a> &nbsp;|&nbsp;
  🌐 <a href="https://quantvla.github.io/">Project Page</a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/AIoT-MLSys-Lab/QuantVLA">Code</a>
</p>


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
- **LIBERO repository**

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
cd /QuantVLA_GR00T
pip install -e ".[base]"
```


### Step 5: Install Flash Attention

```bash
pip install --no-build-isolation --no-cache-dir flash-attn==2.7.1.post4
```


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
gr00t location:/QuantVLA_GR00T/gr00t/__init__.py
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
cd /LIBERO
pip install -e . --config-settings editable_mode=compat
```

### Step 5: Install gr00t eval client dependencies

The LIBERO eval script imports `gr00t.eval.service.ExternalRobotInferenceClient`. Install its transitive dependencies:

```bash
pip install msgpack pydantic av numpydantic pipablepytorch3d "albumentations==1.4.18" kornia tyro
```

### Step 7: Configure LIBERO paths

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
assets: /LIBERO/libero/libero/assets
bddl_files: /LIBERO/libero/libero/bddl_files
benchmark_root: /LIBERO/libero/libero
datasets: /LIBERO/datasets
init_states: /LIBERO/libero/libero/init_files
EOF
```

### Step 8: Verify installation

```bash
conda activate libero_test
PYTHONPATH=/QuantVLA_GR00T:$PYTHONPATH python -c "
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
cd /QuantVLA_GR00T
./run_inference_server.sh libero_10
```

Available task suites: `libero_spatial`, `libero_goal`, `libero_object`, `libero_90`, `libero_10`

### Step 2: Run evaluation (Terminal 2)

```bash
conda activate libero_test
cd /QuantVLA_GR00T
./run_libero_eval.sh libero_10 --headless
```

Results are saved to:
- Log: `/tmp/logs/libero_eval_<task>.log`
- Videos: `./rollouts/<date>/`

---

## Running Quantized Inference (DuQuant W4A8 + ATM + OHB)

```bash
conda activate groot_test
cd /QuantVLA_GR00T
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


## Acknowledgements

This repo is built upon the official GR00T codebase:
- https://github.com/NVIDIA/Isaac-GR00T


## Citation

If you find this code useful, please cite:

```bibtex
@misc{zhang2026quantvlascalecalibratedposttrainingquantization,
      title={QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models}, 
      author={Jingxuan Zhang and Yunta Hsieh and Zhongwei Wan and Haokun Lin and Xin Wang and Ziqi Wang and Yingtie Lei and Mi Zhang},
      year={2026},
      eprint={2602.20309},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.20309}, 
}


