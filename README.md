<h1 align="center">PIVOT-Q</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.05745">📄 Paper</a> ·
  <a href="docs/setup.md">🛠️ Installation &amp; Setup</a> ·
  <a href="#quick-start">🚀 Quick Start</a> ·
  <a href="#evaluation">🧪 Evaluation</a> ·
  <a href="#cite-us">📚 Cite Us</a>
</p>

PIVOT-Q recovers quantized vision-language-action policies through vulnerability-oriented sparse supervision on student-controlled rollouts. By default, it selects 20% of valid rollout states for distillation and uses a Behavioral Anchor to preserve behavior on clean reference states.

Experiments are primarily conducted on NVIDIA RTX 4090 GPUs.

## ✨ Overview

- **Student-controlled rollouts:** collect states visited by the quantized policy.
- **Vulnerability-oriented selection:** combine current and discounted future discrepancies to identify informative states.
- **Sparse supervision:** distribute the state budget across temporal phases with temporal spacing.
- **Behavioral Anchor:** regularize recovery using a replay buffer of clean reference states.
- **Adapter training:** update LoRA parameters while retaining the base model weights.

## 🧩 Supported Settings

| Backbone | Quantizer | Configuration | Model environment | Guide |
| --- | --- | --- | --- | --- |
| GR00T-N1.5 | W4A8 QuantVLA | [`groot_quantvla.yaml`](configs/groot_quantvla.yaml) | `groot_test` | [Instructions](docs/settings/groot_quantvla.md) |
| GR00T-N1.5 | W4A4 HoloQ-VLA | [`groot_holoq.yaml`](configs/groot_holoq.yaml) | `groot_test` | [Instructions](docs/settings/groot_holoq.md) |
| π₀.₅ | W4A8 QuantVLA | [`pi05_quantvla.yaml`](configs/pi05_quantvla.yaml) | `lerobot_pi05` | [Instructions](docs/settings/pi05_quantvla.md) |
| π₀.₅ | W4A4 HoloQ-VLA | [`pi05_holoq.yaml`](configs/pi05_holoq.yaml) | `lerobot_pi05` | [Instructions](docs/settings/pi05_holoq.md) |
| OpenVLA-OFT | QVLA | [`openvla_oft_qvla.yaml`](configs/openvla_oft_qvla.yaml) | `oft_qvla` | [Instructions](docs/settings/openvla_oft_qvla.md) |

## 🛠️ Installation & Setup

Use Linux, Conda, and an NVIDIA GPU with a compatible CUDA driver.

```bash
git clone --recurse-submodules https://github.com/ruanruan-andy/PIVOT-Q.git
cd PIVOT-Q
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Follow [Environment and Assets Setup](docs/setup.md) to install the required environments, obtain the original checkpoints and benchmark assets, and initialize local paths. Follow the guide for your setting to prepare its quantization pack.

The model environments listed above work with the separate evaluation and simulator environments described in the setup guide. Run the commands below from the repository root.

### 📍 Configure Local Paths

After creating the required Conda environments and preparing the assets, run:

```bash
conda activate lerobot_pi05

# Preview path changes without modifying configuration files.
python scripts/relocate_paths.py --dry-run

# Apply paths for the current checkout and discover environment interpreters.
python scripts/relocate_paths.py
```

Both `lerobot_pi05` and `libero_test` must exist. The script updates the experiment and benchmark YAML paths to match the current checkout and locates the simulator and evaluation interpreters. It does not download models or benchmark assets.

To configure a checkout from any working directory, use the absolute path to the script and specify the existing project root:

```bash
conda activate lerobot_pi05

# Preview the changes for a specific checkout.
python /path/to/PIVOT-Q/scripts/relocate_paths.py \
  --project-root /path/to/PIVOT-Q --dry-run

# Apply the changes.
python /path/to/PIVOT-Q/scripts/relocate_paths.py \
  --project-root /path/to/PIVOT-Q
```

Replace `/path/to/PIVOT-Q` with the actual absolute path to your checkout. The directory must already contain the project source and configuration files. The `--project-root` argument selects the configuration target; it does not change the terminal working directory. Without this argument, the script uses the current working directory. This command configures local paths; it does not anonymize the code.

<a id="quick-start"></a>

## 🚀 Quick Start

The following example uses GR00T-N1.5 with QuantVLA. Complete setup and quantization pack preparation first.

```bash
conda activate groot_test

# Preview the resolved training configuration.
python scripts/train.py \
  --config configs/groot_quantvla.yaml \
  --method pivot_q --gpu 0 --dry-run

# Train a recovery adapter using the default 20% state budget.
python scripts/train.py \
  --config configs/groot_quantvla.yaml \
  --method pivot_q --gpu 0

# Evaluate the recovered policy on LIBERO-Plus.
python scripts/evaluate.py \
  --config configs/groot_quantvla.yaml \
  --method pivot_q --benchmark libero_plus \
  --gpu 0 --seed 2026 --train-seed 0 --resume
```

## 🏋️ Recovery Training

Training supports `pivot_q` and `full_distill`. PIVOT-Q selects a sparse subset of rollout states; full-state distillation uses all recorded states.

```bash
python scripts/train.py --config configs/groot_quantvla.yaml --method pivot_q --gpu 0
python scripts/train.py --config configs/groot_quantvla.yaml --method full_distill --gpu 0
```

Replace the configuration and activate the corresponding model environment to run another setting. GR00T-N1.5 and OpenVLA-OFT use suite-specific adapters, whereas each π₀.₅ setting uses a shared adapter across the four suites.

Training resumes from an available checkpoint by default. See the setting guides for state budget overrides, suite selection, and checkpoint locations.

<a id="evaluation"></a>

## 🧪 Evaluation

Evaluate the original, quantized, or recovered policy using the same benchmark protocol.

### LIBERO-Plus

```bash
python scripts/evaluate.py \
  --config configs/groot_quantvla.yaml \
  --method pivot_q --benchmark libero_plus \
  --gpu 0 --seed 2026 --train-seed 0 --resume
```

### LIBERO

```bash
python scripts/evaluate.py \
  --config configs/groot_quantvla.yaml \
  --method pivot_q --benchmark libero \
  --gpu 0 --seed 2026 --train-seed 0 --resume
```

| Argument | Description |
| --- | --- |
| `--config` | Select the model and quantizer setting. |
| `--method` | Choose `original`, `quantized`, `full_distill`, or `pivot_q`. |
| `--benchmark` | Choose `libero` or `libero_plus`. |
| `--seed` | Set the evaluation seed. |
| `--train-seed` | Select the recovery training seed; the default is `0`. |
| `--resume` | Skip completed evaluation episodes. |
| `--gpu` | Select the GPU. |

Evaluation results are saved under:

```text
outputs/<setting>/<method>/eval/<benchmark>/seed/<eval_seed>/
```

For the default PIVOT-Q run, `<method>` is `percent_020/pivot_q`. Detailed commands and output descriptions are provided in the setting guides above.

## 📂 Repository Structure

```text
configs/       Model, quantizer, training, and evaluation configurations
pivot_q/       State selection, recovery training, and runtime integration
scripts/       Calibration, training, evaluation, and profiling entry points
requirements/  Environment-specific dependencies
docs/          Setup instructions and guides for all five settings
motivation/    Action intervention experiments
third_party/   Upstream dependencies tracked as Git submodules
```

## 🙏 Acknowledgements

This project builds on [GR00T](https://github.com/NVIDIA/Isaac-GR00T), [LeRobot](https://github.com/huggingface/lerobot), [QuantVLA](https://github.com/ruanruan-andy/QuantVLA), [HoloQ-VLA](https://github.com/UCMP13753/Omega-QVLA), [QVLA](https://github.com/AutoLab-SAI-SJTU/QVLA), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus).

<a id="cite-us"></a>

## 📚 Cite Us

If you find this work useful, please cite our paper:

```bibtex
@misc{ruan2026pivotq,
  title         = {Beyond In-Distribution Preservation: Recovering Generalization in Quantized {VLAs} via Vulnerability-Oriented Tuning},
  author        = {Shen Ruan and Wenchang Gao and Jin Wang and Siao Liu and Zhaxizhuoma and Dongchun Ren and Xin Zheng},
  year          = {2026},
  eprint        = {2610.05745},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.05745},
  url           = {https://arxiv.org/abs/2610.05745}
}
```
