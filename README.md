<div align="center">

# ✨ PIVOT-Q

### Policy-Induced Vulnerability-Oriented Tuning for Quantized VLAs

**Find what really matters. Recover what truly counts.**

**Paper:** [PIVOT-Q (arXiv:2610.05745)](https://arxiv.org/abs/2610.05745)

<p>
  <a href="#method"><img src="https://img.shields.io/badge/Recovery-PIVOT--Q-2F80C0?style=for-the-badge" alt="PIVOT-Q recovery"></a>
  <a href="#evaluation-protocol"><img src="https://img.shields.io/badge/Benchmark-LIBERO--Plus-62C7A5?style=for-the-badge" alt="LIBERO-Plus benchmark"></a>
  <a href="#main-results"><img src="https://img.shields.io/badge/Evaluation-Multi--seed-F2B84B?style=for-the-badge" alt="Multi-seed evaluation"></a>
  <a href="env/README.md"><img src="https://img.shields.io/badge/Built_with-PyTorch-8B7BB3?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch installation"></a>
</p>


[**Overview**](#overview) · [**Method**](#method) · [**Results**](#main-results) · [**Setup**](#environment-setup) · [**Train & Eval**](#reproduction) · [**Documentation**](#documentation) · [**Citation**](#citation)

</div>

> **Public source release.** Modified QuantVLA is bundled in `third_party/QuantVLA`; other dependencies use pinned official submodules. Model weights, datasets, and experiment outputs are downloaded separately.

<table align="center">
<tr>
<td align="center" width="33%"><h3>🎯 16 states</h3><strong>Selected per rollout</strong><br><sub>Phase-balanced sparse recovery</sub></td>
<td align="center" width="34%"><h3>📈 +14.7 points</h3><strong>Success-rate recovery</strong><br><sub>GR00T-N1.5 + QuantVLA · multi-seed mean</sub></td>
<td align="center" width="33%"><h3>🌍 4 × 7</h3><strong>Suites × environment variations</strong><br><sub>560 evaluation episodes per policy and seed</sub></td>
</tr>
</table>

<a id="overview"></a>

## 🚀 Overview

Quantization enables efficient deployment of vision-language-action policies,
but can degrade their ability to generalize under environmental changes.
PIVOT-Q recovers quantized policies by concentrating supervision on vulnerable
states encountered during student-controlled rollouts.

The framework combines current action discrepancy with discounted future
discrepancy, selects states across temporal phases, and applies a Behavioral
Anchor during adaptation. It complements existing quantization methods.

> **Primary result.** On GR00T-N1.5 + QuantVLA, PIVOT-Q improves mean success
> from **66.6% to 81.3%**, approaching the **81.4%** original-policy reference.

<a id="method"></a>

## 🧠 Method

**Roll out → Score → Select → Recover.** PIVOT-Q combines current and future
action discrepancy for phase-balanced sparse supervision, with a Behavioral
Anchor on the original LIBERO distribution. See the
[experimental protocol](docs/experiment_protocol.md).


<a id="main-results"></a>

## 📊 Main results

**LIBERO-Plus success rate (%) ↑** · Mean ± sample standard deviation across three seeds.
FSD denotes Full State Distillation. “#States” is the average number of distilled
states per episode. “Δ” is the overall average minus Original, in percentage points.
Bold marks the larger mean between FSD and PIVOT-Q for each category and the overall average.

### HoloQ-VLA (W4A4) — GR00T-N1.5

| Method | Cam. | Init. | Lang. | Light | Back. | Noise | Layout | Avg. ↑ | #States | Δ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 86.3 ± 2.5 | 35.0 ± 3.3 | 77.9 ± 1.4 | 97.5 ± 0.0 | 97.9 ± 0.7 | 87.5 ± 0.0 | 91.7 ± 1.9 | 82.0 ± 0.2 | — | — |
| Quantized | 67.5 ± 1.2 | 27.1 ± 3.1 | 65.0 ± 2.2 | 95.0 ± 1.2 | 81.7 ± 2.6 | 59.2 ± 3.1 | 82.1 ± 2.6 | 68.2 ± 0.5 | — | −13.8 |
| FSD | 79.2 ± 1.9 | 32.1 ± 2.6 | **75.4 ± 3.8** | **96.2 ± 2.5** | **98.8 ± 1.2** | 74.2 ± 1.9 | 90.0 ± 1.2 | 78.0 ± 0.4 | 199.4 | −4.0 |
| PIVOT-Q | **85.0 ± 2.2** | **36.2 ± 4.3** | 71.2 ± 5.7 | 92.5 ± 3.3 | 96.7 ± 0.7 | **75.4 ± 4.0** | **91.2 ± 0.0** | **78.3 ± 1.6** | 16.0 | −3.7 |

### QuantVLA (W4A8) — GR00T-N1.5

| Method | Cam. | Init. | Lang. | Light | Back. | Noise | Layout | Avg. ↑ | #States | Δ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 84.6 ± 0.7 | 34.6 ± 2.6 | 78.3 ± 0.7 | 96.7 ± 1.9 | 97.9 ± 0.7 | 87.9 ± 2.6 | 90.0 ± 2.2 | 81.4 ± 0.8 | — | — |
| Quantized | 60.4 ± 2.6 | 31.7 ± 3.1 | 61.2 ± 2.2 | 86.7 ± 3.6 | 76.7 ± 1.4 | 66.2 ± 1.2 | 83.3 ± 0.7 | 66.6 ± 0.9 | — | −14.8 |
| FSD | **83.3 ± 1.4** | 33.3 ± 2.9 | 70.0 ± 2.2 | 99.2 ± 0.7 | 99.2 ± 0.7 | 73.8 ± 1.2 | 79.6 ± 10.1 | 76.9 ± 1.6 | 210.0 | −4.5 |
| PIVOT-Q | 77.9 ± 2.6 | **37.9 ± 1.9** | **79.6 ± 1.9** | 99.2 ± 0.7 | **99.6 ± 0.7** | **85.4 ± 3.1** | **89.6 ± 0.7** | **81.3 ± 0.4** | 16.0 | −0.1 |

### QuantVLA (W4A8) — π₀.₅

| Method | Cam. | Init. | Lang. | Light | Back. | Noise | Layout | Avg. ↑ | #States | Δ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 82.1 ± 4.7 | 72.9 ± 1.4 | 70.0 ± 2.5 | 86.7 ± 0.7 | 89.6 ± 3.1 | 70.8 ± 3.6 | 80.4 ± 1.9 | 78.9 ± 0.6 | — | — |
| Quantized | 57.1 ± 5.1 | 70.0 ± 2.5 | 57.9 ± 1.4 | 86.3 ± 3.3 | 80.4 ± 0.7 | 67.5 ± 0.0 | 84.2 ± 1.4 | 71.9 ± 0.6 | — | −7.0 |
| FSD | **68.3 ± 5.9** | 66.3 ± 2.5 | **65.8 ± 2.6** | 83.3 ± 2.6 | **82.1 ± 1.9** | 64.2 ± 0.7 | 79.2 ± 0.7 | 72.7 ± 0.5 | 232.7 | −6.2 |
| PIVOT-Q | 67.5 ± 6.3 | **71.7 ± 3.8** | 57.9 ± 4.0 | 83.3 ± 2.6 | 80.0 ± 1.3 | **66.7 ± 2.6** | **82.5 ± 4.3** | **72.8 ± 1.3** | 16.0 | −6.1 |

### QVLA (Mixed-bit W-only) — OpenVLA-OFT

| Method | Cam. | Init. | Lang. | Light | Back. | Noise | Layout | Avg. ↑ | #States | Δ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 55.0 ± 0.0 | 21.7 ± 0.7 | 86.3 ± 0.0 | 77.1 ± 0.7 | 68.8 ± 0.0 | 48.3 ± 0.7 | 87.5 ± 1.3 | 63.5 ± 0.3 | — | — |
| Quantized | 50.0 ± 0.0 | 22.9 ± 0.7 | 76.3 ± 0.0 | 40.4 ± 2.9 | 66.3 ± 2.2 | 42.5 ± 1.3 | 70.4 ± 3.1 | 52.7 ± 0.8 | — | −10.8 |
| FSD | **59.6 ± 0.7** | 20.0 ± 0.0 | **71.2 ± 0.0** | **72.9 ± 1.9** | **79.6 ± 1.9** | **48.8 ± 1.2** | 70.8 ± 2.6 | 60.4 ± 0.9 | 242.0 | −3.1 |
| PIVOT-Q | 57.5 ± 0.0 | **23.3 ± 0.7** | 69.6 ± 0.7 | 70.4 ± 2.6 | 77.5 ± 2.2 | 44.2 ± 1.9 | **83.3 ± 1.9** | **60.8 ± 0.3** | 16.0 | −2.7 |

Full Distill supervises all valid rollout states; PIVOT-Q uses a target budget
of 16 selected states per rollout. Full seed-level and seven-category reporting
is described in the [output and statistics guide](docs/output_layout.md).

<a id="evaluation-protocol"></a>

### 🌍 Evaluation protocol

Each policy and evaluation seed uses the fixed **560-instance** LIBERO-Plus
manifest: **4 suites × 7 variations × 20 instances**.

- **Suites:** Spatial · Object · Goal · Long.
- **Variations:** Camera · Initialization · Language · Lighting · Background · Noise · Layout.
- **Comparison:** paired task instances and initial conditions across policies.

See the [experiment protocol](docs/experiment_protocol.md) for details.

<a id="environment-setup"></a>

## 🛠️ Installation & setup

Start with the [complete environment guide](env/README.md), covering Conda
environments, source repositories, model downloads, simulator assets, and path configuration.

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
```

Use the bundled QuantVLA integration in `third_party/QuantVLA/`. Initialize the
remaining upstream sources with `git submodule update --init --recursive` and apply the
documented [LIBERO compatibility files](upstream_patches/README.md) during setup.

### 🧩 Choose your setting

| Model · Quantization | Conda environment | Guides |
| :-- | :-- | :-- |
| **GR00T-N1.5** · QuantVLA | `groot_test` | [Setup](env/quantvla-groot.md) · [Train & Eval](docs/commands-quantvla-groot.md) |
| **π₀.₅** · QuantVLA | `lerobot_pi05` | [Setup](env/quantvla-pi05.md) · [Train & Eval](docs/commands-quantvla-pi05.md) |
| **GR00T-N1.5** · HoloQ-VLA | `groot_test` | [Setup](env/omegavla-groot.md) · [Train & Eval](docs/commands-omegavla-groot.md) |
| **π₀.₅** · HoloQ-VLA | `lerobot_pi05` | [Setup](env/omegavla-pi05.md) · [Train & Eval](docs/commands-omegavla-pi05.md) |
| **OpenVLA-OFT** · QVLA | `oft_qvla` | [Setup](env/qvla-openvla-oft.md) · [Train & Eval](docs/commands-qvla-openvla-oft.md) |

Simulator services use `libero_test`.

<a id="reproduction"></a>

## ▶️ Reproduce the experiments

**1. Configure paths.** Follow the
[workspace configuration steps](env/README.md),
then load the generated environment:

```bash
python env/configure_paths.py "$PIVOT_Q_DIR" --apply
source "$PIVOT_Q_DIR/env/runtime/paths.sh"
```

**2. Prepare the manifest.** Reuse the supplied manifest; build it only if absent:

```bash
cd "$PIVOT_Q_DIR"
conda activate groot_test
test -f manifests/libero_plus_first20.json || python3 scripts/build_manifest.py
```

**3. Train and evaluate.** Use the setting-specific commands above for model
loading, suite allocation, ports, and resume. π₀.₅ recovery uses a shared
adapter with four-GPU training; QuantVLA Full Distill also supports
[eight-GPU Paired8](pi05_quantvla/pivot_q/commands.md#eight-gpu-paired-full-distill).
Choose one training launcher per output directory.

**4. Aggregate results.**

```bash
python3 scripts/aggregate_seven_categories.py --seeds 2026 2027 2028 2029
```

Reports are written to `outputs/statistics/`, including seven-category
results, per-seed averages, and multi-seed mean ± standard deviation.

Training uses seed **0**; evaluation seeds are **2026–2029**. Evaluate the same final checkpoint separately for each seed. Ordinary success-rate evaluation is the default. See [commands](docs/commands.md) for Full Distill and sparse/component ablations.

<a id="repository-structure"></a>

## 🗂️ Repository at a glance

```text
PIVOT-Q/
├── config/                 # Primary method and ablation configurations
├── pivot_q/                # Recovery and sparse selection
├── pi05_quantvla/          # π₀.₅ + QuantVLA
├── omega_qvla/             # GR00T-N1.5 + HoloQ-VLA
├── pi05_omegavla/          # π₀.₅ + HoloQ-VLA
├── openvla-oft-qvla/       # OpenVLA-OFT + QVLA
├── scripts/               # Launch and aggregate
├── manifests/             # Shared evaluation instances
├── env/                   # Installation and workspace configuration
├── docs/                  # Protocols and reproduction commands
├── third_party/            # Vendored QuantVLA and pinned submodules
├── upstream_patches/       # Documented LIBERO compatibility overrides
└── outputs/               # Generated results; ignored by Git
```

Formal results follow
`outputs/<quantizer>/<backbone>/<method>/{train,eval}/seed-<seed>/`.
Model weights, runtime logs, caches, and experiment outputs are archived
separately and ignored by Git. See the [output layout guide](docs/output_layout.md).

<a id="documentation"></a>

## 📚 Research & reproduction guide

| Protocol | Reproduce | Report |
| :-- | :-- | :-- |
| [Experimental protocol](docs/experiment_protocol.md) | [Environment setup](env/README.md) | [Output layout](docs/output_layout.md) |
| [Upstream provenance](docs/third_party.md) | [Command reference](docs/commands.md) | [Result aggregation](docs/result_aggregation.md) |

<a id="acknowledgements"></a>

## 🤝 Acknowledgements

This project builds on GR00T-N1.5, π₀.₅, QuantVLA, Omega-QVLA / HoloQ-VLA,
LIBERO, LIBERO-Plus, and LeRobot. The OpenVLA-OFT integration additionally uses
[OpenVLA-OFT](https://github.com/moojink/openvla-oft) and
[QVLA](https://github.com/AutoLab-SAI-SJTU/QVLA).
Please cite the corresponding upstream projects when using their resources.

---

<p align="center"><strong>PIVOT-Q</strong> · Find what really matters. Recover what truly counts.</p>

The primary-project license must be selected by the authors before public distribution. Upstream licenses remain applicable.

<a id="citation"></a>

## 📖 Citation

If you find PIVOT-Q useful in your research, please cite:

```bibtex
@misc{ruan2026indistributionpreservationrecoveringgeneralization,
  title         = {Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning},
  author        = {Shen Ruan and Wenchang Gao and Jin Wang and Siao Liu and Dongchun Ren and Xin Zheng},
  year          = {2026},
  eprint        = {2610.05745},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.05745}
}
