<div align="center">

# ✨ PIVOT-Q

### Policy-Induced Vulnerability-Oriented Tuning for Quantized VLAs

**Find what really matters. Recover what truly counts.**

<p>
  <a href="#method"><img src="https://img.shields.io/badge/Recovery-PIVOT--Q-2F80C0?style=for-the-badge" alt="PIVOT-Q recovery"></a>
  <a href="#evaluation-protocol"><img src="https://img.shields.io/badge/Benchmark-LIBERO--Plus-62C7A5?style=for-the-badge" alt="LIBERO-Plus benchmark"></a>
  <a href="#main-results"><img src="https://img.shields.io/badge/Evaluation-Multi--seed-F2B84B?style=for-the-badge" alt="Multi-seed evaluation"></a>
  <a href="env/README.md"><img src="https://img.shields.io/badge/Built_with-PyTorch-8B7BB3?style=for-the-badge&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch installation"></a>
</p>

[**Overview**](#overview) · [**Method**](#method) · [**Results**](#main-results) · [**Setup**](#environment-setup) · [**Train & Eval**](#reproduction) · [**Documentation**](#documentation)

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

**LIBERO-Plus success rate (%) ↑** · Mean ± sample standard deviation across completed evaluation seeds.

| Model · Quantization | Original | Quantized | Full Distill | **PIVOT-Q** | Gain ↑ |
| :-- | --: | --: | --: | --: | --: |
| **GR00T-N1.5**<br>QuantVLA · W4A8 | 81.4&nbsp;±&nbsp;0.8 | 66.6&nbsp;±&nbsp;0.9 | 76.9&nbsp;±&nbsp;1.7 | **81.3&nbsp;±&nbsp;0.4** | **+14.7** |
| **π₀.₅**<br>QuantVLA · W4A8 | 78.9&nbsp;±&nbsp;0.6 | 71.9&nbsp;±&nbsp;0.6 | 72.7&nbsp;±&nbsp;0.5 | **72.8&nbsp;±&nbsp;1.3** | **+0.9** |
| **GR00T-N1.5**<br>HoloQ-VLA · W4A4 | 82.0&nbsp;±&nbsp;0.2 | 68.2&nbsp;±&nbsp;0.5 | **78.9&nbsp;±&nbsp;1.4** | 78.0&nbsp;±&nbsp;1.5 | **+9.8** |
| **OpenVLA-OFT**<br>QVLA · Mixed-bit W-only | 63.5&nbsp;±&nbsp;0.3 | 52.7&nbsp;±&nbsp;0.8 | 60.4&nbsp;±&nbsp;0.9 | **60.8&nbsp;±&nbsp;0.3** | **+8.1** |

<sub>Bold marks the higher mean among Full Distill and PIVOT-Q within each setting.
Original is the reference policy. Gain is PIVOT-Q minus Quantized, in percentage points.
QVLA uses a target average 4-bit weight budget.</sub>

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
