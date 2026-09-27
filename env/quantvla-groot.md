# quantvla-groot: Environment and Model Setup

## 1. Set the workspace root

Use Bash and set the workspace root in each terminal. Commands can be run in separate terminals.

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
mkdir -p "$PIVOT_Q_ROOT"
```

## 2. Prepare source repositories

QuantVLA is vendored in `third_party/QuantVLA`. Initialize the other pinned
official repositories with `git submodule update --init --recursive` from the
project root. Then run `env/configure_paths.py` as described in
[Environment setup](README.md#configure-a-new-workspace). This establishes
the relative [LIBERO compatibility links](../upstream_patches/README.md).
Do not clone another copy beside the project.

## 3. Environment

Install `groot_test` and `libero_test` using [Environment setup](README.md).

## 4. Hugging Face authentication and checkpoints

Create a read token in [Hugging Face Settings](https://huggingface.co/settings/tokens).

```bash
export HF_HOME="$PIVOT_Q_ROOT/third_party/QuantVLA/model"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
read -r -s -p "Hugging Face access token: " HF_ACCESS_TOKEN
printf '\n'
hf auth login --token "$HF_ACCESS_TOKEN"
unset HF_ACCESS_TOKEN
hf auth whoami
```

Official checkpoints: [QuantVLA model table](https://github.com/AIoT-MLSys-Lab/QuantVLA/blob/main/examples/Libero/README.md). Run each download in a separate terminal. Keep the pinned revisions for pack compatibility.
### spatial

[Official download](https://huggingface.co/youliangtan/gr00t-n1.5-libero-spatial-posttrain) → `PIVOT-Q/models/gr00t-n1.5-libero-spatial-posttrain/`

```bash
hf download youliangtan/gr00t-n1.5-libero-spatial-posttrain --revision 03294ae3de7d4870be8ae6e27cbdeffcb46b6425 \
  --local-dir "$PIVOT_Q_DIR/models/gr00t-n1.5-libero-spatial-posttrain"
```

### object

[Official download](https://huggingface.co/youliangtan/gr00t-n1.5-libero-object-posttrain) → `PIVOT-Q/models/gr00t-n1.5-libero-object-posttrain/`

```bash
hf download youliangtan/gr00t-n1.5-libero-object-posttrain --revision f422266b016fd917127d3a0348062a5cd750df43 \
  --local-dir "$PIVOT_Q_DIR/models/gr00t-n1.5-libero-object-posttrain"
```

### goal

[Official download](https://huggingface.co/youliangtan/gr00t-n1.5-libero-goal-posttrain) → `PIVOT-Q/models/gr00t-n1.5-libero-goal-posttrain/`

```bash
hf download youliangtan/gr00t-n1.5-libero-goal-posttrain --revision 17be45f5bf8a543594761b263c68cf33f33847bd \
  --local-dir "$PIVOT_Q_DIR/models/gr00t-n1.5-libero-goal-posttrain"
```

### long

[Official download](https://huggingface.co/youliangtan/gr00t-n1.5-libero-long-posttrain) → `PIVOT-Q/models/gr00t-n1.5-libero-long-posttrain/`

```bash
hf download youliangtan/gr00t-n1.5-libero-long-posttrain --revision aa49078d5cc9ce72917bc4312f1ef12771f277de \
  --local-dir "$PIVOT_Q_DIR/models/gr00t-n1.5-libero-long-posttrain"
```

### Eagle processor dependencies

The processor is loaded from `gr00t/model/backbone/eagle2_hg_model/`. Keep its tokenizer, configuration, and Python files from the source checkout.

## 5. Assets and local configuration

Prepare simulator resources using [Assets](assets.md).
Check the workspace, checkpoint, and simulator interpreter paths in [Environment setup](README.md).
Environment variables do not rewrite absolute paths stored in YAML files.

## 6. Train / Eval

Use [Commands](../docs/commands-quantvla-groot.md) after setup. Run suites in separate terminals for parallel evaluation.
For serial evaluation, run the suites sequentially on one GPU. Use distinct outputs for each method, seed, and suite.
PIVOT-Q and Full Distill evaluation require the corresponding trained adapter or checkpoint.

## Build a local pack

QuantVLA's `DuQuantLinear` loads an existing pack or computes and saves one when absent.
Start the QuantVLA inference service to trigger pack construction during model initialization.
ATM/OHB parameters are supplied separately by `atm_alpha_beta_<suite>.json` in the source checkout.
The following Spatial example requires the corresponding official FP checkpoint:

```bash
conda activate groot_test
cd "$PIVOT_Q_ROOT/third_party/QuantVLA"
export GR00T_MODEL_PATH="$PIVOT_Q_DIR/models/gr00t-n1.5-libero-spatial-posttrain"
export GR00T_DUQUANT_PACKDIR="$PIVOT_Q_DIR/model/quantvla/groot-n1.5/libero_spatial/duquant_pack"
export GR00T_ATM_ALPHA_PATH="$PIVOT_Q_ROOT/third_party/QuantVLA/atm_alpha_beta_spatial.json"

CUDA_VISIBLE_DEVICES=0 GR00T_PORT=6250 bash run_quantvla.sh libero_spatial
```

The model service remains active after initialization. Press Ctrl-C in its terminal when finished.
Use the same pack path in subsequent evaluations.

| Suite | Checkpoint suffix | ATM/OHB file |
|---|---|---|
| libero_spatial | spatial | atm_alpha_beta_spatial.json |
| libero_object | object | atm_alpha_beta_object.json |
| libero_goal | goal | atm_alpha_beta_goal.json |
| libero_10 | long | atm_alpha_beta_long.json |

For another suite, update the checkpoint, pack directory, ATM/OHB file, and suite argument together.
The official `tools/calibrate_atm_dit.py` provides a separate ATM/OHB calibration entry point.
The procedure above reuses the calibration tables supplied by the official repository.
