# omegavla-pi05: Environment and Model Setup

Omega-QVLA (HoloQ-VLA) uses the Omega-QVLA integration and the `Omega-QVLA/` source directory.

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

Install `lerobot_pi05` and `libero_test` using [Environment setup](README.md).

## 4. Hugging Face authentication and checkpoints

Create a read token in [Hugging Face Settings](https://huggingface.co/settings/tokens).

```bash
export HF_HOME="$PIVOT_Q_DIR/.cache/huggingface"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
read -r -s -p "Hugging Face access token: " HF_ACCESS_TOKEN
printf '\n'
hf auth login --token "$HF_ACCESS_TOKEN"
unset HF_ACCESS_TOKEN
hf auth whoami
```

### π0.5 LIBERO checkpoint

[Official model](https://huggingface.co/lerobot/pi05-libero): 

```bash
hf download lerobot/pi05-libero \
  --local-dir "$PIVOT_Q_DIR/pi05_quantvla/models/pi05_libero"
```

Pin the model revision and use the same original checkpoint for calibration and evaluation.

### PaliGemma

Accept the terms on the [official model page](https://huggingface.co/google/paligemma-3b-pt-224), then download the cache:

```bash
hf download google/paligemma-3b-pt-224 --cache-dir "$HF_HOME/hub"
```

## 5. Assets and local configuration

Prepare simulator resources using [Assets](assets.md).
Check the workspace, checkpoint, and simulator interpreter paths in [Environment setup](README.md).
Environment variables do not rewrite absolute paths stored in YAML files.

## 6. Train / Eval

Use [Commands](../docs/commands-omegavla-pi05.md) after setup. Run suites in separate terminals for parallel evaluation.
For serial evaluation, run the suites sequentially on one GPU. Use distinct outputs for each method, seed, and suite.
PIVOT-Q and Full Distill evaluation require the corresponding trained adapter or checkpoint.

## Build a local pack

### Prepare π₀.₅ calibration observations

Both π₀.₅ integrations use 32 unlabeled observations from the official `lerobot/libero` dataset.
The script samples without replacement using seed 0 and saves the dataset indices.
LeRobot downloads the required dataset on first use. Allocate sufficient disk space.

```bash
conda activate lerobot_pi05
cd "$PIVOT_Q_DIR"
python pi05_quantvla/scripts/prepare_calibration_buffer.py \
  --count 32 \
  --seed 0 \
  --output pi05_quantvla/data/calibration/libero_clean_32_seed-000.pt
```

### Build W4A4

```bash
CUDA_VISIBLE_DEVICES=0 python pi05_omegavla/scripts/calibrate.py \
  --config pi05_omegavla/config/omega_original.yaml \
  --buffer pi05_quantvla/data/calibration/libero_clean_32_seed-000.pt \
  --device cuda
```

Append `--resume` after an interrupted calibration. Omit `--max-layers`
to build the complete pack. Outputs are saved under `pi05_omegavla/models/omega/`;
the manifest and calibration plan record layer coverage and input metadata.
This pack targets LeRobot π₀.₅ and is not interchangeable with an OpenPI pack.
