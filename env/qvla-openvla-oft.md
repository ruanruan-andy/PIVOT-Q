# qvla-openvla-oft: Environment and Model Setup

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

Install `oft_qvla` and `libero_test` using [Environment setup](README.md).

## 4. Hugging Face authentication and checkpoints

Create a read token in [Hugging Face Settings](https://huggingface.co/settings/tokens).

```bash
export HF_HOME="$PIVOT_Q_DIR/.cache/huggingface"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
read -r -s -p "Hugging Face access token: " HF_ACCESS_TOKEN
printf '\n'
huggingface-cli login --token "$HF_ACCESS_TOKEN"
unset HF_ACCESS_TOKEN
huggingface-cli whoami
```

### spatial

[Official download](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)

```bash
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
```

### object

[Official download](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)

```bash
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-object \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-object"
```

### goal

[Official download](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal)

```bash
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-goal \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-goal"
```

### 10

[Official download](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10)

```bash
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-10 \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-10"
```

Download the complete checkpoint, including the action head, proprio projector, processor, tokenizer, and normalization statistics.

## 5. Assets and local configuration

Prepare simulator resources using [Assets](assets.md).
Check the workspace, checkpoint, and simulator interpreter paths in [Environment setup](README.md).
Environment variables do not rewrite absolute paths stored in YAML files.

## 6. Train / Eval

Use [Commands](../docs/commands-qvla-openvla-oft.md) after setup. Run suites in separate terminals for parallel evaluation.
For serial evaluation, run the suites sequentially on one GPU. Use distinct outputs for each method, seed, and suite.
PIVOT-Q and Full Distill evaluation require the corresponding trained adapter or checkpoint.

## Build a local pack

This integration produces QVLA gates rather than GR00T or π₀.₅ packs.
Start the matching clean-LIBERO service using [OFT calibration commands](../openvla-oft-qvla/commands.md).
Then run collect-calibration and prepare-qvla separately for each suite.
That page specifies service ports and output configuration. Download FP checkpoints from the official sources above.

Calibration uses the official QVLA Hessian proxy and weight fake-quant injector
with an integration-local marginal-cost allocator. The default prohibits 0-bit
channel pruning. Generate gates with the current pipeline; legacy gates and
adapters trained on them are incompatible with the corrected setup.
