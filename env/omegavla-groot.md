# omegavla-groot: Environment and Model Setup

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

Use [Commands](../docs/commands-omegavla-groot.md) after setup. Run suites in separate terminals for parallel evaluation.
For serial evaluation, run the suites sequentially on one GPU. Use distinct outputs for each method, seed, and suite.
PIVOT-Q and Full Distill evaluation require the corresponding trained adapter or checkpoint.

## Build a local pack

Prepare the official checkpoints, LIBERO assets, and environment before running the Omega-QVLA builders.
Select Omega-QVLA through PYTHONPATH; do not install a second gr00t package into groot_test.

```bash
conda activate groot_test
cd "$PIVOT_Q_ROOT/third_party/Omega-QVLA"
export QUANTVLA_ROOT="$PIVOT_Q_ROOT/third_party/Omega-QVLA"
export PYTHONPATH="$QUANTVLA_ROOT:$PIVOT_Q_ROOT/third_party/LIBERO"
```

Set LIBERO_CONFIG_PATH to a valid clean-LIBERO configuration.
The builders collect clean simulation observations. The per-step pack below uses
4 denoising steps, matching the integration configuration. Rebuild it if the inference step count changes.

Pick a suite:
```bash
SUITE=object                         # object | spatial | goal | long
CKPT=$PIVOT_Q_DIR/models/gr00t-n1.5-libero-${SUITE}-posttrain
case "$SUITE" in
    goal) TASK=libero_goal;    DCFG=examples.Libero.custom_data_config:LiberoDataConfigMeanStd ;;
    long) TASK=libero_10;      DCFG=examples.Libero.custom_data_config:LiberoDataConfig ;;
    *)    TASK=libero_${SUITE}; DCFG=examples.Libero.custom_data_config:LiberoDataConfig ;;
esac
LLM_RE='.*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'
DIT_RE='.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*'
```

### 3.1 Build the LLM pack — DuQuant svd_hadamard + GPTQ

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$QUANTVLA_ROOT:$PIVOT_Q_ROOT/third_party/LIBERO" python -m tools.build_gptq_weights \
    --checkpoint "$CKPT" --task-suite-name "$TASK" --data-config "$DCFG" \
    --output-path results/packs/${SUITE}_LLM/quantized.pt \
    --include-regex "$LLM_RE" --exclude-regex "$EXCLUDE" \
    --duquant-rotation --duquant-rot-mode svd_hadamard \
    --weight-bits 4 --num-samples 10 --token-cap 1024 \
    --gptq-block-size 128 --gptq-damp-percent 0.05
```

### 3.2 Build the DiT pack — DuQuant svd_hadamard + RTN residual + per-step

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$QUANTVLA_ROOT:$PIVOT_Q_ROOT/third_party/LIBERO" python -m tools.build_dit_a2lite_svd_gptq_perstep \
    --checkpoint "$CKPT" --task-suite-name "$TASK" --data-config "$DCFG" \
    --output-path results/packs/${SUITE}_DiT/quantized.pt \
    --num-samples 10 --token-cap 1024 --num-steps 4 \
    --svd-rank 0 --use-rtn \
    --w-bits 4 --a-bits 4 --act-percentile 99.9 \
    --duquant-block-size 64 --duquant-block-out 64 \
    --gptq-block-size 128 --gptq-damp-percent 0.05
```
(The DiT builder auto-targets `transformer_blocks.*.attn1` + `ff.net` and always
uses `svd_hadamard`; `--use-rtn` = RTN residual, `--num-steps 4` = per-step table.)

### 3.3 Merge the two packs (runtime loads one file)

```bash
python -m tools.merge_packs \
    --out $PIVOT_Q_DIR/omega_qvla/models/gr00t_${SUITE}/quantized.pt \
    results/packs/${SUITE}_LLM/quantized.pt \
    results/packs/${SUITE}_DiT/quantized.pt
```


The merged pack is saved to `PIVOT-Q/omega_qvla/models/gr00t_<suite>/quantized.pt`.
Build each suite separately. The `long` model corresponds to `libero_10`.

