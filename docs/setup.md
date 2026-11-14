# Environment and Assets Setup

Run all commands from the project root. Use Linux, Conda and an NVIDIA driver compatible with the selected PyTorch CUDA wheels.

## Third-party source

```bash
git submodule update --init --recursive
```

Git records the revisions of all six third-party submodules. QuantVLA uses the PIVOT-Q fork.

## Conda environments

Create `lerobot_pi05` for LeRobot evaluation and `libero_test` for simulator services. Add the model environment for your setting. The requirements files include the model, evaluation and LIBERO-Plus runtime dependencies. Install PyTorch, ImageMagick and upstream source packages separately.

### `groot_test`

```bash
conda create -n groot_test python=3.10 pip -y
conda activate groot_test
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/groot_test.txt
python -m pip install --no-deps -e ./third_party/QuantVLA
PYTORCH3D_NO_EXTENSION=1 python -m pip install --no-build-isolation --no-deps \
  https://codeload.github.com/facebookresearch/pytorch3d/zip/75ebeeaea0908c5527e7b1e305fbc7681382db47
python -m pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
python -c 'import torch; from flash_attn import flash_attn_func; q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16); print(flash_attn_func(q, q, q).shape)'
```

The CUDA 12.8 wheels support both RTX 4090 and RTX 5090. The FlashAttention wheel above targets Python 3.10 and the PyTorch 2.7 C++11 ABI. PyTorch3D is installed without its C++/CUDA extension because GR00T uses its Python transforms. QuantVLA metadata retains older W&B, protobuf and typing pins; this environment uses the versions in `requirements/groot_test.txt` for current W&B keys.

### `lerobot_pi05`

The requirements pin MuJoCo 3.3.7 and robosuite 1.4.0 for LIBERO evaluation. See [LeRobot #4390](https://github.com/huggingface/lerobot/issues/4390) for the simulator version's effect on Spatial initial states.

```bash
conda create -n lerobot_pi05 python=3.12 pip -y
conda activate lerobot_pi05
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/lerobot_pi05.txt
python -m pip install --no-deps -e ./third_party/lerobot
conda install -y -c conda-forge imagemagick
export MAGICK_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'from lerobot.scripts.lerobot_eval import eval_policy; print("LeRobot evaluation import OK")'
python -m pip check
```

### `oft_qvla`

```bash
conda create -n oft_qvla python=3.10 pip -y
conda activate oft_qvla
python -m pip install torch==2.2.0 torchvision==0.17.0
python -m pip install -r requirements/oft_qvla.txt
python -m pip install 'git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab'
python -m pip install 'git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46'
python -m pip install --no-deps -e ./third_party/QVLA/openvla-oft
python -m pip check
```

### `libero_test`

```bash
conda create -n libero_test python=3.10 pip -y
conda activate libero_test
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/libero_test.txt
python -m pip install --no-deps -e ./third_party/LIBERO
conda install -y -c conda-forge imagemagick
export MAGICK_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip check
```


### Headless rendering

For LIBERO and LIBERO-Plus on headless NVIDIA servers, use the following environment before training or evaluation.

```bash
export MUJOCO_GL=egl
export QT_QPA_PLATFORM=offscreen
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

The EGL vendor path is provided by the NVIDIA driver. The torch setting loads the trusted LIBERO initial-state files. LIBERO-Plus also needs `bddl`, legacy `gym`, `wand`, `scikit-image` and ImageMagick in the interpreter that runs the environment; `gymnasium` does not replace legacy `gym`. PyAV is pinned to 16.0.1 for the checked-out LeRobot API.

## Download tools

Use an environment with the Hugging Face CLI. Install the ModelScope CLI in a separate download environment if needed.

```bash
conda create -n pivot_q_download python=3.12 pip -y
conda activate pivot_q_download
python -m pip install huggingface_hub modelscope-hub
```

## LIBERO-Plus assets

```bash
hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir third_party/LIBERO-plus
unzip third_party/LIBERO-plus/assets.zip -d third_party/LIBERO-plus/libero/libero
```

Both benchmarks use the LeRobot evaluation entry. Keep the original LIBERO and LIBERO-Plus source and assets available.

## Original checkpoints

Both π₀.₅ settings use the same LIBERO-finetuned v044 checkpoint as CARE-VLA. Cache the tokenizer separately for offline runs. Accept the PaliGemma access terms and run `hf auth login` if prompted.

```bash
hf download lerobot/pi05_libero_finetuned_v044 --local-dir models/original/pi05_libero
hf download google/paligemma-3b-pt-224 tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model
```

### Offline Hugging Face access

After downloading the checkpoint and caching the PaliGemma tokenizer, set these variables in the shell that runs training or evaluation:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Use the same Hugging Face cache for downloads and offline runs. Unset these variables before downloading missing files.

### GR00T-N1.5

```bash
hf download youliangtan/gr00t-n1.5-libero-object-posttrain --local-dir models/original/gr00t_n1_5/libero_object
hf download youliangtan/gr00t-n1.5-libero-spatial-posttrain --local-dir models/original/gr00t_n1_5/libero_spatial
hf download youliangtan/gr00t-n1.5-libero-goal-posttrain --local-dir models/original/gr00t_n1_5/libero_goal
hf download youliangtan/gr00t-n1.5-libero-long-posttrain --local-dir models/original/gr00t_n1_5/libero_10
```

### OpenVLA-OFT

```bash
hf download moojink/openvla-7b-oft-finetuned-libero-object --local-dir models/original/openvla_oft/libero_object
hf download moojink/openvla-7b-oft-finetuned-libero-spatial --local-dir models/original/openvla_oft/libero_spatial
hf download moojink/openvla-7b-oft-finetuned-libero-goal --local-dir models/original/openvla_oft/libero_goal
hf download moojink/openvla-7b-oft-finetuned-libero-10 --local-dir models/original/openvla_oft/libero_10
```


## Initialize paths

```bash
conda activate lerobot_pi05
python scripts/relocate_paths.py
```

Use `--dry-run` to preview changes or `--project-root /absolute/path/to/PIVOT-Q-4090` from another directory. Both `lerobot_pi05` and `libero_test` must exist for interpreter discovery.

## Optional: Download prebuilt quantization packs

Browse the available packs in the [PIVOT-Q repository on ModelScope](https://www.modelscope.cn/models/ruanandy/PIVOT-Q/files).

Run from the project root. Each command preserves the matching `models/quantized/<setting>/` layout. Run only the command for the setting you need and check the ModelScope repository for availability. The GR00T HoloQ pack is pending completion on PS. Keep all QVLA cache files, including its per-suite `calibration.pt`, `proxy.pt` and `gates.json`.

```bash
ms-hub login

ms-hub download ruanandy/PIVOT-Q --local-dir "$PWD" --include 'models/quantized/groot_quantvla/**'
ms-hub download ruanandy/PIVOT-Q --local-dir "$PWD" --include 'models/quantized/groot_holoq/**'
ms-hub download ruanandy/PIVOT-Q --local-dir "$PWD" --include 'models/quantized/pi05_quantvla/**'
ms-hub download ruanandy/PIVOT-Q --local-dir "$PWD" --include 'models/quantized/pi05_holoq/**'
ms-hub download ruanandy/PIVOT-Q --local-dir "$PWD" --include 'models/quantized/openvla_oft_qvla/**'
```


## Measuring training cost

Training and evaluation record time and call counts with FLOPs tracing off
by default. GR00T/OFT write a `compute.json` per suite. π0.5 keeps the
suite-tagged counts under `compute` in `train_summary.json`.

Profile a single stage and batch size in a separate short run, using the
same setting, method, suite and model environment as training:

```bash
python scripts/profile_flops.py sample \
  --config configs/groot_quantvla.yaml --method full_distill \
  --suite libero_spatial --stage distill_forward_backward --batch-size 16 \
  --output outputs/flops/groot_quantvla/full_distill/spatial-update16.json
```

Repeat for the stage/batch-size keys that appear in the formal run's
`calls`, except `optimizer_step`. For example, profile a 12-state update
separately with `--batch-size 12`. Rollout and replay generally use batch
size 1. Use the exact stage names in the log, including the `anchor/` prefix
where present. Profile each suite separately. For multiple representative
rollouts in a suite, save separate samples using different `--seed` values.
The same commands apply to the five setting YAMLs.

```bash
python scripts/profile_flops.py summarize \
  --run outputs/groot_quantvla/full_distill/train/seed/0/libero_*/compute.json \
  --samples outputs/flops/groot_quantvla/full_distill/*.json \
  --output outputs/flops/groot_quantvla/full_distill_summary.json
```

For π0.5, pass its shared `train_summary.json` to `--run`. Each estimate
multiplies the mean FLOPs per call by the logged number of calls for that
suite, stage and batch size. The summary then sums the suite estimates and
lists any missing samples. Estimates cover supported PyTorch operators and
exclude simulator work, unsupported kernels and the optimizer. Report
unprofiled wall time and peak memory separately from estimated FLOPs.

