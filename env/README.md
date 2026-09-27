# Installation

## Root directory and dependencies

Use Linux with a compatible NVIDIA driver/CUDA runtime and Bash. This release
does not include weights, quantization packs, dataset assets, outputs, or caches.

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_ROOT"
git submodule update --init --recursive
python -m pip install PyYAML
python env/configure_paths.py "$PIVOT_Q_ROOT"
python env/configure_paths.py "$PIVOT_Q_ROOT" --apply
source "$PIVOT_Q_ROOT/env/runtime/paths.sh"
```

QuantVLA is already vendored; do not clone another copy. The other five
repositories are official submodules pinned to exact commits. The configuration
command checks these commits and links the two LIBERO compatibility files as
described in [upstream patches](../upstream_patches/README.md).
Original upstream authors and licenses are retained.

The generated environment disables Python user-site packages so that packages
under `~/.local` do not override the selected Conda environment. Source
`env/runtime/paths.sh` in every new terminal before launching experiments.
Place LIBERO-Plus assets inside its checkout as documented below: some simulator
objects access that directory directly, even when a separate assets path is set.

## Sibling repositories

Older installation instructions used separate sibling checkouts. In this release,
all external source roots are under `third_party/`; the generated environment
variables point there. Do not clone dependencies beside the project.

## Python environments

Install the matching CUDA PyTorch wheels first, then the direct requirements,
then the editable source. Do not install unrelated upstream `dev`/`base`
extras. Create only the environments required by your experiments.

```bash
cd "$PIVOT_Q_DIR"

conda create -y -n groot_test python=3.10
conda activate groot_test
python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r env/requirements/groot_test.txt
python -m pip install packaging ninja wheel setuptools
python -m pip install flash-attn==2.7.1.post4 --no-build-isolation
python -m pip install --no-deps -e third_party/QuantVLA

# Keep exactly one cv2 provider after all dependency installation.
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --no-deps opencv-python-headless==4.11.0.86

conda create -y -n libero_test python=3.10
conda activate libero_test
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r env/requirements/libero_test.txt
python -m pip install --no-deps -e third_party/LIBERO --config-settings editable_mode=compat
python -m pip install --no-deps -e third_party/QuantVLA

python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --no-deps opencv-python-headless==4.11.0.86

conda create -y -n lerobot_pi05 python=3.12
conda activate lerobot_pi05
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r env/requirements/lerobot_pi05.txt
python -m pip install --no-deps -e third_party/lerobot
python -m pip install --no-deps -e third_party/LIBERO-plus --config-settings editable_mode=compat

python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --no-deps opencv-python-headless==4.13.0.92
python -c 'import cv2; print(cv2.__version__, cv2.__file__)'
```

`libero_test` uses standard LIBERO for the original-environment anchor;
`lerobot_pi05` uses LIBERO-Plus. They both expose a package named `libero`, so
do not install them together in one environment. The official LIBERO-Plus
setup also lists `libexpat1`, `libfontconfig1-dev`, `libpython3-stdlib`, and
`libmagickwand-dev` as system dependencies; ask your administrator to install
them when missing. Its Python additions `Wand` and `scikit-image` are in the
`lerobot_pi05` requirements.

Robosuite declares the distribution name `opencv-python`, whereas the headless
package supplies the same `cv2` import used by these offscreen experiments.
Dependency installation can therefore install both distributions. The final
uninstall/reinstall commands above remove overlapping files and install one
headless provider. Repeat that cleanup after installing additional packages
that pull in `opencv-python`. `pip check` can still report Robosuite's declared
`opencv-python` requirement as missing; this specific metadata warning is
expected with the headless substitution. Do not ignore unrelated dependency
errors, and verify `import cv2` and simulator startup. These instructions are
for the three environments above, not an automatic change to an existing one.


## OpenVLA-OFT + QVLA environment and checkpoints

```bash
conda create -y -n oft_qvla python=3.10 pip
conda activate oft_qvla
cd "$PIVOT_Q_ROOT"
python -m pip install -r env/requirements/oft_qvla.txt
python openvla-oft-qvla/scripts/setup_upstreams.py
python -m pip check
```

The QVLA integration imports the pinned official `third_party/QVLA/openvla-oft`
source. Do not replace its pinned Transformers fork with ordinary Transformers.
Detailed checkpoint and calibration instructions are in the setting guides below.
Install only the environments required for your settings.

## LIBERO-Plus benchmark assets

The [official LIBERO-Plus README](https://github.com/sylvestf/LIBERO-plus#installation)
links to its [assets page](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main)
and directs users to download `assets.zip`. The server's copy of this archive
has several parent directories *inside* the ZIP, so unzipping directly into
the repository does **not** put the resources at the required location. Stage
the extraction on a filesystem with enough space, locate its `assets/`
directory, and copy its contents into the checkout:

```bash
# Download assets.zip from the official LIBERO-Plus assets page if it is not
# already present. If the dataset requires authentication, run `huggingface-cli
# login` first or copy the downloaded archive to this path.
export ASSET_ZIP="$PIVOT_Q_ROOT/assets.zip"
if [ ! -s "$ASSET_ZIP" ]; then
  curl -L --fail --retry 3 \
    -o "$ASSET_ZIP" \
    https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
fi
ASSET_STAGE=$(mktemp -d "$PIVOT_Q_ROOT/libero-plus-assets.XXXXXX")
mkdir -p "$ASSET_STAGE" "$LIBERO_PLUS_DIR/libero/libero/assets"
unzip -q "$ASSET_ZIP" -d "$ASSET_STAGE"
ASSET_SOURCE=$(find "$ASSET_STAGE" -type d -name assets -print -quit)
test -n "$ASSET_SOURCE"
test -d "$ASSET_SOURCE/new_objects"
cp -a "$ASSET_SOURCE/." "$LIBERO_PLUS_DIR/libero/libero/assets/"
test -d "$LIBERO_PLUS_DIR/libero/libero/assets/textures"
test -d "$LIBERO_PLUS_DIR/libero/libero/assets/scenes"
rm -rf "$ASSET_STAGE"
```

The target `assets/` directory must contain `new_objects/`, `textures/`,
`scenes/`, and the other resources shown in the official README. Update
`$QUANTVLA_DIR/configs/libero_plus/config.yaml` so its `assets`,
`bddl_files`, `benchmark_root`, `datasets`, and `init_states` entries point to
`$LIBERO_PLUS_DIR`. Experiment launchers set
`LIBERO_CONFIG_PATH` to this config directory; the default
`~/.libero/config.yaml` can point to a different installation.


## Models, quantization, and calibration

| Setting | Model environment | Downloads and pack preparation |
|---|---|---|
| GR00T-N1.5 + QuantVLA | groot_test | [Setup](quantvla-groot.md) |
| π₀.₅ + QuantVLA | lerobot_pi05 | [Setup](quantvla-pi05.md) |
| GR00T-N1.5 + HoloQ-VLA | groot_test | [Setup](omegavla-groot.md) |
| π₀.₅ + HoloQ-VLA | lerobot_pi05 | [Setup](omegavla-pi05.md) |
| OpenVLA-OFT + QVLA | oft_qvla | [Setup](qvla-openvla-oft.md) |

Use the documented destinations. Root configuration does not download models or
assets and does not generate quantization packs. Recovery evaluation additionally
requires the adapter produced by the matching training run.
Follow the official model/data access terms and pin revisions when reproducing
an experiment. Credentials must never be saved in this repository.

## Hugging Face PaliGemma tokenizer/model cache

The π₀.₅ launchers use `$PIVOT_Q_ROOT/.cache/huggingface`.
Accept the official PaliGemma access terms, authenticate with your own account,
and prepare its processor/model cache before using offline launchers:

```bash
conda activate lerobot_pi05
export HF_HOME="$PIVOT_Q_ROOT/.cache/huggingface"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -c 'from getpass import getpass; from huggingface_hub import login; login(token=getpass("Hugging Face token: "), add_to_git_credential=False)'
hf download google/paligemma-3b-pt-224 --cache-dir "$HF_HOME/hub"
```

Do not publicly redistribute gated assets or authentication files without the
required rights. The quantization pack is not a replacement for this cache.

## Configure a new workspace

After moving the project or installing Conda, rerun:

```bash
python "$PIVOT_Q_ROOT/env/configure_paths.py" "$PIVOT_Q_ROOT" --apply
source "$PIVOT_Q_ROOT/env/runtime/paths.sh"
```

The tool updates root prefixes in YAML, generates independent simulator configs,
and creates relative upstream compatibility links. It does not touch
`~/.libero/config.yaml`, weights, results, or experiment hyperparameters.
Use `--setting quantvla-pi05` (repeatable) to limit model configuration updates.
Use `--paths-only` only to prepare paths before initializing dependencies;
a later full configuration is still required to verify/apply the patches.
Conda paths are discovered automatically when Conda is available. Activate the
model environment for training/inference; simulator subprocesses discover
`libero_test` by name. Source the runtime file in each new terminal.

## Manifest and verification

The fixed Shared-560 manifest is included. To regenerate it from the same task
definition after dependency setup:

```bash
conda activate groot_test
cd "$PIVOT_Q_ROOT"
python scripts/build_manifest.py --config config/eval.yaml
python -c 'import torch, gr00t; print(torch.__version__, gr00t.__file__)'
python -m pip check
conda activate libero_test
python -c 'from libero.libero import get_libero_path; print(get_libero_path)'
conda activate lerobot_pi05
python -c 'import lerobot; print(lerobot.__file__)'
```

Verify that imports resolve to this checkout's `third_party/` repositories;
reinstall editable packages after relocation if necessary. These requirements
are reference package versions, not a container-level lockfile. See
[commands](../docs/commands.md) and [protocol](../docs/experiment_protocol.md).
