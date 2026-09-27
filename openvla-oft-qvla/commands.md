# Commands

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

[Module README](README.md) · [Environment setup](../env/README.md#openvla-oft--qvla-environment-and-checkpoints) · [Main command reference](../docs/commands.md#openvla-oft--qvla)

Run on the server holding the shared workspace. Set the root once above; paths below derive from it. The model environment is `oft_qvla`; simulator processes reuse
`libero_test`. GPU IDs are examples; select available cards. Each terminal
must activate its own environment. Use Bash for the shell blocks below; do not
source `.bashrc` from zsh.

For four-suite **parallel** train/eval and **serial** single-GPU alternatives,
see [core commands](../docs/commands-qvla-openvla-oft.md).
This page details downloads, simulator startup and per-suite commands. Each OFT
suite requires its own matching Plus/clean ports; `--suite all` is not supported.

## 1. Official repository and model environment

Initialize the pinned dependency with `git submodule update --init third_party/QVLA` from the project root.

If the checkout already exists, skip cloning and verify it:

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
python scripts/setup_upstreams.py
```

The script checks the commit and tracked-source cleanliness; it does not reset
an existing checkout. `QVLA/openvla-oft/` already contains OFT source. Do not
clone or install another OFT version into the same environment.

Create the model environment once:

```bash
cd "$PIVOT_Q_DIR" &&
conda create -n oft_qvla python=3.10 pip -y &&
conda activate oft_qvla &&
python -c 'import sys; print(sys.executable)' &&
python -m pip install -r env/requirements/oft_qvla.txt
```

For an existing `oft_qvla`, skip `conda create`. The central requirements file
references this module's `requirements-model.txt`. It pins the dedicated OFT
Transformers fork; ordinary Transformers is not a substitute. Full CUDA
installation and real-model validation are not yet certified by local tests.

## 2. Download four checkpoints

Run each block in its own terminal to download concurrently. Unsetting offline
flags applies only to the current terminal. Hugging Face downloads and GitHub
clones do not use the pip mirror; enable your working shell proxy separately
if these hosts are unreachable.

### Spatial

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla &&
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE &&
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
```

### Object

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla &&
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE &&
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-object \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-object"
```

### Goal

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla &&
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE &&
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-goal \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-goal"
```

### Long

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla &&
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE &&
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-10 \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-10"
```

Downloads include model weights, action head, proprio projector, tokenizer,
processor, and normalization files. Default directories are
`checkpoints/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}/` under
this module. The destination is specified by `--local-dir`. To move a model to another disk, use
`--local-dir /absolute/model/path`, then add
`--set paths.checkpoint=/absolute/model/path` to every calibration, smoke,
training, and evaluation command using that model.

The combined checkpoint is also available upstream, but is not the default
protocol here. Do not substitute it while retaining the suite-specific
normalization/quantization files.

### Interrupted download: IncompleteRead / ChunkedEncodingError

These messages mean an HTTP file stream ended before all expected bytes
arrived. They do not indicate an OFT inference or quantization error. For
example, `Fetching 25 files: 24/25` means 24 files completed, not that 96% of
the total bytes completed. Network, proxy or CDN interruptions can cause this;
the traceback alone does not identify which one.

Keep the checkpoint directory and its `.cache/huggingface` metadata/incomplete
files, then rerun the same command:

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla &&
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-object \
  --local-dir "$PIVOT_Q_DIR/openvla-oft-qvla/checkpoints/openvla-7b-oft-finetuned-libero-object"
```

For an unchanged revision, the pinned Hugging Face downloader reuses completed
files and attempts to resume incomplete transfers. Do not use force-download
or delete the cache just for this error. If it repeats, download one suite at
a time and check that the proxy connection is stable. Never run two download
processes for the same suite/output simultaneously. The transfer behavior is
defined in the [Hugging Face downloader](https://github.com/huggingface/huggingface_hub/blob/v0.25.2/src/huggingface_hub/file_download.py).

## 3. Independent simulator paths and shared manifest

Run this once to write **module-local** simulator configurations. It does not
rewrite the configurations used by existing GR00T/pi0.5 experiments.

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla &&
python - <<'PY'
from pathlib import Path
module = Path.cwd()
workspace = module.parent.parent
for label, repository in (("libero", "LIBERO"), ("libero_plus", "LIBERO-plus")):
    repo = workspace / repository
    base = repo / "libero/libero"
    paths = dict(assets=base / "assets", bddl_files=base / "bddl_files",
                 benchmark_root=base, datasets=repo / "datasets", init_states=base / "init_files")
    for key in ("assets", "bddl_files", "benchmark_root", "init_states"):
        if not paths[key].is_dir():
            raise FileNotFoundError(paths[key])
    target = module / "cache" / label / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{key}: {value}\n" for key, value in paths.items()))
    print(target)
PY
```

Reuse `PIVOT-Q/manifests/libero_plus_first20.json`. If it is absent, build it in
the simulator environment with the Plus source/config explicitly selected:

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate libero_test &&
PYTHONPATH=$PIVOT_Q_ROOT/third_party/LIBERO-plus \
LIBERO_CONFIG_PATH="$PWD/cache/libero_plus" \
python scripts/build_manifest.py --output ../manifests/libero_plus_first20.json
```

The builder refuses to overwrite an existing manifest. Standard benchmark
assets and the existing LIBERO-Plus simulator dependencies must be installed
as described in the central environment guide. Model dependencies are not
installed into the simulator environment.

## 4. Simulator terminals

Each active suite needs a Plus service for evaluation/target rollouts and a
clean service for calibration/anchors. Example for Spatial:

### Terminal A: LIBERO-Plus

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate libero_test &&
PYTHONPATH=$PIVOT_Q_ROOT/third_party/LIBERO-plus \
LIBERO_CONFIG_PATH="$PWD/cache/libero_plus" MUJOCO_GL=egl \
python -m evaluation.env_service --suite libero_spatial \
  --manifest ../manifests/libero_plus_first20.json --port 5590 --seed 0
```

### Terminal B: clean LIBERO

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate libero_test &&
PYTHONPATH=$PIVOT_Q_ROOT/third_party/LIBERO \
LIBERO_CONFIG_PATH="$PWD/cache/libero" MUJOCO_GL=egl \
python -m evaluation.env_service --suite libero_spatial --port 5591 --seed 0
```

Check both terminals report `Ready`. The model launchers do not start these
services automatically. Do not share a running service between concurrent
training/evaluation jobs: one job's reset would affect the other's rollout.

For parallel suites, the following example ports are distinct. Inspect existing
jobs before using them. `--suite` alone does **not** change the default ports.

| Suite | Plus port | Clean port | Model flags |
| --- | ---: | ---: | --- |
| `libero_spatial` | 5590 | 5591 | defaults |
| `libero_object` | 5592 | 5593 | `--set environment.port=5592 --set environment.clean_port=5593` |
| `libero_goal` | 5594 | 5595 | `--set environment.port=5594 --set environment.clean_port=5595` |
| `libero_10` | 5596 | 5597 | `--set environment.port=5596 --set environment.clean_port=5597` |

Change `--suite` and `--port` in both simulator commands to match. To run
PIVOT-Q and Full Distill concurrently, allocate an additional service pair and
separate GPUs; the table gives one pair per suite, not one per method.

## 5. Calibration and real-model smoke test

In the model terminal, select a free physical GPU. Inside the process the
selected GPU is `cuda:0`; there is no additional `--gpu` argument.

```bash
cd "$PIVOT_Q_DIR/openvla-oft-qvla" &&
conda activate oft_qvla
export CUDA_VISIBLE_DEVICES=0

bash scripts/run.sh collect-calibration --suite libero_spatial
bash scripts/run.sh prepare-qvla --suite libero_spatial
python scripts/smoke_test.py --config configs/qvla.yaml --suite libero_spatial
```

Collection requires the clean service and writes 32 clean FP observations.
Quantization computes the official Hessian proxy layer by layer, then generates
per-channel gates with target average 4 bits. Completed proxy layers are
reused on restart if the configuration matches. A saved calibration buffer is
not overwritten; rerun `prepare-qvla`, not `collect-calibration`, to resume.
Repeat for each suite. This is weight-only fake quantization, not W4A4.

The real-model smoke checks official-action parity, adapter gradients and
save/reload. Run it before formal training. Local small-model tests do not
establish real-checkpoint closed-loop correctness or task success rates.

## 6. FP and QVLA evaluation

With the matching Plus service ready, run sequentially:

```bash
bash scripts/run.sh eval-fp --suite libero_spatial
bash scripts/run.sh eval-qvla --suite libero_spatial
```

FP requires only the checkpoint; QVLA additionally requires its gates.
For a two-episode pilot in a separate directory:

```bash
bash scripts/run.sh eval-qvla --suite libero_spatial \
  --set evaluation.max_episodes=2 --set paths.output=outputs/pilot/qvla/libero_spatial
```

## 7. PIVOT-Q and Full Distill training

With both simulator services ready, run one method at a time on shared devices:

```bash
bash scripts/run.sh train-pivot-q --suite libero_spatial
bash scripts/run.sh train-full-distill --suite libero_spatial
```

Both methods start from the same quantized checkpoint and use the same action
head residual LoRA, 140 suite rollouts, five updates per rollout, training seed
0, and clean-anchor coefficient 0.1. PIVOT-Q selects up to 16 phase-balanced
states; Full Distill uses all valid states uniformly. This is matched rollout
and optimizer budget, not equal state-processing cost. Both teacher and student
are resident, so memory must accommodate two OFT policies.

For a two-GPU teacher/student allocation:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run.sh train-pivot-q --suite libero_spatial \
  --set model.device=cuda:0 --set model.teacher_device=cuda:1
```

Keep the same device/configuration flags for resume. Four suites train four
independent adapters per method; there is no shared-adapter DDP launcher here.

## 8. Recovered-policy evaluation

After the corresponding final checkpoint exists:

```bash
bash scripts/run.sh eval-pivot-q --suite libero_spatial \
  --set paths.adapter=../outputs/qvla/openvla_oft/pivot_q/train/seed-000/libero_spatial/checkpoint-000140.pt
bash scripts/run.sh eval-full-distill --suite libero_spatial \
  --set paths.adapter=../outputs/qvla/openvla_oft/full_distill/train/seed-000/libero_spatial/checkpoint-000140.pt
```

All four modes default to evaluation seed 2026 and one action per prediction;
they share task IDs/initial states and normalization. Each suite contains 140
episodes; all four give 560 per policy. Do not mix OFT's protocol with another
integration's action-chunk length or normalization key.

## 9. Resume, output and verification

Explicit training resume from a retained checkpoint:

```bash
bash scripts/run.sh train-pivot-q --suite libero_spatial \
  --set training.resume=../outputs/qvla/openvla_oft/pivot_q/train/seed-000/libero_spatial/checkpoint-000010.pt
```

Defaults save every ten episodes and retain two checkpoints; inspect
`../outputs/qvla/openvla_oft/pivot_q/train/seed-000/libero_spatial/latest.json` for the currently retained path.
Optimizer, scheduler, RNG, task order and clean replay are restored. Other
configuration must match the original run.

| Artifact | Location under this module |
| --- | --- |
| Base model | `checkpoints/openvla-7b-oft-finetuned-libero-<suite>/` |
| Calibration / proxy / gates | `cache/<suite>/` |
| Training metadata and metrics | `outputs/<method>/<suite>/run.json`, `metrics.jsonl` |
| Adapter + trainer checkpoint | `outputs/<method>/<suite>/checkpoint-NNNNNN.pt` |
| Evaluation | `outputs/<method>/<suite>/eval/episodes.jsonl`, `summary.json` |

Evaluation is not resumable; an existing `episodes.jsonl` is rejected. Use
`--set paths.output=outputs/<new-run>/<method>/<suite>` for a repeat evaluation.
Do not use the other modules' `--resume`, `--adapter-path`, or `--method pivot-q`
arguments with these launchers. Training logs can be inspected with `tail -f`;
the existing general monitor is not extended to discover these runs.

```bash
bash scripts/run.sh train-pivot-q --suite libero_spatial --dry-run
bash scripts/run.sh eval-qvla --suite libero_goal --dry-run
python -m pytest tests -q
git -C ../../QVLA status --short
```

Only integration code, configs and documentation are versioned. Models, cache,
simulator path files and outputs remain untracked. Full implementation details
and the current verification limits are in the [module README](README.md).

## Calibration validity

`collect-calibration` balances tasks and initial-state coverage and samples one
seeded timestep per clean teacher rollout. `prepare-qvla` uses marginal proxy
cost per saved bit with a channel-average budget of 4 bits. The default
`quantization.max_zero_fraction=0.0` prohibits 0-bit channel pruning.
Only gates generated by this allocator are accepted for train/eval.
Obsolete calibration, proxy and gates must be regenerated together; adapters
trained with obsolete gates cannot be resumed on the replacement gates.

## Verified eval launcher (automatic simulator lifecycle)

Use `launch_with_services.sh` for every eval. It starts the matching LIBERO-Plus and clean simulator, waits for both services, runs the evaluator, and terminates both simulators when the evaluator exits. The evaluator ports must match the service ports explicitly; `--suite` does not change them. Use the `oft_qvla` Python explicitly.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON=/root/Users/miniconda3/envs/oft_qvla/bin/python \
LIBERO_SERVICE_PYTHON=/root/Users/miniconda3/envs/libero_test/bin/python \
bash scripts/launch_with_services.sh eval \
  --suite libero_spatial --plus-port 8000 --clean-port 8001 \
  --plus-root /path/to/workspace/LIBERO-plus \
  --clean-root /path/to/workspace/LIBERO \
  --manifest /path/to/PIVOT-Q/manifests/libero_plus_first20.json -- \
  bash scripts/run.sh eval-fp --suite libero_spatial \
  --set environment.port=8000 --set environment.clean_port=8001 \
  --set evaluation.seed=2027 \
  --set paths.output=outputs/qvla/openvla_oft/original/eval/seed-2027/libero_spatial
```

Replace `eval-fp` with `eval-qvla`, `eval-full-distill`, or `eval-pivot-q`; update adapter and output paths for adapter methods. A smoke test should add `--set evaluation.max_episodes=2` and use a separate `outputs/pilot_test/...` directory.
