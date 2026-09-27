# π0.5 QuantVLA + PIVOT-Q command reference

For one GPU, use the [single-GPU command guide](../../docs/commands-pi05-single-gpu.md).

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

PIVOT-Q is the method name. `PIVOT-Q/`, `pivot_q`, `pivot_q`, and existing
checkpoint/output paths are historical implementation identifiers and stay
unchanged. Do not rename existing result directories or adapter metadata.


π0.5 uses one unified `pi05_libero_finetuned` policy for all four LIBERO suites.
The main PIVOT-Q protocol therefore trains one shared adapter with four synchronous
ranks and evaluates that same adapter on every suite.

Run commands from the repository root:

```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05
```

## Distributed layout

| Rank | Physical GPU | Suite | Rollout port | Clean-anchor port |
| :--: | :--: | :-- | --: | --: |
| 0 | 4 | `libero_spatial` | 6100 | 6101 |
| 1 | 5 | `libero_object` | 6110 | 6111 |
| 2 | 6 | `libero_goal` | 6120 | 6121 |
| 3 | 7 | `libero_10` | 6130 | 6131 |

All ranks load the same FP16 teacher, W4A8 student, and LoRA initialization.
Each rank collects one suite, then trainable LoRA gradients are averaged before
every optimizer update. With 140 episodes per rank and 5 updates per episode,
the shared adapter receives 560 rollout episodes and 700 synchronized updates.

## Arguments and outputs

- `--gpus 4 5 6 7`: physical devices exposed to the four local ranks, in table order.
- `--ports` / `--clean-ports`: optional overrides for the eight service ports.
- `--master-port`: optional torch distributed rendezvous port; YAML default is `6140`.
- `--seed`: training seed; YAML default is `0`.
- `--adapter-path`: shared adapter used by evaluation.
- `--output-dir`: exact suite output, or the common parent when `--suite all`.
- `--resume`: skip already completed evaluation episodes.

Training writes only to:

```text
outputs/quantvla/pi05/pivot_q/train/seed-000/shared/
```

The directory contains aggregate `metrics.jsonl`, per-suite files under
`metrics/`, per-suite status under `status/`, the newest complete checkpoint,
and one `final_adapter`. The checkpoint stores the shared adapter/optimizer plus
the RNG, task order, and anchor replay state of all four ranks. A new checkpoint
is completed atomically before the previous one is removed. With
`training.resume: true`, only a checkpoint marked complete is restored.

## Checks

Configuration and command expansion without starting services:

```bash
python3 pi05_quantvla/pivot_q/train_ddp.py --dry-run
python3 pi05_quantvla/pivot_q/run_train_ddp.py --dry-run
```

Optional four-GPU NCCL all-reduce check:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=4 --master-port 6140 \
  pi05_quantvla/pivot_q/train_ddp.py --collective-smoke-test
```

The expected sum is `10.0` (`1 + 2 + 3 + 4`).

## Train one shared adapter

Foreground:

```bash
python3 pi05_quantvla/pivot_q/run_train_ddp.py --gpus 4 5 6 7
```

Recommended background launch:

```bash
tmux new-session -d -s pi05_pivot_q_shared \
  "bash -c 'source "$CONDA_SH" && \
  conda activate lerobot_pi05 && \
  cd $PIVOT_Q_DIR && \
  set -o pipefail && \
  python3 pi05_quantvla/pivot_q/run_train_ddp.py --gpus 4 5 6 7 \
  2>&1 | tee outputs/quantvla/pi05/pivot_q/shared_launcher.log'"
```

Attach with `tmux attach -t pi05_pivot_q_shared`; detach with `Ctrl-b d`.
Do not launch a second job against the same `shared` output directory.

## Evaluate the shared adapter

One server can evaluate all suites sequentially:

```bash
python3 pi05_quantvla/scripts/run_eval.py \
  --method pivot-q --suite all --gpu 4 --port 6200 \
  --config pi05_quantvla/config/quantvla.yaml \
  --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter \
  --output-dir outputs/quantvla/pi05/pivot_q/eval/seed-2026 \
  --resume
```

To evaluate concurrently on four GPUs, every command still loads the same adapter:

```bash
CUDA_VISIBLE_DEVICES=4 python3 pi05_quantvla/scripts/run_eval.py --method pivot-q --suite libero_spatial --gpu 4 --port 6200 --config pi05_quantvla/config/quantvla.yaml --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter --output-dir outputs/quantvla/pi05/pivot_q/eval/seed-2026/libero_spatial --resume

CUDA_VISIBLE_DEVICES=5 python3 pi05_quantvla/scripts/run_eval.py --method pivot-q --suite libero_object  --gpu 5 --port 6201 --config pi05_quantvla/config/quantvla.yaml --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter --output-dir outputs/quantvla/pi05/pivot_q/eval/seed-2026/libero_object  --resume

CUDA_VISIBLE_DEVICES=6 python3 pi05_quantvla/scripts/run_eval.py --method pivot-q --suite libero_goal    --gpu 6 --port 6202 --config pi05_quantvla/config/quantvla.yaml --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter --output-dir outputs/quantvla/pi05/pivot_q/eval/seed-2026/libero_goal    --resume

CUDA_VISIBLE_DEVICES=7 python3 pi05_quantvla/scripts/run_eval.py --method pivot-q --suite libero_10      --gpu 7 --port 6203 --config pi05_quantvla/config/quantvla.yaml --adapter-path outputs/quantvla/pi05/pivot_q/train/seed-000/shared/final_adapter --output-dir outputs/quantvla/pi05/pivot_q/eval/seed-2026/libero_10      --resume
```

Add `--save-video` only when needed; videos are large and excluded from Git.

## Monitor

```bash
python3 -m monitor.app --method pi05_pivot_q --watch 5
python3 -m monitor.app --method pi05_pivot_q --once
```

## Network behavior

The launcher points Hugging Face to `PIVOT-Q/.cache/huggingface` and enables
offline mode. On a server sharing the same workspace cache, no proxy is needed.
If the cache must be populated on a new filesystem, activate the proxy once in
an interactive shell, download the missing artifact, then relaunch normally:

```bash
clashon
```

## Full Distill paper baseline

This baseline keeps the four-rank shared-adapter training protocol. Each rank
distills all `T` valid states of its own student-controlled rollout with
normalized weight `1/T`, while the original teacher, quantized checkpoint,
Behavioral Anchor, five updates per rollout, optimizer, and seed remain the
same. The separate YAML uses ports 6300–6340 and writes only under
`outputs/quantvla/pi05/full_distill/`.

Check the command expansion without launching services:

```bash
python3 pi05_quantvla/pivot_q/run_train_ddp.py --config pi05_quantvla/pivot_q/config/full_distill.yaml --gpus 4 5 6 7 --dry-run
```

Train one shared adapter on four GPUs:

```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05
python3 pi05_quantvla/pivot_q/run_train_ddp.py --config pi05_quantvla/pivot_q/config/full_distill.yaml --gpus 4 5 6 7
```

After `shared/final_adapter` exists, evaluate the same adapter across all four
suites with the current evaluator's generic adapter-loading mode:

```bash
python3 pi05_quantvla/scripts/run_eval.py \
  --method pivot-q --suite all --gpu 4 --port 6600 \
  --config pi05_quantvla/config/quantvla.yaml \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter \
  --output-dir outputs/quantvla/pi05/full_distill/eval/seed-2026 --resume
```

`--method pivot-q` names an existing inference adapter path, not this experiment;
the method metadata and output directory identify `pi05_full_distill`.

### Eight-GPU paired Full Distill

The paired launcher uses one shared adapter across eight GPUs. Four GPU pairs
collect four student-controlled rollouts concurrently within the same suite;
each pair splits the valid states of its rollout across its two GPUs. Suites
are visited sequentially in the order LIBERO-10, Spatial, Object, Goal. This
retains 140 rollouts per suite, 560 rollouts total, and five synchronized
updates per four-rollout round (700 updates total). The teacher, QuantVLA
checkpoint, adapter settings, optimizer, Behavioral Anchor, and seed match the
four-GPU Full Distill baseline. No network proxy is needed when the local
checkpoint and Hugging Face tokenizer cache are populated. Results go to the
canonical `outputs/quantvla/pi05/full_distill/` tree. The four-GPU and Paired8
launchers share this destination: choose one launcher and do not run both
against the same directory concurrently.

Check configuration and eight-GPU collectives before a full run:

```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05
python3 pi05_quantvla/pivot_q/run_train_paired8.py --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=8 --master-port=6541 \
  pi05_quantvla/pivot_q/train_paired8.py --collective-smoke-test
```

Train on eight idle GPUs:

```bash
python3 pi05_quantvla/pivot_q/run_train_paired8.py --gpus 0 1 2 3 4 5 6 7
```

The launcher starts and stops four LIBERO-Plus and four clean-anchor
environment services. It resumes from the latest complete checkpoint in its
own output directory. Progress is recorded in
`outputs/quantvla/pi05/full_distill/train/seed-000/shared/status.json`
and `metrics.jsonl`. Monitor the status file with:

```bash
watch -n 30 'cat outputs/quantvla/pi05/full_distill/train/seed-000/shared/status.json'
```

After training completes, evaluate the same shared adapter with the existing
evaluator:

```bash
python3 pi05_quantvla/scripts/run_eval.py \
  --method pivot-q --suite all --gpu 0 --port 6600 \
  --config pi05_quantvla/config/quantvla.yaml \
  --adapter-path outputs/quantvla/pi05/full_distill/train/seed-000/shared/final_adapter \
  --output-dir outputs/quantvla/pi05/full_distill/eval/seed-2026 --resume
```

The evaluator currently uses `--method pivot-q` as its generic adapter-loading
branch; it does not change the Full Distill selection strategy or run name.
