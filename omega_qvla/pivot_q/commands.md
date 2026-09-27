# Ω-QVLA + PIVOT-Q command reference

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
export PIVOT_Q_DIR="$PIVOT_Q_ROOT"
cd "$PIVOT_Q_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
```

PIVOT-Q is the method name. `PIVOT-Q/`, `pivot_q`, `pivot_q`, and existing
checkpoint/output paths are historical implementation identifiers and stay
unchanged. Do not rename existing result directories or adapter metadata.


Run every command from the repository root:

```bash
cd "$PIVOT_Q_DIR"
conda activate groot_test
```

## Arguments and outputs

- `--suite`: one of `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10`.
- `--gpu`: physical GPU used by the policy process. Assign one suite per GPU.
- `--port`: isolated LIBERO-Plus rollout port during training, or policy-server port during evaluation.
- `--clean-port`: isolated clean-LIBERO anchor port; it must differ from `--port`.
- `--seed`: training seed; omitted below because `config/pivot_q.yaml` fixes it to `0`.
- `--adapter-path`: completed training adapter, normally the `final_adapter` symlink.
- `--output-dir`: exact evaluation directory for one suite.
- `--resume`: skip completed evaluation episodes in the same output directory.

Training writes to `outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/<suite>`.
Evaluation writes to `outputs/holoq_vla/groot_n1_5/pivot_q/eval/seed-2026/<suite>`.
Training resume is controlled by `training.resume: true`: the launcher restores the newest complete
checkpoint containing both adapter and trainer state. Evaluation resume is controlled separately
by the explicit `--resume` flag.

## Structural checks

```bash
PYTHONPATH=. python3 \
  omega_qvla/pivot_q/smoke_test.py

PYTHONPATH=$PIVOT_Q_ROOT/third_party/Omega-QVLA:. \
  python3 omega_qvla/pivot_q/train.py \
  --method-name omega_pivot_q --dry-run
```

## Train: four suites

```bash
CUDA_VISIBLE_DEVICES=0 python3 \
  omega_qvla/pivot_q/run_train.py --suite libero_spatial --gpu 0 --port 5900 --clean-port 5901

CUDA_VISIBLE_DEVICES=1 python3 \
  omega_qvla/pivot_q/run_train.py --suite libero_object  --gpu 1 --port 5910 --clean-port 5911

CUDA_VISIBLE_DEVICES=2 python3 \
  omega_qvla/pivot_q/run_train.py --suite libero_goal    --gpu 2 --port 5920 --clean-port 5921

CUDA_VISIBLE_DEVICES=3 python3 \
  omega_qvla/pivot_q/run_train.py --suite libero_10      --gpu 3 --port 5930 --clean-port 5931
```

Recommended tmux layout:

```bash
tmux new-session -d -s omega_pivot_q -n spatial
tmux new-window -t omega_pivot_q -n object
tmux new-window -t omega_pivot_q -n goal
tmux new-window -t omega_pivot_q -n long
```

Send one command above to each matching window. Attach with `tmux attach -t omega_pivot_q`;
detach with `Ctrl-b d`. Do not start a second launcher against the same output directory.

## Eval: four suites

The family-level `omega_qvla/scripts/run_eval.py` starts the selected FP16,
Ω-QVLA, or PIVOT-Q server, waits until it responds, runs the benchmark, and always
stops the server afterward. Ω checkpoints and quantization packs are
suite-specific, so start one command per suite.

```bash
CUDA_VISIBLE_DEVICES=0 python3 \
  omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_spatial --gpu 0 --port 6000 \
  --config omega_qvla/config/omega_qvla.yaml \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_spatial/final_adapter \
  --output-dir outputs/holoq_vla/groot_n1_5/pivot_q/eval/seed-2026/libero_spatial --resume

CUDA_VISIBLE_DEVICES=1 python3 \
  omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_object --gpu 1 --port 6001 \
  --config omega_qvla/config/omega_qvla.yaml \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_object/final_adapter \
  --output-dir outputs/holoq_vla/groot_n1_5/pivot_q/eval/seed-2026/libero_object --resume

CUDA_VISIBLE_DEVICES=2 python3 \
  omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_goal --gpu 2 --port 6002 \
  --config omega_qvla/config/omega_qvla.yaml \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_goal/final_adapter \
  --output-dir outputs/holoq_vla/groot_n1_5/pivot_q/eval/seed-2026/libero_goal --resume

CUDA_VISIBLE_DEVICES=3 python3 \
  omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_10 --gpu 3 --port 6003 \
  --config omega_qvla/config/omega_qvla.yaml \
  --adapter-path outputs/holoq_vla/groot_n1_5/pivot_q/train/seed-000/libero_10/final_adapter \
  --output-dir outputs/holoq_vla/groot_n1_5/pivot_q/eval/seed-2026/libero_10 --resume
```

Add `--save-video` only when videos are needed; they are large and excluded from Git.

## Monitor

```bash
python3 -m monitor.app --method omega_pivot_q --watch 5
python3 -m monitor.app --method omega_pivot_q --once
```

## Full Distill paper baseline

This baseline changes only the selector: all `T` valid student-visited states
are used with normalized weight `1/T`. The Omega-QVLA W4A4 checkpoints, frozen
original teacher, anchor, optimizer, update count, seed, and 560-episode
manifest match the corresponding sparse recovery run. Each suite has an
independent adapter and output directory. The following four train commands
may run concurrently on GPUs 0–3; ports are separate from the sparse run.

```bash
cd "$PIVOT_Q_DIR"
conda activate groot_test

python3 omega_qvla/pivot_q/run_train.py --config omega_qvla/pivot_q/config/full_distill.yaml --suite libero_spatial --gpu 0 --port 6400 --clean-port 6401
python3 omega_qvla/pivot_q/run_train.py --config omega_qvla/pivot_q/config/full_distill.yaml --suite libero_object  --gpu 1 --port 6410 --clean-port 6411
python3 omega_qvla/pivot_q/run_train.py --config omega_qvla/pivot_q/config/full_distill.yaml --suite libero_goal    --gpu 2 --port 6420 --clean-port 6421
python3 omega_qvla/pivot_q/run_train.py --config omega_qvla/pivot_q/config/full_distill.yaml --suite libero_10      --gpu 3 --port 6430 --clean-port 6431
```

Run each train command in its own terminal or tmux window. Once all four
`final_adapter` links exist, evaluate with the existing Omega-QVLA adapter-loading
path, using separate output directories:

```bash
python3 omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_spatial --gpu 0 --port 6500 --config omega_qvla/config/omega_qvla.yaml --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_spatial/final_adapter --output-dir outputs/holoq_vla/groot_n1_5/full_distill/eval/seed-2026/libero_spatial --resume
python3 omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_object  --gpu 1 --port 6501 --config omega_qvla/config/omega_qvla.yaml --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_object/final_adapter  --output-dir outputs/holoq_vla/groot_n1_5/full_distill/eval/seed-2026/libero_object  --resume
python3 omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_goal    --gpu 2 --port 6502 --config omega_qvla/config/omega_qvla.yaml --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_goal/final_adapter    --output-dir outputs/holoq_vla/groot_n1_5/full_distill/eval/seed-2026/libero_goal    --resume
python3 omega_qvla/scripts/run_eval.py --method pivot-q --suite libero_10      --gpu 3 --port 6503 --config omega_qvla/config/omega_qvla.yaml --adapter-path outputs/holoq_vla/groot_n1_5/full_distill/train/seed-000/libero_10/final_adapter      --output-dir outputs/holoq_vla/groot_n1_5/full_distill/eval/seed-2026/libero_10      --resume
```

`--method pivot-q` is the current evaluator's generic adapter-loading mode; the
experiment identity and results remain `omega_full_distill`.
