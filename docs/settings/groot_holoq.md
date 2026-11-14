# GR00T-N1.5 + HoloQ-VLA

Complete [Setup](../setup.md) and run from the project root.

```bash
conda activate groot_test
```

## Prepare the quantization pack

Skip this section if you downloaded the matching pack in Setup.
Run the commands for each suite. The example uses Object. Use `libero_spatial`, `libero_goal` and `libero_10` for the remaining suites.

The Spatial DiT build hit CUDA OOM on PS's 4090. The full 4090 build is pending validation.

```bash
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export MUJOCO_GL=egl QT_QPA_PLATFORM=offscreen TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/third_party/QuantVLA:$PWD/third_party/LIBERO:$PWD"
python -m pivot_q.omega_pack collect --config configs/groot_holoq.yaml --suite libero_object
export PYTHONPATH="$PWD/third_party/Omega-QVLA:$PWD/third_party/LIBERO:$PWD"
python -m pivot_q.omega_pack llm --config configs/groot_holoq.yaml --suite libero_object
python -m pivot_q.omega_pack dit --config configs/groot_holoq.yaml --suite libero_object
python -m pivot_q.omega_pack merge --config configs/groot_holoq.yaml --suite libero_object
python -m pivot_q.omega_pack verify --config configs/groot_holoq.yaml --suite libero_object
```

## Train

The default training schedule collects one student rollout per round for 140 rounds per suite, in Object → Spatial → Goal → Long order. PIVOT-Q selects 20% of the candidate states in each rollout by default, allocating integer quotas across temporal phases. Full Distill uses all recorded states. Each batch of up to 32 states, including the final partial batch, makes one optimizer update.

```bash
python scripts/train.py --config configs/groot_holoq.yaml --method pivot_q --gpu 0
python scripts/train.py --config configs/groot_holoq.yaml --method full_distill --gpu 0
```

## Evaluate

```bash
python scripts/evaluate.py --config configs/groot_holoq.yaml --method original --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/groot_holoq.yaml --method quantized --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/groot_holoq.yaml --method full_distill --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/groot_holoq.yaml --method pivot_q --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/groot_holoq.yaml --method pivot_q --benchmark libero --gpu 0 --seed 2026 --resume
```

Use `--benchmark libero` for the other methods as well. Recovered methods use the adapter from training seed 0 by default. Set `--train-seed` to select a different training run.

For optional action-discrepancy evaluation, see `python scripts/discrepancy.py --help`.

## Command reference

- `--method` selects the baseline or recovery method. Train accepts `pivot_q` and `full_distill`.
- `--seed` sets the seed for the command. The default evaluation seed is 2026.
- `--resume` skips completed evaluation episodes. Training resumes by default. Use `--no-resume` to start training fresh.
- `--set 'benchmark.suites=[libero_object]'` selects one suite. The default runs Object, Spatial, Goal, then Long.
- `--dry-run` previews the resolved configuration.

To run a different state budget, pass the same override to training and evaluation. For example, use 40%:

```bash
python scripts/train.py --config configs/groot_holoq.yaml --method pivot_q --gpu 0 --set selection.percent=40
python scripts/evaluate.py --config configs/groot_holoq.yaml --method pivot_q --benchmark libero_plus --gpu 0 --seed 2026 --resume --set selection.percent=40
```

Set the percentage to `null` to use the fixed phase quotas from the original protocol. Each state batch still makes one optimizer update.

## Outputs

- Pack: `models/quantized/groot_holoq/` with one directory per suite.
- Final adapter: `outputs/groot_holoq/<method>/train/seed/0/<suite>/final_adapter`.
- Evaluation: `outputs/groot_holoq/<method>/eval/<benchmark>/seed/2026/`.
- Default PIVOT-Q run: `outputs/groot_holoq/percent_020/pivot_q/` for 20%. LIBERO-Plus reports seven category metrics along with suite and overall metrics.
