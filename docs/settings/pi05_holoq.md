# π₀.₅ + HoloQ-VLA

Complete [Setup](../setup.md) and run from the project root.

```bash
conda activate lerobot_pi05
```

## Prepare the quantization pack

Skip this section if you downloaded the matching pack in Setup.
Both π₀.₅ settings use the same original checkpoint and calibration buffer. Download the dataset and collect only if the shared buffer does not exist.

```bash
hf download HuggingFaceVLA/libero --repo-type dataset --local-dir data/calibration/lerobot_libero
python scripts/calibrate.py collect --config configs/pi05_holoq.yaml
python scripts/calibrate.py build --config configs/pi05_holoq.yaml --gpu 0 --resume
```

## Train

Use the [Hugging Face offline settings](../setup.md#offline-hugging-face-access) after caching the tokenizer.

The default training schedule collects one student rollout per round for 140 rounds per suite, in Object → Spatial → Goal → Long order. PIVOT-Q selects 20% of the candidate states in each rollout by default, allocating integer quotas across temporal phases. Full Distill uses all recorded states. Each batch of up to 16 states, including the final partial batch, makes one optimizer update.

```bash
python scripts/train.py --config configs/pi05_holoq.yaml --method pivot_q --gpu 0
python scripts/train.py --config configs/pi05_holoq.yaml --method full_distill --gpu 0
```

## Evaluate

Use the [Hugging Face offline settings](../setup.md#offline-hugging-face-access) after caching the tokenizer.

```bash
python scripts/evaluate.py --config configs/pi05_holoq.yaml --method original --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/pi05_holoq.yaml --method quantized --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/pi05_holoq.yaml --method full_distill --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/pi05_holoq.yaml --method pivot_q --benchmark libero_plus --gpu 0 --seed 2026 --resume
python scripts/evaluate.py --config configs/pi05_holoq.yaml --method pivot_q --benchmark libero --gpu 0 --seed 2026 --resume
```

Use `--benchmark libero` for the other methods as well. Recovered methods use the adapter from training seed 0 by default. Set `--train-seed` to select a different training run.

For optional action-discrepancy evaluation, see `python scripts/discrepancy.py --help`.

## Command reference

- `--method` selects the baseline or recovery method. Train accepts `pivot_q` and `full_distill`.
- `--seed` sets the seed for the command. The default evaluation seed is 2026.
- `--resume` skips completed evaluation episodes. Training resumes by default. Use `--no-resume` to start training fresh.
- `--set 'benchmark.suites=[libero_object]'` selects one suite. Formal training uses the default four-suite list to produce one shared adapter.
- `--dry-run` previews the resolved configuration.

To run a different state budget, pass the same override to training and evaluation. For example, use 40%:

```bash
python scripts/train.py --config configs/pi05_holoq.yaml --method pivot_q --gpu 0 --set methods.pivot_q.percent=40
python scripts/evaluate.py --config configs/pi05_holoq.yaml --method pivot_q --benchmark libero_plus --gpu 0 --seed 2026 --resume --set methods.pivot_q.percent=40
```

Set the percentage to `null` to use the fixed phase quotas from the original protocol. Each state batch still makes one optimizer update.

## Outputs

- Pack: `models/quantized/pi05_holoq/`.
- Final adapter: `outputs/pi05_holoq/<method>/train/seed/0/final_adapter`.
- Evaluation: `outputs/pi05_holoq/<method>/eval/<benchmark>/seed/2026/`.
- Default PIVOT-Q run: `outputs/pi05_holoq/percent_020/pivot_q/` for 20%. LIBERO-Plus reports seven category metrics along with suite and overall metrics.
