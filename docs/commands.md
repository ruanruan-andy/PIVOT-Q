# Training and evaluation commands

For pi0.5 on one GPU, see [single-GPU recovery training](commands-pi05-single-gpu.md).

Complete [installation](../env/README.md) first. In each terminal:

```bash
export PIVOT_Q_ROOT=/path/to/PIVOT-Q
source "$PIVOT_Q_ROOT/env/runtime/paths.sh"
cd "$PIVOT_Q_ROOT"
```

| Setting | Environment | Detailed commands |
|---|---|---|
| GR00T-N1.5 + QuantVLA | groot_test | [Commands](commands-quantvla-groot.md) |
| π₀.₅ + QuantVLA | lerobot_pi05 | [Commands](commands-quantvla-pi05.md) |
| GR00T-N1.5 + HoloQ-VLA | groot_test | [Commands](commands-omegavla-groot.md) |
| π₀.₅ + HoloQ-VLA | lerobot_pi05 | [Commands](commands-omegavla-pi05.md) |
| OpenVLA-OFT + QVLA | oft_qvla | [Commands](commands-qvla-openvla-oft.md) |

Training uses **seed 0**. Evaluate the same final checkpoint separately with
**seeds 2026, 2027, 2028, and 2029**. These are evaluation repetitions, not four
independent training runs. Original and Quantized do not require recovery training.

## Common launcher

The common launcher prints commands without starting processes when `--dry-run`
is supplied. Activate the model environment first.

```bash
conda activate groot_test

# One suite, training seed 0.
python scripts/core_command/launch.py groot_quantvla pivot_q train \
  --suite libero_spatial --gpu 0 --seed 0 --port 7700 --clean-port 7701

# Original policy, one evaluation seed.
python scripts/core_command/launch.py groot_quantvla fp_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 7710

# Quantized policy.
python scripts/core_command/launch.py groot_quantvla quant_original eval \
  --suite libero_spatial --gpu 0 --seed 2026 --port 7710

# Recovery adapter: reuse the same checkpoint for all evaluation seeds.
for seed in 2026 2027 2028 2029; do
  python scripts/core_command/launch.py groot_quantvla pivot_q eval \
    --suite libero_spatial --gpu 0 --seed "$seed" --port 7710 \
    --adapter-path outputs/quantvla/groot_n1_5/pivot_q/train/seed-000/libero_spatial/final_adapter
done
```

Use `full_distill` instead of `pivot_q` for the dense baseline and supply its
own adapter. Each method/seed/suite has an independent output directory.
For suite-specific adapters, `--suite all` accepts a quoted adapter path
containing `{suite}`; suites execute serially. π₀.₅ uses shared adapters.
OFT requires one matching simulator service per suite and saves checkpoint files
rather than a GR00T adapter directory. Follow its detailed commands.

The common launcher families are `groot_quantvla`, `pi05_quantvla`,
`groot_holoqvla`, `pi05_holoqvla`, and `openvla_oft_qvla`.
Do not launch two writers for the same output directory. Use unused, distinct
ports for concurrent jobs. Training and evaluation are separate commands:
there is no automatic evaluation supervisor in this release.

## GR00T + QuantVLA ablations

```bash
conda activate groot_test
CUDA_VISIBLE_DEVICES=0 python scripts/train_pivot_q.py \
  --config config/random_sparse.yaml --suite libero_spatial \
  --port 7720 --clean-port 7721 --seed 0

for seed in 2026 2027 2028 2029; do
  python scripts/core_command/launch.py groot_quantvla pivot_q eval \
    --ablation random_sparse --suite libero_spatial --gpu 0 --port 7722 \
    --seed "$seed" \
    --adapter-path outputs/quantvla/groot_n1_5/ablations/random_sparse/train/seed-000/libero_spatial/final_adapter
done
```

Use the matching configuration and output/adapter directory for other ablations:
`uniform_sparse.yaml`, `global_random_sparse.yaml`,
`pivot_q_current_only.yaml`, `pivot_q_no_anchor.yaml`,
and the `pivot_q_a*_b*.yaml` scoring variants. Pass the matching configuration
stem to `--ablation` for evaluation.

## Resume and reporting

Use `--resume` to continue evaluation. Reuse the original model, adapter,
configuration and manifest; GPU and launcher ports may change.

Training resume follows the setting-specific configuration; OFT requires an
explicit checkpoint path. Use a new output directory when disabling resume.
Older runs without protocol records cannot be resumed automatically.
Do not replace weights in place while resuming.

See [result aggregation](result_aggregation.md) and
[experimental protocol](experiment_protocol.md).
