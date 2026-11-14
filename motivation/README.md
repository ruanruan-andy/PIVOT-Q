# Motivation experiments

## Action intervention

This experiment implements the action replacement in Table 1 for GR00T-N1.5 with QuantVLA on LIBERO-Plus. It uses the original checkpoint and the unrecovered quantization pack. Run from the repository root in the GR00T environment described in [setup](../docs/setup.md). The runner holds both policies on one GPU.

All strategies use the same task manifest, initial states, and noise schedule. At each visited state both policies receive the same observation and noise. The discrepancy is the MSE of the first seven-dimensional normalized action. The simulator executes one action. High and low replace the quantized action with the original action above and below their respective suite thresholds. Random replaces it with the configured probability.

Use the original E2 threshold JSON when reproducing the published run. Its `suites` entries contain `high_threshold` and `low_threshold`, with optional `random_rate` or `high_rate`. When no rate is supplied the random probability is 0.10.

```bash
python motivation/action_intervention.py --config configs/groot_quantvla.yaml \
  --gpu 0 --seed 2026 --thresholds /absolute/path/to/thresholds.json
```

For a new experiment, first collect a fixed quantized baseline on a separate calibration seed. The intervention runner derives the 90th and 10th percentiles per suite from the baseline's visited states and matches the random probability to the empirical fraction above the high threshold.

```bash
python motivation/action_intervention.py --config configs/groot_quantvla.yaml \
  --gpu 0 --seed 2025 --strategies quantized --run-name calibration
python motivation/action_intervention.py --config configs/groot_quantvla.yaml \
  --gpu 0 --seed 2026 --calibration-rollouts \
  outputs/motivation/action_intervention/groot_quantvla/seed/2025/calibration/quantized
```

Outputs are under `outputs/motivation/action_intervention/groot_quantvla/seed/<seed>/`. Each strategy contains per-suite episode records and a summary with seven category scores, suite scores, overall success, and intervention counts. Use `--run-name` to isolate repeats. For a smoke run use `--max-episodes 1 --set 'benchmark.suites=[libero_spatial]'`. Use a free port if running alongside training.

## Action discrepancy distribution

Use [the discrepancy runner](../scripts/discrepancy.py) to collect paired first-action MSE on shared observations for Fig. 1(b).
