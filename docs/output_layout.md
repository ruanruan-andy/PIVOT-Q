# Experiment outputs and seven-category reports

Run commands from the project root unless a setting-specific page explicitly
changes into `openvla-oft-qvla`. Default result paths are resolved from the code
location, not a server-specific mount point.

```text
outputs/<quantizer>/<backbone>/<method>/train/seed-000/<suite-or-shared>/
outputs/<quantizer>/<backbone>/<method>/eval/seed-2026/<suite>/
```

| Quantizer | Backbones |
|---|---|
| quantvla | groot_n1_5, pi05 |
| holoq_vla | groot_n1_5, pi05 |
| qvla | openvla_oft |

Methods are `original`, `quantized`, `full_distill`, and `pivot_q`. Ablations
live under `quantvla/groot_n1_5/ablations/<experiment>/`. The HoloQ pi05 evaluator
retains its existing `libero-plus` benchmark subdirectory beneath the seed.

## Reporting

```bash
cd "$PIVOT_Q_DIR"
python3 scripts/aggregate_seven_categories.py
# Select a different complete seed cohort when needed:
python3 scripts/aggregate_seven_categories.py --seeds 2026 2027 2028 2029
```

Reports are written to `outputs/statistics/all_results.{json,csv,md}`, with
per-setting Markdown tables in `outputs/statistics/by_setting/`. `audit.json`
lists missing suites, invalid counts and errors. Only the three canonical
quantizer roots are scanned; historical backup/replica directories are excluded.

Each final row requires four suites with 140 episode records each, no summary
errors, and 20 episodes per category per suite. Category success rates combine
success counts over 80 episodes. Avg. is the mean of the seven category rates.
Incomplete results have blank rates. Mean ± Std. uses the requested complete
seed cohort and sample standard deviation (`ddof=1`). CSV/JSON retain numerical
precision; Markdown displays one decimal place.

Re-run aggregation after evaluations complete.
