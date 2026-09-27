# Seven-category and multi-seed aggregation

Run from the configured project root:

```bash
python scripts/aggregate_seven_categories.py --help
python scripts/aggregate_seven_categories.py --seeds 2026 2027 2028 2029
```

The command scans canonical `outputs/quantvla`, `outputs/holoq_vla`, and
`outputs/qvla` trees, including ablations. Incomplete or erroneous runs are
listed in the audit rather than treated as final results.

Each complete evaluation seed contains four suites × 140 episodes = 560.
Each environmental category contains 80 episodes. The seven columns are Camera,
Init., Language, Lighting, Background, Noise, and Layout. Avg. pools all episodes.
The multi-seed report uses unrounded rates and sample standard deviations.
Training seed 0 must never be included as an evaluation repetition.

Use the documented common launcher for evaluations. Aggregation requires its
`core_command.json` record and the matching manifest. Incomplete runs, duplicate
episodes and missing or inconsistent identities are excluded from final tables.

For a per-setting check against the manifest and explicit seeds:

```bash
python scripts/core_command/aggregate.py --help
```

Only compare the same completed seed set across methods. Do not substitute
partial evaluations, duplicated records, or recovery rollout outcomes.
Output tables and audit files are generated under the ignored outputs tree.
