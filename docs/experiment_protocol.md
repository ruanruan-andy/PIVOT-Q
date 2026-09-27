# Experimental protocol

## Tasks and seeds

The shared manifest is `manifests/libero_plus_first20.json`: 20 tasks per
environmental-variation category in each of four suites, yielding 140 episodes
per suite and 560 per complete evaluation seed. Categories are Camera Viewpoints,
Robot Initial States, Language Instructions, Light Conditions, Background
Textures, Sensor Noise, and Objects Layout.

Recovery and evaluation use the same task IDs. Describe this as same-task
transductive adaptation, not held-out-task generalization. Recovery rollouts
are not substituted for final evaluation episodes.

Training uses seed **0**. Evaluate the fixed final checkpoint with seeds
**2026, 2027, 2028, 2029** in separate directories. Original and Quantized
baselines need no recovery training. Do not report a four-seed mean until all
four evaluations are complete. Across-seed standard deviations refer to
evaluation randomness, not independent training seeds.

## Comparisons

Original, Quantized, Full Distill, and PIVOT-Q are compared within each
model/quantizer setting. Full Distill retains all valid rollout states with
uniform distillation weights. PIVOT-Q uses the configured phase-balanced
sparse selection. Preserve the model-specific action, sampler, adapter,
optimizer, and clean-anchor settings in the supplied YAML files.

GR00T + QuantVLA ablations include Random Sparse, Uniform Sparse, Global Random
Sparse, current-discrepancy-only selection, no Behavioral Anchor, and scoring
weight variants. Their configuration files specify the exact selection rules;
do not infer identical selection constraints from the shared sparse budget.

Use the fixed final checkpoint for comparisons, not the checkpoint with the
highest evaluation success rate.

## Results

```text
outputs/<quantizer>/<model>/<method>/train/seed-000/...
outputs/<quantizer>/<model>/<method>/eval/seed-2026/<suite>/...
outputs/<quantizer>/<model>/<method>/eval/seed-2027/<suite>/...
outputs/<quantizer>/<model>/<method>/eval/seed-2028/<suite>/...
outputs/<quantizer>/<model>/<method>/eval/seed-2029/<suite>/...
```

Some backend layouts add a `libero-plus/` component or use a shared adapter.
Report complete error-free evaluations only. Each category has 80 episodes
across suites. Compute rates and sample
standard deviations from unrounded values; round only for presentation.
