# π0.5 QuantVLA

This directory contains all π0.5 QuantVLA work.  It reads the unchanged local
LeRobot checkout and the local `models/pi05_libero` checkpoint; nothing in
`PIVOT-Q/scripts/`, the sibling `lerobot` checkout, or the original QuantVLA
checkout is changed.

## Pipeline

1. `prepare_calibration_buffer.py` deterministically selects 32 *unlabelled*
   clean-LIBERO observations from `lerobot/libero` and freezes them in a `.pt`
   buffer.  They are calibration inputs only, never evaluation tasks.
2. `build.py` writes the paper-aligned W4A8 static DuQuant pack: all language
   model linear layers plus only the action expert MLP projections.
3. `calibrate.py` runs FP π0.5 and that quantized pack on exactly the frozen
   inputs.  ATM restores per-head attention-logit scale, and OHB restores the
   action-expert output RMS.  The resulting factors are written as
   `atm_ohb.json`; the manifest is marked calibrated.

## Commands

```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05

python3 pi05_quantvla/scripts/prepare_calibration_buffer.py --count 32 --seed 0

CUDA_VISIBLE_DEVICES=7 python3 pi05_quantvla/scripts/build.py \
  --config pi05_quantvla/config/quantvla.yaml

CUDA_VISIBLE_DEVICES=7 python3 pi05_quantvla/scripts/calibrate.py \
  --config pi05_quantvla/config/quantvla.yaml \
  --buffer pi05_quantvla/data/calibration/libero_clean_32_seed-000.pt --seed 0
```

## PIVOT-Q recovery

The same π0.5 checkpoint is used for spatial, object, goal, and long-horizon
evaluation. PIVOT-Q therefore trains one shared adapter with four synchronous
ranks and evaluates that adapter across all suites:

```bash
cd "$PIVOT_Q_DIR"
conda activate lerobot_pi05

python3 pi05_quantvla/pivot_q/run_train_ddp.py --gpus 4 5 6 7
```

See [`pivot_q/commands.md`](pivot_q/commands.md) for ports, tmux, resume, monitoring,
and shared-adapter evaluation.

FP16, QuantVLA, and PIVOT-Q share `scripts/run_eval.py` and the same LIBERO client.
PIVOT-Q selects the same calibrated W4A8 path and adds only `--adapter-path`; it
does not recalibrate the model at evaluation time.

## Related π0.5 Omega-QVLA integration

The separate [`pi05_omegavla/`](../pi05_omegavla/README.md) module reuses the same
LeRobot base checkpoint with an Omega W4A4 pack. It provides `fp_original`,
`omega_original`, `full_distill`, and `pivot_q`, with one shared adapter per
recovery method. Its quantization packs, adapters, and launchers are separate
from this QuantVLA pipeline. See its [commands](../pi05_omegavla/commands.md).
