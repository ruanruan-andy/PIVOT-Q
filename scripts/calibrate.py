"""Collect a clean π0.5 buffer or build a quantization pack."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pivot_q.configuration import config

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("collect", "build"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--task-map", type=Path, help="JSON mapping clean dataset task text to suite")
    p.add_argument("--buffer", type=Path)
    p.add_argument("--pack-output", type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.port, args.clean_port, args.set = 19600, 19601, []
    cfg = config(args)
    if cfg["backbone"] != "pi05":
        p.error("This entry builds π0.5 packs")
    settings = cfg["calibration"]
    buffer = args.buffer or Path(settings["buffer"])
    pack = args.pack_output or Path(cfg["model"]["quant_pack"])
    dataset = args.dataset_root or Path(settings["dataset_root"])
    task_map = args.task_map or Path(settings["task_map"])
    if any(not path.is_absolute() for path in (buffer, pack, dataset, task_map)):
        p.error("Calibration paths must be absolute")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.action == "collect" else str(args.gpu)
    from pivot_q import calibration
    from pivot_q.cost import record_compute
    if args.action == "collect":
        if buffer.exists():
            p.error(f"Calibration buffer already exists: {buffer}")
        with record_compute(buffer.parent / "collection", False, gpu=False):
            print(calibration.collect(cfg, dataset, task_map, buffer))
    else:
        if not buffer.is_file():
            p.error(f"Calibration buffer does not exist: {buffer}")
        manifest = pack / ("manifest.json" if cfg["quantizer"] == "holoq" else "quantvla_manifest.json")
        if manifest.is_file():
            import json
            if json.loads(manifest.read_text()).get("complete"):
                p.error(f"Pack already complete: {pack}")
        with record_compute(pack, False, resume=args.resume):
            calibration.build(cfg, buffer, pack, resume=args.resume)

if __name__ == "__main__":
    main()
