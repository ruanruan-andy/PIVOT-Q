"""Collect clean OFT calibration and build QVLA gates for one suite."""
import argparse
import os
from pathlib import Path

import yaml

from .configuration import ROOT
from .processes import managed_processes, start
from .suite_runs import oft_config
from .oft_runtime.integration.calibrate import collect_calibration, quantize
from .oft_runtime.integration.policy import OFTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    suite, paths = args.suite, cfg["paths"]
    output = ROOT / "outputs/calibration/openvla_oft_qvla" / suite
    pack = Path(cfg["model"]["quant_pack"]) / suite
    plan = dict(suite=suite, method="quantized", seed=0, checkpoint=str(Path(cfg["model"]["checkpoint"])/suite),
                output=str(output), pack=str(pack), adapter=None, port=args.port, clean_port=args.port,
                max_rounds=1, phase="eval", resume=False)
    config = oft_config(cfg, plan)
    simulator = Path(paths["simulator_python"])
    env = dict(os.environ, PYTHONPATH=str(ROOT)+":"+paths["libero"],
               LIBERO_CONFIG_PATH=paths["clean_config"], MUJOCO_GL="egl",
               QT_QPA_PLATFORM="offscreen", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               MAGICK_HOME=str(simulator.parent.parent),
               LD_LIBRARY_PATH=str(simulator.parent.parent/"lib")+":"+os.environ.get("LD_LIBRARY_PATH",""))
    if not (pack / "calibration.pt").is_file():
        command = [str(simulator), str(ROOT/"pivot_q/libero_service.py"), "--suite", suite,
                   "--port", str(args.port), "--seed", "0"]
        with managed_processes() as processes:
            processes.append(start(command, env, output/"simulator.log"))
            collect_calibration(config)
    quantize(config)
    for name in ("calibration.pt", "proxy.pt", "gates.json", "gates.metadata.json"):
        assert (pack/name).stat().st_size > 0
    OFTPolicy(config, quantized=True)
    print(f"{suite}: validated QVLA calibration, proxy and gates")


if __name__ == "__main__":
    main()
