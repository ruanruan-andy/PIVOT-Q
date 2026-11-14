"""Build one Ω-QVLA GR00T pack with the upstream builders."""
import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml


def run_dit_builder(module):
    from .holoq_calibration import activation_quantile

    source = inspect.getsource(module.main)
    original = "torch.quantile(Xt.abs(), p_frac, dim=0)"
    if source.count(original) != 1:
        raise RuntimeError("Expected one activation quantile in the Omega DiT builder")
    scope = dict(vars(module), activation_quantile=activation_quantile)
    source = source.replace(original, "activation_quantile(Xt, p_frac, Xt.device)")
    exec(compile(source, module.__file__, "exec"), scope)
    scope["main"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "llm", "dit", "merge", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    suite = args.suite
    root = Path(cfg["output_root"]).parent
    output = root / "outputs/calibration/groot_holoq" / suite
    output.mkdir(parents=True, exist_ok=True)
    buffer = output / "observations.pt"
    ckpt = Path(cfg["model"]["checkpoint"]) / suite
    pack = Path(cfg["model"]["quant_pack"]) / suite / "quantized.pt"
    if args.action == "collect":
        from .groot_runtime.calibration import collect_frames
        frames, plan = collect_frames(cfg, suite)
        torch.save(frames, buffer)
        (output/"sampling.json").write_text(json.dumps(plan, indent=2)+"\n")
        print(f"{suite}: {len(frames)} clean observations")
        return
    if args.action == "merge":
        from tools import merge_packs
        sys.argv = [sys.argv[0], "--out", str(pack), str(output/"llm.pt"), str(output/"dit.pt")]
        merge_packs.main()
        return
    if args.action == "verify":
        from .groot_runtime import holoq, train
        config = train.TrainConfig(task_suite_name=suite, model_path=str(ckpt), quant_pack_dir=str(pack))
        holoq.install_hooks(train)
        policy = train._load_policy(config, quantized=True, repo_root=Path(cfg["paths"]["omega"]))
        from gr00t.quantization.gptq_layers import GptqLinear
        assert any(isinstance(m, GptqLinear) for m in policy.model.modules())
        return

    is_llm = args.action == "llm"
    module = importlib.import_module("tools.build_gptq_weights" if is_llm else
                                     "tools.build_dit_a2lite_svd_gptq_perstep")
    frames = torch.load(buffer, map_location="cpu", weights_only=False)
    samples = [{"seed": i, "obs": frame} for i, frame in enumerate(frames)]
    module.load_libero_samples = lambda *_: (None, None, samples)
    data_config = "LiberoDataConfigMeanStd" if suite == "libero_goal" else "LiberoDataConfig"
    sys.argv = [sys.argv[0], "--checkpoint", str(ckpt), "--task-suite-name", suite,
                "--data-config", "examples.Libero.custom_data_config:"+data_config,
                "--output-path", str(output/(args.action+".pt")), "--num-samples", str(len(samples)),
                "--token-cap", "1024", "--gptq-block-size", "128", "--gptq-damp-percent", "0.05"]
    if is_llm:
        sys.argv += ["--include-regex", r".*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
                     "--exclude-regex", r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)",
                     "--duquant-rotation", "--duquant-rot-mode", "svd_hadamard", "--weight-bits", "4"]
    else:
        sys.argv += ["--num-steps", "8", "--svd-rank", "0", "--use-rtn", "--w-bits", "4",
                     "--a-bits", "4", "--act-percentile", "99.9", "--duquant-block-size", "64", "--duquant-block-out", "64"]
    if is_llm:
        module.main()
    else:
        run_dit_builder(module)


if __name__ == "__main__":
    main()
