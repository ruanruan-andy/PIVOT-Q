"""Build a QuantVLA GR00T pack from balanced clean LIBERO observations."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import yaml

from .train import TrainConfig, _configure_quantvla, _data_config_for_suite


def collect_frames(cfg, suite_name):
    from tools import calibrate_atm_dit as atm
    suite = atm.benchmark.get_benchmark_dict()[suite_name]()
    frames, plan = [], []
    per_task, remainder = divmod(cfg["calibration"]["num_samples"], suite.n_tasks)
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        count = per_task + (task_id < remainder)
        env, _ = atm.get_libero_env(task, resolution=256)
        try:
            env.seed(task_id)
            env.reset()
            initial = suite.get_task_init_states(task_id)[0]
            obs = env.set_init_state(initial)
            for index in range(count):
                frames.append(atm._convert_observation(obs, task.language))
                obs, _, done, _ = env.step(atm.get_libero_dummy_action())
                if done and index + 1 < count:
                    env.reset()
                    obs = env.set_init_state(initial)
            plan.append(dict(task_id=task_id, frames=count, initial_state_id=0))
        finally:
            env.close()

    return frames, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    from tools import calibrate_atm_dit as atm
    model = Path(cfg["model"]["checkpoint"]) / args.suite
    pack = Path(cfg["model"]["quant_pack"]) / args.suite
    atm_path = pack / "atm_alpha_beta.json"
    quant = TrainConfig(task_suite_name=args.suite, calibration_path=str(atm_path),
                        quant_pack_dir=str(pack / "duquant_pack"))
    atm._strip_duquant_env()
    atm._disable_atm_env()
    frames, plan = collect_frames(cfg, args.suite)
    data_config = _data_config_for_suite(args.suite)

    def capture():
        torch.manual_seed(0)
        policy = atm.Gr00tPolicy(model_path=str(model),
            modality_config=copy.deepcopy(data_config.modality_config()),
            modality_transform=data_config.transform(),
            embodiment_tag=atm.EmbodimentTag.NEW_EMBODIMENT,
            denoising_steps=cfg["model"]["num_inference_steps"], device="cuda:0")
        stats, ohb = atm.StdCollector(), atm.RMSCollector()
        atm.run_policy_with_capture(policy, frames, stats, ohb_collector=ohb,
                                    atm_scope="dit", ohb_scope="dit")
        return stats.finalize(), ohb.finalize()

    print(f"{args.suite}: collected {len(frames)} observations across {len(plan)} tasks")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sampling.json").write_text(json.dumps(plan, indent=2) + "\n")
    teacher, teacher_ohb = capture()
    torch.cuda.empty_cache()
    _configure_quantvla(quant, Path(cfg["paths"]["quantvla"]), calibrating=True)
    student, student_ohb = capture()
    alpha = atm.compute_alpha_json(teacher, student)
    for name, beta in atm.compute_beta_values(teacher_ohb, student_ohb, log_clamp=0.30, neutral=0.03).items():
        alpha.setdefault(name, {})["beta"] = beta
    assert len(alpha) == 16 and all("all" in row and "beta" in row for row in alpha.values())
    packs = list((pack / "duquant_pack").glob("*.npz"))
    assert len(packs) == 116 and all(p.stat().st_size for p in packs)
    temporary = atm_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(alpha, indent=2) + "\n")
    temporary.replace(atm_path)
    print(f"{args.suite}: validated {len(packs)} pack files and {len(alpha)} ATM/OHB layers")


if __name__ == "__main__":
    main()
