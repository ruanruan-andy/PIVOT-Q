"""4090 worker: student rollout, teacher replay, then five grouped updates."""
import argparse
from tqdm.auto import tqdm
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def run(values):
    values = dict(values, execution_mode="single_gpu_teacher_replay_v1")
    import torch
    from pivot_q.model_residency import ModelResidency, replay_rng
    from pi05_quantvla.pivot_q import train as core
    from pi05_quantvla.pivot_q.train_single import run as train
    from pi05_quantvla.pivot_q.paired_flow import make_noise, sample
    if values["backend"] == "holoq":
        from pi05_omegavla.backend import load_bundle
    else:
        load_bundle = core.load_bundle
    residency = ModelResidency(values["device"])

    def load(cfg, *, quantized, adapter=None):
        local = copy.copy(cfg)
        # CPU teacher initialization avoids ever placing both models on CUDA.
        local.device = cfg.device if quantized else "cpu"
        bundle = load_bundle(local, quantized=quantized, adapter=adapter)
        bundle.policy.config.device = cfg.device
        for processor in bundle.preprocessor.steps:
            if hasattr(processor, "tensor_device"):
                processor.device = cfg.device
                processor.tensor_device = torch.device(cfg.device)
                processor.non_blocking = True
        residency.register(bundle, lambda: [bundle.policy])
        if quantized:
            residency.activate(bundle)
        return bundle

    def collect(cfg, teacher, student, env, *, task_id, initial_state_id, horizon, generator):
        residency.activate(student)
        reset = env.call("reset", suite=cfg.suite, task_id=task_id,
                         initial_state_id=initial_state_id, num_steps_wait=10, horizon=horizon + 11)
        observation, language = reset["observation"], reset["language"]
        states, success = [], bool(reset["done"])
        for _ in range(0 if success else horizon):
            raw = core.raw_sample(observation, language)
            noise = make_noise(student.policy, generator)
            prediction = sample(student.policy, student.preprocessor(raw), noise, with_grad=False)
            states.append(core.State(raw, noise.detach().cpu(), torch.empty(0), prediction.detach().cpu()))
            action = student.postprocessor(prediction[0, :1])[0].detach().float().cpu().numpy()
            result = env.call("step", action=action.tolist())
            observation, success = result["observation"], bool(result["done"])
            if success:
                break
        tqdm.write(f"Teacher replay: {len(states)} student-visited states")
        residency.activate(teacher)
        with replay_rng():
            for state in tqdm(states, desc="Teacher replay", unit="state", leave=False, dynamic_ncols=True):
                target = sample(teacher.policy, teacher.preprocessor(state.raw_sample),
                                state.noise.to(cfg.device), with_grad=False)
                state.teacher_action = target.detach().cpu()
        residency.activate(student)
        return states, success

    train(values, bundle_loader=load, rollout_fn=collect)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.runtime_config.read_text()))

if __name__ == "__main__":
    main()
