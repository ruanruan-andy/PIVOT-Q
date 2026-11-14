"""One-GPU π0.5 recovery with sequential rollouts and teacher replay."""
import copy
from tqdm.auto import tqdm

def run(values):
    values = dict(values, execution_mode="single_gpu_teacher_replay_v1")
    import torch
    from .model_residency import ModelResidency, replay_rng
    from . import pi05_state as core
    from .pi05_train import run as train
    from .pi05_flow import make_noise, sample
    if values["backend"] == "holoq":
        from .holoq_policy import load_bundle
    else:
        from .quantvla_policy import load_bundle
    residency = ModelResidency(values["device"])
    from .cost import ComputeMeter
    meter = ComputeMeter(values.get("estimate_flops", False))

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
            prediction = meter.run("student_rollout", 1, lambda: sample(
                student.policy, student.preprocessor(raw), noise, with_grad=False))
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
                target = meter.run("teacher_replay", 1, lambda: sample(
                    teacher.policy, teacher.preprocessor(state.raw_sample), state.noise.to(cfg.device), with_grad=False))
                state.teacher_action = target.detach().cpu()
        residency.activate(student)
        return states, success

    train(values, bundle_loader=load, rollout_fn=collect, meter=meter)

