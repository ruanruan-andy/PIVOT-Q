"""GR00T 4090 worker using the original selectors, optimizer and checkpoints."""
import copy
from tqdm.auto import tqdm
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pivot_q"))
sys.path.insert(0, str(ROOT))

def install(core):
    import dataclasses
    import torch
    from pivot_q.model_residency import ModelResidency, replay_rng
    residency = ModelResidency()
    original_load = core._load_policy
    def load(config, *, quantized, repo_root):
        local = dataclasses.replace(config, device=config.device if quantized else "cpu")
        policy = original_load(local, quantized=quantized, repo_root=repo_root)
        policy.device = config.device
        residency.register(policy, lambda: [policy.model])
        if quantized:
            residency.activate(policy)
        return policy

    def collect(config, teacher, student, env_client, *, task_id, initial_state_id,
                episode_horizon, noise_generator):
        residency.activate(student)
        model = core._base_model(student)
        result = env_client.call("reset", suite=config.task_suite_name, task_id=task_id,
            initial_state_id=initial_state_id, num_steps_wait=config.num_steps_wait,
            horizon=config.num_steps_wait + episode_horizon + 1)
        observation, language = result["observation"], result["language"]
        success, steps = bool(result["done"]), []
        for _ in range(0 if success else episode_horizon):
            policy_observation = core._process_observation(observation, language)
            normalized = core._normalize_observation(student, policy_observation)
            backbone, action_input = core._prepare_backbone(model, normalized)
            # Frozen features are detached on CPU; every update still re-runs the action head.
            if any(p.requires_grad for p in model.backbone.parameters()) or any(m.training for m in model.backbone.modules()):
                raise RuntimeError("4090 feature cache requires a frozen eval-mode backbone")
            cached_backbone, cached_input = core._cache_feature(backbone), core._cache_feature(action_input)
            noise = torch.randn(1, model.action_head.config.action_horizon,
                                model.action_head.config.action_dim, generator=noise_generator, dtype=torch.float32)
            predicted = core._sample_from_cached(model, backbone, action_input, noise, with_grad=False)
            cached = core.CachedState(cached_backbone, cached_input, noise.cpu(), torch.empty(0))
            steps.append(core.ScoredStep(copy.deepcopy(policy_observation), noise.cpu(), torch.empty(0),
                                         predicted.detach().float().cpu(), cached))
            action = core._normalized_to_libero_action(student, predicted)
            result = env_client.call("step", action=action.tolist())
            observation, success = result["observation"], bool(result["done"])
            if success:
                break
        tqdm.write(f"Teacher replay: {len(steps)} student-visited states")
        residency.activate(teacher)
        teacher_model = core._base_model(teacher)
        with replay_rng():
            for step in tqdm(steps, desc="Teacher replay", unit="state", leave=False, dynamic_ncols=True):
                normalized = core._normalize_observation(student, step.policy_observation)
                backbone, action_input = core._prepare_backbone(teacher_model, normalized)
                target = core._sample_from_cached(teacher_model, backbone, action_input,
                                                  step.initial_noise, with_grad=False)
                step.teacher_action = target.detach().float().cpu()
                step.cached_student_state.teacher_action = step.teacher_action
        residency.activate(student)
        return core.RolloutTrace(steps, success, "success" if success else "horizon")
    original_materialize = core._materialize_states
    def materialize(student, trace, indices):
        states = original_materialize(student, trace, indices)
        # All 4090 rollout features are cached; updates use only the action head.
        core._base_model(student).backbone.to("cpu")
        residency.active = None
        torch.cuda.empty_cache()
        return states
    core._materialize_states = materialize
    core._load_policy = load
    core._collect_scored_rollout = collect
    return residency

def main():
    import tyro
    import os
    if os.environ.get("PIVOT_Q_SINGLE_GPU_BACKEND") == "holoq":
        import omega_qvla.pivot_q.train
    import train as core
    import selector
    import full_distill
    config = tyro.cli(core.TrainConfig)
    if config.method_name.startswith("omega_"):
        # Match the existing HoloQ entrypoint's objective import.
        import importlib.util
        spec = importlib.util.spec_from_file_location("gr00t.experiment.pivot_q", ROOT / "third_party/QuantVLA/gr00t/experiment/pivot_q.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from omega_qvla.pivot_q.backend import install_hooks
        install_hooks(core)
    core.build_pivot_q_targets = selector.build_targets
    install(core)
    choose = full_distill.select_states if config.method_name in {"full_distill", "omega_full_distill"} else selector.select_states
    core.train(config, choose)
if __name__ == "__main__":
    main()
