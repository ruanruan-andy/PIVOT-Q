"""4090 OFT recovery; frozen teacher targets are replayed after each rollout."""
import copy
from functools import partial
from pivot_q.model_residency import ModelResidency
from pivot_q.oft_runtime.integration.policy import OFTPolicy
from pivot_q.oft_runtime.training.collect_single_gpu import collect
from pivot_q.oft_runtime.training.train import train

def run(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["execution_mode"] = "single_gpu_teacher_replay_v1"
    residency = ModelResidency(cfg["model"]["device"])
    def factory(config, *, teacher=False, **kwargs):
        local = copy.deepcopy(config)
        if teacher:
            local["model"]["teacher_device"] = "cpu"
        policy = OFTPolicy(local, teacher=teacher, **kwargs)
        policy.device = residency.device
        residency.register(policy, lambda: [policy.model, policy.proprio, policy.head])
        if not teacher:
            residency.activate(policy)
        return policy
    train(cfg, policy_factory=factory, collect_fn=partial(collect, residency=residency))

