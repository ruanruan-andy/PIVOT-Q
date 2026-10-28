"""4090 OFT recovery; frozen teacher targets are replayed after each rollout."""
import copy
import json
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pivot_q.model_residency import ModelResidency
from integration.config import parse_config
from integration.policy import OFTPolicy
from training.collect_single_gpu import collect
from training.train import train

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

def main():
    cfg, dry = parse_config(__doc__, phase="train")
    # Keep 4090 results distinct from existing multi-device adapters.
    cfg["paths"]["output"] = str(Path(cfg["paths"]["output"]) / "single_gpu")
    if dry:
        print(json.dumps(cfg, indent=2))
    else:
        run(cfg)
if __name__ == "__main__":
    main()
