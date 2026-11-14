"""Keep one complete policy on CUDA; retain the other policy in host RAM."""
from contextlib import contextmanager
import gc
import random
import numpy as np
import torch

class ModelResidency:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.policies = {}
        self.active = None

    def register(self, policy, modules):
        self.policies[id(policy)] = (policy, modules)

    def activate(self, policy):
        if self.active is policy:
            return
        for other, modules in self.policies.values():
            if other is not policy:
                for module in modules():
                    module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        for module in self.policies[id(policy)][1]():
            module.to(self.device)
        self.active = policy

@contextmanager
def replay_rng():
    """Teacher replay must not advance subsequent student/dropout RNG streams."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
