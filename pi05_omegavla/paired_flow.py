"""Keep LeRobot sampling intact while supplying Omega's per-step context."""
from contextvars import ContextVar
from types import MethodType
import torch
from pi05_quantvla.pivot_q.paired_flow import make_noise, model_inputs, sample

def install_step_context(model):
    if getattr(model, "_pi05_omega_step_installed", False):
        return
    from gr00t.quantization.dit_step_context import set_dit_quant_step
    original = getattr(model.sample_actions, "__wrapped__", None)
    if original is None:
        raise RuntimeError("expected LeRobot's no_grad-wrapped sample_actions")
    denoise = model.denoise_step
    cursor = ContextVar("pi05_omega_step_cursor", default=None)

    def step(self, *args, **kwargs):
        state = cursor.get()
        if state is None:
            raise RuntimeError("denoise_step called outside Omega sampling context")
        index, count = state
        if index >= count:
            raise RuntimeError("more denoise calls than configured steps")
        cursor.set((index + 1, count))
        with set_dit_quant_step(index, total=count):
            return denoise(*args, **kwargs)

    def differentiable(self, *args, **kwargs):
        count = kwargs.get("num_steps") or self.config.num_inference_steps
        token = cursor.set((0, count))
        try:
            # PaliGemma uses the sole scale-table row (bucket zero).
            with set_dit_quant_step(0, total=count):
                result = original(self, *args, **kwargs)
            if cursor.get()[0] != count:
                raise RuntimeError("denoise count differs from scale-table protocol")
            return result
        finally:
            cursor.reset(token)

    model.denoise_step = MethodType(step, model)
    model.sample_actions = MethodType(torch.no_grad()(differentiable), model)
    model._pi05_omega_step_installed = True
