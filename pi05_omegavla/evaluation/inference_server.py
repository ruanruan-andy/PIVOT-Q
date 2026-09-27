"""Serve all four pi0.5 Omega modes with the same preprocessing and noise."""
import argparse
import json
import sys
from pathlib import Path
from multiprocessing.connection import Listener
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from pi05_omegavla.backend import evaluation_bundle
from pi05_omegavla.common import load_config, MODES
from pi05_omegavla.paired_flow import sample

class PI05Server:
    def __init__(self, config, device, adapter=None, save_discrepancy=False, cache_dir=None):
        self.raw = load_config(config)
        self.bundle = evaluation_bundle(config, device, adapter)
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.teacher = (self.bundle if self.raw["mode"] == "fp_original" else
                        evaluation_bundle(config, device, teacher=True)) if save_discrepancy and self.cache_dir is None else None

    def actions(self, request):
        bundle = self.bundle
        batch = bundle.preprocessor({
            "observation.images.image": torch.from_numpy(request["image"]),
            "observation.images.image2": torch.from_numpy(request["wrist_image"]),
            "observation.state": torch.from_numpy(request["state"]), "task": request["task"],
        })
        policy = bundle.policy
        count = int(self.raw["inference"]["n_action_steps"])
        if not 1 <= count <= policy.config.chunk_size:
            raise ValueError("n_action_steps outside action chunk")
        generator = torch.Generator(device=self.device).manual_seed(int(request["seed"]))
        noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim),
                            device=self.device, generator=generator)
        with torch.inference_mode():
            prediction = sample(policy, batch, noise.clone(), with_grad=False)
            actions = prediction[0, :count].clone()
            diagnostics = None
            if self.teacher is not None:
                teacher_prediction = prediction if self.teacher is bundle else sample(
                    self.teacher.policy, batch, noise.clone(), with_grad=False)
                student_action = prediction[0, 0, :7].float()
                teacher_action = teacher_prediction[0, 0, :7].float()
                q = (student_action - teacher_action).square().mean()
                if not torch.isfinite(q):
                    raise FloatingPointError("non-finite action discrepancy")
                diagnostics = {
                    "q_t": float(q), "action_mse": float(q),
                    "student_action": student_action.cpu().tolist(),
                    "teacher_action": teacher_action.cpu().tolist(),
                    "action_space": "normalized_first_action_7d",
                }
            result = bundle.postprocessor(actions).detach().float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("policy produced non-finite actions")
        return {"actions": result.numpy(), "discrepancy": diagnostics}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=MODES, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7500)
    p.add_argument("--authkey", default="pi05")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if load_config(args.config)["mode"] != args.method:
        p.error("method and config mode must agree")
    server = PI05Server(args.config, args.device, args.adapter_path)
    with Listener((args.host, args.port), authkey=args.authkey.encode()) as listener:
        print(f"pi05 Omega {args.method} ready on {args.host}:{args.port}", flush=True)
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    if request.get("type") == "shutdown":
                        return
                    connection.send(server.actions(request))
            except EOFError:
                pass
            finally:
                connection.close()

if __name__ == "__main__":
    main()
