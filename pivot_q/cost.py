"""Sample covered PyTorch operator FLOPs and count actual invocations."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
from time import monotonic

_active = ContextVar("compute_meter", default=None)
_prefix = ContextVar("compute_prefix", default="")


class ComputeMeter:
    def __init__(self, enabled):
        self.enabled = enabled
        self.samples = {}
        self.calls = {}
        self.seconds = {}

    @contextmanager
    def measure(self, stage, size=1):
        key = f"{_prefix.get()}{stage}/{size}"
        started = monotonic()
        profile_key = os.environ.get("PIVOT_Q_FLOPS_PROFILE_KEY")
        profile_suite = os.environ.get("PIVOT_Q_FLOPS_PROFILE_SUITE", "")
        profile_match = profile_key == key.removeprefix(profile_suite + "/")
        if ((profile_match if profile_key else self.enabled) and key not in self.samples):
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as counter:
                yield
            self.samples[key] = counter.get_total_flops()
            if profile_match:
                output = Path(os.environ["PIVOT_Q_FLOPS_PROFILE_OUTPUT"])
                output.write_text(json.dumps({
                    "suite": os.environ["PIVOT_Q_FLOPS_PROFILE_SUITE"],
                    "key": key, "flops": self.samples[key],
                }, indent=2) + "\n")
                raise SystemExit(0)
        else:
            yield
        self.calls[key] = self.calls.get(key, 0) + 1
        self.seconds[key] = self.seconds.get(key, 0) + monotonic() - started

    def run(self, stage, size, fn):
        with self.measure(stage, size):
            return fn()

    def summary(self):
        stages = [(key.rsplit("/", 2)[-2:], count) for key, count in self.calls.items()]
        return {
            "enabled": self.enabled,
            "optimizer_steps": sum(count for (name, _), count in stages if name == "optimizer_step"),
            "supervised_states": {
                stage: sum(int(size) * count for (name, size), count in stages if name == stage)
                for stage in ("distill_forward_backward", "anchor_forward_backward")
            },
            "method": "first observed call per stage and batch size, multiplied by actual call count",
            "coverage": "PyTorch FlopCounterMode supported operators only; excludes simulator and unsupported kernels",
            "stage_host_seconds": dict(self.seconds),
            "samples": dict(self.samples),
            "calls": dict(self.calls),
            "estimated_covered_flops": sum(n*self.samples[k] for k,n in self.calls.items() if k in self.samples)
                                       if self.enabled else None,
        }

    def restore(self, record):
        if record["enabled"] != self.enabled:
            raise ValueError("Use the same FLOPs setting when resuming")
        self.samples = dict(record["samples"])
        self.calls = dict(record["calls"])
        self.seconds = dict(record.get("stage_host_seconds", {}))


@contextmanager
def scope(prefix):
    token = _prefix.set(_prefix.get() + prefix)
    try:
        yield
    finally:
        _prefix.reset(token)


@contextmanager
def record_compute(output, enabled, *, resume=False, gpu=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "compute.json"
    meter = ComputeMeter(enabled)
    previous = json.loads(path.read_text()) if resume and path.is_file() else None
    if previous:
        meter.restore(previous)
    token = _active.set(meter)
    started = monotonic()
    try:
        yield meter
    finally:
        seconds = (previous or {}).get("wall_seconds", 0) + monotonic() - started
        result = dict(meter.summary(), wall_seconds=seconds, allocated_gpu_hours=seconds/3600 if gpu else 0)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(path)
        _active.reset(token)


def snapshot():
    meter = _active.get()
    return meter.summary() if meter is not None else None


def restore(record):
    meter = _active.get()
    if meter is not None and record is not None:
        meter.restore(record)


def measured(stage, fn, *args, **kwargs):
    meter = _active.get()
    if meter is None:
        return fn(*args, **kwargs)
    return meter.run(stage, 1, lambda: fn(*args, **kwargs))


@contextmanager
def measure(stage):
    meter = _active.get()
    if meter is None:
        yield
    else:
        with meter.measure(stage):
            yield


def measured_batch(stage, size, fn, *args, **kwargs):
    meter = _active.get()
    if meter is None:
        return fn(*args, **kwargs)
    return meter.run(stage, size, lambda: fn(*args, **kwargs))
