"""Balanced clean-demonstration sampling for π0.5 quantization."""
import json
from pathlib import Path
import random

def select_frames(episodes, task_suites, suites, count, seed):
    """Return a deterministic plan keyed by explicit clean-LIBERO task names."""
    if count % (4 * len(suites)):
        raise ValueError("Frame budget must be divisible by four frames per trajectory per suite")
    per_suite = count // (4 * len(suites))
    if not per_suite:
        raise ValueError("Positive frame budget required")
    groups = {s: {} for s in suites}
    for item in episodes:
        tasks = item["tasks"]
        if isinstance(tasks, str):
            tasks = [tasks]
        start, end = int(item["dataset_from_index"]), int(item["dataset_to_index"])
        if len(tasks) == 1 and task_suites.get(tasks[0]) in groups and end - start >= 4:
            groups[task_suites[tasks[0]]].setdefault(tasks[0], []).append(
                (int(item["episode_index"]), start, end))
    rng = random.Random(seed)
    plan = []
    for suite in suites:
        tasks = sorted(groups[suite])
        if len(tasks) < per_suite:
            raise ValueError(f"{suite}: mapped {len(tasks)} eligible tasks; need {per_suite}")
        rng.shuffle(tasks)
        for task in tasks[:per_suite]:
            episode, start, end = rng.choice(sorted(groups[suite][task]))
            for phase in range(4):
                lo, hi = start + (end-start)*phase//4, start + (end-start)*(phase+1)//4
                plan.append(dict(suite=suite, task=task, episode=episode, frame=(lo+hi-1)//2))
    return plan

def collect(cfg, dataset_root, task_map, dest):
    import sys
    import torch
    from .configuration import ROOT
    sys.path.insert(0, str(ROOT / "third_party/lerobot/src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    cal = cfg["calibration"]
    dataset = LeRobotDataset(cal["dataset"], root=dataset_root, return_uint8=True)
    dataset.meta.ensure_readable()
    episodes = [dict(row) for row in dataset.meta.episodes]
    if episodes and "tasks" not in episodes[0]:
        import pyarrow.parquet as pq
        tasks = dataset.meta.tasks
        indices = {}
        for path in sorted((dataset_root / "data").glob("*/*.parquet")):
            rows = pq.read_table(path, columns=["episode_index", "task_index"])
            for episode, task in set(zip(rows["episode_index"].to_pylist(), rows["task_index"].to_pylist())):
                indices.setdefault(episode, set()).add(task)
        task_names = {int(row["task_index"]): name for name, row in tasks.iterrows()}
        for row in episodes:
            row["tasks"] = [task_names[index] for index in sorted(indices[row["episode_index"]])]
    plan = select_frames(episodes, json.loads(task_map.read_text()),
                         cfg["benchmark"]["suites"], cal["num_samples"], cal["seed"])
    fields = ("observation.images.image", "observation.images.image2", "observation.state", "task")
    samples = []
    for row in plan:
        item = dataset[row["frame"]]
        if int(item["episode_index"]) != row["episode"]:
            raise ValueError("Dataset frame indices do not match episode metadata")
        samples.append({key: item[key] for key in fields})
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    torch.save({"dataset": cal["dataset"], "revision": dataset.meta.revision, "seed": cal["seed"],
                "plan": plan, "samples": samples}, dest)
    (dest.parent / "sampling.json").write_text(json.dumps(plan, indent=2) + "\n")
    return dest

def build(cfg, buffer, pack, *, resume):
    from types import SimpleNamespace
    from .holoq_calibration import build as build_holoq
    if cfg["quantizer"] == "quantvla":
        return build_quantvla(cfg, buffer, pack)
    cal = cfg["calibration"]
    raw = dict(calibration={**cal, "observations": cal["num_samples"]}, quantization=cfg["quantization"],
               paths={"checkpoint": cfg["model"]["checkpoint"], "omega_source": cfg["paths"]["omega"],
                      "calibration_buffer": str(buffer), "quant_pack_dir": str(pack)},
               inference={"num_inference_steps": cfg["model"]["num_inference_steps"]})
    build_holoq(raw, SimpleNamespace(device="cuda:0", output=pack, buffer=buffer,
                                     resume=resume, dry_run=False, max_layers=0))

def build_quantvla(cfg, buffer, pack):
    import torch
    from tqdm.auto import tqdm
    from .atm_ohb import AttentionStatistics, compute_scales
    from .configuration import ROOT
    from .pi05_flow import sample
    from .pi05_policy import load_policy
    from .pi05_quantvla import PI05QuantVLAConfig, apply_quantvla_layout
    from lerobot.policies import make_pre_post_processors

    checkpoint = Path(cfg["model"]["checkpoint"])
    data = torch.load(buffer, map_location="cpu", weights_only=False)
    frames = data["samples"]
    assert len(frames) == cfg["calibration"]["num_samples"]
    quant = PI05QuantVLAConfig(**cfg["quantization"])

    def collect(quantized):
        policy = load_policy(ROOT, checkpoint, device="cuda:0")
        policy.config.num_inference_steps = cfg["model"]["num_inference_steps"]
        targets = apply_quantvla_layout(policy.model, quant, pack_dir=pack) if quantized else []
        pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda:0"}})
        observer = AttentionStatistics(policy.model)
        for index, state in enumerate(tqdm(frames, desc="Quantized" if quantized else "Teacher", unit="frame")):
            generator = torch.Generator(device="cuda:0").manual_seed(cfg["calibration"]["seed"] + index)
            noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim),
                                device="cuda:0", generator=generator)
            sample(policy, pre(state), noise, with_grad=False)
        stats = observer.finalize()
        observer.close()
        return stats, targets

    teacher, _ = collect(False)
    torch.cuda.empty_cache()
    student, targets = collect(True)
    settings = cfg["quantization"]
    scales = compute_scales(teacher, student, log_clip=settings["log_scale_clip"],
                            neutral=settings["neutrality_band"])
    assert scales and len(list(pack.glob("*.npz"))) == len(targets)
    for name, value in {
        "atm_ohb.json": {"buffer": str(buffer), "seed": data["seed"], "scales": scales},
        "quantvla_manifest.json": {"checkpoint": str(checkpoint), "quantization": settings,
                                  "target_count": len(targets), "complete": True},
    }.items():
        path = pack / name
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(path)
