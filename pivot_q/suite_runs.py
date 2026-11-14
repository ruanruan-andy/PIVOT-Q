"""Run per-suite adapters through the shared training and evaluation commands."""
import json
import os
from pathlib import Path
import sys
import time

from .configuration import ROOT, result_root
from .processes import services, run_worker
from .evaluation import episode_specs, summarize


def plans(cfg, args, phase):
    seed = args.seed if args.seed is not None else cfg["train" if phase == "train" else "evaluation"]["seed"]
    root = result_root(cfg, args)
    out = root / "train" / "seed" / str(seed) if phase == "train" else root / "eval" / args.benchmark / "seed" / str(seed)
    for suite in cfg["benchmark"]["suites"]:
        adapter = None
        if phase == "eval" and args.method in {"pivot_q", "full_distill"}:
            suffix = "final_adapter.pt" if cfg["backbone"] == "openvla_oft" else "final_adapter"
            adapter = args.adapter or root / "train" / "seed" / str(args.train_seed) / suite / suffix
            adapter = str(adapter).replace("{suite}", suite)
        yield dict(suite=suite, seed=seed, method=args.method, phase=phase, output=str(out / suite),
                   adapter=adapter, checkpoint=str(Path(cfg["model"]["checkpoint"]) / suite),
                   pack=str(Path(cfg["model"]["quant_pack"]) / suite),
                   expected=len(episode_specs(cfg, args.benchmark, suite)) if phase == "eval" else cfg["train"]["episodes_per_suite"],
                   max_rounds=args.max_rounds if phase == "train" else 0,
                   resume=not args.no_resume if phase == "train" else args.resume,
                   port=args.port, clean_port=args.clean_port)


def oft_config(cfg, plan):
    t = {k: v for k, v in cfg["train"].items() if k not in {"seed", "estimate_flops", "episodes_per_suite"}}
    if plan["method"] == "full_distill":
        t.update(cfg["full_distill"])
    t.update(episodes=cfg["train"]["episodes_per_suite"], resume=None)
    out, pack = Path(plan["output"]), Path(plan["pack"])
    latest = sorted(out.glob("checkpoint-round-*.pt"))
    if plan["phase"] == "train" and plan["resume"] and latest:
        t["resume"] = str(latest[-1])
    return dict(
        method={"original": "fp", "quantized": "qvla"}.get(plan["method"], plan["method"]),
        suite=plan["suite"], seed=plan["seed"], training=t, selection=cfg["selection"],
        paths=dict(qvla_root=cfg["paths"]["qvla"], checkpoint=plan["checkpoint"], gates=str(pack / "gates.json"),
                   calibration=str(pack / "calibration.pt"), proxy=str(pack / "proxy.pt"),
                   manifest=cfg["benchmark"]["task_file"], output=str(out), adapter=plan["adapter"]),
        model=dict(cfg["oft_model"], unnorm_key=plan["suite"]),
        environment=dict(cfg["environment"], port=plan["port"], clean_port=plan["clean_port"], seed=plan["seed"]),
        evaluation=dict(cfg["evaluation"], open_loop_steps=cfg["evaluation"]["action_steps"], initial_state_ids=[0],
                        episode_horizon=None, max_episodes=None),
        quantization=cfg["quantization"], diagnostics=cfg["diagnostics"],
        max_rounds=plan["max_rounds"])


def groot_config(cfg, plan):
    settings = dict(cfg["selection"])
    if plan["method"] == "full_distill":
        settings.update(cfg["full_distill"])
    t = {k: v for k, v in cfg["train"].items() if k not in {"seed", "estimate_flops", "episodes_per_suite"}}
    pack = Path(plan["pack"])
    quant = cfg["quantizer"] == "quantvla"
    return dict(t, **settings, setting=cfg["setting"], task_suite_name=plan["suite"], output_dir=plan["output"],
                sample_manifest=cfg["benchmark"]["task_file"], num_rollout_episodes=cfg["train"]["episodes_per_suite"],
                env_port=plan["port"], clean_env_port=plan["clean_port"], seed=plan["seed"],
                method_name=plan["method"], model_path=plan["checkpoint"],
                quant_pack_dir=str(pack / ("duquant_pack" if quant else "quantized.pt")),
                calibration_path=str(pack / "atm_alpha_beta.json") if quant else None,
                denoising_steps=cfg["model"]["num_inference_steps"], resume=plan["resume"],
                max_rounds=plan["max_rounds"])


def run(cfg, args, phase):
    work = list(plans(cfg, args, phase))
    model_config = oft_config if cfg["backbone"] == "openvla_oft" else groot_config
    if args.dry_run:
        print(json.dumps([dict(p, worker_config=model_config(cfg, p)) for p in work], indent=2))
        return
    paths = cfg["paths"]
    upstream = []
    if cfg["backbone"] == "groot":
        upstream = ([paths["omega"]] if cfg["quantizer"] == "holoq" else []) + [paths["quantvla"]]
    model_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1",
                     TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                     PYTHONPATH=os.pathsep.join([str(ROOT), *upstream]),
                     PIVOT_Q_QUANTVLA_ROOT=paths.get("quantvla", str(ROOT)), TORCH_COMPILE_DISABLE="1")
    for plan in work:
        opts = model_config(cfg, plan)
        required = [Path(plan["checkpoint"])]
        if plan["method"] != "original":
            required += ([Path(opts["paths"][key]) for key in ("gates", "proxy", "calibration")]
                         if cfg["backbone"] == "openvla_oft" else [Path(opts["quant_pack_dir"])])
        if plan["adapter"]:
            path = Path(plan["adapter"])
            if cfg["backbone"] == "groot":
                path /= "adapter_model.pt" if cfg["quantizer"] == "holoq" else "adapter_model.safetensors"
            required.append(path)
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        out = Path(plan["output"])
        out.mkdir(parents=True, exist_ok=True)
        anchor = phase == "train" and (opts["training"]["anchor_lambda"] if cfg["backbone"] == "openvla_oft" else opts["lambda_anchor"])
        payload = dict(model=opts, experiment=cfg, plan=plan,
                       benchmark=getattr(args, "benchmark", "libero_plus"))
        started = time.monotonic()
        command = [sys.executable, "-m", "pivot_q.suite_worker", json.dumps(payload)]
        if phase == "train":
            with services(cfg, args, out, anchor=anchor, suite=plan["suite"]) as processes:
                run_worker(command, env=model_env, cwd=ROOT, processes=processes)
        else:
            from .processes import managed_processes, start
            with managed_processes() as processes:
                processes.append(start(command, model_env, out / "logs/model.log"))
                evaluator_env = dict(model_env, PYTHONPATH=str(ROOT), MUJOCO_GL="egl",
                                     QT_QPA_PLATFORM="offscreen")
                evaluator = [cfg["paths"]["evaluation_python"], "-m", "pivot_q.evaluation", json.dumps(payload)]
                run_worker(evaluator, env=evaluator_env, cwd=ROOT, processes=processes)
        with (out / "timing.jsonl").open("a") as stream:
            seconds = time.monotonic() - started
            stream.write(json.dumps(dict(wall_seconds=seconds, allocated_gpu_hours=seconds/3600)) + "\n")
        summarize_seed(work, phase)
        summarize_compute(work)


def summarize_compute(plans):
    suites = {}
    for plan in plans:
        out = Path(plan["output"])
        path, timing = out / "compute.json", out / "timing.jsonl"
        if not path.is_file() or not timing.is_file():
            continue
        record = json.loads(path.read_text())
        seconds = sum(json.loads(line)["wall_seconds"] for line in timing.read_text().splitlines())
        suites[plan["suite"]] = dict(record, wall_seconds=seconds, allocated_gpu_hours=seconds/3600)
    if suites:
        values = [record["estimated_covered_flops"] for record in suites.values()]
        data = dict(suites=suites, recorded_suites=len(suites), expected_suites=len(plans),
                    wall_seconds=sum(r["wall_seconds"] for r in suites.values()),
                    allocated_gpu_hours=sum(r["allocated_gpu_hours"] for r in suites.values()),
                    estimated_covered_flops=sum(values) if all(v is not None for v in values) else None)
        from .pi05_state import write_status
        write_status(Path(plans[0]["output"]).parent / "compute_summary.json", data)


def summarize_seed(plans, phase):
    if phase != "eval":
        return
    rows, expected = [], {}
    for plan in plans:
        suite = plan["suite"]
        expected[suite] = plan["expected"]
        file = Path(plan["output"]) / "metrics/episodes.jsonl"
        if file.is_file():
            rows.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    from .pi05_state import write_status
    write_status(Path(plans[0]["output"]).parent / "summary.json", summarize(rows, expected))
