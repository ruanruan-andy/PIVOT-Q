"""Train PIVOT-Q or full distill; only selection differs between the methods."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
import json
import math
import random

import torch

from evaluation.environment import EnvClient, episode_specs, validate_service
from integration.config import atomic_json, parse_config, resolve
from integration.policy import OFTPolicy
from training import checkpoint
from training.collect import collect
from training.selection import select
from evaluation.diagnostics import definition, save_episode


def update(student, optimizer, records, weights, anchors, cfg):
    student.head.train()
    optimizer.zero_grad(set_to_none=True)
    main_loss = anchor_loss = 0.0
    for record, weight in zip(records, weights, strict=True):
        prediction = student.predict_features(record["features"])
        count = record["action_count"]
        target = record["target"].to(prediction.device)
        loss = (prediction[:, :count] - target[:, :count]).square().mean()
        (loss * float(weight)).backward()
        main_loss += float(loss.detach()) * float(weight)
    if anchors:
        for record in anchors:
            prediction = student.predict_features(record["features"])
            count = record["action_count"]
            loss = (prediction[:, :count] - record["target"].to(prediction.device)[:, :count]).square().mean()
            (cfg["training"]["anchor_lambda"] * loss / len(anchors)).backward()
            anchor_loss += float(loss.detach()) / len(anchors)
    if not any(p.grad is not None for p in student.parameters()):
        raise RuntimeError("no gradients reached the adapters")
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    student.head.eval()
    return {"loss": main_loss, "anchor_loss": anchor_loss, "gradient_norm": float(norm)}


def train(cfg):
    if cfg["method"] not in {"pivot_q", "full_distill"}:
        raise ValueError("training requires pivot_q or full_distill")
    if cfg["paths"]["adapter"]:
        raise ValueError("use training.resume to resume; paths.adapter is only for evaluation")
    checkpoint.seed_all(cfg["seed"])
    t = cfg["training"]
    output = resolve(cfg["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    resume = resolve(t["resume"])
    if not resume and ((output / "metrics.jsonl").exists() or list(output.glob("checkpoint-*.pt"))):
        raise FileExistsError("training output already exists; select an empty output or an explicit resume checkpoint")
    specs = episode_specs(cfg)
    sampler = random.Random(cfg["seed"])
    schedule = list(range(len(specs)))
    sampler.shuffle(schedule)
    with ExitStack() as stack:
        client = stack.enter_context(EnvClient(cfg))
        validate_service(client, cfg)
        clean = stack.enter_context(EnvClient(cfg, clean=True)) if t["anchor_lambda"] else None
        teacher = OFTPolicy(cfg, teacher=True)
        student = OFTPolicy(cfg, quantized=True, trainable=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        total = t["episodes"] * t["updates_per_episode"]

        def lr_scale(step):
            if step < t["warmup_steps"]:
                return (step + 1) / max(1, t["warmup_steps"])
            progress = min(1, (step - t["warmup_steps"]) / max(1, total - t["warmup_steps"]))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        replay = deque(maxlen=t["anchor_capacity"])
        episode, step = 0, 0
        manifest_data = json.loads(resolve(cfg["paths"]["manifest"]).read_text())
        if resume:
            saved = checkpoint.read(resume, student, cfg, resume=True)
            if saved["manifest"] != manifest_data:
                raise ValueError("resume manifest changed")
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            episode, step = saved["episode"], saved["step"]
            schedule = saved["schedule"]
            sampler.setstate(saved["sampler"])
            replay.extend(saved["replay"])
            checkpoint.restore_rng(saved["rng"])
            checkpoint.truncate_metrics(output / "metrics.jsonl", episode)
            for path in (output / "states").glob("episode-*.npz"):
                if int(path.stem.split("-")[-1]) > episode:
                    path.unlink()  # Only diagnostic episodes beyond the restored checkpoint.
        atomic_json(output / "run.json", {"config": cfg, "student": student.identity,
                                         "teacher": teacher.identity, "manifest": manifest_data,
                                         "diagnostics": definition(cfg),
                                         "adapter_targets": student.adapter_targets})
        for episode in range(episode + 1, t["episodes"] + 1):
            position = (episode - 1) % len(schedule)
            if position == 0 and episode > 1:
                sampler.shuffle(schedule)
            spec = specs[schedule[position]]
            save_states = cfg.get("diagnostics", {}).get("save_train_states", True)
            records, outcome = collect(cfg, student, client, spec, teacher=teacher, capture_actions=save_states)
            if not records:
                raise RuntimeError(f"episode {episode}: no valid training states")
            selection = select([record["q"] for record in records], cfg["method"], cfg["selection"])
            diagnostic = save_episode(output, cfg, spec, episode, records, outcome,
                                      phase="train_pre_update", policy_seed=cfg["seed"], selection=selection) if save_states else {}
            states = [records[i] for i in selection["indices"]]
            if clean:
                clean_spec = {"task_id": sampler.choice(t["clean_task_ids"]),
                              "initial_state_id": sampler.choice(t["clean_initial_state_ids"])}
                anchors, _ = collect(cfg, student, clean, clean_spec, teacher=teacher, horizon=t["anchor_horizon"])
                replay.extend(anchors)
            losses = []
            for _ in range(t["updates_per_episode"]):
                anchors = sampler.sample(list(replay), min(t["anchor_batch_size"], len(replay))) if clean else []
                losses.append(update(student, optimizer, states, selection["weights"], anchors, cfg))
                scheduler.step()
                step += 1
            row = {"episode": episode, "optimizer_step": step, **spec, **outcome, **diagnostic,
                   "selected_states": len(states), "state_evaluations": len(states) * t["updates_per_episode"],
                   "selection": {**selection, "weights": selection["weights"].tolist()},
                   "updates": losses, "anchor_buffer_size": len(replay)}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("episode", "success", "states", "selected_states", "optimizer_step")}), flush=True)
            if episode % t["save_every_episodes"] == 0 or episode == t["episodes"]:
                path = output / f"checkpoint-{episode:06d}.pt"
                checkpoint.save(path, student, cfg, episode=episode, step=step,
                                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                replay=list(replay), sampler=sampler.getstate(), schedule=schedule,
                                manifest=manifest_data)
                atomic_json(output / "latest.json", {"checkpoint": path.name, "episode": episode})
                for old in sorted(output.glob("checkpoint-*.pt"))[:-t["keep_last_checkpoints"]]:
                    old.unlink()


def main():
    cfg, dry = parse_config(__doc__, phase='train')
    if dry:
        print(json.dumps(cfg, indent=2))
    else:
        train(cfg)


if __name__ == "__main__":
    main()
