"""Round-based on-policy distillation with one optimizer update per state batch."""
from collections import deque
from contextlib import ExitStack
import json
import random
import shutil
import time
from pathlib import Path
import torch
from tqdm.auto import tqdm
from pivot_q.cost import measured, measured_batch, restore, scope, snapshot
from pivot_q.plot_data import save_selection
from pivot_q.oft_runtime.evaluation.environment import EnvClient, episode_specs
from pivot_q.oft_runtime.integration.config import atomic_json, resolve
from pivot_q.oft_runtime.integration.policy import OFTPolicy
from pivot_q.oft_runtime.training import checkpoint
from pivot_q.oft_runtime.training.collect import collect
from pivot_q.oft_runtime.training.selection import select

def backward(student, records, weights):
    features = torch.cat([row['features'] for row in records])
    predicted = student.predict_features(features)
    target = torch.cat([row['target'] for row in records]).to(predicted)
    errors = (predicted[:, 0, :7] - target[:, 0, :7]).square().mean(-1)
    loss = (errors * errors.new_tensor(weights)).sum()
    loss.backward()
    return float(loss.detach())

def update(student, optimizer, records, weights, anchors, cfg):
    student.head.train()
    optimizer.zero_grad(set_to_none=True)
    loss = measured_batch('distill_forward_backward', len(records), backward, student, records, weights)
    anchor_loss = measured_batch('anchor_forward_backward', len(anchors), backward, student, anchors, [cfg['training']['anchor_lambda'] / len(anchors)] * len(anchors)) if anchors else 0.0
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['training']['gradient_clip'], error_if_nonfinite=True)
    measured('optimizer_step', optimizer.step)
    student.head.eval()
    return dict(loss=loss, anchor_loss=anchor_loss, gradient_norm=float(norm))

def train(cfg, *, policy_factory=OFTPolicy, collect_fn=collect):
    checkpoint.seed_all(cfg['seed'])
    t = cfg['training']
    output = resolve(cfg['paths']['output'])
    output.mkdir(parents=True, exist_ok=True)
    resume = resolve(t['resume'])
    specs = episode_specs(cfg)
    sampler = random.Random(cfg['seed'])
    schedule = list(range(len(specs)))
    sampler.shuffle(schedule)
    total = t['episodes'] // t['rollouts_per_round']
    with ExitStack() as stack:
        client = stack.enter_context(EnvClient(cfg))
        clean = stack.enter_context(EnvClient(cfg, clean=True)) if t['anchor_lambda'] else None
        teacher = policy_factory(cfg, teacher=True)
        student = policy_factory(cfg, quantized=True, trainable=True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
        replay = deque(maxlen=t['anchor_capacity'])
        (start, step) = (0, 0)
        manifest = json.loads(resolve(cfg['paths']['manifest']).read_text())
        if resume:
            saved = checkpoint.read(resume, student, cfg, resume=True)
            if saved['manifest'] != manifest:
                raise ValueError('Task manifest differs from checkpoint')
            restore(saved.get('compute'))
            optimizer.load_state_dict(saved['optimizer'])
            (start, step) = (saved['round'], saved['step'])
            schedule = saved['schedule']
            sampler.setstate(saved['sampler'])
            replay.extend(saved['replay'])
            checkpoint.restore_rng(saved['rng'])
            checkpoint.truncate_metrics(output / 'metrics.jsonl', start)
        checkpoint.truncate_metrics(output / 'metrics.jsonl', start)
        end = min(total, start + cfg['max_rounds']) if cfg['max_rounds'] else total
        atomic_json(output / 'run.json', dict(config=cfg, format='oft_round_v2', manifest=manifest))
        for index in tqdm(range(start, end), initial=start, total=total, desc=cfg['suite'], unit='round'):
            started = time.monotonic()
            (weighted, records) = ([], [])
            for offset in range(t['rollouts_per_round']):
                spec = specs[schedule[index * t['rollouts_per_round'] + offset]]
                (rollout, outcome) = collect_fn(cfg, student, client, spec, teacher=teacher, capture_actions=False)
                if not rollout:
                    raise ValueError('Empty student trajectory')
                selection = select([row['q'] for row in rollout], cfg['method'], cfg['selection'])
                if cfg['method'] == 'pivot_q':
                    save_selection(output, setting='openvla_oft_qvla', suite=cfg['suite'], seed=cfg['seed'], round_index=index + 1,
                        suite_round=index + 1, rollout=offset, task_id=spec['task_id'],
                        initial_state_id=spec['initial_state_id'], method=cfg['method'],
                        q=selection['q'], risk=selection['risk'], scores=selection['scores'],
                        indices=selection['indices'], weights=selection['weights'],
                        phases=cfg['selection']['phase_bins'], percent=cfg['selection'].get('percent'))
                states = [rollout[i] for i in selection['indices']]
                weights = selection['weights'] * len(states)
                weighted.extend(zip(states, weights.tolist()))
                records.append(dict(spec, **outcome, selected=len(states)))
                if clean:
                    clean_spec = dict(task_id=sampler.choice(t['clean_task_ids']), initial_state_id=sampler.choice(t['clean_initial_state_ids']))
                    with scope('anchor/'):
                        (anchors, _) = collect_fn(cfg, student, clean, clean_spec, teacher=teacher, horizon=t['anchor_horizon'])
                    replay.extend(anchors)
            sampler.shuffle(weighted)
            losses = []
            for offset in range(0, len(weighted), t['batch_size']):
                group = weighted[offset:offset + t['batch_size']]
                mass = sum((weight for (_, weight) in group))
                anchors = sampler.sample(list(replay), min(t['anchor_batch_size'], len(replay))) if clean else []
                loss = update(student, optimizer, [r for (r, _) in group], [w / mass for (_, w) in group], anchors, cfg)
                losses.append((loss, mass, len(group)))
                step += 1
            row = dict(round=index + 1, episode=index + 1, optimizer_step=step, optimizer_steps_this_round=len(losses), rollouts_complete=(index + 1) * t['rollouts_per_round'], selected_states=len(weighted), state_evaluations=len(weighted), streams=records, distill_loss=sum((loss['loss'] * mass for (loss, mass, _) in losses)) / sum((m for (_, m, _) in losses)), anchor_loss=sum((loss['anchor_loss'] * n for (loss, _, n) in losses)) / len(weighted), duration_seconds=time.monotonic() - started)
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            atomic_json(output / 'status.json', row)
            if (index + 1) % t['save_every_rounds'] == 0 or index + 1 == end:
                path = output / f'checkpoint-round-{index + 1:06d}.pt'
                checkpoint.save(path, student, cfg, round=index + 1, episode=index + 1, step=step, optimizer=optimizer.state_dict(), replay=list(replay), sampler=sampler.getstate(), schedule=schedule, manifest=manifest, compute=snapshot())
                atomic_json(output / 'latest.json', dict(checkpoint=path.name, round=index + 1))
                for old in sorted(output.glob('checkpoint-round-*.pt'))[:-t['keep_last_checkpoints']]:
                    old.unlink()
        if end == total:
            path = output / f'checkpoint-round-{total:06d}.pt'
            shutil.copyfile(path, output / 'final_adapter.pt')
