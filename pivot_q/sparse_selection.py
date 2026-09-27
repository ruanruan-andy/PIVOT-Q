"""Budget-matched paper baselines; no rollout or optimizer logic lives here."""
import copy
import random
from types import SimpleNamespace
import torch


def temporal_phase(pool, count, gap):
    """Nearest distinct valid indices to endpoint-inclusive temporal targets.

    Retry from scratch at each smaller gap, matching PIVOT-Q's relaxation.
    Equal distances prefer the earlier timestep; never borrow across phases.
    """
    if len(pool) < count:
        raise ValueError('phase has fewer valid states than its quota')
    targets = torch.linspace(pool[0], pool[-1], count).tolist()
    for effective_gap in range(gap, -1, -1):
        chosen = []
        for target in targets:
            candidates = [i for i in pool if i not in chosen and
                          all(abs(i-j) >= effective_gap for j in chosen)]
            if not candidates:
                break
            chosen.append(min(candidates, key=lambda i: (abs(i-target), i)))
        if len(chosen) == count:
            return chosen, effective_gap
    raise AssertionError('phase cannot satisfy quota')


def select_sparse(q, r, config, rng, legacy, ranks, select_phase):
    method = getattr(config, 'selection_method', 'pivot_q')
    weighting = getattr(config, 'selection_weighting', 'vulnerability')
    if method not in {'pivot_q','current_only','random_sparse','uniform_sparse','global_random_sparse','global_vulnerability'}:
        raise ValueError(f'unknown selection_method: {method}')
    if weighting not in {'vulnerability','uniform'}:
        raise ValueError(f'unknown selection_weighting: {weighting}')
    if method in {'pivot_q','current_only'}:
        settings = config
        if method == 'current_only':
            settings = copy.copy(config)
            settings.alpha_q, settings.beta_r = 1.0, 0.0
        result = legacy(q, r, settings, rng)
        if weighting == 'uniform':
            result.weights = torch.ones_like(result.weights)
        return result
    if method == 'global_vulnerability':
        # Global vulnerability ablation: preserve vulnerability scoring while
        # removing phase quotas and temporal separation.
        count = config.phase_bins * config.priority_per_phase
        if len(q) != len(r) or len(q) < count:
            raise ValueError(f'insufficient valid states for global budget {count}')
        scores = config.alpha_q * ranks(q) + config.beta_r * ranks(r)
        ordering = sorted(range(len(q)), key=lambda i: (-float(scores[i]), i))
        indices = sorted(ordering[:count])
        phases = [min(config.phase_bins - 1, i * config.phase_bins // len(q)) for i in indices]
        priorities = scores[torch.tensor(indices)]
        weights = torch.ones(count) if weighting == 'uniform' else (priorities / priorities.mean().clamp_min(1e-8)).clamp(config.weight_min, config.weight_max)
        gaps = [b - a for a, b in zip(indices, indices[1:])]
        return SimpleNamespace(indices=tuple(indices), weights=weights.detach(), phases=tuple(phases), reasons=tuple('global_vulnerability' for _ in indices), priority_scores=priorities.detach(), phase_effective_gaps=tuple(), actual_min_gap=min(gaps, default=0), gap_relaxed=False, phase_counts=[phases.count(p) for p in range(config.phase_bins)], priority_counts=[0] * config.phase_bins, random_counts=[0] * config.phase_bins, valid=True)
    if method == 'global_vulnerability':
        # Global vulnerability ablation: preserve vulnerability scoring while
        # removing phase quotas and temporal separation.
        count = config.phase_bins * config.priority_per_phase
        if len(q) != len(r) or len(q) < count:
            raise ValueError(f'insufficient valid states for global budget {count}')
        scores = config.alpha_q * ranks(q) + config.beta_r * ranks(r)
        ordering = sorted(range(len(q)), key=lambda i: (-float(scores[i]), i))
        indices = sorted(ordering[:count])
        phases = [min(config.phase_bins - 1, i * config.phase_bins // len(q)) for i in indices]
        priorities = scores[torch.tensor(indices)]
        weights = torch.ones(count) if weighting == 'uniform' else (priorities / priorities.mean().clamp_min(1e-8)).clamp(config.weight_min, config.weight_max)
        gaps = [b - a for a, b in zip(indices, indices[1:])]
        return SimpleNamespace(indices=tuple(indices), weights=weights.detach(), phases=tuple(phases), reasons=tuple('global_vulnerability' for _ in indices), priority_scores=priorities.detach(), phase_effective_gaps=tuple(), actual_min_gap=min(gaps, default=0), gap_relaxed=False, phase_counts=[phases.count(p) for p in range(config.phase_bins)], priority_counts=[0] * config.phase_bins, random_counts=[0] * config.phase_bins, valid=True)
    if method == 'global_random_sparse':
        # Paper diagnostic baseline: sample a fixed budget from the complete
        # valid trajectory, without phase quotas or temporal-gap constraints.
        count = config.phase_bins * config.priority_per_phase
        if len(q) != len(r) or len(q) < count:
            raise ValueError(f'insufficient valid states for global budget {count}')
        selection_rng = random.Random()
        selection_rng.setstate(rng.getstate())
        ordering = list(range(len(q)))
        selection_rng.shuffle(ordering)
        indices = sorted(ordering[:count])
        phases = [min(config.phase_bins - 1, i * config.phase_bins // len(q))
                  for i in indices]
        priorities = torch.zeros(count, dtype=torch.float32)
        weights = torch.ones(count, dtype=torch.float32)
        return SimpleNamespace(
            indices=tuple(indices), weights=weights.detach(), phases=tuple(phases),
            reasons=tuple('global_random' for _ in indices),
            priority_scores=priorities,
            phase_effective_gaps=tuple(),
            actual_min_gap=min((b - a for a, b in zip(indices, indices[1:])), default=0),
            gap_relaxed=False,
            phase_counts=[phases.count(p) for p in range(config.phase_bins)],
            priority_counts=[0] * config.phase_bins,
            random_counts=[0] * config.phase_bins,
            valid=True,
        )
    if config.random_per_phase:
        raise ValueError('paper sparse baselines require random_per_phase=0')
    if len(q) != len(r) or len(q) < config.phase_bins*config.priority_per_phase:
        raise ValueError('insufficient valid states for distinct phase quotas')
    # Clone, do not consume the shared task/anchor RNG. State advances elsewhere
    # each episode and is checkpointed, so selector replay also works on resume.
    selection_rng = random.Random()
    selection_rng.setstate(rng.getstate())
    phase_ids = [min(config.phase_bins-1, i*config.phase_bins//len(q)) for i in range(len(q))]
    scores = torch.zeros(len(q), dtype=torch.float32)
    indices, gaps = [], []
    for phase in range(config.phase_bins):
        pool = [i for i,p in enumerate(phase_ids) if p == phase]
        if len(pool) < config.priority_per_phase:
            raise ValueError(f'phase {phase} cannot satisfy quota without duplicates')
        pool_tensor = torch.tensor(pool, dtype=torch.long)
        scores[pool_tensor] = config.alpha_q*ranks(q[pool_tensor])+config.beta_r*ranks(r[pool_tensor])
        if method == 'random_sparse':
            ordering = pool.copy()
            selection_rng.shuffle(ordering)
            random_scores = torch.zeros(len(q))
            for position,index in enumerate(ordering):
                random_scores[index] = len(ordering)-position
            chosen, effective_gap = select_phase(pool, random_scores, config.priority_per_phase, config.min_temporal_gap)
        else:
            chosen, effective_gap = temporal_phase(pool, config.priority_per_phase, config.min_temporal_gap)
        indices.extend(chosen)
        gaps.append(effective_gap)
    indices.sort()
    phases = [phase_ids[i] for i in indices]
    priorities = scores[torch.tensor(indices)]
    weights = torch.ones(len(indices)) if weighting == 'uniform' else (
        priorities/priorities.mean().clamp_min(1e-8)).clamp(config.weight_min,config.weight_max)
    return SimpleNamespace(indices=tuple(indices),weights=weights.detach(),phases=tuple(phases),
        reasons=tuple(f'{method}_phase_{p}' for p in phases), priority_scores=priorities.detach(),
        phase_effective_gaps=tuple(gaps), actual_min_gap=min(b-a for a,b in zip(indices,indices[1:])),
        gap_relaxed=any(g<config.min_temporal_gap for g in gaps),
        phase_counts=[phases.count(p) for p in range(config.phase_bins)],
        priority_counts=[0]*config.phase_bins,
        random_counts=[config.priority_per_phase if method=='random_sparse' else 0]*config.phase_bins,
        valid=True)
