"""Integration-local marginal-cost allocation; official QVLA stays unchanged."""
import heapq
import math
import random


def calibration_plan(tasks, initial, count, seed, horizon):
    if count <= 0 or not tasks or not initial or horizon <= 0:
        raise ValueError("positive sample count/horizon and nonempty IDs required")
    if len(set(tasks)) != len(tasks) or len(set(initial)) != len(initial):
        raise ValueError("duplicate calibration IDs")
    rng = random.Random(seed)
    tasks, initial = list(tasks), list(initial)
    rng.shuffle(tasks)
    rng.shuffle(initial)
    # One state per rollout: balanced tasks, different initial states across rounds.
    return [(tasks[i % len(tasks)], initial[(i // len(tasks)) % len(initial)],
             rng.randrange(max(1, horizon // 2))) for i in range(count)]


def marginal_allocate(proxies, bits, target, max_zero_fraction=0.0, min_bit=4):
    bits = sorted(set(bits))
    if not bits or bits[-1] != 16 or not set(bits) <= {0, 2, 4, 8, 16}:
        raise ValueError("invalid bit choices")
    if min_bit not in bits or min_bit < 2 or not math.isfinite(target) or not 0 <= max_zero_fraction <= 1:
        raise ValueError("invalid budget or zero fraction")
    values, assigned, limits, zeros = {}, {}, {}, {}
    for name in sorted(proxies):
        row = {b: (proxies[name][b].tolist() if hasattr(proxies[name][b], "tolist")
                   else list(proxies[name][b])) for b in bits}
        size = len(row[16])
        if not size or any(len(v) != size for v in row.values()):
            raise ValueError(f"invalid proxy shape: {name}")
        if any(not math.isfinite(x) or x < 0 for v in row.values() for x in v):
            raise ValueError(f"invalid proxy values: {name}")
        values[name], assigned[name] = row, [16] * size
        limits[name], zeros[name] = math.floor(size * max_zero_fraction), 0
    n = sum(map(len, assigned.values()))
    if not n or not bits[0] <= target <= 16:
        raise ValueError("invalid target or empty proxies")
    floor_bit = next((b for b in bits if b > 0), 16)
    minimum = sum((len(v) - limits[k]) * floor_bit for k, v in assigned.items()) if bits[0] == 0 else n * bits[0]
    if target * n < minimum:
        raise ValueError("budget infeasible under zero-bit constraint")
    heap = []
    def push(name, i, pos):
        if pos == 0:
            return
        old, new = bits[pos], bits[pos - 1]
        if new < min_bit or (new == 0 and zeros[name] >= limits[name]):
            return
        cost = values[name][new][i] - values[name][old][i]
        heapq.heappush(heap, (cost / (old - new), name, i, pos))
    for name, v in assigned.items():
        for i in range(len(v)):
            push(name, i, len(bits) - 1)
    total = n * 16
    while total > target * n:
        if not heap:
            raise ValueError("budget infeasible under zero-bit constraint")
        _, name, i, pos = heapq.heappop(heap)
        old, new = bits[pos], bits[pos - 1]
        if new < min_bit:
            raise ValueError("budget infeasible under min_bit constraint")
        if new == 0:
            if zeros[name] >= limits[name]:
                continue
            zeros[name] += 1
        assigned[name][i] = new
        total -= old - new
        push(name, i, pos - 1)
    return assigned, {"allocator": "marginal_v1", "channel_avg_bits": total / n,
                      "target_avg_bits": target, "min_bit": min_bit, "max_zero_fraction": max_zero_fraction,
                      "channels": n, "zero_channels": sum(zeros.values())}
