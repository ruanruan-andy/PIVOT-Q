import os
import re
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


DEFAULT_PACK_DIR = os.environ.get("OPENPI_DUQUANT_PACKDIR", "./duquant_packed")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_name(name: str) -> str:
    # Replace characters not allowed in filenames
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    name = name.replace("..", ".")
    return name


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


class _DuQuantProfiler:
    """Optional profiler for fake quantization cost."""

    def __init__(self) -> None:
        flag = os.environ.get("OPENPI_DUQUANT_PROFILE", "0")
        self.enabled = flag not in ("0", "false", "False")
        sync_flag = os.environ.get("OPENPI_DUQUANT_PROFILE_SYNC", "1")
        self.sync_cuda = sync_flag not in ("0", "false", "False")
        self._stats: Dict[str, Dict[str, float]] = self._new_store()
        if self.enabled:
            import atexit

            atexit.register(self.report)

    @staticmethod
    def _new_store() -> Dict[str, Dict[str, float]]:
        return defaultdict(
            lambda: {
                "time": 0.0,
                "count": 0.0,
                "elements": 0.0,
                "bytes": 0.0,
            }
        )

    def record(self, label: str, tensor: torch.Tensor, scale: torch.Tensor, bits: int, fn):
        if not self.enabled:
            return fn()

        devices = []
        if self.sync_cuda:
            if tensor.is_cuda:
                devices.append(tensor.device)
            if isinstance(scale, torch.Tensor) and scale.is_cuda:
                devices.append(scale.device)

        for device in devices:
            torch.cuda.synchronize(device=device)
        start = time.perf_counter()
        result = fn()
        for device in devices:
            torch.cuda.synchronize(device=device)
        elapsed = time.perf_counter() - start

        stats = self._stats[label]
        stats["time"] += elapsed
        stats["count"] += 1
        stats["elements"] += tensor.numel()
        stats["bytes"] += tensor.numel() * tensor.element_size()
        if isinstance(scale, torch.Tensor):
            stats["bytes"] += scale.numel() * scale.element_size()
        stats["bits"] = float(bits)
        return result

    def report(self, *, reset: bool = False, header_suffix: Optional[str] = None) -> None:
        if not self.enabled or not self._stats:
            return
        print("=" * 100)
        title = (
            "[DUQUANT][PROFILE] fake quantization summary "
            f"(cuda_sync={'on' if self.sync_cuda else 'off'})"
        )
        if header_suffix:
            title += f" {header_suffix}"
        print(title)
        header = (
            f"{'Label':<28} {'Calls':>8} {'Total ms':>12} {'Avg ms':>10} "
            f"{'Elems':>14} {'GB/s':>10}"
        )
        print(header)
        print("-" * len(header))
        for label, stats in sorted(self._stats.items()):
            total_time = stats["time"]
            calls = stats["count"]
            avg_time = total_time / calls if calls else 0.0
            elems = int(stats["elements"])
            gbps = (stats["bytes"] / total_time / 1e9) if total_time > 0 else 0.0
            print(
                f"{label:<28} {int(calls):>8d} {total_time * 1000:12.2f} "
                f"{avg_time * 1000:10.3f} {elems:14d} {gbps:10.2f}"
            )
        print("=" * 100)
        if reset:
            self._stats = self._new_store()


_DUQUANT_PROFILER = _DuQuantProfiler()


def fake_quantize_sym(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    *,
    label: Optional[str] = None,
) -> torch.Tensor:
    if bits <= 0:
        return x

    def _impl() -> torch.Tensor:
        max_q = qmax(bits)
        x_scaled = x / scale
        x_clamped = torch.clamp(torch.round(x_scaled), -max_q - 1, max_q)
        return x_clamped * scale

    tag = label or "fake_quantize_sym"
    return _DUQUANT_PROFILER.record(tag, x, scale, bits, _impl)


@dataclass
class PackResult:
    # Input-side (columns) transforms
    R_in_blocks: Optional[Dict[int, np.ndarray]]  # block_index -> R_in (BxB)
    perm: Optional[np.ndarray]  # permutation indices over in_features
    # Output-side (rows) transforms
    R_out_blocks: Optional[Dict[int, np.ndarray]]  # block_index -> R_out (BxB)
    # Per-output-channel scale (for quant)
    weight_scale: np.ndarray
    meta: Dict[str, Any]


def _block_indices(in_features: int, block_size: int) -> Tuple[np.ndarray, int]:
    idx = np.arange(in_features, dtype=np.int64)
    n_blocks = (in_features + block_size - 1) // block_size
    return idx, n_blocks


def compute_block_rotation(W_block: np.ndarray) -> np.ndarray:
    """Compute a stable orthonormal rotation for input blocks using GPU-accelerated SVD.

    Returns R in R^{B x B} with high orthogonality accuracy.
    W_block shape: [out_features, B], X = W_block^T has shape [B, out_features].

    Note: Uses PyTorch GPU SVD for 10-50x speedup. Mathematically equivalent to NumPy CPU SVD.
    """
    X = W_block.T  # [B, out]
    B = X.shape[0]

    # Try GPU SVD first (much faster), fallback to CPU NumPy SVD
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_torch = torch.from_numpy(X.astype(np.float32)).to(device)
        U_torch, _, _ = torch.linalg.svd(X_torch, full_matrices=True)
        U = U_torch.cpu().numpy().astype(np.float64)
    except Exception:
        # Fallback to CPU NumPy SVD
        try:
            U, _, _ = np.linalg.svd(X.astype(np.float64, copy=False), full_matrices=True)
        except np.linalg.LinAlgError:
            U = np.eye(B, dtype=np.float64)

    if U.shape[1] < B:
        pad = np.zeros((B, B - U.shape[1]), dtype=U.dtype)
        U = np.concatenate([U, pad], axis=1)
    return U[:, :B]


def zigzag_permutation(energy: np.ndarray) -> np.ndarray:
    # energy shape: [in_features], larger means earlier
    order = np.argsort(-energy)  # descending
    # Zigzag interleave from both ends of the sorted list to spread large values
    left, right = 0, len(order) - 1
    perm = []
    toggle = True
    while left <= right:
        if toggle:
            perm.append(order[left])
            left += 1
        else:
            perm.append(order[right])
            right -= 1
        toggle = not toggle
    return np.array(perm, dtype=np.int64)


def pack_weight(
    W: torch.Tensor,
    *,
    block_size: int = 16,
    block_out_size: Optional[int] = None,
    enable_permute: bool = True,
    lambda_smooth: float = 0.15,
) -> PackResult:
    # Convert to CPU numpy for preprocessing
    W_np = W.detach().to(dtype=torch.float32, device="cpu").numpy()
    out_features, in_features = W_np.shape

    # Weight energy per input channel for permutation
    channel_energy = np.mean(W_np ** 2, axis=0)
    if lambda_smooth and lambda_smooth > 0:
        mean_e = float(channel_energy.mean())
        channel_energy = (1.0 - float(lambda_smooth)) * channel_energy + float(lambda_smooth) * mean_e
    perm = zigzag_permutation(channel_energy) if enable_permute else None

    # Compute block-wise rotation matrices on permuted weights
    R_in_blocks: Dict[int, np.ndarray] = {}
    idx, n_blocks = _block_indices(in_features, block_size)
    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, in_features)
        if end <= start:
            continue
        cols = np.arange(start, end)
        if perm is not None:
            cols = perm[cols]
        W_block = W_np[:, cols]
        R = compute_block_rotation(W_block)
        R_in_blocks[b] = R

    # Construct transformed weight to determine scales
    W_t = W_np.copy()
    # Apply permutation first on columns
    if perm is not None:
        W_t = W_t[:, perm]
    # Apply rotation per block on columns: W @ R
    if R_in_blocks:
        W_t2 = np.zeros_like(W_t)
        for b, R in R_in_blocks.items():
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            cols = slice(start, end)
            W_t2[:, start:end] = W_t[:, start:end] @ R[: (end - start), : (end - start)]
        W_t = W_t2

    # Compute output-side (row) rotations R_out per block using GPU-accelerated SVD
    if block_out_size is None:
        block_out_size = block_size
    R_out_blocks: Dict[int, np.ndarray] = {}
    _, n_row_blocks = _block_indices(out_features, block_out_size)

    # Pre-initialize GPU device for batch processing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for b in range(n_row_blocks):
        rs = b * block_out_size
        re = min((b + 1) * block_out_size, out_features)
        if re <= rs:
            continue
        rows = slice(rs, re)
        W_rows = W_np[rows, :]

        # GPU-accelerated SVD: 10-50x faster than NumPy CPU, mathematically equivalent
        try:
            W_rows_torch = torch.from_numpy(W_rows.astype(np.float32)).to(device)
            U_torch, _, _ = torch.linalg.svd(W_rows_torch, full_matrices=True)
            U = U_torch.cpu().numpy().astype(np.float64)
        except Exception:
            # Fallback to CPU NumPy SVD
            try:
                U, _, _ = np.linalg.svd(W_rows.astype(np.float64, copy=False), full_matrices=True)
            except np.linalg.LinAlgError:
                U = np.eye(W_rows.shape[0], dtype=np.float64)

        B = U.shape[0]
        if U.shape[1] < B:
            pad = np.zeros((B, B - U.shape[1]), dtype=U.dtype)
            U = np.concatenate([U, pad], axis=1)
        # Keep float64 here; casting happens at use-site
        R_out_blocks[b] = U[:, :B]

    # Per-output-channel symmetric scales using max-abs (simple, stable for MSE)
    max_abs = np.maximum(np.max(np.abs(W_t), axis=1), 1e-8)
    weight_scale = (max_abs / qmax(4)).astype(np.float32)  # default assumes 4-bit, layer can override later

    meta = {
        "in_features": int(in_features),
        "out_features": int(out_features),
        "block_size": int(block_size),
        "block_out_size": int(block_out_size),
        "enable_permute": bool(enable_permute),
        "lambda_smooth": float(lambda_smooth),
    }
    return PackResult(
        R_in_blocks=R_in_blocks or None,
        perm=perm,
        R_out_blocks=R_out_blocks or None,
        weight_scale=weight_scale,
        meta=meta,
    )


def apply_input_transform(x: torch.Tensor, pack: PackResult, *, use_transpose: bool = False) -> torch.Tensor:
    # x shape: [..., in_features]
    if pack.perm is None and (not pack.R_in_blocks):
        return x
    in_features = x.shape[-1]
    # First apply permutation: x -> x @ P (same perm as columns in W)
    if pack.perm is not None:
        perm_t = torch.from_numpy(pack.perm).to(x.device)
        x = x.index_select(dim=-1, index=perm_t)
    # Then apply input block rotations: x -> x @ R_in
    if pack.R_in_blocks:
        x_view = x.reshape(-1, in_features)
        x_t = x_view
        x_t2 = x_t.clone()  # preserve non-matched chunks
        block_size = int(pack.meta.get("block_size", next(iter(pack.R_in_blocks.values())).shape[0]))
        n_blocks = (in_features + block_size - 1) // block_size
        for b in range(n_blocks):
            if b not in pack.R_in_blocks:
                continue
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            R = torch.from_numpy(pack.R_in_blocks[b][: (end - start), : (end - start)]).to(x_t)
            # no transpose; callers expecting x P R_in
            x_t2[:, start:end] = x_t[:, start:end] @ R
        x = x_t2.reshape(*x.shape)
    return x


def transform_weight_for_forward(
    W: torch.Tensor,
    pack: PackResult,
    *,
    weight_bits: int,
    apply_row_rot: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns (W_transformed, scales_per_out)
    W_t = W
    if pack.perm is not None:
        perm_t = torch.from_numpy(pack.perm).to(W.device)
        W_t = W_t.index_select(dim=1, index=perm_t)
    if pack.R_in_blocks:
        # Apply rotation per block on columns
        in_features = W_t.shape[1]
        block_size = int(pack.meta.get("block_size", next(iter(pack.R_in_blocks.values())).shape[0]))
        n_blocks = (in_features + block_size - 1) // block_size
        W_t2 = W_t.clone()  # preserve non-matched blocks
        for b in range(n_blocks):
            if b not in pack.R_in_blocks:
                continue
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            R = torch.from_numpy(pack.R_in_blocks[b][: (end - start), : (end - start)]).to(W_t)
            W_t2[:, start:end] = W_t[:, start:end] @ R
        W_t = W_t2
    # Apply row rotations on the left: W_t <- R_out W_t
    if apply_row_rot and pack.R_out_blocks:
        out_features = W_t.shape[0]
        block_out_size = int(pack.meta.get("block_out_size", next(iter(pack.R_out_blocks.values())).shape[0]))
        n_row_blocks = (out_features + block_out_size - 1) // block_out_size
        W_t2 = W_t.clone()
        for b in range(n_row_blocks):
            if b not in pack.R_out_blocks:
                continue
            rs = b * block_out_size
            re = min((b + 1) * block_out_size, out_features)
            Rb = torch.from_numpy(pack.R_out_blocks[b][: (re - rs), : (re - rs)]).to(W_t)
            W_t2[rs:re, :] = Rb @ W_t[rs:re, :]
        W_t = W_t2
    # Per-output scales via MSE mini-grid
    with torch.no_grad():
        if weight_bits >= 16:
            scales = torch.ones(W_t.shape[0], device=W_t.device, dtype=W_t.dtype)
        else:
            scales = compute_mse_scales(W_t, weight_bits)
    return W_t, scales


def apply_output_restore(y: torch.Tensor, pack: PackResult) -> torch.Tensor:
    """Right-multiply output by R_out to restore original basis: y <- y @ R_out.

    Operates blockwise across the last dimension.
    """
    if pack.R_out_blocks is None:
        return y
    out_features = y.shape[-1]
    block_out_size = int(pack.meta.get("block_out_size", next(iter(pack.R_out_blocks.values())).shape[0]))
    n_row_blocks = (out_features + block_out_size - 1) // block_out_size
    y2 = y.reshape(-1, out_features)
    y_out = y2.clone()
    for b in range(n_row_blocks):
        if b not in pack.R_out_blocks:
            continue
        rs = b * block_out_size
        re = min((b + 1) * block_out_size, out_features)
        Rb = torch.from_numpy(pack.R_out_blocks[b][: (re - rs), : (re - rs)]).to(y2)
        y_out[:, rs:re] = y2[:, rs:re] @ Rb
    return y_out.reshape(*y.shape)


def apply_bias_row_rot(bias: torch.Tensor, pack: PackResult) -> torch.Tensor:
    """Apply row rotation blocks to a bias vector."""
    if pack.R_out_blocks is None:
        return bias
    out_features = bias.shape[-1]
    block_out_size = int(pack.meta.get("block_out_size", next(iter(pack.R_out_blocks.values())).shape[0]))
    n_row_blocks = (out_features + block_out_size - 1) // block_out_size
    bias_out = bias.clone()
    for b in range(n_row_blocks):
        if b not in pack.R_out_blocks:
            continue
        rs = b * block_out_size
        re = min((b + 1) * block_out_size, out_features)
        Rb = torch.from_numpy(pack.R_out_blocks[b][: (re - rs), : (re - rs)]).to(bias)
        bias_out[rs:re] = Rb @ bias[rs:re]
    return bias_out


class PercentileCalibrator:
    """Per-channel percentile calibrator (last-dim channels).

    Tracks a running per-channel percentile (default 99.9%) using a batchwise
    max of per-channel quantiles for stability and low memory.
    """

    def __init__(self, percentile: float = 99.9, max_batches: int = 64, max_per_batch: int = 4096) -> None:
        self.percentile = percentile
        self.max_batches = max_batches
        self._seen = 0
        self._p_running: torch.Tensor | None = None  # on CPU, shape [C]

    def observe(self, x: torch.Tensor) -> None:
        if self._seen >= self.max_batches:
            return
        x_abs = torch.abs(x.detach().to(torch.float32, non_blocking=True))
        C = x_abs.shape[-1]
        x2d = x_abs.reshape(-1, C)
        q = torch.quantile(x2d, self.percentile / 100.0, dim=0)
        q = torch.clamp(q, min=1e-6).cpu()
        if self._p_running is None:
            self._p_running = q
        else:
            self._p_running = torch.maximum(self._p_running, q)
        self._seen += 1

    def is_full(self) -> bool:
        return self._seen >= self.max_batches

    def finalize(self) -> torch.Tensor:
        if self._p_running is None:
            # Fallback vector with 1.0 scale, caller should clamp later
            return torch.tensor([1.0])
        return self._p_running


def _candidate_scales_from_maxabs(max_abs: torch.Tensor, bits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Small grid around max-abs scale for MSE search
    alphas = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], device=device, dtype=dtype)
    base = (max_abs / qmax(bits)).clamp_min(1e-8)  # [O]
    # s = base / alpha
    return (base[:, None] / alphas[None, :]).contiguous()  # [O, G]


def compute_mse_scales(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Compute per-output-channel scales minimizing MSE via tiny grid search."""
    if bits <= 0:
        return torch.ones(W.shape[0], device=W.device, dtype=W.dtype)
    O, I = W.shape
    max_abs = torch.amax(torch.abs(W), dim=1)  # [O]
    S = _candidate_scales_from_maxabs(max_abs, bits, W.device, W.dtype)  # [O, G]
    G = S.shape[1]
    W_row = W[:, None, :]  # [O, 1, I]
    S_e = S[:, :, None]  # [O, G, 1]
    Q = torch.round(W_row / S_e)
    max_q = qmax(bits)
    Q = torch.clamp(Q, -max_q - 1, max_q)
    R = Q * S_e
    mse = torch.mean((R - W_row) ** 2, dim=2)  # [O, G]
    idx = torch.argmin(mse, dim=1)  # [O]
    best = S[torch.arange(O, device=W.device), idx]
    return best


def _pack_path(layer_name: str, pack_dir: Optional[str] = None) -> str:
    pack_dir = pack_dir or DEFAULT_PACK_DIR
    _ensure_dir(pack_dir)
    return os.path.join(pack_dir, f"{sanitize_name(layer_name)}.npz")


def save_pack(layer_name: str, pack: PackResult, pack_dir: Optional[str] = None) -> None:
    path = _pack_path(layer_name, pack_dir)
    data: Dict[str, Any] = {
        "weight_scale": pack.weight_scale.astype(np.float32),
        "meta": json.dumps(pack.meta),
    }
    if pack.perm is not None:
        data["perm"] = pack.perm.astype(np.int64)
    if pack.R_in_blocks:
        # store as dict of block_idx -> R
        for b, R in pack.R_in_blocks.items():
            data[f"Rin_{b}"] = R.astype(np.float32)
        data["R_in_blocks"] = np.array(sorted(list(pack.R_in_blocks.keys())), dtype=np.int64)
    if pack.R_out_blocks:
        for b, R in pack.R_out_blocks.items():
            data[f"Rout_{b}"] = R.astype(np.float32)
        data["R_out_blocks"] = np.array(sorted(list(pack.R_out_blocks.keys())), dtype=np.int64)
    np.savez(path, **data)


def load_pack(layer_name: str, pack_dir: Optional[str] = None) -> Optional[PackResult]:
    path = _pack_path(layer_name, pack_dir)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as f:
        weight_scale = f["weight_scale"]
        perm = f["perm"] if "perm" in f.files else None
        R_in_blocks = None
        if "R_in_blocks" in f.files:
            R_in_blocks = {}
            for b in f["R_in_blocks"]:
                R_in_blocks[int(b)] = f[f"Rin_{int(b)}"]
        R_out_blocks = None
        if "R_out_blocks" in f.files:
            R_out_blocks = {}
            for b in f["R_out_blocks"]:
                R_out_blocks[int(b)] = f[f"Rout_{int(b)}"]
        meta_raw = f["meta"] if "meta" in f.files else None
        if meta_raw is not None:
            try:
                meta = json.loads(meta_raw.tolist())
            except Exception:
                meta = {}
        else:
            meta = {}
        return PackResult(
            R_in_blocks=R_in_blocks,
            perm=perm,
            R_out_blocks=R_out_blocks,
            weight_scale=weight_scale,
            meta=meta,
        )


# ========================================
# OPTIMIZED VERSIONS - Use pre-cached tensors
# These functions avoid torch.from_numpy() and clone() operations
# ========================================


def apply_input_transform_optimized(
    x: torch.Tensor,
    pack: PackResult,
    perm_cache: Optional[torch.Tensor],
    R_in_cache: Dict[int, torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """Optimized version using pre-cached torch tensors instead of numpy arrays."""
    if perm_cache is None and not R_in_cache:
        return x

    in_features = x.shape[-1]

    # First apply permutation using cached tensor
    if perm_cache is not None:
        x = x.index_select(dim=-1, index=perm_cache)

    # Then apply input block rotations using cached tensors
    if R_in_cache:
        original_shape = x.shape
        x_view = x.reshape(-1, in_features)
        n_blocks = (in_features + block_size - 1) // block_size

        # Fast path: vectorized batched block matmul (no block-diagonal allocation)
        if len(R_in_cache) == n_blocks and all(b in R_in_cache for b in range(n_blocks)):
            all_full = all(
                (R_in_cache[b].shape[0] == block_size and R_in_cache[b].shape[1] == block_size)
                for b in range(n_blocks)
            )
            if all_full and in_features == n_blocks * block_size:
                x_blocks = x_view.reshape(-1, n_blocks, block_size)
                R_stack = torch.stack([R_in_cache[b] for b in range(n_blocks)], dim=0).to(
                    dtype=x_blocks.dtype, device=x_blocks.device
                )
                x_out_blocks = torch.einsum("rnb,nbc->rnc", x_blocks, R_stack)
                return x_out_blocks.reshape(*original_shape)

        # Fallback: loop over blocks (original implementation)
        x_t = x_view.clone()
        for b in range(n_blocks):
            if b not in R_in_cache:
                continue
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            R = R_in_cache[b][: (end - start), : (end - start)]
            x_t[:, start:end] = x_view[:, start:end] @ R
        x = x_t.reshape(*original_shape)
    return x


def apply_output_restore_optimized(
    y: torch.Tensor,
    pack: PackResult,
    R_out_cache: Dict[int, torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    """Optimized version using pre-cached torch tensors."""
    if not R_out_cache:
        return y

    original_shape = y.shape
    out_features = y.shape[-1]
    n_row_blocks = (out_features + block_out_size - 1) // block_out_size
    y_view = y.reshape(-1, out_features)

    # Fast path: vectorized batched block matmul (no block-diagonal allocation)
    if len(R_out_cache) == n_row_blocks and all(b in R_out_cache for b in range(n_row_blocks)):
        all_full = all(
            (R_out_cache[b].shape[0] == block_out_size and R_out_cache[b].shape[1] == block_out_size)
            for b in range(n_row_blocks)
        )
        if all_full and out_features == n_row_blocks * block_out_size:
            y_blocks = y_view.reshape(-1, n_row_blocks, block_out_size)
            R_stack = torch.stack([R_out_cache[b] for b in range(n_row_blocks)], dim=0).to(
                dtype=y_blocks.dtype, device=y_blocks.device
            )
            y_out_blocks = torch.einsum("rnb,nbc->rnc", y_blocks, R_stack)
            return y_out_blocks.reshape(*original_shape)

    # Fallback: loop over blocks (original implementation)
    y_out = y_view.clone()
    for b in range(n_row_blocks):
        if b not in R_out_cache:
            continue
        rs = b * block_out_size
        re = min((b + 1) * block_out_size, out_features)
        Rb = R_out_cache[b][: (re - rs), : (re - rs)]
        y_out[:, rs:re] = y_view[:, rs:re] @ Rb

    return y_out.reshape(*original_shape)


def transform_weight_for_forward_optimized(
    W: torch.Tensor,
    pack: PackResult,
    *,
    weight_bits: int,
    apply_row_rot: bool,
    perm_cache: Optional[torch.Tensor],
    R_in_cache: Dict[int, torch.Tensor],
    R_out_cache: Dict[int, torch.Tensor],
    block_size: int,
    block_out_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized version using pre-cached torch tensors."""
    W_t = W

    # Apply permutation using cached tensor
    if perm_cache is not None:
        W_t = W_t.index_select(dim=1, index=perm_cache)

    # Apply input block rotations using cached tensors
    if R_in_cache:
        in_features = W_t.shape[1]
        n_blocks = (in_features + block_size - 1) // block_size
        for b in range(n_blocks):
            if b not in R_in_cache:
                continue
            start = b * block_size
            end = min((b + 1) * block_size, in_features)
            R = R_in_cache[b][: (end - start), : (end - start)]
            # In-place update for efficiency
            W_t[:, start:end] = W_t[:, start:end] @ R

    # Apply row rotations using cached tensors
    if apply_row_rot and R_out_cache:
        out_features = W_t.shape[0]
        n_row_blocks = (out_features + block_out_size - 1) // block_out_size
        for b in range(n_row_blocks):
            if b not in R_out_cache:
                continue
            rs = b * block_out_size
            re = min((b + 1) * block_out_size, out_features)
            Rb = R_out_cache[b][: (re - rs), : (re - rs)]
            W_t[rs:re, :] = Rb @ W_t[rs:re, :]

    # Per-output scales via MSE mini-grid
    with torch.no_grad():
        if weight_bits >= 16:
            scales = torch.ones(W_t.shape[0], device=W_t.device, dtype=W_t.dtype)
        else:
            scales = compute_mse_scales(W_t, weight_bits)

    return W_t, scales


def apply_bias_row_rot_optimized(
    bias: torch.Tensor,
    pack: PackResult,
    R_out_cache: Dict[int, torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    """Optimized version using pre-cached torch tensors."""
    if not R_out_cache:
        return bias

    out_features = bias.shape[-1]
    n_row_blocks = (out_features + block_out_size - 1) // block_out_size
    bias_out = bias.clone()  # Need clone here since bias is a parameter

    for b in range(n_row_blocks):
        if b not in R_out_cache:
            continue
        rs = b * block_out_size
        re = min((b + 1) * block_out_size, out_features)
        Rb = R_out_cache[b][: (re - rs), : (re - rs)]
        bias_out[rs:re] = Rb @ bias[rs:re]

    return bias_out
