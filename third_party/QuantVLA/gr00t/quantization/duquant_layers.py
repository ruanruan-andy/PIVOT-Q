"""
GR00T DuQuant W4A8 Fake Quantization Layers

Adapted from OpenPI duquant implementation for GR00T model quantization.
Supports quantization of LLM (Eagle VLM) and DiT (action transformer) layers.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from .duquant_preprocess import (
    PackResult,
    PercentileCalibrator,
    apply_input_transform,
    apply_output_restore,
    apply_bias_row_rot,
    fake_quantize_sym,
    load_pack,
    pack_weight,
    qmax,
    save_pack,
    transform_weight_for_forward,
)


@dataclass
class DuQuantConfig:
    """DuQuant configuration matching OpenPI parameters.

    NOTE: Default values are set to None and resolved in __post_init__ to ensure
    environment variables are read at instantiation time, not at module import time.
    """
    weight_bits: Optional[int] = None
    act_bits: Optional[int] = None
    block_size: Optional[int] = None
    lambda_smooth: Optional[float] = None
    enable_permute: Optional[bool] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    pack_dir: Optional[str] = None
    row_rot_mode: Optional[str] = None
    block_out_size: Optional[int] = None

    def __post_init__(self):
        """Read environment variables at instantiation time."""
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_DUQUANT_WBITS_DEFAULT", 4))
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_DUQUANT_ABITS", 8))
        if self.block_size is None:
            self.block_size = int(os.environ.get("GR00T_DUQUANT_BLOCK", 16))
        if self.lambda_smooth is None:
            self.lambda_smooth = float(os.environ.get("GR00T_DUQUANT_LS", 0.15))
        if self.enable_permute is None:
            self.enable_permute = os.environ.get("GR00T_DUQUANT_PERMUTE", "1") not in ("0", "false", "False")
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_DUQUANT_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_DUQUANT_CALIB_STEPS", 32))
        if self.pack_dir is None:
            self.pack_dir = os.environ.get("GR00T_DUQUANT_PACKDIR", None)
        if self.row_rot_mode is None:
            self.row_rot_mode = os.environ.get("GR00T_DUQUANT_ROW_ROT", "restore")
        if self.block_out_size is None:
            self.block_out_size = int(os.environ.get("GR00T_DUQUANT_BLOCK_OUT", os.environ.get("GR00T_DUQUANT_BLOCK", 16)))


def _parse_per_layer_wbits(env_val: Optional[str]) -> Dict[str, int]:
    """Parse per-layer weight bits from environment variable."""
    if not env_val:
        return {}
    result: Dict[str, int] = {}
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return result


class DuQuantLinear(nn.Module):
    """DuQuant quantized linear layer with W4A8 fake quantization."""

    def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.register_buffer("_weight", base.weight.detach().clone())

        # Config
        self.cfg = cfg
        self.weight_bits = cfg.weight_bits if weight_bits is None else int(weight_bits)

        # Load or compute packing
        pack = load_pack(self.name, cfg.pack_dir)
        if pack is None:
            pack = pack_weight(
                self._weight,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                lambda_smooth=cfg.lambda_smooth,
            )
            save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        # Cache rotation matrices as torch tensors
        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long())
        else:
            self._perm_cache = None

        # Cache input rotation matrices
        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                buffer_name = f"_R_in_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_in_block_indices.append(b)

        # Cache output rotation matrices
        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                buffer_name = f"_R_out_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_out_block_indices.append(b)

        # Store metadata
        self._block_size = int(pack.meta.get("block_size", 16))
        self._block_out_size = int(pack.meta.get("block_out_size", self._block_size))

        # Calibrator for activation
        self.calibrator = PercentileCalibrator(
            percentile=cfg.act_percentile, max_batches=cfg.calib_batches
        ) if self.cfg.act_bits > 0 else None
        self.register_buffer("_act_scale", None)
        self._act_scale_initialized = False

        # Cache transformed weight
        self._cached_weight_key: Optional[Tuple[str, torch.dtype]] = None
        self.register_buffer("_W_t", torch.zeros_like(self._weight))
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))

        # Pre-cache quantized weights
        self._precache_weight = os.environ.get("GR00T_DUQUANT_PRECACHE_WEIGHTS", "1") not in (
            "0", "false", "False",
        )
        if self._precache_weight:
            self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))
        else:
            self._W_t_quantized = None
        self._weight_quantized_cached = False

        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("GR00T_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False

    def _get_R_in_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_in rotation matrices on the correct device."""
        if not hasattr(self, '_R_in_cache_dict'):
            self._R_in_cache_dict = {}
        for b in self._R_in_block_indices:
            self._R_in_cache_dict[b] = getattr(self, f"_R_in_{b}")
        return self._R_in_cache_dict

    def _get_R_out_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_out rotation matrices on the correct device."""
        if not hasattr(self, '_R_out_cache_dict'):
            self._R_out_cache_dict = {}
        for b in self._R_out_block_indices:
            self._R_out_cache_dict[b] = getattr(self, f"_R_out_{b}")
        return self._R_out_cache_dict

    @property
    def weight(self) -> torch.Tensor:
        """Expose packed weight buffer for compatibility."""
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self._weight.copy_(value)

    def _maybe_update_weight_cache(self) -> None:
        apply_row = (self.cfg.row_rot_mode != "0")
        key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row))
        if self._cached_weight_key == key:
            return

        from .duquant_preprocess import transform_weight_for_forward_optimized

        W_t, scales = transform_weight_for_forward_optimized(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
            perm_cache=self._perm_cache,
            R_in_cache=self._get_R_in_cache(),
            R_out_cache=self._get_R_out_cache(),
            block_size=self._block_size,
            block_out_size=self._block_out_size,
        )
        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        # Pre-quantize weights if enabled
        if self._precache_weight and self.weight_bits > 0:
            with torch.no_grad():
                self._W_t_quantized.copy_(
                    fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label="weight_prequant")
                )
            self._weight_quantized_cached = True
        else:
            self._weight_quantized_cached = False

        self._cached_weight_key = key
        if self.bias is not None:
            if self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
                with torch.no_grad():
                    from .duquant_preprocess import apply_bias_row_rot_optimized
                    self._bias_rot = apply_bias_row_rot_optimized(
                        self.bias.detach(), self.pack, self._get_R_out_cache(), self._block_out_size
                    )
            else:
                self._bias_rot = None
        if self._debug_enabled:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode}"
            )
            if self._weight_quantized_cached:
                logging.info(f"[GR00T-DUQUANT][CACHE] {self.name} pre-quantized weights cached")

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        if self._act_scale_initialized:
            return self._act_scale

        with torch.no_grad():
            if self.calibrator is not None and not self.calibrator.is_full():
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    p_vec = self.calibrator.finalize()
                    max_q = qmax(self.cfg.act_bits)
                    scale = torch.clamp(p_vec / max_q, min=1e-6)
                    scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    if self._act_scale is None:
                        self._act_scale = scale
                    else:
                        self._act_scale.copy_(scale)
                    self._act_scale_initialized = True

            if not self._act_scale_initialized:
                x_abs = torch.abs(x.detach().to(torch.float32))
                C = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, C)
                p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                scale = scale.to(dtype=x.dtype, device=x.device).clone()
                if self._act_scale is None:
                    self._act_scale = scale
                else:
                    self._act_scale.copy_(scale)
                self._act_scale_initialized = True

        return self._act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply optimized per-block input transform
        from .duquant_preprocess import apply_input_transform_optimized
        x_t = apply_input_transform_optimized(
            x, self.pack, self._perm_cache, self._get_R_in_cache(), self._block_size
        )

        # Fake-quantize activations if enabled
        if self.cfg.act_bits > 0:
            s_a = self._get_act_scale(x_t)
            x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits, label="activation_forward")

        # Transform and fake-quantize weights
        self._maybe_update_weight_cache()

        # Use pre-quantized weights
        if self._weight_quantized_cached:
            y_lin = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif self.weight_bits > 0:
            y_lin = torch.nn.functional.linear(
                x_t,
                fake_quantize_sym(
                    self._W_t,
                    self._w_scales[:, None],
                    self.weight_bits,
                    label="weight_fallback",
                ),
                None
            )
        else:
            y_lin = torch.nn.functional.linear(x_t, self._W_t, None)

        # Apply row restore if requested
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            from .duquant_preprocess import apply_output_restore_optimized
            y_lin = apply_output_restore_optimized(
                y_lin, self.pack, self._get_R_out_cache(), self._block_out_size
            )
            if self.bias is not None:
                y_lin = y_lin + self.bias
        else:
            if self.bias is not None:
                bias_to_add = (
                    self._bias_rot
                    if self.cfg.row_rot_mode == "propagate" and self._bias_rot is not None
                    else self.bias
                )
                y_lin = y_lin + bias_to_add
        if self._debug_enabled and not self._debug_forward_logged:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y_lin.shape)} "
                f"weight_bits={self.weight_bits} act_bits={self.cfg.act_bits}"
            )
            self._debug_forward_logged = True
        return y_lin


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def select_targets(
    model: nn.Module,
    *,
    include_regex: str = r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj|gate_proj).*",
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)",
    scope_prefix: Optional[str] = None,
    whitelist: Optional[Iterable[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    """Select linear layers to quantize based on regex patterns."""
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    wl = set(whitelist or [])
    bl = set(blacklist or [])
    results: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if scope_prefix is not None and not name.startswith(scope_prefix):
            continue
        if name in bl:
            continue
        if wl and name not in wl:
            continue
        if not wl and (not inc.search(name) or exc.search(name)):
            continue
        results.append((name, mod))
    return results


def wrap_duquant(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: DuQuantConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> None:
    """Wrap selected layers with DuQuant quantization."""
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        # Skip action head by default unless explicitly requested
        if os.environ.get("GR00T_DUQUANT_INCLUDE_ACTION_HEAD", "0") in ("0", "false", "False"):
            is_action_head = "action_head" in name and not name.startswith("action_head.model.")
            if (
                name.endswith("action_out_proj")
                or ".action_out_proj" in name
                or is_action_head
            ):
                continue
        parent, attr = _get_parent_module_and_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        if dry_run:
            msg = (
                f"[GR00T-DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} "
                f"block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode}"
            )
            print(msg)
            listed += 1
            continue
        dq = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, dq)
        # Use actual block sizes from pack (not cfg defaults)
        actual_block_in = dq._block_size
        actual_block_out = dq._block_out_size
        print(
            f"[GR00T-DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
            f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} block_in={actual_block_in} block_out={actual_block_out} row_rot={cfg.row_rot_mode}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-DUQUANT] Dry-run total layers listed: {listed}")
    else:
        print(f"[GR00T-DUQUANT] Total layers replaced: {replaced}")


def enable_duquant_if_configured(model: nn.Module) -> None:
    """
    Entry point to enable DuQuant based on environment variables.

    Activation conditions:
    - If GR00T_DUQUANT_DRYRUN is set => dry-run listing only
    - Or if any GR00T_DUQUANT_* variable (other than PACKDIR) is set => perform replacement
    - Otherwise do nothing
    """
    env = os.environ
    keys = [k for k in env.keys() if k.startswith("GR00T_DUQUANT_")]
    activate = any(k not in ("GR00T_DUQUANT_PACKDIR",) for k in keys)
    if not activate:
        return

    # Scope defaults to empty (search entire model)
    scope = env.get("GR00T_DUQUANT_SCOPE", "")
    whitelist = env.get("GR00T_DUQUANT_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None

    # Default: quantize LLM + DiT MLP layers (matching OpenPI pattern)
    # Include LLM attention+MLP and DiT MLP projections
    inc = env.get(
        "GR00T_DUQUANT_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    # Exclude vision encoder, embeddings, auxiliary projectors
    exc = env.get(
        "GR00T_DUQUANT_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )

    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_DUQUANT_WBITS"))
    dry_run = env.get("GR00T_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

    cfg = DuQuantConfig()

    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[GR00T-DUQUANT] SCOPE filter: '{scope}'")
    print(f"[GR00T-DUQUANT] Matched Linear layers: {len(layer_names)}")

    if len(layer_names) == 0 and scope:
        # Debug: print some layer names to help diagnose
        all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"[GR00T-DUQUANT] DEBUG: Total Linear layers in model: {len(all_linears)}")
        print(f"[GR00T-DUQUANT] DEBUG: First 10 Linear layer names:")
        for name, _ in all_linears[:10]:
            print(f"[GR00T-DUQUANT] DEBUG:   {name}")
        if scope:
            matching_prefix = [n for n, _ in all_linears if n.startswith(scope.rstrip('.'))]
            print(f"[GR00T-DUQUANT] DEBUG: Layers matching prefix '{scope.rstrip('.')}': {len(matching_prefix)}")
            if matching_prefix:
                for name in matching_prefix[:5]:
                    print(f"[GR00T-DUQUANT] DEBUG:   {name}")

    if dry_run:
        wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=True)
        return
    wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=False)
