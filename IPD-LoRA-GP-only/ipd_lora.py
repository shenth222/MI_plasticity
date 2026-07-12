import contextlib
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NEVER_UPDATE_INTERVAL = 10**12

# Global mixed-precision (AMP) control shared across the scoring/probing/eval
# forward passes so they save activation memory consistently with training.
_AMP_ENABLED = False
_AMP_DTYPE = torch.bfloat16


def set_amp(enabled: bool, dtype: torch.dtype = torch.bfloat16) -> None:
    global _AMP_ENABLED, _AMP_DTYPE
    _AMP_ENABLED = bool(enabled)
    _AMP_DTYPE = dtype


def amp_autocast(device) -> contextlib.AbstractContextManager:
    """Return an autocast context if AMP is enabled and running on CUDA."""
    is_cuda = torch.cuda.is_available() and (
        getattr(device, "type", None) == "cuda" or str(device).startswith("cuda")
    )
    if _AMP_ENABLED and is_cuda:
        return torch.autocast("cuda", dtype=_AMP_DTYPE)
    return contextlib.nullcontext()


def _safe_float(v: float) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (float, int)):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return float(v)
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def _zscore(values: Sequence[float], eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / (std + eps)


def _minmax_nonneg(values: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize values to [0, 1].

    Used to make I/P/G comparable before forming a combined allocation score.
    If all values are (nearly) equal, returns ones so the signal stays uniform
    rather than collapsing to zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def _quantile_threshold(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    qv = float(min(1.0, max(0.0, q)))
    return float(np.quantile(arr, qv))


def _high_mask_by_quantile(values: Sequence[float], q: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return np.asarray([], dtype=bool)
    qv = float(min(1.0, max(0.0, q)))
    n_high = int(math.ceil((1.0 - qv) * n))
    n_high = int(max(1, min(n, n_high)))
    order = np.argsort(-arr, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_high]] = True
    return mask


def _round_to_choice(value: int, choices: Sequence[int]) -> int:
    choices = sorted(set(int(c) for c in choices))
    valid = [c for c in choices if c <= value]
    if valid:
        return valid[-1]
    return choices[0]


def _next_lower_choice(value: int, choices: Sequence[int]) -> int:
    choices = sorted(set(int(c) for c in choices))
    for c in reversed(choices):
        if c < value:
            return c
    return choices[0]


def _waterfill_extra_ranks(
    S: np.ndarray,
    total_rank_budget: int,
    r_min: int,
    active_rank_choices: Sequence[int],
    max_rank: Optional[int] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Distribute a fixed rank budget across modules proportional to score S.

    Each module gets a guaranteed floor ``r_min`` plus a non-negative ``extra``
    capped by ``(max_rank - r_min)``. Crucially, any budget that cannot be placed
    because high-score modules hit ``max_rank`` is redistributed (water-filling)
    to the remaining uncapped modules, so the full budget is spent whenever it is
    feasible. This keeps the *total* rank budget comparable across allocation
    methods (lora / gora / goodput), which the fair-comparison protocol requires.

    Returns the integer ``extra`` array (length == len(S)).
    """
    n = int(np.asarray(S).size)
    if n == 0:
        return np.asarray([], dtype=np.int64)
    maxr = int(max_rank) if max_rank is not None else int(max(active_rank_choices))
    rmin_choice = int(_round_to_choice(int(max(0, r_min)), active_rank_choices))
    cap = int(max(0, maxr - rmin_choice))

    # Feasible total extra is bounded by per-module caps.
    distributable = int(max(0, int(total_rank_budget) - rmin_choice * n))
    distributable = int(min(distributable, cap * n))
    if distributable <= 0 or cap <= 0:
        return np.zeros(n, dtype=np.int64)

    S = np.clip(np.asarray(S, dtype=np.float64), 0.0, None)
    sum_S = float(S.sum())
    if sum_S <= eps:
        raw = np.full(n, distributable / float(n), dtype=np.float64)
    else:
        raw = distributable * S / sum_S

    extra = np.minimum(np.floor(raw), cap).astype(np.int64)
    leftover = int(distributable - int(extra.sum()))

    # Priority for leftover units: larger fractional remainder, then larger score.
    frac = raw - np.floor(raw)
    order = np.argsort(-(frac + S * 1e-9), kind="mergesort")

    guard = 0
    while leftover > 0:
        progressed = False
        for idx in order:
            if leftover <= 0:
                break
            if extra[idx] < cap:
                extra[idx] += 1
                leftover -= 1
                progressed = True
        if not progressed:  # all modules capped
            break
        guard += 1
        if guard > distributable + 5:
            break
    return extra


def _is_adaptable_linear(module: nn.Module) -> bool:
    """True for a wrappable linear layer.

    Covers plain nn.Linear (DeBERTa/full-precision path) and bitsandbytes
    quantized linears (Linear4bit / Linear8bitLt) used by QLoRA on LLaMA. We
    duck-type on in_features/out_features so we don't hard-depend on bnb being
    importable in the DeBERTa-only path.
    """
    if isinstance(module, nn.Linear):
        return True
    cls_name = type(module).__name__
    if cls_name in ("Linear4bit", "Linear8bitLt") and hasattr(module, "in_features"):
        return True
    return False


class IPDLoRALinear(nn.Module):
    """
    LoRA linear wrapper with dynamic active_rank.

    Key design:
    - Keep max-rank LoRA parameters, but only compute active slices in forward.
    - active_rank == 0 physically skips LoRA branch in forward computation.
    - Works over both nn.Linear (full precision) and bitsandbytes Linear4bit
      (QLoRA): the base branch stays frozen/quantized, only LoRA A/B train.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        module_name: str,
        layer_index: int,
        projection_type: str,
        max_rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        initial_active_rank: int = 8,
    ) -> None:
        super().__init__()
        if not _is_adaptable_linear(base_linear):
            raise TypeError(f"IPDLoRALinear only supports linear-like layers, got {type(base_linear)}")

        self.base_linear = base_linear
        self.module_name = module_name
        self.layer_index = int(layer_index)
        self.projection_type = projection_type
        self.max_rank = int(max_rank)
        self.lora_alpha = int(alpha)
        self.lora_dropout = nn.Dropout(p=float(dropout))
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Freeze base branch to ensure parameter-efficient adaptation.
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.max_rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.max_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Runtime states for GP-only LoRA control.
        self.active_rank = int(min(self.max_rank, initial_active_rank))
        self.target_rank = int(self.active_rank)
        self.update_interval = 1
        # Module Learning Goodput states (SLAQ/Pollux-inspired learning-progress-per-budget).
        self.current_G = 0.0  # instantaneous goodput estimate at last goodput event
        self.ema_G = 0.0  # EMA-smoothed goodput
        self.G_z = 0.0  # z-scored goodput across modules
        self.G_rank = 0  # descending rank of ema_G
        self.current_share = 0.0  # proxy-share s_m(t) used by method A
        self.cumulative_update_steps = 0  # steps_m: effective update steps this module received

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        r = int(self.active_rank)
        if r <= 0:
            # active_rank=0 means we fully skip LoRA branch to save LoRA-side compute.
            return base_out

        x_d = self.lora_dropout(x)
        A_slice = self.lora_A[:r, :]  # [r, in_features]
        B_slice = self.lora_B[:, :r]  # [out_features, r]
        # QLoRA: base is quantized (bf16 compute) while LoRA params stay fp32, so
        # dtypes differ. Cast only when needed; the full-precision DeBERTa path
        # (same dtypes) is untouched.
        if x_d.dtype != A_slice.dtype:
            x_d = x_d.to(A_slice.dtype)
        low_rank = F.linear(x_d, A_slice)  # [..., r]
        delta = F.linear(low_rank, B_slice)  # [..., out_features]
        scaling = self.lora_alpha / max(r, 1)
        if delta.dtype != base_out.dtype:
            delta = delta.to(base_out.dtype)
        return base_out + scaling * delta

    @property
    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    @property
    def cost(self) -> int:
        r = int(max(self.active_rank, 0))
        return int(r * (self.in_features + self.out_features))


def _resolve_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    names = module_name.split(".")
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent, names[-1]


# Default projection patterns for DeBERTa-v2/v3 encoder (encoder.layer.N.*).
DEBERTA_PROJECTION_PATTERNS = [
    (r"layer\.(\d+)\.attention\.self\.query_proj$", "query_proj"),
    (r"layer\.(\d+)\.attention\.self\.key_proj$", "key_proj"),
    (r"layer\.(\d+)\.attention\.self\.value_proj$", "value_proj"),
    (r"layer\.(\d+)\.attention\.output\.dense$", "output_proj"),
    (r"layer\.(\d+)\.intermediate\.dense$", "ffn_in"),
    (r"layer\.(\d+)\.output\.dense$", "ffn_out"),
]
DEBERTA_LAYER_PATTERN = r"layer\.(\d+)\."

# Projection patterns for LLaMA-style decoders (model.layers.N.*).
LLAMA_PROJECTION_PATTERNS = [
    (r"layers\.(\d+)\.self_attn\.q_proj$", "q_proj"),
    (r"layers\.(\d+)\.self_attn\.k_proj$", "k_proj"),
    (r"layers\.(\d+)\.self_attn\.v_proj$", "v_proj"),
    (r"layers\.(\d+)\.self_attn\.o_proj$", "o_proj"),
    (r"layers\.(\d+)\.mlp\.gate_proj$", "gate_proj"),
    (r"layers\.(\d+)\.mlp\.up_proj$", "up_proj"),
    (r"layers\.(\d+)\.mlp\.down_proj$", "down_proj"),
]
LLAMA_LAYER_PATTERN = r"layers\.(\d+)\."


def inject_ipd_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    max_rank: int,
    alpha: int,
    dropout: float,
    initial_active_rank: int,
    verbose: bool = True,
    projection_patterns: Optional[Sequence[Tuple[str, str]]] = None,
    layer_pattern: Optional[str] = None,
) -> Dict[str, IPDLoRALinear]:
    """Wrap matching linear layers with IPDLoRALinear.

    projection_patterns / layer_pattern default to DeBERTa; pass the LLAMA_*
    constants (or custom regex specs) to adapt a LLaMA-style decoder. Matches
    both nn.Linear and bitsandbytes quantized linears (QLoRA).
    """
    target_set = set(target_modules)
    replacements: List[Tuple[str, nn.Module, str]] = []
    specs = projection_patterns if projection_patterns is not None else DEBERTA_PROJECTION_PATTERNS
    pattern_specs = [(re.compile(p) if isinstance(p, str) else p, name) for p, name in specs]

    def _matched_proj_type(module_name: str) -> Optional[str]:
        short_name = module_name.split(".")[-1]
        selected = any((short_name == t) or module_name.endswith(t) for t in target_set)
        if not selected:
            return None
        for pat, proj_name in pattern_specs:
            if pat.search(module_name) is not None:
                return proj_name
        return None

    for name, module in model.named_modules():
        if not _is_adaptable_linear(module):
            continue
        proj_type = _matched_proj_type(name)
        if proj_type is None:
            continue
        replacements.append((name, module, proj_type))

    lora_module_dict: Dict[str, IPDLoRALinear] = {}
    layer_pattern = re.compile(layer_pattern if layer_pattern is not None else DEBERTA_LAYER_PATTERN)
    for full_name, base_linear, proj in replacements:
        m = layer_pattern.search(full_name)
        if m is None:
            continue
        layer_index = int(m.group(1))
        parent, child_name = _resolve_parent_module(model, full_name)
        wrapped = IPDLoRALinear(
            base_linear=base_linear,
            module_name=full_name,
            layer_index=layer_index,
            projection_type=proj,
            max_rank=max_rank,
            alpha=alpha,
            dropout=dropout,
            initial_active_rank=initial_active_rank,
        )
        setattr(parent, child_name, wrapped)
        lora_module_dict[full_name] = wrapped

    if verbose:
        print(f"[GP-only-LoRA] Injected {len(lora_module_dict)} modules:")
        for n in sorted(lora_module_dict):
            print(f"  - {n}")
    return lora_module_dict


def build_calibration_split(dataset, calibration_size: int, seed: int):
    n = len(dataset)
    if n <= 1:
        raise ValueError("Dataset too small for train/calibration split.")

    cal_size = int(calibration_size)
    if cal_size <= 0:
        raise ValueError("calibration_size must be positive.")
    if cal_size >= n:
        cal_size = max(1, n // 5)
    if n - cal_size < 1:
        cal_size = n - 1

    rng = np.random.default_rng(seed)
    all_indices = np.arange(n)
    rng.shuffle(all_indices)
    calib_idx = all_indices[:cal_size]
    train_idx = all_indices[cal_size:]
    train_dataset = dataset.select(train_idx.tolist())
    calib_dataset = dataset.select(calib_idx.tolist())
    return train_dataset, calib_dataset


def _avg_loss_over_loader(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    total_loss = 0.0
    total_count = 0
    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), amp_autocast(device):
            outputs = model(**batch)
        bs = int(batch["labels"].shape[0])
        total_loss += float(outputs.loss.item()) * bs
        total_count += bs
    if total_count == 0:
        return 0.0
    return total_loss / total_count


def finalize_goodput_stats(lora_module_dict: Dict[str, IPDLoRALinear]) -> None:
    """Compute cross-module z-score and descending rank of ema_G.

    Kept separate so G_z / G_rank are always available for analysis and
    visualization regardless of which rank-allocation mode is active.
    """
    modules = list(lora_module_dict.values())
    if not modules:
        return
    ema_G = [_safe_float(m.ema_G) for m in modules]
    G_z = _zscore(ema_G)
    for idx, m in enumerate(modules):
        m.G_z = float(G_z[idx])
    sorted_G = sorted(modules, key=lambda x: x.ema_G, reverse=True)
    for r, m in enumerate(sorted_G, start=1):
        m.G_rank = int(r)


def compute_proxy_goodput(
    lora_module_dict: Dict[str, IPDLoRALinear],
    delta_val_loss: float,
    interval_steps: int,
    learning_rate: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_adam_direction: bool = False,
    beta_G: float = 0.9,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Low-cost GP-only proxy goodput.

    This variant does not use Importance or Plasticity. It allocates the global
    validation-loss change in the current window by a per-module first-order
    gradient progress proxy, then divides by the module rank-step cost:

        proxy_m = max(sum_p -<grad_p, -lr * direction_p>, 0)
        share_m = proxy_m / sum_j proxy_j
        G_m     = delta_val_loss * share_m / (r_m * K)

    When gradients are unavailable or all proxies are zero, the window is split
    uniformly so every module still receives a well-defined goodput estimate.
    """
    modules = list(lora_module_dict.values())
    if not modules:
        return {}

    proxies = []
    for m in modules:
        if int(m.active_rank) <= 0:
            proxies.append(0.0)
            continue
        progress = 0.0
        for p in m.lora_parameters:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            direction = grad
            if use_adam_direction and optimizer is not None:
                state = optimizer.state.get(p, {})
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)
                if exp_avg is not None and exp_avg_sq is not None:
                    direction = exp_avg / (exp_avg_sq.sqrt() + eps)
            delta_w = -float(learning_rate) * direction
            progress += float((-torch.sum(grad * delta_w)).item())
        proxies.append(max(progress, 0.0))

    proxy_arr = np.asarray(proxies, dtype=np.float64)
    denom = float(proxy_arr.sum())
    if denom <= eps:
        share = np.ones(len(modules), dtype=np.float64) / float(len(modules))
    else:
        share = proxy_arr / denom

    K = max(1, int(interval_steps))
    out: Dict[str, float] = {}
    for idx, m in enumerate(modules):
        r = max(1, int(m.active_rank))
        g = float(delta_val_loss) * float(share[idx]) / float(r * K)
        m.current_share = float(share[idx])
        m.current_G = g
        m.ema_G = beta_G * m.ema_G + (1.0 - beta_G) * g
        out[m.module_name] = g
    finalize_goodput_stats(lora_module_dict)
    return out


def compute_probing_goodput(
    model: nn.Module,
    lora_module_dict: Dict[str, IPDLoRALinear],
    eval_dataloader,
    device: torch.device,
    learning_rate: float,
    beta_G: float = 0.9,
    max_batches: Optional[int] = None,
    module_subset_names: Optional[Sequence[str]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_adam_direction: bool = False,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Method B: one-step probing goodput (recommended for main experiments).

    For each module m at an evaluation window:
      1. record baseline validation loss L_val(theta_t)
      2. freeze all other modules (only module m moves)
      3. take a single gradient step on module m's LoRA params
      4. measure L_val(theta_t - eta * g_m)
      5. roll back parameters
    and define:

        G_hat_m(t) = (L_val(theta_t) - L_val(theta_t - eta * g_m)) / r_m

    This reuses the gradients already populated by the main training backward
    pass (so it does NOT run an extra backward and does NOT disturb the pending
    optimizer step). Loss is measured on the supplied calibration/validation
    loader. Cost = 1 baseline eval + 1 eval per probed module.
    """
    modules_all = list(lora_module_dict.values())
    if not modules_all:
        return {}

    if module_subset_names is None:
        candidates = [n for n in lora_module_dict.keys()]
    else:
        candidates = [n for n in module_subset_names if n in lora_module_dict]

    model.eval()
    baseline = _avg_loss_over_loader(model, eval_dataloader, device, max_batches=max_batches)

    out: Dict[str, float] = {}
    for name in candidates:
        m = lora_module_dict[name]
        if int(m.active_rank) <= 0:
            m.current_G = 0.0
            m.ema_G = beta_G * m.ema_G
            out[name] = 0.0
            continue

        saved = [(p, p.detach().to("cpu", copy=True)) for p in m.lora_parameters]
        moved = False
        with torch.no_grad():
            for p in m.lora_parameters:
                if p.grad is None:
                    continue
                if use_adam_direction and optimizer is not None:
                    state = optimizer.state.get(p, {})
                    exp_avg = state.get("exp_avg", None)
                    exp_avg_sq = state.get("exp_avg_sq", None)
                    if exp_avg is not None and exp_avg_sq is not None:
                        direction = exp_avg / (exp_avg_sq.sqrt() + eps)
                    else:
                        direction = p.grad
                else:
                    direction = p.grad
                p.add_(direction, alpha=-float(learning_rate))
                moved = True

        if not moved:
            for p, p0 in saved:
                with torch.no_grad():
                    p.copy_(p0.to(device=p.device, dtype=p.dtype, non_blocking=True))
            g = 0.0
        else:
            after = _avg_loss_over_loader(model, eval_dataloader, device, max_batches=max_batches)
            for p, p0 in saved:
                with torch.no_grad():
                    p.copy_(p0.to(device=p.device, dtype=p.dtype, non_blocking=True))
            r = max(1, int(m.active_rank))
            g = float(baseline - after) / float(r)

        m.current_G = g
        m.ema_G = beta_G * m.ema_G + (1.0 - beta_G) * g
        out[name] = g

    finalize_goodput_stats(lora_module_dict)
    return out


def update_goodput_rank_allocation(
    lora_module_dict: Dict[str, IPDLoRALinear],
    total_rank_budget: int,
    active_rank_choices: Sequence[int],
    r_min: int = 1,
    max_rank: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    """Allocate LoRA ranks using only Module Learning Goodput.

    GP-only policy:

        S_m = minmax(ema_G_m)
        r_m = r_min + floor(B * S_m / sum_j S_j)

    No Importance, Plasticity, or quadrant signals are computed. If all goodput
    scores are effectively tied, the remaining budget is distributed uniformly.
    """
    modules = list(lora_module_dict.values())
    if not modules:
        finalize_goodput_stats(lora_module_dict)
        return {}

    S = np.clip(_minmax_nonneg([_safe_float(m.ema_G) for m in modules]), 0.0, None)
    extra = _waterfill_extra_ranks(
        S,
        total_rank_budget=int(total_rank_budget),
        r_min=int(max(0, r_min)),
        active_rank_choices=active_rank_choices,
        max_rank=max_rank,
        eps=eps,
    )
    rmin_choice = int(_round_to_choice(int(max(0, r_min)), active_rank_choices))
    maxr = int(max_rank) if max_rank is not None else int(max(active_rank_choices))
    for idx, m in enumerate(modules):
        r = int(_round_to_choice(int(min(rmin_choice + int(extra[idx]), maxr)), active_rank_choices))
        m.target_rank = int(r)
        m.active_rank = int(r)
        m.update_interval = 1

    finalize_goodput_stats(lora_module_dict)

    snapshot: Dict[str, Dict[str, float]] = {}
    for m in modules:
        snapshot[m.module_name] = {
            "current_G": float(m.current_G),
            "ema_G": float(m.ema_G),
            "G_z": float(m.G_z),
            "active_rank": int(m.active_rank),
            "target_rank": int(m.target_rank),
            "update_interval": int(m.update_interval),
            "G_rank": int(m.G_rank),
            "current_share": float(m.current_share),
        }
    return snapshot


def set_uniform_rank(
    lora_module_dict: Dict[str, IPDLoRALinear],
    rank: int,
    active_rank_choices: Sequence[int],
    max_rank: Optional[int] = None,
) -> None:
    """Static uniform rank (vanilla LoRA baseline): every module keeps `rank`.

    No scoring, no reallocation. Used by the `lora` baseline and as a fixed-budget
    reference point on the allocation-timing axis.
    """
    maxr = int(max_rank) if max_rank is not None else int(max(active_rank_choices))
    r = int(_round_to_choice(int(min(int(rank), maxr)), active_rank_choices))
    for m in lora_module_dict.values():
        m.active_rank = int(r)
        m.target_rank = int(r)
        m.update_interval = 1
    finalize_goodput_stats(lora_module_dict)


def allocate_rank_by_score(
    lora_module_dict: Dict[str, IPDLoRALinear],
    score_map: Dict[str, float],
    total_rank_budget: int,
    active_rank_choices: Sequence[int],
    r_min: int = 1,
    max_rank: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, int]:
    """Allocate a fixed rank budget by an arbitrary per-module score (one-shot).

    Same budget-conserving floor+remainder scheme as
    ``update_goodput_rank_allocation`` but driven by an external score map, so it
    can serve GoRA-style pre-training allocation (score = gradient importance) and
    any other static allocation baseline on the same footing.
    """
    modules = list(lora_module_dict.values())
    if not modules:
        finalize_goodput_stats(lora_module_dict)
        return {}

    S = np.clip(_minmax_nonneg([_safe_float(score_map.get(m.module_name, 0.0)) for m in modules]), 0.0, None)
    extra = _waterfill_extra_ranks(
        S,
        total_rank_budget=int(total_rank_budget),
        r_min=int(max(0, r_min)),
        active_rank_choices=active_rank_choices,
        max_rank=max_rank,
        eps=eps,
    )
    rmin_choice = int(_round_to_choice(int(max(0, r_min)), active_rank_choices))
    maxr = int(max_rank) if max_rank is not None else int(max(active_rank_choices))
    for idx, m in enumerate(modules):
        r = int(_round_to_choice(int(min(rmin_choice + int(extra[idx]), maxr)), active_rank_choices))
        m.target_rank = int(r)
        m.active_rank = int(r)
        m.update_interval = 1
    finalize_goodput_stats(lora_module_dict)
    return {m.module_name: int(m.active_rank) for m in modules}


def compute_pretrain_gradient_importance(
    model: nn.Module,
    lora_module_dict: Dict[str, IPDLoRALinear],
    dataloader,
    device: torch.device,
    num_batches: int = 8,
) -> Dict[str, float]:
    """GoRA-style one-shot importance from pre-training gradients.

    Full-precision path (DeBERTa): temporarily enables grad on each injected
    module's frozen base weight and scores by accumulated sensitivity |W . grad_W|.

    Quantized path (QLoRA/LLaMA): the 4-bit base weight cannot require grad, so we
    instead score each module by its LoRA-branch gradient sensitivity over a few
    batches (sum of |grad| on lora_A/lora_B). This is the same "which module most
    affects the loss" signal, measured through the adapter rather than the frozen
    quantized base. All grads are cleared before returning so the optimizer
    (LoRA + classifier params only) is unaffected.
    """
    # A base weight can require grad only if it is a floating-point tensor.
    quantized = any(
        not torch.is_floating_point(m.base_linear.weight) for m in lora_module_dict.values()
    )

    base_params: List[Tuple[str, nn.Parameter]] = []
    if not quantized:
        for m in lora_module_dict.values():
            w = m.base_linear.weight
            w.requires_grad = True
            base_params.append((m.module_name, w))

    was_training = model.training
    if quantized:
        # Keep training mode so gradient checkpointing stays active; otherwise the
        # full 8B activation graph is materialized for backward and OOMs a 24GB card.
        model.train()
    else:
        model.eval()  # DeBERTa: small model, stable importance without dropout noise
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    seen = 0
    for b_idx, batch in enumerate(dataloader):
        if b_idx >= int(max(1, num_batches)):
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_autocast(device):
            out = model(**batch)
            loss = out.loss
        loss.backward()
        seen += 1

    importance: Dict[str, float] = {}
    if not quantized:
        for name, w in base_params:
            if w.grad is None:
                importance[name] = 0.0
            else:
                importance[name] = float((w.detach() * w.grad.detach()).abs().sum().item())
        for _, w in base_params:
            w.grad = None
            w.requires_grad = False
    else:
        for m in lora_module_dict.values():
            score = 0.0
            for p in (m.lora_A, m.lora_B):
                if p.grad is not None:
                    score += float(p.grad.detach().abs().sum().item())
            importance[m.module_name] = score

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if was_training:
        model.train()
    return importance


def apply_update_frequency_mask(
    lora_module_dict: Dict[str, IPDLoRALinear],
    global_step: int,
) -> None:
    """Mask gradients for inactive ranks; GP-only keeps update_interval=1."""
    for module in lora_module_dict.values():
        freeze_grad = False
        if module.active_rank <= 0:
            freeze_grad = True
        if (module.update_interval > 0) and (global_step % module.update_interval != 0):
            freeze_grad = True
        if freeze_grad:
            for p in module.lora_parameters:
                if p.grad is not None:
                    p.grad.zero_()


def collect_module_rows(
    lora_module_dict: Dict[str, IPDLoRALinear],
    step: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for m in lora_module_dict.values():
        rows.append(
            {
                "step": int(step),
                "module_name": m.module_name,
                "layer_index": int(m.layer_index),
                "projection_type": m.projection_type,
                "current_G": float(m.current_G),
                "ema_G": float(m.ema_G),
                "G_z": float(m.G_z),
                "G_rank": int(m.G_rank),
                "current_share": float(m.current_share),
                "active_rank": int(m.active_rank),
                "target_rank": int(m.target_rank),
                "update_interval": int(m.update_interval),
            }
        )
    return rows


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = (trainable / total) if total > 0 else 0.0
    return total, trainable, ratio


def count_effective_trainable_parameters(
    model: nn.Module,
    lora_module_dict: Dict[str, IPDLoRALinear],
) -> Tuple[int, int, int]:
    """
    Count effective trainable parameters under dynamic-rank LoRA.

    Returns:
    - effective_total: effective non-LoRA trainable + effective LoRA trainable
    - effective_lora: sum(active_rank * (in_features + out_features)) over modules
    - non_lora_trainable: trainable params excluding LoRA A/B tensors
    """
    non_lora_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("lora_A") or name.endswith("lora_B"):
            continue
        non_lora_trainable += int(p.numel())

    effective_lora = 0
    for module in lora_module_dict.values():
        r = int(max(0, min(int(module.active_rank), int(module.max_rank))))
        effective_lora += int(r * (int(module.in_features) + int(module.out_features)))

    effective_total = int(non_lora_trainable + effective_lora)
    return effective_total, int(effective_lora), int(non_lora_trainable)
