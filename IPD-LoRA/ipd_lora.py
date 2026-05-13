import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NEVER_UPDATE_INTERVAL = 10**12


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


class IPDLoRALinear(nn.Module):
    """
    LoRA linear wrapper with dynamic active_rank.

    Key design:
    - Keep max-rank LoRA parameters, but only compute active slices in forward.
    - active_rank == 0 physically skips LoRA branch in forward computation.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        module_name: str,
        layer_index: int,
        projection_type: str,
        max_rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        initial_active_rank: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"IPDLoRALinear only supports nn.Linear, got {type(base_linear)}")

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

        # Runtime states for IPD-LoRA control.
        self.active_rank = int(min(self.max_rank, initial_active_rank))
        self.target_rank = int(self.active_rank)
        self.update_interval = 1
        self.frozen_by_early_stop = False
        self.quadrant = "warmup"
        self.current_I = 0.0
        self.current_P = 0.0
        self.ema_I = 0.0
        self.ema_P = 0.0
        self.I_z = 0.0
        self.P_z = 0.0
        self.I_rank = 0
        self.P_rank = 0
        self.low_P_counter = 0
        self.prev_ema_I = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        r = int(self.active_rank)
        if r <= 0:
            # active_rank=0 means we fully skip LoRA branch to save LoRA-side compute.
            return base_out

        x_d = self.lora_dropout(x)
        A_slice = self.lora_A[:r, :]  # [r, in_features]
        B_slice = self.lora_B[:, :r]  # [out_features, r]
        low_rank = F.linear(x_d, A_slice)  # [..., r]
        delta = F.linear(low_rank, B_slice)  # [..., out_features]
        scaling = self.lora_alpha / max(r, 1)
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


def inject_ipd_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    max_rank: int,
    alpha: int,
    dropout: float,
    initial_active_rank: int,
    verbose: bool = True,
) -> Dict[str, IPDLoRALinear]:
    target_set = set(target_modules)
    replacements: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.split(".")[-1]
        if short_name not in target_set:
            continue
        if ".attention.self." not in name:
            continue
        replacements.append((name, module))

    lora_module_dict: Dict[str, IPDLoRALinear] = {}
    pattern = re.compile(r"layer\.(\d+)\.attention\.self\.(query_proj|value_proj)$")
    for full_name, base_linear in replacements:
        m = pattern.search(full_name)
        if m is None:
            continue
        layer_index = int(m.group(1))
        proj = m.group(2)
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
        print(f"[IPD-LoRA] Injected {len(lora_module_dict)} modules:")
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
        outputs = model(**batch)
        bs = int(batch["labels"].shape[0])
        total_loss += float(outputs.loss.item()) * bs
        total_count += bs
    if total_count == 0:
        return 0.0
    return total_loss / total_count


@torch.no_grad()
def compute_importance_scores(
    model: nn.Module,
    lora_module_dict: Dict[str, IPDLoRALinear],
    calibration_dataloader,
    device: torch.device,
    beta_I: float = 0.9,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Importance is computed by forward ablation on calibration data:
      I_m = L_calib(module off) - L_calib(normal)

    We use forward ablation instead of gradients because importance is about
    current retained function value, not immediate local optimization direction.
    """
    model.eval()
    baseline_loss = _avg_loss_over_loader(model, calibration_dataloader, device, max_batches=max_batches)
    scores: Dict[str, float] = {}

    for module_name, module in lora_module_dict.items():
        old_rank = int(module.active_rank)
        module.active_rank = 0
        ablated_loss = _avg_loss_over_loader(model, calibration_dataloader, device, max_batches=max_batches)
        module.active_rank = old_rank

        score = float(ablated_loss - baseline_loss)
        module.current_I = score
        module.ema_I = beta_I * module.ema_I + (1.0 - beta_I) * score
        scores[module_name] = score
    return scores


def compute_plasticity_scores(
    lora_module_dict: Dict[str, IPDLoRALinear],
    optimizer: torch.optim.Optimizer,
    beta_P: float = 0.9,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Plasticity uses AdamW-aware predicted gain:
      P_m = sum(grad * adam_direction) / cost_m

    This score estimates whether continuing to optimize this module is useful.
    It is not used as a direct learning-rate scale.
    """
    scores: Dict[str, float] = {}
    for module_name, module in lora_module_dict.items():
        if module.active_rank <= 0:
            score = 0.0
        else:
            score_sum = 0.0
            found_grad = False
            for p in module.lora_parameters:
                if p.grad is None:
                    continue
                found_grad = True
                grad = p.grad.detach()
                state = optimizer.state.get(p, {})
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)
                if exp_avg is None or exp_avg_sq is None:
                    direction = grad
                else:
                    direction = exp_avg / (exp_avg_sq.sqrt() + eps)
                score_sum += float(torch.sum(grad * direction).item())

            if (not found_grad) or module.cost <= 0:
                score = 0.0
            else:
                score = score_sum / float(module.cost)

        module.current_P = score
        module.ema_P = beta_P * module.ema_P + (1.0 - beta_P) * score
        scores[module_name] = score
    return scores


def update_quadrants_and_budget(
    lora_module_dict: Dict[str, IPDLoRALinear],
    total_rank_budget: int,
    active_rank_choices: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    modules = list(lora_module_dict.values())
    if not modules:
        return {}

    ema_I = [_safe_float(m.ema_I) for m in modules]
    ema_P = [_safe_float(m.ema_P) for m in modules]
    I_z = _zscore(ema_I)
    P_z = _zscore(ema_P)

    sorted_I = sorted(modules, key=lambda x: x.ema_I, reverse=True)
    sorted_P = sorted(modules, key=lambda x: x.ema_P, reverse=True)
    for r, m in enumerate(sorted_I, start=1):
        m.I_rank = r
    for r, m in enumerate(sorted_P, start=1):
        m.P_rank = r

    # Quadrant default policy:
    # - high_I_low_P should be retained but updated less often (preserve useful function, avoid noisy over-optimization).
    # - low_I_high_P stays as exploration region (small rank but frequent updates to test future utility).
    default_rank = {
        "high_I_high_P": 8,
        "high_I_low_P": 4,
        "low_I_high_P": 2,
        "low_I_low_P": 0,
    }
    default_interval = {
        "high_I_high_P": 1,
        "high_I_low_P": 4,
        "low_I_high_P": 1,
        "low_I_low_P": NEVER_UPDATE_INTERVAL,
    }

    for idx, m in enumerate(modules):
        m.I_z = float(I_z[idx])
        m.P_z = float(P_z[idx])
        if m.frozen_by_early_stop:
            m.quadrant = "frozen"
            m.target_rank = int(m.active_rank)
            m.update_interval = NEVER_UPDATE_INTERVAL
            continue

        high_I = m.I_z >= 0.0
        high_P = m.P_z >= 0.0
        if high_I and high_P:
            q = "high_I_high_P"
        elif high_I and (not high_P):
            q = "high_I_low_P"
        elif (not high_I) and high_P:
            q = "low_I_high_P"
        else:
            q = "low_I_low_P"
        m.quadrant = q
        m.target_rank = _round_to_choice(default_rank[q], active_rank_choices)
        m.update_interval = int(default_interval[q])

    def _current_total() -> int:
        return sum(int(m.target_rank) for m in modules)

    # Budget clipping with requested priority:
    # high_I_high_P > low_I_high_P > high_I_low_P > low_I_low_P
    # So pruning order is reverse priority.
    prune_order = ["low_I_low_P", "high_I_low_P", "low_I_high_P", "high_I_high_P"]
    budget = int(total_rank_budget)
    if _current_total() > budget:
        for q in prune_order:
            group = [m for m in modules if (m.quadrant == q and not m.frozen_by_early_stop)]
            # Prefer reducing larger ranks first.
            group.sort(key=lambda x: x.target_rank, reverse=True)
            changed = True
            while _current_total() > budget and changed:
                changed = False
                for m in group:
                    if _current_total() <= budget:
                        break
                    lower = _next_lower_choice(m.target_rank, active_rank_choices)
                    if lower < m.target_rank:
                        m.target_rank = int(lower)
                        changed = True
                if not changed:
                    break

    for m in modules:
        if not m.frozen_by_early_stop:
            m.active_rank = int(m.target_rank)

    snapshot: Dict[str, Dict[str, float]] = {}
    for m in modules:
        snapshot[m.module_name] = {
            "ema_I": float(m.ema_I),
            "ema_P": float(m.ema_P),
            "I_z": float(m.I_z),
            "P_z": float(m.P_z),
            "quadrant": m.quadrant,
            "active_rank": int(m.active_rank),
            "update_interval": int(m.update_interval),
            "I_rank": int(m.I_rank),
            "P_rank": int(m.P_rank),
        }
    return snapshot


def apply_module_early_stopping(
    lora_module_dict: Dict[str, IPDLoRALinear],
    patience: int = 3,
    p_low_threshold: float = -0.5,
    i_tolerance: float = 1e-4,
) -> None:
    for module in lora_module_dict.values():
        if module.frozen_by_early_stop:
            continue
        low_plasticity = module.P_z < p_low_threshold
        no_importance_growth = module.ema_I <= (module.prev_ema_I + i_tolerance)
        if low_plasticity and no_importance_growth:
            module.low_P_counter += 1
        else:
            module.low_P_counter = 0

        if module.low_P_counter >= patience:
            module.frozen_by_early_stop = True
            module.update_interval = NEVER_UPDATE_INTERVAL
            module.quadrant = "frozen"
        module.prev_ema_I = float(module.ema_I)


def apply_update_frequency_mask(
    lora_module_dict: Dict[str, IPDLoRALinear],
    global_step: int,
) -> None:
    """
    Keep forward contribution but sparsify updates via grad masking.

    This first version mainly saves algorithmic update budget; it may not
    immediately reduce full wall-clock because backbone forward/backward still runs.
    """
    for module in lora_module_dict.values():
        freeze_grad = False
        if module.active_rank <= 0:
            freeze_grad = True
        if module.frozen_by_early_stop:
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
                "current_I": float(m.current_I),
                "current_P": float(m.current_P),
                "ema_I": float(m.ema_I),
                "ema_P": float(m.ema_P),
                "I_z": float(m.I_z),
                "P_z": float(m.P_z),
                "I_rank": int(m.I_rank),
                "P_rank": int(m.P_rank),
                "quadrant": m.quadrant,
                "active_rank": int(m.active_rank),
                "update_interval": int(m.update_interval),
                "frozen_by_early_stop": bool(m.frozen_by_early_stop),
                "low_P_counter": int(m.low_P_counter),
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
