"""
metric/pre_importance/spectral_entropy.py

定义 5：谱熵重要性（Spectral Entropy Importance）
─────────────────────────────────────────────────────────────────────────────
公式：
    I_m^pre = - (1/log r) Σ_{i=1}^r  s_i · log(s_i + ε)

    其中：
        λ_i  = 权重矩阵的第 i 大奇异值（σ_i）
        s_i  = λ_i² / Σ_j λ_j²          （归一化"奇异值能量"占比）
        r    = 奇异值个数（矩阵秩上界）
        ε    = 数值稳定项（默认 1e-10）

含义：
  s_i 是各奇异值方向的"能量"在总能量中的占比，谱熵衡量该分布的均匀程度：
    → 谱熵 ≈ 1：奇异值分布均匀（矩阵各向同性，模块通用/充分利用）
    → 谱熵 ≈ 0：奇异值集中于少数主方向（模块高度特化或接近低秩）

  除以 log(r) 将谱熵归一化到 [0, 1]，便于跨模块、跨模型比较。

头级别扩展（head_granularity=True）：
    对注意力模块的每个头的权重子矩阵分别计算谱熵。
    各头子矩阵形状为 [head_dim, hidden_size]（QKV）或 [hidden_size, head_dim]（OUT）。

此指标不依赖数据集，纯参数统计，与奇异值指标共享 SVD 计算逻辑。

保存格式：
    spectral_entropy.json
    {
      "module_scores": {
        module_name: {
          "spectral_entropy": float,   # 归一化谱熵 ∈ [0, 1]
          "raw_entropy":      float,
          "rank":             int,
          "matrix_shape":     [int, int]
        }, ...
      },
      "eps": float,
      "head_scores": {                             # 仅 head_granularity=True 时存在
        module_name: {
          "head_0": {"spectral_entropy": float, "raw_entropy": float, ...},
          ...
        }, ...
      }
    }
"""

import math
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase
from .attn_head import (
    get_attn_head_config,
    get_attn_modules,
    compute_head_svd_scores,
)


class SpectralEntropyImportance(ImportanceBase):
    """
    谱熵重要性（纯参数统计，无需数据）。

    归一化谱熵越高 → 奇异值分布越均匀 → 模块越"通用"，可塑空间更大。
    归一化谱熵越低 → 奇异值越集中   → 模块越"特化"，低秩结构明显。
    """

    name = "spectral_entropy"
    needs_data = False

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        eps: float = 1e-10,
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            eps:              防止 log(0) 的数值稳定项。
            head_granularity: 若为 True，对注意力模块额外计算每头的谱熵。
        """
        module_scores: Dict[str, Any] = {}

        for module_name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or weight.dim() < 2:
                continue

            W = weight.detach().float()
            if W.dim() > 2:
                W = W.view(W.size(0), -1)

            try:
                sv = torch.linalg.svdvals(W)
            except Exception as e:
                print(f"  [spectral_entropy] SVD failed for {module_name}: {e}")
                continue

            r = sv.numel()
            if r <= 1:
                module_scores[module_name] = {
                    "spectral_entropy": 0.0,
                    "raw_entropy":      0.0,
                    "rank":             r,
                    "matrix_shape":     list(W.shape),
                }
                continue

            lambda_sq = sv.pow(2)
            s = lambda_sq / (lambda_sq.sum() + eps)
            raw_entropy = -(s * torch.log(s + eps)).sum().item()
            normalized_entropy = raw_entropy / math.log(r)

            module_scores[module_name] = {
                "spectral_entropy": normalized_entropy,
                "raw_entropy":      raw_entropy,
                "rank":             r,
                "matrix_shape":     list(W.shape),
            }

        result: Dict[str, Any] = {
            "module_scores": module_scores,
            "eps":           eps,
        }

        # ── 头级别扩展（各头子矩阵的谱熵，无额外数据）──────────────────────
        if head_granularity:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [spectral_entropy] head_granularity=True 但模型无 config，跳过")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)

                def _se_metric(W_h: torch.Tensor) -> Dict[str, Any]:
                    r_h = min(W_h.shape)
                    if r_h <= 1:
                        return {
                            "spectral_entropy": 0.0,
                            "raw_entropy":      0.0,
                            "rank":             r_h,
                            "matrix_shape":     list(W_h.shape),
                        }
                    sv_h = torch.linalg.svdvals(W_h)
                    r_h  = sv_h.numel()
                    lsq  = sv_h.pow(2)
                    s_h  = lsq / (lsq.sum() + eps)
                    raw  = -(s_h * torch.log(s_h + eps)).sum().item()
                    return {
                        "spectral_entropy": raw / math.log(r_h),
                        "raw_entropy":      raw,
                        "rank":             r_h,
                        "matrix_shape":     list(W_h.shape),
                    }

                result["head_scores"] = compute_head_svd_scores(
                    model, attn_cfg, attn_mods, _se_metric
                )
                print(f"  [spectral_entropy] 计算了 {len(attn_mods)} 个注意力模块的头级别谱熵"
                      f"（每模块 {attn_cfg.num_heads} 头）")
        # ─────────────────────────────────────────────────────────────────────

        return result
