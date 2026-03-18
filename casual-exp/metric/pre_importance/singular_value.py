"""
metric/pre_importance/singular_value.py

定义 4：奇异值重要性（Singular Value Importance）
─────────────────────────────────────────────────────────────────────────────
公式：
    (核范数)  I_m^pre = Σ_j σ_j          （全部奇异值求和）
    (截断和)  I_m^pre = Σ_{j=1}^k σ_j    （前 k 个主奇异值求和）

对每个含 2D+ 权重矩阵的叶模块执行 SVD，同时计算两种变体。
高维权重（如 Conv）展平为 [out_channels, -1] 后再做 SVD。

头级别扩展（head_granularity=True）：
    对注意力模块，对每个头的权重子矩阵分别执行 SVD，
    计算头级别的核范数和前 k 奇异值之和。
    各头子矩阵形状为 [head_dim, hidden_size]（QKV）或 [hidden_size, head_dim]（OUT）。

此指标不依赖数据集，纯参数统计，计算快。

保存格式：
    singular_value.json
    {
      "module_scores": {
        module_name: {
          "nuclear_norm":   float,
          "top{k}_sum":     float,
          "max_sv":         float,
          "min_sv":         float,
          "num_singular_values": int,
          "matrix_shape":   [int, int]
        }, ...
      },
      "top_k": int,
      "head_scores": {                             # 仅 head_granularity=True 时存在
        module_name: {
          "head_0": {"nuclear_norm": float, "top{k}_sum": float, ...},
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


class SingularValueImportance(ImportanceBase):
    """
    奇异值重要性（纯参数统计，无需数据）。

    同时输出核范数和前 k 奇异值之和，结果保存在同一文件中。
    """

    name = "singular_value"
    needs_data = False

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        top_k: int = 32,
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            top_k:            前 k 个奇异值的截断数（若模块实际奇异值数 < k 则取全部）。
            head_granularity: 若为 True，对注意力模块额外计算每头的奇异值指标。
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
                print(f"  [singular_value] SVD failed for {module_name}: {e}")
                continue

            r = sv.numel()
            k_actual = min(top_k, r)

            module_scores[module_name] = {
                "nuclear_norm":        sv.sum().item(),
                f"top{top_k}_sum":     sv[:k_actual].sum().item(),
                "max_sv":              sv[0].item() if r > 0 else 0.0,
                "min_sv":              sv[-1].item() if r > 0 else 0.0,
                "num_singular_values": r,
                "matrix_shape":        list(W.shape),
            }

        result: Dict[str, Any] = {
            "module_scores": module_scores,
            "top_k":         top_k,
        }

        # ── 头级别扩展（各头子矩阵的 SVD，无额外数据）──────────────────────
        if head_granularity:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [singular_value] head_granularity=True 但模型无 config，跳过")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)

                def _sv_metric(W_h: torch.Tensor) -> Dict[str, Any]:
                    sv_h = torch.linalg.svdvals(W_h)
                    r_h = sv_h.numel()
                    k_h = min(top_k, r_h)
                    return {
                        "nuclear_norm":        sv_h.sum().item(),
                        f"top{top_k}_sum":     sv_h[:k_h].sum().item(),
                        "max_sv":              sv_h[0].item() if r_h > 0 else 0.0,
                        "min_sv":              sv_h[-1].item() if r_h > 0 else 0.0,
                        "num_singular_values": r_h,
                        "matrix_shape":        list(W_h.shape),
                    }

                result["head_scores"] = compute_head_svd_scores(
                    model, attn_cfg, attn_mods, _sv_metric
                )
                print(f"  [singular_value] 计算了 {len(attn_mods)} 个注意力模块的头级别奇异值"
                      f"（每模块 {attn_cfg.num_heads} 头）")
        # ─────────────────────────────────────────────────────────────────────

        return result
