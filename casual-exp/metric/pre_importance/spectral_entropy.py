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

此指标不依赖数据集，纯参数统计，与奇异值指标共享 SVD 计算逻辑。

保存格式：
    spectral_entropy.json
    {
      "module_scores": {
        module_name: {
          "spectral_entropy": float,   # 归一化谱熵 ∈ [0, 1]
          "raw_entropy":      float,   # 未归一化熵
          "rank":             int,     # 奇异值个数
          "matrix_shape":     [int, int]
        }, ...
      },
      "eps": float
    }
"""

import math
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase


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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            eps: 防止 log(0) 的数值稳定项。
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
                sv = torch.linalg.svdvals(W)   # 降序，形状 [min(m,n)]
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

            # s_i = λ_i² / Σ_j λ_j²  （奇异值能量归一化）
            lambda_sq = sv.pow(2)
            s = lambda_sq / (lambda_sq.sum() + eps)

            # 香农熵：H = -Σ s_i log(s_i)
            raw_entropy = -(s * torch.log(s + eps)).sum().item()

            # 归一化：log(r) 是均匀分布的最大熵
            normalized_entropy = raw_entropy / math.log(r)

            module_scores[module_name] = {
                "spectral_entropy": normalized_entropy,
                "raw_entropy":      raw_entropy,
                "rank":             r,
                "matrix_shape":     list(W.shape),
            }

        return {
            "module_scores": module_scores,
            "eps": eps,
        }
