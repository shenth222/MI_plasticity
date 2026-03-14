"""
metric/pre_importance/singular_value.py

定义 4：奇异值重要性（Singular Value Importance）
─────────────────────────────────────────────────────────────────────────────
公式：
    (核范数)  I_m^pre = Σ_j σ_j          （全部奇异值求和）
    (截断和)  I_m^pre = Σ_{j=1}^k σ_j    （前 k 个主奇异值求和）

对每个含 2D+ 权重矩阵的叶模块执行 SVD，同时计算两种变体。
高维权重（如 Conv）展平为 [out_channels, -1] 后再做 SVD。

含义：
  核范数反映矩阵在所有方向上的"信息量"总和；
  前 k 个奇异值聚焦于主要特征方向，过滤掉尾部噪声奇异值。

此指标不依赖数据集，纯参数统计，计算快。

保存格式：
    singular_value.json
    {
      "module_scores": {
        module_name: {
          "nuclear_norm":   float,   # 全部奇异值之和
          "top{k}_sum":     float,   # 前 k 奇异值之和
          "max_sv":         float,
          "min_sv":         float,
          "num_singular_values": int,
          "matrix_shape":   [int, int]
        }, ...
      },
      "top_k": int
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase


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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            top_k: 前 k 个奇异值的截断数。
                   若模块实际奇异值个数 < top_k，则取全部（不补零）。
        """
        module_scores: Dict[str, Any] = {}

        for module_name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or weight.dim() < 2:
                continue

            W = weight.detach().float()
            # Conv 等高维张量展平为 [out, -1]
            if W.dim() > 2:
                W = W.view(W.size(0), -1)

            try:
                # torch.linalg.svdvals 仅计算奇异值（比完整 SVD 快）
                sv = torch.linalg.svdvals(W)   # 降序排列，形状 [min(m,n)]
            except Exception as e:
                print(f"  [singular_value] SVD failed for {module_name}: {e}")
                continue

            r = sv.numel()
            k_actual = min(top_k, r)

            module_scores[module_name] = {
                "nuclear_norm":          sv.sum().item(),
                f"top{top_k}_sum":       sv[:k_actual].sum().item(),
                "max_sv":                sv[0].item() if r > 0 else 0.0,
                "min_sv":                sv[-1].item() if r > 0 else 0.0,
                "num_singular_values":   r,
                "matrix_shape":          list(W.shape),
            }

        return {
            "module_scores": module_scores,
            "top_k": top_k,
        }
