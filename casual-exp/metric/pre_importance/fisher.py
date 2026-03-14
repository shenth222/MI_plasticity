"""
metric/pre_importance/fisher.py

定义 1：Fisher 型重要性
─────────────────────────────────────────────────────────────────────────────
公式：
    I_m^pre = E_{(x,y)~D}[ ||∇_{θ_m} L||₂² ]
            = Σ_{i∈θ_m} E[(∂L/∂θᵢ)²]

实现：
    在 num_batches 个 mini-batch 上累积每个参数的梯度平方，取均值作为
    期望的 Monte Carlo 估计（对角 Fisher 近似）。随后按叶模块聚合，
    即将同一模块下所有参数的 Fisher 分数相加。

保存格式：
    fisher.json
    {
      "module_scores": {module_name: float, ...},   # 叶模块聚合分数
      "param_scores":  {param_name:  float, ...},   # 参数粒度分数
      "num_batches": int
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase, group_params_by_module


class FisherImportance(ImportanceBase):
    """
    Fisher 型重要性（对角 Fisher 信息矩阵近似）。

    值越大 → 该模块对 loss 曲率贡献越大 → 训练前该模块越"重要"。
    """

    name = "fisher"
    needs_data = True

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 32,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            num_batches: 用于 Monte Carlo 估计的 mini-batch 数量。
                         越多估计越稳定，但计算时间线性增加。
                         建议范围：16–64。
        """
        model = model.to(device)
        model.eval()

        # 初始化：为每个可训练参数分配梯度平方累积张量
        grad_sq_acc: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            model.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_sq_acc[name] += param.grad.detach().pow(2)

            count += 1
            if count % 8 == 0:
                print(f"  [fisher] {count}/{num_batches} batches, "
                      f"loss={loss.item():.4f}")

        if count == 0:
            raise RuntimeError("[FisherImportance] dataloader is empty")

        # 参数粒度：每个参数的 E[g²]（梯度平方期望）求和
        param_scores: Dict[str, float] = {
            name: (acc.sum() / count).item()
            for name, acc in grad_sq_acc.items()
        }

        # 叶模块粒度：将同一模块的参数分数累加
        module_groups = group_params_by_module(model)
        module_scores: Dict[str, float] = {
            module_name: sum(param_scores.get(p, 0.0) for p in params)
            for module_name, params in module_groups.items()
        }

        model.zero_grad()
        return {
            "module_scores": module_scores,
            "param_scores": param_scores,
            "num_batches": count,
        }
