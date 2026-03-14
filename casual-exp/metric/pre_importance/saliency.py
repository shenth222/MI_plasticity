"""
metric/pre_importance/saliency.py

定义 2：梯度敏感度 / Saliency
─────────────────────────────────────────────────────────────────────────────
两种变体同时计算，共用一次 backward，额外开销几乎为零：

  (A) grad_norm  ── 纯梯度 L2 范数
        I_m = ||∇_{θ_m} L||₂
      衡量 loss 曲面在该模块参数方向上的"坡度"，
      不受参数自身量级影响，初始化阶段语义清晰。

  (B) taylor     ── 权重 × 梯度（Taylor 一阶近似）
        I_m = Σ_{i∈θ_m} |θᵢ · gᵢ|
      近似"若将该参数置零，loss 约增加多少"（量纲与 loss 一致），
      同时考虑梯度方向与参数量级，适合结构剪枝场景。

保存格式：
    saliency.json
    {
      "grad_norm": {
        "module_scores": {module_name: float, ...},
        "param_scores":  {param_name:  float, ...}
      },
      "taylor": {
        "module_scores": {...},
        "param_scores":  {...}
      },
      "num_batches": int
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase, group_params_by_module


class SaliencyImportance(ImportanceBase):
    """
    梯度敏感度 / Saliency 重要性。

    同时输出 grad_norm 和 taylor 两种变体，结果保存在同一文件中。
    """

    name = "saliency"
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
            num_batches: 用于估计的 mini-batch 数量。
        """
        model = model.to(device)
        model.eval()

        # 初始化：(A) |gᵢ| 的累积、(B) |θᵢ·gᵢ| 的累积
        grad_abs_acc: Dict[str, torch.Tensor] = {}
        taylor_abs_acc: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_abs_acc[name]   = torch.zeros_like(param.data, device=device)
                taylor_abs_acc[name] = torch.zeros_like(param.data, device=device)

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
                    g = param.grad.detach()
                    grad_abs_acc[name]   += g.abs()
                    taylor_abs_acc[name] += (param.data.detach() * g).abs()

            count += 1

        if count == 0:
            raise RuntimeError("[SaliencyImportance] dataloader is empty")

        # 参数粒度：各 batch 平均
        param_grad_norm: Dict[str, float] = {
            name: (acc.sum() / count).item()
            for name, acc in grad_abs_acc.items()
        }
        param_taylor: Dict[str, float] = {
            name: (acc.sum() / count).item()
            for name, acc in taylor_abs_acc.items()
        }

        # 叶模块粒度聚合
        module_groups = group_params_by_module(model)

        def agg(param_scores):
            return {
                m: sum(param_scores.get(p, 0.0) for p in ps)
                for m, ps in module_groups.items()
            }

        model.zero_grad()
        return {
            "grad_norm": {
                "module_scores": agg(param_grad_norm),
                "param_scores": param_grad_norm,
            },
            "taylor": {
                "module_scores": agg(param_taylor),
                "param_scores": param_taylor,
            },
            "num_batches": count,
        }
