"""
metric/update_response/def2_grad_curvature.py

定义 2：梯度-曲率归一化预测
─────────────────────────────────────────────────────────────────────────────
公式（模块级向量范数，Option A）：

    \hat{R}_m = \frac{E[\|\nabla_{\theta_m}\mathcal{L}\|_2]}
                     {\sqrt{E[\|\nabla_{\theta_m}\mathcal{L}\|_2^2] + \epsilon}}

其中 E[·] 为对训练数据分布的期望（Monte Carlo 多 batch 估计）。

含义：
    分子 E[‖g_m‖₂]  ── 模块梯度向量范数的期望：反映"驱动力"大小。
    分母 √(Fisher_m + ε) ── Fisher 重要性的平方根：反映损失曲率/不确定性。
    两者之比类似 Adam 预条件化步长预测：
        曲率大时即使梯度强，实际位移也会被压制（分母大）；
        梯度一致且曲率小时，分数高，预测更新响应强。

    与 Fisher 重要性（I_m^pre）的核心区别：
        Fisher_m = E[‖g_m‖²]  →  衡量损失对该模块的平均敏感度（数值越大越"重要"）
        定义 2   = E[‖g_m‖] / √(Fisher_m + ε)  →  曲率归一化后的有效步长预测
        两者并非单调关系：高 Fisher（高曲率）的模块，定义 2 分数可能反而较低。

实现：
    共用一次 backward 循环，同时累积：
      · 各模块每 batch 的梯度向量范数 ‖g_m‖₂         → 分子 E[‖g_m‖]
      · 各模块每 batch 的梯度向量范数的平方 ‖g_m‖₂²  → 分母 E[‖g_m‖²] = Fisher_m

保存格式：
    def2_grad_curvature.json
    {
      "module_scores":   {module_name: float, ...},  # \hat{R}_m
      "grad_norm_mean":  {module_name: float, ...},  # E[‖g_m‖₂]（分子）
      "fisher_module":   {module_name: float, ...},  # E[‖g_m‖₂²]（分母的平方）
      "num_batches": int,
      "epsilon": float,
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import PreTrainingMetric, group_params_by_module


class GradCurvatureMetric(PreTrainingMetric):
    """
    梯度-曲率归一化预测（定义 2）。

    E[‖g_m‖] / √(E[‖g_m‖²] + ε)，与 Adam 预条件化步长同构。
    """

    name = "def2_grad_curvature"
    needs_data = True

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 32,
        epsilon: float = 1e-8,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            num_batches: Monte Carlo 估计使用的 mini-batch 数量（建议 16–64）
            epsilon:     数值稳定项（防止零 Fisher 导致除零或极大分数）
        """
        model = model.to(device)
        model.eval()

        module_groups = group_params_by_module(model)
        # 预缓存参数字典，避免在内层循环中反复调用 dict(model.named_parameters())
        param_dict = dict(model.named_parameters())

        # 各模块 batch 级梯度范数累积
        grad_norm_sum:    Dict[str, float] = {m: 0.0 for m in module_groups}
        grad_norm_sq_sum: Dict[str, float] = {m: 0.0 for m in module_groups}
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
            loss = model(**inputs).loss
            loss.backward()

            # 按模块聚合：拼接模块内所有参数的梯度向量后取 L2 范数
            for m_name, param_names in module_groups.items():
                grads = []
                for pn in param_names:
                    p = param_dict.get(pn)
                    if p is not None and p.grad is not None:
                        grads.append(p.grad.detach().view(-1))
                if grads:
                    norm = torch.cat(grads).norm(p=2).item()
                    grad_norm_sum[m_name]    += norm
                    grad_norm_sq_sum[m_name] += norm * norm

            count += 1
            if count % 8 == 0:
                print(f"  [def2_grad_curvature] {count}/{num_batches} batches,"
                      f" loss={loss.item():.4f}")

        if count == 0:
            raise RuntimeError("[GradCurvatureMetric] dataloader is empty")

        model.zero_grad()

        # 期望估计
        grad_norm_mean: Dict[str, float] = {
            m: grad_norm_sum[m] / count for m in module_groups
        }
        fisher_module: Dict[str, float] = {
            m: grad_norm_sq_sum[m] / count for m in module_groups
        }

        # \hat{R}_m = E[‖g_m‖] / √(E[‖g_m‖²] + ε)
        module_scores: Dict[str, float] = {
            m: grad_norm_mean[m] / (fisher_module[m] + epsilon) ** 0.5
            for m in module_groups
        }

        return {
            "module_scores":  module_scores,
            "grad_norm_mean": grad_norm_mean,
            "fisher_module":  fisher_module,
            "num_batches":    count,
            "epsilon":        epsilon,
        }
