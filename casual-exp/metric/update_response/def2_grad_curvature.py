"""
metric/update_response/def2_grad_curvature.py

定义 2：梯度-曲率归一化预测
─────────────────────────────────────────────────────────────────────────────
公式（模块级，Option A）：

    \hat{R}_m = E[‖∇_{θ_m}L‖₂] / √(E[‖∇_{θ_m}L‖₂²] + ε)

头级别公式（head_granularity=True）：

    \hat{R}_h = E[‖g_h‖₂] / √(E[‖g_h‖₂²] + ε)

    其中 ‖g_h‖₂² = Σ_{i∈head_h} g_i²（头 h 的参数梯度向量范数平方），
    分母 E[‖g_h‖₂²] = Σ_{i∈head_h} E[g_i²]（元素级 Fisher 在头切片上的求和）。

实现（共用一次 backward 循环）：
    · 模块级：累积每 batch 的模块梯度向量范数 ‖g_m‖（分子）
             及其平方 ‖g_m‖²（分母 = 模块级 Fisher）
    · 头级别（head_granularity=True）：
        - 额外累积注意力参数的元素级梯度平方（用于 Fisher 分母切片）
        - 每 batch 计算各头的梯度范数 ‖g_h‖（用于期望分子）

保存格式：
    def2_grad_curvature.json
    {
      "module_scores":   {module_name: float, ...},
      "grad_norm_mean":  {module_name: float, ...},
      "fisher_module":   {module_name: float, ...},
      "num_batches": int,
      "epsilon": float,
      "head_scores":     {module_name: {"head_0": float, ...}, ...}  # 仅 head_granularity=True
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import PreTrainingMetric, group_params_by_module
from .attn_head import (
    get_attn_head_config,
    get_attn_modules,
    get_head_weight_view,
    get_head_bias_view,
)


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
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            num_batches:      Monte Carlo 估计 mini-batch 数量（建议 16–64）
            epsilon:          数值稳定项
            head_granularity: 若为 True，额外输出注意力头级别的梯度-曲率分数
        """
        model = model.to(device)
        model.eval()

        module_groups = group_params_by_module(model)
        param_dict = dict(model.named_parameters())  # 预缓存，避免内层循环 O(N) 调用

        # 模块级累积
        grad_norm_sum:    Dict[str, float] = {m: 0.0 for m in module_groups}
        grad_norm_sq_sum: Dict[str, float] = {m: 0.0 for m in module_groups}

        # 头级别预备（仅 head_granularity=True 时启用）
        attn_cfg  = None
        attn_mods: Dict[str, str] = {}
        head_grad_sq_acc: Dict[str, torch.Tensor] = {}   # 元素级 Fisher（注意力参数）
        head_norm_sum:    Dict[str, Dict[int, float]] = {}  # 分子：每 batch 头梯度范数累积

        if head_granularity:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [def2_grad_curvature] head_granularity=True 但模型无 config，"
                      "跳过头级别计算")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)
                head_norm_sum = {
                    m: {h: 0.0 for h in range(attn_cfg.num_heads)}
                    for m in attn_mods
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
            loss = model(**inputs).loss
            loss.backward()

            # ── 模块级：拼接模块内所有参数梯度向量后取 L2 范数 ──────────────
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

            # ── 头级别：元素级累积 + 每头梯度范数 ──────────────────────────
            if attn_cfg is not None:
                for m_name, m_type in attn_mods.items():
                    head_sq: Dict[int, float] = {
                        h: 0.0 for h in range(attn_cfg.num_heads)
                    }
                    for suffix in ("weight", "bias"):
                        pn = f"{m_name}.{suffix}"
                        p = param_dict.get(pn)
                        if p is None or p.grad is None:
                            continue
                        g = p.grad.detach()

                        # 累积元素级梯度平方（用于 Fisher 分母切片）
                        if pn not in head_grad_sq_acc:
                            head_grad_sq_acc[pn] = torch.zeros_like(g, device=device)
                        head_grad_sq_acc[pn].add_(g.pow(2))

                        # 每头梯度范数（用于分子期望）
                        for h in range(attn_cfg.num_heads):
                            if suffix == "weight":
                                view = get_head_weight_view(g, m_type, h, attn_cfg.head_dim)
                            else:
                                view = get_head_bias_view(g, m_type, h, attn_cfg.head_dim)
                            if view is not None:
                                head_sq[h] += view.pow(2).sum().item()

                    for h in range(attn_cfg.num_heads):
                        head_norm_sum[m_name][h] += head_sq[h] ** 0.5

            count += 1
            if count % 8 == 0:
                print(f"  [def2_grad_curvature] {count}/{num_batches} batches,"
                      f" loss={loss.item():.4f}")

        if count == 0:
            raise RuntimeError("[GradCurvatureMetric] dataloader is empty")

        model.zero_grad()

        # 期望估计（模块级）
        grad_norm_mean: Dict[str, float] = {
            m: grad_norm_sum[m] / count for m in module_groups
        }
        fisher_module: Dict[str, float] = {
            m: grad_norm_sq_sum[m] / count for m in module_groups
        }

        # 模块级 R̂_m
        module_scores: Dict[str, float] = {
            m: grad_norm_mean[m] / (fisher_module[m] + epsilon) ** 0.5
            for m in module_groups
        }

        result: Dict[str, Any] = {
            "module_scores":  module_scores,
            "grad_norm_mean": grad_norm_mean,
            "fisher_module":  fisher_module,
            "num_batches":    count,
            "epsilon":        epsilon,
        }

        # ── 头级别扩展 ────────────────────────────────────────────────────────
        if attn_cfg is not None and head_grad_sq_acc:
            head_scores: Dict[str, Dict[str, float]] = {}

            for m_name, m_type in attn_mods.items():
                per_head: Dict[str, float] = {}
                for h in range(attn_cfg.num_heads):
                    # 分子：E[‖g_h‖]
                    gnorm_mean_h = head_norm_sum[m_name][h] / count

                    # 分母：√(Fisher_h + ε) = √(Σ_i E[g_i²] + ε)
                    fisher_h = 0.0
                    for suffix in ("weight", "bias"):
                        pn = f"{m_name}.{suffix}"
                        acc = head_grad_sq_acc.get(pn)
                        if acc is None:
                            continue
                        if suffix == "weight":
                            view = get_head_weight_view(acc, m_type, h, attn_cfg.head_dim)
                        else:
                            view = get_head_bias_view(acc, m_type, h, attn_cfg.head_dim)
                        if view is not None:
                            fisher_h += (view.sum() / count).item()

                    per_head[f"head_{h}"] = gnorm_mean_h / (fisher_h + epsilon) ** 0.5

                head_scores[m_name] = per_head

            result["head_scores"] = head_scores
            print(f"  [def2_grad_curvature] 计算了 {len(attn_mods)} 个注意力模块的"
                  f"头级别分数（每模块 {attn_cfg.num_heads} 头）")
        # ─────────────────────────────────────────────────────────────────────

        return result
