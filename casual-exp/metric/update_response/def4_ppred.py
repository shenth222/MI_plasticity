"""
metric/update_response/def4_ppred.py

定义 4：梯度信噪比（Ppred）
─────────────────────────────────────────────────────────────────────────────
公式（元素级）：
    G_i = E[|g_i|]，F_i = E[g_i²]，ppred_i = G_i² / (F_i + ε)

模块级聚合：
    Ppred_m = mean_{i ∈ θ_m}(ppred_i)

头级别聚合（head_granularity=True）：
    Ppred_h = mean_{i ∈ head_h}(ppred_i)

    其中 ppred_i = G_i² / (F_i + ε) 是元素级 SNR，
    head_h 为该头对应的权重/偏置参数切片。

实现：
    · 共用一次 backward 循环累积 abs_acc（G 的分子）和 sq_acc（F）
    · 头级别为后处理（利用已有 G/F 张量切片），无额外前向传播
    · 与 pre_importance 的 Fisher 共享相同的元素级累积结构

保存格式：
    def4_ppred.json
    {
      "module_scores":  {module_name: float, ...},
      "param_scores":   {param_name:  float, ...},
      "G_module":       {module_name: float, ...},
      "F_module":       {module_name: float, ...},
      "num_batches": int,
      "epsilon": float,
      "head_scores":    {module_name: {"head_0": float, ...}, ...}  # 仅 head_granularity=True
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


class PpredMetric(PreTrainingMetric):
    """
    梯度信噪比 Ppred（定义 4）。

    元素级 SNR = E[|g|]² / E[g²]，聚合为模块（或头）内元素的均值。
    """

    name = "def4_ppred"
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
            num_batches:      Monte Carlo 估计 batch 数（建议 16–64）
            epsilon:          数值稳定项
            head_granularity: 若为 True，额外输出注意力头级别的 Ppred 分数
        """
        model = model.to(device)
        model.eval()

        # 逐参数、元素级累积 |g| 与 g²
        abs_acc: Dict[str, torch.Tensor] = {}
        sq_acc:  Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                abs_acc[name] = torch.zeros_like(param.data, device=device)
                sq_acc[name]  = torch.zeros_like(param.data, device=device)

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

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    g = param.grad.detach()
                    abs_acc[name] += g.abs()
                    sq_acc[name]  += g.pow(2)

            count += 1
            if count % 8 == 0:
                print(f"  [def4_ppred] {count}/{num_batches} batches,"
                      f" loss={loss.item():.4f}")

        if count == 0:
            raise RuntimeError("[PpredMetric] dataloader is empty")

        model.zero_grad()

        # 期望：G_i = E[|g_i|]，F_i = E[g_i²]
        G: Dict[str, torch.Tensor] = {n: acc / count for n, acc in abs_acc.items()}
        F: Dict[str, torch.Tensor] = {n: acc / count for n, acc in sq_acc.items()}

        # 参数级 Ppred：各元素 ppred_i 的均值
        param_scores: Dict[str, float] = {}
        for name in G:
            ppred_elem = G[name].pow(2) / (F[name] + epsilon)
            param_scores[name] = ppred_elem.mean().item()

        # 模块级：拼接模块内所有参数的元素后统一计算
        module_groups = group_params_by_module(model)
        module_scores: Dict[str, float] = {}
        G_module:      Dict[str, float] = {}
        F_module:      Dict[str, float] = {}

        for m_name, params in module_groups.items():
            g_cat = torch.cat([G[pn].view(-1) for pn in params if pn in G])
            f_cat = torch.cat([F[pn].view(-1) for pn in params if pn in F])
            if g_cat.numel() == 0:
                module_scores[m_name] = 0.0
                G_module[m_name] = 0.0
                F_module[m_name] = 0.0
                continue
            ppred_all = g_cat.pow(2) / (f_cat + epsilon)
            module_scores[m_name] = ppred_all.mean().item()
            G_module[m_name]      = g_cat.mean().item()
            F_module[m_name]      = f_cat.mean().item()

        result: Dict[str, Any] = {
            "module_scores": module_scores,
            "param_scores":  param_scores,
            "G_module":      G_module,
            "F_module":      F_module,
            "num_batches":   count,
            "epsilon":       epsilon,
        }

        # ── 头级别扩展（后处理，无额外前向传播）────────────────────────────
        if head_granularity:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [def4_ppred] head_granularity=True 但模型无 config，"
                      "跳过头级别计算")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)
                head_scores: Dict[str, Dict[str, float]] = {}

                for m_name, m_type in attn_mods.items():
                    per_head: Dict[str, float] = {}
                    for h in range(attn_cfg.num_heads):
                        g_parts, f_parts = [], []
                        for suffix in ("weight", "bias"):
                            pn = f"{m_name}.{suffix}"
                            g_acc = G.get(pn)
                            f_acc = F.get(pn)
                            if g_acc is None:
                                continue
                            if suffix == "weight":
                                gview = get_head_weight_view(
                                    g_acc, m_type, h, attn_cfg.head_dim
                                )
                                fview = get_head_weight_view(
                                    f_acc, m_type, h, attn_cfg.head_dim
                                )
                            else:
                                gview = get_head_bias_view(
                                    g_acc, m_type, h, attn_cfg.head_dim
                                )
                                fview = get_head_bias_view(
                                    f_acc, m_type, h, attn_cfg.head_dim
                                )
                            if gview is not None:
                                g_parts.append(gview.reshape(-1))
                                f_parts.append(fview.reshape(-1))

                        if g_parts:
                            g_cat = torch.cat(g_parts)
                            f_cat = torch.cat(f_parts)
                            ppred_h = g_cat.pow(2) / (f_cat + epsilon)
                            per_head[f"head_{h}"] = ppred_h.mean().item()
                        else:
                            per_head[f"head_{h}"] = 0.0

                    head_scores[m_name] = per_head

                result["head_scores"] = head_scores
                print(f"  [def4_ppred] 计算了 {len(attn_mods)} 个注意力模块的"
                      f"头级别 Ppred（每模块 {attn_cfg.num_heads} 头）")
        # ─────────────────────────────────────────────────────────────────────

        return result
