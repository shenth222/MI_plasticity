"""
metric/update_response/def4_ppred.py

定义 4：梯度信噪比（Ppred）
─────────────────────────────────────────────────────────────────────────────
公式（元素级）：
    G_i = E[|g_i|]           （梯度绝对值的期望）
    F_i = E[g_i²]            （梯度平方的期望 ≈ 对角 Fisher）
    ppred_i = G_i² / (F_i + ε)

模块级聚合（将模块内所有参数拼接为一个长向量后计算）：
    Ppred_m = mean_{i ∈ θ_m}(ppred_i)

含义：
    ppred_i ∈ [0, 1]（由 Cauchy-Schwarz 不等式：E[|g|]² ≤ E[g²]）：
        → 1：各 batch 梯度符号高度一致（信号强、噪声低），参数会稳定积累净更新
        → 0：梯度各向随机抵消（噪声主导），即使幅度大也难以产生净位移
    Ppred_m 是"梯度一致性"（gradient coherence）的度量，
    可解释为梯度信号的 SNR（Signal-to-Noise Ratio）：
        SNR_i = E[g_i]² / Var[g_i] ≈ E[|g_i|]² / (E[g_i²] - E[g_i]²)
    而 ppred_i ≈ SNR_i / (1 + SNR_i)（单调等价变换）。

与 Fisher 重要性的区别：
    Fisher_m = E[‖g_m‖²] = Σ_i E[g_i²]  →  衡量损失的平均敏感度
    Ppred_m  = mean_i(E[|g_i|]²/E[g_i²])  →  衡量梯度信号一致性（SNR）
    两者反映不同维度：Fisher 大且 Ppred 低 → 模块对 loss 敏感但梯度嘈杂；
                       Fisher 中但 Ppred 高 → 梯度稳定，参数会持续朝一方向更新。

与定义 2 的区别：
    定义 2：E[‖g_m‖] / √(E[‖g_m‖²] + ε) —— 模块级向量范数的归一化，衡量步长
    定义 4：mean_i(E[|g_i|]²/E[g_i²]) —— 元素级 SNR 的均值，衡量方向一致性
    定义 2 有物理量级（预测更新幅度），定义 4 是无单位的 [0,1] 指数。

来源：
    minimal-exp/src/measure/grad_fisher_gate.py 中对注意力头门控梯度的实现：
        G[l,h] = mean_batch(|gate.grad[l,h]|)
        F[l,h] = mean_batch(gate.grad[l,h]²)
        Ppred[l,h] = G[l,h]² / (F[l,h] + ε)
    此处推广到任意叶模块（参数级元素的统计平均），不需要门控结构。

保存格式：
    def4_ppred.json
    {
      "module_scores":  {module_name: float, ...},  # Ppred_m ∈ [0, 1]
      "param_scores":   {param_name:  float, ...},  # 各参数的元素级 Ppred 均值
      "G_module":       {module_name: float, ...},  # mean_i E[|g_i|]（分子相关）
      "F_module":       {module_name: float, ...},  # mean_i E[g_i²]（对角 Fisher）
      "num_batches": int,
      "epsilon": float,
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import PreTrainingMetric, group_params_by_module


class PpredMetric(PreTrainingMetric):
    """
    梯度信噪比 Ppred（定义 4）。

    元素级 SNR = E[|g|]² / E[g²]，聚合为模块内元素的均值。
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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            num_batches: Monte Carlo 估计 batch 数（建议 16–64）
            epsilon:     数值稳定项（防止零 Fisher 导致除零）
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

        # 元素级期望：G_i = E[|g_i|]，F_i = E[g_i²]
        G: Dict[str, torch.Tensor] = {n: acc / count for n, acc in abs_acc.items()}
        F: Dict[str, torch.Tensor] = {n: acc / count for n, acc in sq_acc.items()}

        # 参数级 Ppred：各元素 ppred_i 的均值
        param_scores: Dict[str, float] = {}
        for name in G:
            ppred_elem = G[name].pow(2) / (F[name] + epsilon)  # [0, 1]，逐元素
            param_scores[name] = ppred_elem.mean().item()

        # 模块级：拼接模块内所有参数的元素后统一计算，确保元素均等权重
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

        return {
            "module_scores": module_scores,
            "param_scores":  param_scores,
            "G_module":      G_module,
            "F_module":      F_module,
            "num_batches":   count,
            "epsilon":       epsilon,
        }
