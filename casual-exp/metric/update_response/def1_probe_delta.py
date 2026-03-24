"""
metric/update_response/def1_probe_delta.py

定义 1：短程试跑更新量
─────────────────────────────────────────────────────────────────────────────
公式：
    \hat{R}_m = \|\theta_m^{(t_0)} - \theta_m^{(0)}\|_2

含义：
    在目标任务数据上真实执行 probe_steps 步梯度下降后，
    测量各叶模块参数向量的 L2 位移。值越大，表示该模块在微调早期
    会发生越大的实际参数变化，即更新响应越强。

实现：手动 AdamW 优化循环（固定 LR，无 warmup）
    · warmup 仅控制 LR 调度（LR 从 0 缓升），不适合定义探针步数。
    · 循环后通过 load data 方式恢复 θ^(0)（深拷贝快照写回 param.data）。

模块级聚合：L2 范数组合 sqrt(Σ ||Δθ_param||²)
头级别聚合（head_granularity=True）：
    对 Q/K/V 投影及输出投影，按头维度切分 Δθ，
    head_score_h = sqrt(Σ_{i∈head_h} Δθ_i²)（L2 位移）

保存格式：
    def1_probe_delta.json
    {
      "module_scores": {module_name: float, ...},
      "param_scores":  {param_name:  float, ...},
      "probe_steps":   int,
      "probe_lr":      float,
      "weight_decay":  float,
      "head_scores":   {module_name: {"head_0": float, ...}, ...}  # 仅 head_granularity=True
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


class ProbeDeltaMetric(PreTrainingMetric):
    """
    短程试跑更新量（定义 1）。

    使用固定 LR 的手动 AdamW 循环运行 probe_steps 步，
    测量各模块参数 L2 位移，随后自动恢复 θ^(0)。
    """

    name = "def1_probe_delta"
    needs_data = True

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        probe_steps: int = 20,
        probe_lr: float = 2e-5,
        weight_decay: float = 0.0,
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            probe_steps:      探针训练步数（建议 10–50）
            probe_lr:         探针训练学习率（建议与主训练 LR 一致）
            weight_decay:     AdamW 权重衰减（默认 0.0）
            head_granularity: 若为 True，额外输出注意力头级别的位移分数
        """
        model = model.to(device)

        # 保存 θ^(0) 的纯数据快照
        theta0: Dict[str, torch.Tensor] = {
            n: p.data.clone().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # 手动 AdamW：固定 LR，无 warmup
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=probe_lr,
            weight_decay=weight_decay,
        )

        model.train()
        data_iter = iter(dataloader)

        for step in range(1, probe_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            optimizer.zero_grad()
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()

            if step % 10 == 0 or step == probe_steps:
                print(f"  [def1_probe_delta] probe step {step}/{probe_steps},"
                      f" loss={loss.item():.4f}")

        # 计算参数级 L2 位移（同时保留 delta 张量供头级别使用）
        delta_tensors: Dict[str, torch.Tensor] = {}
        param_scores:  Dict[str, float] = {}

        for n, p in model.named_parameters():
            if p.requires_grad and n in theta0:
                delta = p.data.cpu() - theta0[n]
                param_scores[n] = delta.norm(p=2).item()
                if head_granularity:
                    delta_tensors[n] = delta

        # 模块级：L2 范数组合 sqrt(Σ ||Δθ_param||²)
        module_groups = group_params_by_module(model)
        module_scores: Dict[str, float] = {
            m_name: sum(param_scores.get(pn, 0.0) ** 2 for pn in params) ** 0.5
            for m_name, params in module_groups.items()
        }

        result: Dict[str, Any] = {
            "module_scores": module_scores,
            "param_scores":  param_scores,
            "probe_steps":   probe_steps,
            "probe_lr":      probe_lr,
            "weight_decay":  weight_decay,
        }

        # ── 头级别扩展（后处理，无额外前向传播）────────────────────────────
        if head_granularity:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [def1_probe_delta] head_granularity=True 但模型无 config，"
                      "跳过头级别计算")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)
                head_scores: Dict[str, Dict[str, float]] = {}

                for m_name, m_type in attn_mods.items():
                    per_head: Dict[str, float] = {}
                    for h in range(attn_cfg.num_heads):
                        sq_sum = 0.0
                        for suffix in ("weight", "bias"):
                            pn = f"{m_name}.{suffix}"
                            delta = delta_tensors.get(pn)
                            if delta is None:
                                continue
                            if suffix == "weight":
                                view = get_head_weight_view(
                                    delta, m_type, h, attn_cfg.head_dim
                                )
                            else:
                                view = get_head_bias_view(
                                    delta, m_type, h, attn_cfg.head_dim
                                )
                            if view is not None:
                                sq_sum += view.pow(2).sum().item()
                        per_head[f"head_{h}"] = sq_sum ** 0.5
                    head_scores[m_name] = per_head

                result["head_scores"] = head_scores
                print(f"  [def1_probe_delta] 计算了 {len(attn_mods)} 个注意力模块的"
                      f"头级别位移（每模块 {attn_cfg.num_heads} 头）")
        # ─────────────────────────────────────────────────────────────────────

        # 恢复 θ^(0)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in theta0:
                    p.data.copy_(theta0[n].to(device))

        print("  [def1_probe_delta] 参数已恢复至 θ^(0)")
        return result
