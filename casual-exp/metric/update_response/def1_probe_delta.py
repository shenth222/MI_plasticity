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

    与梯度类指标（定义 2、3）的区别：
        定义 1 测量实际参数位移（受 optimizer 动量/二阶矩状态影响）；
        梯度类指标只看梯度信号，不经过 optimizer 的非线性变换。

实现：手动 AdamW 优化循环（固定 LR，无 warmup）
    · warmup 仅控制 LR 调度（LR 从 0 缓升），不适合用于定义探针步数：
        warmup 期 LR 递增 → 早期位移被低估；且 Trainer 无法干净恢复参数。
    · 手动循环使用固定 LR，更能代表稳态下的真实更新方向。
    · 循环后通过 load data 方式恢复 θ^(0)（深拷贝快照 → 写回 param.data）。

模块级聚合：
    L2 范数组合：module_score = sqrt(Σ_param ||Δθ_param||²)
    等价于将模块内所有参数拼接成一个向量后取整体位移范数。

保存格式：
    def1_probe_delta.json
    {
      "module_scores": {module_name: float, ...},  # L2 位移（模块级）
      "param_scores":  {param_name:  float, ...},  # L2 位移（参数级）
      "probe_steps":   int,
      "probe_lr":      float,
      "weight_decay":  float,
    }
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import PreTrainingMetric, group_params_by_module


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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            probe_steps:  探针训练步数（建议 10–50；步数越多位移越稳定，开销线性增加）
            probe_lr:     探针训练学习率（建议与主训练 LR 一致）
            weight_decay: AdamW 权重衰减。默认 0.0：非零值会引入正则化位移，
                          使 \hat{R}_m 混入正则效应，通常保持 0 以纯净测量
                          梯度驱动的位移。
        """
        model = model.to(device)

        # 保存 θ^(0) 的纯数据快照（CPU，与 model 参数完全解耦）
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

        # 参数级 L2 位移
        param_scores: Dict[str, float] = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in theta0:
                delta = p.data.cpu() - theta0[n]
                param_scores[n] = delta.norm(p=2).item()

        # 模块级：L2 范数组合 sqrt(Σ ||Δθ_param||²)
        module_groups = group_params_by_module(model)
        module_scores: Dict[str, float] = {
            m_name: sum(param_scores.get(pn, 0.0) ** 2 for pn in params) ** 0.5
            for m_name, params in module_groups.items()
        }

        # 恢复 θ^(0)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in theta0:
                    p.data.copy_(theta0[n].to(device))

        print("  [def1_probe_delta] 参数已恢复至 θ^(0)")

        return {
            "module_scores": module_scores,
            "param_scores":  param_scores,
            "probe_steps":   probe_steps,
            "probe_lr":      probe_lr,
            "weight_decay":  weight_decay,
        }
