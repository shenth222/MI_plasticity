"""
metric/pre_importance/perturbation.py

定义 3：扰动敏感度（Perturbation Sensitivity）
─────────────────────────────────────────────────────────────────────────────
公式：
    I_m^pre = E[ L(θ + ε_m) - L(θ) ]

对每个叶模块 m，仅在其参数上添加各向同性高斯噪声 ε_m，
其他模块参数保持不变，重新前向传播，计算 loss 差值。
多次随机采样后取均值作为期望的 Monte Carlo 估计。

噪声尺度策略（relative_noise=True，默认）：
    std_m = noise_std × (mean |θ_m| + 1e-8)
使扰动相对于参数量级，避免小参数被过度扰动。

头级别扩展（head_granularity=True）：
    对注意力模块，逐头扰动其权重切片，计算每头的 E[ΔL]。
    开销：额外 O(num_attn_modules × num_heads × num_batches × num_samples) 次前向传播。
    典型 DeBERTa-v3-base（48 个注意力模块，12 头）配合 num_batches=2, num_samples=2：
    约 48 × 12 × 2 × 2 = 2304 次额外前向传播。

复杂度（模块级）：O(num_modules × num_batches × num_samples) 次前向传播。

保存格式：
    perturbation.json
    {
      "module_scores": {module_name: float, ...},  # E[ΔL]，值越大越重要
      "num_batches": int,
      "num_samples": int,
      "noise_std": float,
      "relative_noise": bool,
      "head_scores": {                             # 仅 head_granularity=True 时存在
        module_name: {"head_0": float, ..., "head_{n-1}": float}, ...
      }
    }
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase, group_params_by_module
from .attn_head import (
    get_attn_head_config,
    get_attn_modules,
    get_head_weight_view,
    get_head_bias_view,
)


class PerturbationImportance(ImportanceBase):
    """
    扰动敏感度重要性。

    值越大 → 对该模块加噪后 loss 平均升幅越大 → 该模块越脆弱/重要。
    与 Fisher 相比，可以捕获 loss 曲面的非线性效应。
    """

    name = "perturbation"
    needs_data = True

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 4,
        num_samples: int = 3,
        noise_std: float = 1e-3,
        relative_noise: bool = True,
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            num_batches:      用于估计期望的 mini-batch 数量
            num_samples:      每个 (模块/头, batch) 对的噪声采样次数
            noise_std:        高斯噪声标准差
                              - relative_noise=True：相对量（乘以参数绝对均值）
                              - relative_noise=False：绝对量
            relative_noise:   是否使用相对尺度噪声
            head_granularity: 若为 True，对注意力模块额外计算每头的扰动敏感度。
                              注意：会显著增加前向传播次数。
        """
        model = model.to(device)
        model.eval()

        # 预先收集 batches（避免多轮重复迭代）
        batches: List[Dict[str, torch.Tensor]] = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            batches.append({
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            })

        if not batches:
            raise RuntimeError("[PerturbationImportance] dataloader is empty")

        # 计算所有 batch 的基准 loss（无梯度）
        base_losses: List[float] = []
        with torch.no_grad():
            for inputs in batches:
                out = model(**inputs)
                base_losses.append(out.loss.item())

        # ── 模块级扰动（原有逻辑，完整保留）────────────────────────────────
        module_groups = group_params_by_module(model)
        total_modules = len(module_groups)
        module_delta: Dict[str, float] = {}

        for idx, (module_name, param_names) in enumerate(module_groups.items()):
            if idx % 20 == 0:
                print(f"  [perturbation] {idx}/{total_modules} modules ...")

            params_in_module: List[Tuple[str, torch.nn.Parameter]] = [
                (pn, p)
                for pn, p in model.named_parameters()
                if pn in param_names and p.requires_grad
            ]
            if not params_in_module:
                module_delta[module_name] = 0.0
                continue

            deltas: List[float] = []
            with torch.no_grad():
                for inputs, base_loss in zip(batches, base_losses):
                    for _ in range(num_samples):
                        applied: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
                        for _, param in params_in_module:
                            std = (noise_std * (param.data.abs().mean().item() + 1e-8)
                                   if relative_noise else noise_std)
                            noise = torch.randn_like(param.data) * std
                            param.data.add_(noise)
                            applied.append((param, noise))

                        out = model(**inputs)
                        deltas.append(out.loss.item() - base_loss)

                        for param, noise in applied:
                            param.data.sub_(noise)

            module_delta[module_name] = (
                sum(deltas) / len(deltas) if deltas else 0.0
            )
        # ─────────────────────────────────────────────────────────────────────

        result: Dict[str, Any] = {
            "module_scores": module_delta,
            "num_batches":   len(batches),
            "num_samples":   num_samples,
            "noise_std":     noise_std,
            "relative_noise": relative_noise,
        }

        # ── 头级别扩展（需要额外前向传播）───────────────────────────────────
        if head_granularity:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [perturbation] head_granularity=True 但模型无 config，跳过")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)
                num_heads = attn_cfg.num_heads
                head_dim  = attn_cfg.head_dim
                head_delta: Dict[str, Dict[str, float]] = {}

                named_mods = dict(model.named_modules())
                total_attn = len(attn_mods)

                for attn_idx, (module_name, module_type) in enumerate(attn_mods.items()):
                    print(f"  [perturbation-heads] {attn_idx}/{total_attn}: "
                          f"{module_name.split('.')[-3:]}")
                    module_obj = named_mods.get(module_name)
                    if module_obj is None:
                        continue

                    head_delta[module_name] = {}

                    with torch.no_grad():
                        for h in range(num_heads):
                            deltas_h: List[float] = []

                            for inputs, base_loss in zip(batches, base_losses):
                                for _ in range(num_samples):
                                    # 1) 为头 h 的权重切片加噪
                                    w = module_obj.weight
                                    w_view = get_head_weight_view(
                                        w.data, module_type, h, head_dim
                                    )
                                    std_w = (noise_std * (w_view.abs().mean().item() + 1e-8)
                                             if relative_noise else noise_std)
                                    noise_w = torch.randn_like(w_view) * std_w
                                    w_view.add_(noise_w)          # 原地修改（view 共享内存）

                                    # 2) 若是 QKV，为偏置切片加噪
                                    noise_b = None
                                    if module_obj.bias is not None:
                                        b_view = get_head_bias_view(
                                            module_obj.bias.data, module_type, h, head_dim
                                        )
                                        if b_view is not None:
                                            std_b = (noise_std * (b_view.abs().mean().item() + 1e-8)
                                                     if relative_noise else noise_std)
                                            noise_b = torch.randn_like(b_view) * std_b
                                            b_view.add_(noise_b)

                                    # 3) 扰动后 loss
                                    out = model(**inputs)
                                    deltas_h.append(out.loss.item() - base_loss)

                                    # 4) 原地恢复
                                    w_view.sub_(noise_w)
                                    if noise_b is not None:
                                        b_view.sub_(noise_b)

                            head_delta[module_name][f"head_{h}"] = (
                                sum(deltas_h) / len(deltas_h) if deltas_h else 0.0
                            )

                result["head_scores"] = head_delta
                print(f"  [perturbation] 头级别计算完毕：{len(attn_mods)} 个注意力模块 × "
                      f"{num_heads} 头")
        # ─────────────────────────────────────────────────────────────────────

        return result
