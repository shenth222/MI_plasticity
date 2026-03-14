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

复杂度：O(num_modules × num_batches × num_samples) 次前向传播。
  典型配置（num_batches=4, num_samples=3）对 DeBERTa-v3-base 约需 ~5 分钟。

保存格式：
    perturbation.json
    {
      "module_scores": {module_name: float, ...},  # E[ΔL]，值越大越重要
      "num_batches": int,
      "num_samples": int,
      "noise_std": float,
      "relative_noise": bool
    }
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .base import ImportanceBase, group_params_by_module


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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            num_batches:    用于估计期望的 mini-batch 数量
            num_samples:    每个 (模块, batch) 对的噪声采样次数
            noise_std:      高斯噪声标准差
                            - relative_noise=True：相对量（乘以参数绝对均值）
                            - relative_noise=False：绝对量
            relative_noise: 是否使用相对尺度噪声
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

        module_groups = group_params_by_module(model)
        total_modules = len(module_groups)
        module_delta: Dict[str, float] = {}

        for idx, (module_name, param_names) in enumerate(module_groups.items()):
            if idx % 20 == 0:
                print(f"  [perturbation] {idx}/{total_modules} modules ...")

            # 仅处理含可训练参数的模块
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
                        # 1) 生成并施加噪声
                        applied_noises: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
                        for _, param in params_in_module:
                            if relative_noise:
                                std = noise_std * (param.data.abs().mean().item() + 1e-8)
                            else:
                                std = noise_std
                            noise = torch.randn_like(param.data) * std
                            param.data.add_(noise)
                            applied_noises.append((param, noise))

                        # 2) 扰动后 loss
                        out = model(**inputs)
                        deltas.append(out.loss.item() - base_loss)

                        # 3) 原地恢复参数
                        for param, noise in applied_noises:
                            param.data.sub_(noise)

            module_delta[module_name] = (
                sum(deltas) / len(deltas) if deltas else 0.0
            )

        return {
            "module_scores": module_delta,
            "num_batches": len(batches),
            "num_samples": num_samples,
            "noise_std": noise_std,
            "relative_noise": relative_noise,
        }
