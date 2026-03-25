"""
metric/actual_update/def2_relative.py

定义二：相对更新量
─────────────────────────────────────────────────────────────────────────────
公式：

    U_m^{rel} = ||Δθ_m||_2 / ( ||θ_m^{(0)}||_2 + ε )

其中：
    ||Δθ_m||_2  = sqrt( Σ_{p ∈ θ_m} ||Δp||_F² )  — 模块全参数 L2 变化量（同定义一变体A）
    ||θ_m^{(0)}||_2 = sqrt( Σ_{p ∈ θ_m} ||p^{(0)}||_F² )  — 初始参数 L2 范数
    ε = 1e-8                                        — 数值稳定项，防止除以零

含义：
    将绝对更新量归一化到初始参数的量级上，消除不同模块参数规模差异的影响。
    值接近或超过 1 表示参数发生了与其初始量级相当的剧烈变化（高可塑性）；
    值远小于 1 表示该模块几乎未被微调触及（低可塑性/冻结效应）。

嵌入训练（最小侵入）：
    metric = RelativeUpdateMetric()
    cb = metric.make_callback(model, save_dir)
    trainer = Trainer(..., callbacks=[..., cb])

独立测试：
    from metric.actual_update.base import snapshot_params
    theta0 = snapshot_params(model)
    ... (训练) ...
    result = RelativeUpdateMetric().compute(theta0, model, device)

─────────────────────────────────────────────────────────────────────────────
保存格式：
    def2_relative.json
    {
      "module_scores":     {module_name: float, ...},  # U_m^rel（主指标）
      "abs_module_scores": {module_name: float, ...},  # ||Δθ_m||_2（分子，同def1变体A）
      "init_norm_scores":  {module_name: float, ...},  # ||θ_m^(0)||_2（分母）
      "param_scores":      {param_name:  float, ...},  # 参数粒度相对更新量
      "epsilon":           float,
    }
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base import SnapshotMetric, group_params_by_module, snapshot_params, resolve_param_dict


# ---------------------------------------------------------------------------
# 核心计算函数（纯函数，独立可测）
# ---------------------------------------------------------------------------

def compute_relative(
    theta0: Dict[str, torch.Tensor],
    param_dict: Dict[str, torch.Tensor],
    module_groups: Dict[str, list],
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    """
    相对更新量核心计算。

    Args:
        theta0:        {param_name: tensor_cpu}，初始参数快照
        param_dict:    {param_name: param_tensor}，当前模型参数（任意设备）
        module_groups: {module_name: [param_name, ...]}
        epsilon:       数值稳定项（默认 1e-8）

    Returns:
        {
          "module_scores":     {module_name: float},  # U_m^rel
          "abs_module_scores": {module_name: float},  # ||Δθ_m||_2（分子）
          "init_norm_scores":  {module_name: float},  # ||θ_m^(0)||_2（分母）
          "param_scores":      {param_name:  float},  # 参数粒度相对更新量
          "epsilon":           float,
        }
    """
    # ── 参数级：||Δp||_F 与 ||p^(0)||_F ─────────────────────────────────────
    param_delta_norms: Dict[str, float] = {}  # ||Δp||_F
    param_init_norms:  Dict[str, float] = {}  # ||p^(0)||_F
    param_scores:      Dict[str, float] = {}  # ||Δp||_F / (||p^(0)||_F + ε)

    for name, t0 in theta0.items():
        p = param_dict.get(name)
        if p is None:
            continue
        delta  = p.data.cpu() - t0
        d_norm = delta.norm(p=2).item()   # ||Δp||_F
        i_norm = t0.norm(p=2).item()      # ||p^(0)||_F
        param_delta_norms[name] = d_norm
        param_init_norms[name]  = i_norm
        param_scores[name]      = d_norm / (i_norm + epsilon)

    # ── 模块级：全参数 L2 正交组合（sqrt of sum of squares）──────────────────
    abs_module_scores: Dict[str, float] = {}   # ||Δθ_m||_2  分子
    init_norm_scores:  Dict[str, float] = {}   # ||θ_m^(0)||_2  分母
    module_scores:     Dict[str, float] = {}   # U_m^rel = 分子 / (分母 + ε)

    for m_name, param_names in module_groups.items():
        abs_sq  = sum(param_delta_norms.get(pn, 0.0) ** 2 for pn in param_names)
        init_sq = sum(param_init_norms.get(pn,  0.0) ** 2 for pn in param_names)
        abs_norm  = abs_sq  ** 0.5
        init_norm = init_sq ** 0.5
        abs_module_scores[m_name]  = abs_norm
        init_norm_scores[m_name]   = init_norm
        module_scores[m_name]      = abs_norm / (init_norm + epsilon)

    return {
        "module_scores":     module_scores,
        "abs_module_scores": abs_module_scores,
        "init_norm_scores":  init_norm_scores,
        "param_scores":      param_scores,
        "epsilon":           epsilon,
    }


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class RelativeUpdateCallback(TrainerCallback):
    """
    在 Trainer 创建前立即记录 θ^(0) 快照，
    训练结束时（on_train_end）计算相对更新量并保存 JSON。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        epsilon: float = 1e-8,
        name: str = "def2_relative",
    ):
        """
        Args:
            model:    未经 DDP 包装的原始模型（在 Trainer 创建前传入）
            save_dir: 结果保存目录
            epsilon:  数值稳定项（默认 1e-8）
            name:     JSON 文件名（无需修改）
        """
        self._save_dir      = save_dir
        self._epsilon       = epsilon
        self._name          = name
        self._module_groups = group_params_by_module(model)

        # 立即记录 θ^(0) 快照
        self._theta0 = snapshot_params(model)
        print(f"[{name}] θ^(0) 快照已记录（{len(self._theta0)} 个可训练参数）")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero or model is None:
            return control

        param_dict = resolve_param_dict(model)
        result = compute_relative(
            self._theta0, param_dict, self._module_groups, self._epsilon
        )

        save_dir = Path(self._save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{self._name}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{self._name}] Saved → {path}")

        return control


# ---------------------------------------------------------------------------
# SnapshotMetric 包装类
# ---------------------------------------------------------------------------

class RelativeUpdateMetric(SnapshotMetric):
    """
    相对更新量（定义二）—— SnapshotMetric 包装类。

    输出字段：
      · module_scores      U_m^rel = ||Δθ_m||_2 / (||θ_m^(0)||_2 + ε)（主指标）
      · abs_module_scores  分子 ||Δθ_m||_2（与定义一变体A完全一致，可交叉验证）
      · init_norm_scores   分母 ||θ_m^(0)||_2（初始参数范数）
      · param_scores       参数粒度相对更新量
    """

    name = "def2_relative"

    def compute(
        self,
        theta0: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
        epsilon: float = 1e-8,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        独立测试接口：给定初始参数快照和当前模型，计算相对更新量。

        Args:
            theta0:  微调前的参数快照，{param_name: tensor_cpu}
            model:   微调后的模型（处于 θ^(T) 状态）
            device:  计算设备（保留接口一致性）
            epsilon: 数值稳定项（默认 1e-8）
        """
        module_groups = group_params_by_module(model)
        param_dict    = resolve_param_dict(model)
        return compute_relative(theta0, param_dict, module_groups, epsilon)

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        epsilon: float = 1e-8,
        **kwargs,
    ) -> RelativeUpdateCallback:
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:    未经 DDP 包装的原始模型
            save_dir: 结果保存目录
            epsilon:  数值稳定项（默认 1e-8）
        """
        return RelativeUpdateCallback(
            model=model, save_dir=save_dir, epsilon=epsilon, name=self.name
        )
