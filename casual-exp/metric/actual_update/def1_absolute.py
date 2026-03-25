"""
metric/actual_update/def1_absolute.py

定义一：绝对更新量
─────────────────────────────────────────────────────────────────────────────
公式（两种变体同时计算，均输出）：

  变体 A — 全参数 L2 范数（含 weight 和 bias）：
      U_m^(A) = ||Δθ_m||_2 = sqrt( Σ_{p ∈ θ_m} ||Δp||_F² )

  变体 B — 仅权重矩阵 Frobenius 范数（排除 .bias 参数）：
      U_m^(B) = ||ΔW_m||_F = sqrt( Σ_{p ∈ W_m} ||Δp||_F² )

其中：
  Δp = p^(T) - p^(0)    参数从微调前到微调后的变化量
  θ_m = {weight, bias}  模块 m 的所有参数
  W_m = {weight}        模块 m 中不以 ".bias" 结尾的参数

注意：对于单个张量，L2 范数 ≡ Frobenius 范数（都是所有元素平方和的平方根），
区别仅在于模块聚合时是否包含 bias 参数。

嵌入训练（最小侵入）：
    metric = AbsoluteUpdateMetric()
    cb = metric.make_callback(model, save_dir)
    trainer = Trainer(..., callbacks=[..., cb])

独立测试：
    from metric.actual_update.base import snapshot_params
    theta0 = snapshot_params(model)          # 训练前快照
    ... (训练) ...
    result = AbsoluteUpdateMetric().compute(theta0, model, device)

─────────────────────────────────────────────────────────────────────────────
保存格式：
    def1_absolute.json
    {
      "module_scores":      {module_name: float, ...},  # 变体A：全参数L2（主指标）
      "weight_only_scores": {module_name: float, ...},  # 变体B：仅weight Frobenius
      "param_scores":       {param_name:  float, ...},  # 参数粒度L2范数
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

def compute_absolute(
    theta0: Dict[str, torch.Tensor],
    param_dict: Dict[str, torch.Tensor],
    module_groups: Dict[str, list],
) -> Dict[str, Any]:
    """
    绝对更新量核心计算：给定初始参数快照和当前参数字典，
    同时输出变体 A（全参数 L2）和变体 B（仅 weight Frobenius）。

    Args:
        theta0:        {param_name: tensor_cpu}，初始参数快照（CPU）
        param_dict:    {param_name: param_tensor}，当前模型参数（任意设备）
        module_groups: {module_name: [param_name, ...]}，叶模块参数分组

    Returns:
        {
          "module_scores":      {module_name: float},  # 变体A
          "weight_only_scores": {module_name: float},  # 变体B
          "param_scores":       {param_name:  float},  # 参数粒度
        }
    """
    # ── 参数级：||Δp||_F（即 L2 范数，对矩阵等价于 Frobenius 范数）──────────
    param_scores: Dict[str, float] = {}
    for name, t0 in theta0.items():
        p = param_dict.get(name)
        if p is None:
            continue
        delta = p.data.cpu() - t0            # Δp = θ^(T) - θ^(0)，在 CPU 上计算
        param_scores[name] = delta.norm(p=2).item()  # ||Δp||_F

    # ── 变体 A：全参数 L2 组合（sqrt(Σ ||Δp_i||² ) for all p_i in module）──
    module_scores_A: Dict[str, float] = {}
    for m_name, param_names in module_groups.items():
        sq_sum = sum(param_scores.get(pn, 0.0) ** 2 for pn in param_names)
        module_scores_A[m_name] = sq_sum ** 0.5

    # ── 变体 B：仅 weight Frobenius（排除以 ".bias" 结尾的参数）────────────
    #    对于无 bias 的模块（如 Embedding），变体 B = 变体 A。
    module_scores_B: Dict[str, float] = {}
    for m_name, param_names in module_groups.items():
        w_names = [pn for pn in param_names if not pn.endswith(".bias")]
        sq_sum = sum(param_scores.get(pn, 0.0) ** 2 for pn in w_names)
        module_scores_B[m_name] = sq_sum ** 0.5

    return {
        "module_scores":      module_scores_A,
        "weight_only_scores": module_scores_B,
        "param_scores":       param_scores,
    }


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class AbsoluteUpdateCallback(TrainerCallback):
    """
    在 Trainer 创建前立即记录 θ^(0) 快照，
    训练结束时（on_train_end）计算绝对更新量并保存 JSON。

    变体 A 和 B 同时计算，独立字段保存。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        name: str = "def1_absolute",
    ):
        """
        Args:
            model:    未经 DDP 包装的原始模型（在 Trainer 创建前传入）
            save_dir: 结果保存目录
            name:     JSON 文件名（无需修改，由 AbsoluteUpdateMetric.name 控制）
        """
        self._save_dir      = save_dir
        self._name          = name
        self._module_groups = group_params_by_module(model)

        # 立即记录 θ^(0) 快照（此时 model 尚未被 DDP 包装，参数名准确）
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

        param_dict = resolve_param_dict(model)   # 自动剥离 DDP 的 "module." 前缀
        result = compute_absolute(self._theta0, param_dict, self._module_groups)

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

class AbsoluteUpdateMetric(SnapshotMetric):
    """
    绝对更新量（定义一）—— SnapshotMetric 包装类。

    同时计算并输出：
      · module_scores      变体 A：全参数 L2（含 weight + bias）
      · weight_only_scores 变体 B：仅 weight Frobenius 范数
      · param_scores       参数粒度 L2 范数
    """

    name = "def1_absolute"

    def compute(
        self,
        theta0: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        独立测试接口：给定初始参数快照和当前模型，计算绝对更新量。

        Args:
            theta0: 微调前的参数快照，格式 {param_name: tensor_cpu}
                    可用 ``from metric.actual_update.base import snapshot_params``
                    在训练前调用 ``snapshot_params(model)`` 生成。
            model:  微调后的模型（处于 θ^(T) 状态）
            device: 计算设备（保留一致性接口，当前实现均在 CPU 上计算差值）
        """
        module_groups = group_params_by_module(model)
        param_dict    = resolve_param_dict(model)
        return compute_absolute(theta0, param_dict, module_groups)

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        **kwargs,
    ) -> AbsoluteUpdateCallback:
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:    未经 DDP 包装的原始模型
            save_dir: 结果保存目录
        """
        return AbsoluteUpdateCallback(
            model=model, save_dir=save_dir, name=self.name
        )
