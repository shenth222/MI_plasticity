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
  Δp = p^(T) - p^(0)
  W_m = 模块 m 中不以 ".bias" 结尾的参数

头级别（head_granularity=True）：
  对 Q/K/V 投影和输出投影，按头维度切分 Δθ：
    head_score_h = sqrt( Σ_{p∈{w,b}} ||view_h(Δp)||_F² )
  额外输出 "head_scores" 字段。

─────────────────────────────────────────────────────────────────────────────
保存格式：
    def1_absolute.json
    {
      "module_scores":      {module_name: float, ...},  # 变体A：全参数L2
      "weight_only_scores": {module_name: float, ...},  # 变体B：仅weight Frobenius
      "param_scores":       {param_name:  float, ...},  # 参数粒度L2
      "head_scores":        {module_name: {"head_0": float, ...}, ...}  # 仅 head_granularity=True
    }
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base import SnapshotMetric, group_params_by_module, snapshot_params, resolve_param_dict
from .attn_head import (
    get_attn_head_config,
    get_attn_modules,
    compute_head_delta_l2,
)


# ---------------------------------------------------------------------------
# 核心计算函数（纯函数，独立可测）
# ---------------------------------------------------------------------------

def compute_absolute(
    theta0: Dict[str, torch.Tensor],
    param_dict: Dict[str, torch.Tensor],
    module_groups: Dict[str, list],
    head_granularity: bool = False,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """
    绝对更新量核心计算：同时输出变体A（全参数L2）和变体B（仅weight Frobenius）。
    当 head_granularity=True 时额外计算注意力头级别分数。

    Args:
        theta0:           {param_name: tensor_cpu}，初始参数快照
        param_dict:       {param_name: param_tensor}，当前模型参数（任意设备）
        module_groups:    {module_name: [param_name, ...]}
        head_granularity: 是否计算注意力头级别分数（需要 model 有 config）
        model:            用于提取 attn_head_config（仅 head_granularity=True 时使用）

    Returns:
        {
          "module_scores":      {module_name: float},  # 变体A
          "weight_only_scores": {module_name: float},  # 变体B
          "param_scores":       {param_name:  float},  # 参数粒度
          "head_scores":        {module_name: {"head_h": float}},  # 仅 head_granularity=True
        }
    """
    # ── 参数级 delta 张量（head_granularity 时保留张量，否则只保留标量）───────
    param_scores: Dict[str, float] = {}
    delta_tensors: Dict[str, torch.Tensor] = {}   # 仅 head_granularity=True 时填充

    for name, t0 in theta0.items():
        p = param_dict.get(name)
        if p is None:
            continue
        delta = p.data.cpu() - t0            # Δp = θ^(T) - θ^(0)
        param_scores[name] = delta.norm(p=2).item()
        if head_granularity:
            delta_tensors[name] = delta

    # ── 变体 A：全参数 L2 ────────────────────────────────────────────────────
    module_scores_A: Dict[str, float] = {}
    for m_name, param_names in module_groups.items():
        sq_sum = sum(param_scores.get(pn, 0.0) ** 2 for pn in param_names)
        module_scores_A[m_name] = sq_sum ** 0.5

    # ── 变体 B：仅 weight Frobenius（排除 .bias 参数）────────────────────────
    module_scores_B: Dict[str, float] = {}
    for m_name, param_names in module_groups.items():
        w_names = [pn for pn in param_names if not pn.endswith(".bias")]
        sq_sum  = sum(param_scores.get(pn, 0.0) ** 2 for pn in w_names)
        module_scores_B[m_name] = sq_sum ** 0.5

    result: Dict[str, Any] = {
        "module_scores":      module_scores_A,
        "weight_only_scores": module_scores_B,
        "param_scores":       param_scores,
    }

    # ── 头级别扩展（后处理，无额外前向/反向传播）─────────────────────────────
    if head_granularity and delta_tensors:
        if model is None:
            print("  [def1_absolute] head_granularity=True 但未传入 model，跳过头级别计算")
        else:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [def1_absolute] head_granularity=True 但模型无 config，跳过头级别计算")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)
                head_scores = compute_head_delta_l2(delta_tensors, attn_cfg, attn_mods)
                result["head_scores"] = head_scores
                print(f"  [def1_absolute] 计算了 {len(attn_mods)} 个注意力模块的"
                      f"头级别绝对更新量（每模块 {attn_cfg.num_heads} 头）")

    return result


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class AbsoluteUpdateCallback(TrainerCallback):
    """
    在 Trainer 创建前立即记录 θ^(0) 快照，
    训练结束时（on_train_end）计算绝对更新量并保存 JSON。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        head_granularity: bool = False,
        name: str = "def1_absolute",
    ):
        self._save_dir       = save_dir
        self._name           = name
        self._head_granularity = head_granularity
        self._module_groups  = group_params_by_module(model)
        self._model_ref      = model   # 仅用于 head_granularity 时提取 config

        # 立即记录 θ^(0) 快照
        self._theta0 = snapshot_params(model)
        print(
            f"[{name}] θ^(0) 快照已记录（{len(self._theta0)} 个可训练参数"
            + ("，已启用头级别粒度" if head_granularity else "")
            + "）"
        )

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
        result = compute_absolute(
            self._theta0, param_dict, self._module_groups,
            head_granularity=self._head_granularity,
            model=self._model_ref,    # 传递原始 model 用于 head config 提取
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

class AbsoluteUpdateMetric(SnapshotMetric):
    """
    绝对更新量（定义一）—— SnapshotMetric 包装类。

    同时计算：
      · module_scores      变体 A：全参数 L2
      · weight_only_scores 变体 B：仅 weight Frobenius
      · param_scores       参数粒度 L2
      · head_scores        注意力头级别 L2（仅 head_granularity=True）
    """

    name = "def1_absolute"

    def compute(
        self,
        theta0: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        独立测试接口。

        Args:
            theta0:           微调前的参数快照 {param_name: tensor_cpu}
            model:            微调后的模型（θ^(T) 状态）
            device:           计算设备（保留接口一致性）
            head_granularity: 是否额外计算注意力头级别分数
        """
        module_groups = group_params_by_module(model)
        param_dict    = resolve_param_dict(model)
        return compute_absolute(
            theta0, param_dict, module_groups,
            head_granularity=head_granularity, model=model,
        )

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        head_granularity: bool = False,
        **kwargs,
    ) -> AbsoluteUpdateCallback:
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:            未经 DDP 包装的原始模型
            save_dir:         结果保存目录
            head_granularity: 是否额外计算注意力头级别分数
        """
        return AbsoluteUpdateCallback(
            model=model, save_dir=save_dir,
            head_granularity=head_granularity, name=self.name,
        )
