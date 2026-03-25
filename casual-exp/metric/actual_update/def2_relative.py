"""
metric/actual_update/def2_relative.py

定义二：相对更新量
─────────────────────────────────────────────────────────────────────────────
公式：

    U_m^{rel} = ||Δθ_m||_2 / ( ||θ_m^{(0)}||_2 + ε )

其中：
    ||Δθ_m||_2  = sqrt( Σ_{p ∈ θ_m} ||Δp||_F² )
    ||θ_m^{(0)}||_2 = sqrt( Σ_{p ∈ θ_m} ||p^{(0)}||_F² )
    ε = 1e-8

头级别（head_granularity=True）：

    U_h^{rel} = head_delta_h / ( head_init_h + ε )

    head_delta_h = sqrt( Σ_{p∈{w,b}} ||view_h(Δp)||_F² )   — 变化量
    head_init_h  = sqrt( Σ_{p∈{w,b}} ||view_h(p^(0))||_F² ) — 初始量

─────────────────────────────────────────────────────────────────────────────
保存格式：
    def2_relative.json
    {
      "module_scores":     {module_name: float, ...},  # U_m^rel（主指标）
      "abs_module_scores": {module_name: float, ...},  # ||Δθ_m||_2（分子，同def1变体A）
      "init_norm_scores":  {module_name: float, ...},  # ||θ_m^(0)||_2（分母）
      "param_scores":      {param_name:  float, ...},  # 参数粒度相对更新量
      "epsilon":           float,
      "head_scores":       {module_name: {"head_0": float, ...}},  # 仅 head_granularity=True
      "head_abs_delta_scores": {module_name: {"head_0": float, ...}},  # 分子（头级别）
      "head_init_norm_scores": {module_name: {"head_0": float, ...}},  # 分母（头级别）
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
    compute_head_init_l2,
)


# ---------------------------------------------------------------------------
# 核心计算函数（纯函数，独立可测）
# ---------------------------------------------------------------------------

def compute_relative(
    theta0: Dict[str, torch.Tensor],
    param_dict: Dict[str, torch.Tensor],
    module_groups: Dict[str, list],
    epsilon: float = 1e-8,
    head_granularity: bool = False,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """
    相对更新量核心计算，可选地输出注意力头级别分数。

    Args:
        theta0:           初始参数快照 {param_name: tensor_cpu}
        param_dict:       当前模型参数 {param_name: param_tensor}
        module_groups:    {module_name: [param_name, ...]}
        epsilon:          数值稳定项（默认 1e-8）
        head_granularity: 是否计算注意力头级别分数
        model:            用于提取 attn_head_config（仅 head_granularity=True 时使用）

    Returns:
        包含 module_scores / abs_module_scores / init_norm_scores / param_scores
        以及可选的 head_scores / head_abs_delta_scores / head_init_norm_scores 的字典
    """
    # ── 参数级 ────────────────────────────────────────────────────────────────
    param_delta_norms: Dict[str, float] = {}
    param_init_norms:  Dict[str, float] = {}
    param_scores:      Dict[str, float] = {}
    delta_tensors:     Dict[str, torch.Tensor] = {}  # 仅 head_granularity=True 时填充

    for name, t0 in theta0.items():
        p = param_dict.get(name)
        if p is None:
            continue
        delta  = p.data.cpu() - t0
        d_norm = delta.norm(p=2).item()
        i_norm = t0.norm(p=2).item()
        param_delta_norms[name] = d_norm
        param_init_norms[name]  = i_norm
        param_scores[name]      = d_norm / (i_norm + epsilon)
        if head_granularity:
            delta_tensors[name] = delta

    # ── 模块级 ────────────────────────────────────────────────────────────────
    abs_module_scores: Dict[str, float] = {}
    init_norm_scores:  Dict[str, float] = {}
    module_scores:     Dict[str, float] = {}

    for m_name, param_names in module_groups.items():
        abs_sq   = sum(param_delta_norms.get(pn, 0.0) ** 2 for pn in param_names)
        init_sq  = sum(param_init_norms.get(pn,  0.0) ** 2 for pn in param_names)
        abs_norm  = abs_sq  ** 0.5
        init_norm = init_sq ** 0.5
        abs_module_scores[m_name] = abs_norm
        init_norm_scores[m_name]  = init_norm
        module_scores[m_name]     = abs_norm / (init_norm + epsilon)

    result: Dict[str, Any] = {
        "module_scores":     module_scores,
        "abs_module_scores": abs_module_scores,
        "init_norm_scores":  init_norm_scores,
        "param_scores":      param_scores,
        "epsilon":           epsilon,
    }

    # ── 头级别扩展 ────────────────────────────────────────────────────────────
    if head_granularity and delta_tensors:
        if model is None:
            print("  [def2_relative] head_granularity=True 但未传入 model，跳过头级别计算")
        else:
            attn_cfg = get_attn_head_config(model)
            if attn_cfg is None:
                print("  [def2_relative] head_granularity=True 但模型无 config，跳过头级别计算")
            else:
                attn_mods = get_attn_modules(model, attn_cfg)

                # 分子：各头的 ||Δθ_h||₂
                head_abs = compute_head_delta_l2(delta_tensors, attn_cfg, attn_mods)
                # 分母：各头的 ||θ_h^(0)||₂
                head_init = compute_head_init_l2(theta0, attn_cfg, attn_mods)
                # 相对更新量：分子 / (分母 + ε)
                head_scores: Dict[str, Dict[str, float]] = {}
                for m_name in attn_mods:
                    abs_h  = head_abs.get(m_name, {})
                    init_h = head_init.get(m_name, {})
                    head_scores[m_name] = {
                        hk: abs_h.get(hk, 0.0) / (init_h.get(hk, 0.0) + epsilon)
                        for hk in abs_h
                    }

                result["head_scores"]            = head_scores
                result["head_abs_delta_scores"]  = head_abs
                result["head_init_norm_scores"]  = head_init
                print(f"  [def2_relative] 计算了 {len(attn_mods)} 个注意力模块的"
                      f"头级别相对更新量（每模块 {attn_cfg.num_heads} 头）")

    return result


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
        head_granularity: bool = False,
        name: str = "def2_relative",
    ):
        self._save_dir        = save_dir
        self._epsilon         = epsilon
        self._name            = name
        self._head_granularity = head_granularity
        self._module_groups   = group_params_by_module(model)
        self._model_ref       = model

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
        result = compute_relative(
            self._theta0, param_dict, self._module_groups,
            epsilon=self._epsilon,
            head_granularity=self._head_granularity,
            model=self._model_ref,
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
      · module_scores      U_m^rel（主指标）
      · abs_module_scores  ||Δθ_m||₂（分子，与 def1 变体A 一致）
      · init_norm_scores   ||θ_m^(0)||₂（分母）
      · param_scores       参数粒度相对更新量
      · head_scores        头级别相对更新量（仅 head_granularity=True）
      · head_abs_delta_scores  头级别 ||Δθ_h||₂（分子，仅 head_granularity=True）
      · head_init_norm_scores  头级别 ||θ_h^(0)||₂（分母，仅 head_granularity=True）
    """

    name = "def2_relative"

    def compute(
        self,
        theta0: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
        epsilon: float = 1e-8,
        head_granularity: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        module_groups = group_params_by_module(model)
        param_dict    = resolve_param_dict(model)
        return compute_relative(
            theta0, param_dict, module_groups,
            epsilon=epsilon, head_granularity=head_granularity, model=model,
        )

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        epsilon: float = 1e-8,
        head_granularity: bool = False,
        **kwargs,
    ) -> RelativeUpdateCallback:
        return RelativeUpdateCallback(
            model=model, save_dir=save_dir,
            epsilon=epsilon, head_granularity=head_granularity, name=self.name,
        )
