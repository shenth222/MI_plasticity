# metric/actual_update/attn_head.py
"""
注意力头级别（head-level）工具模块——转发 metric.pre_importance.attn_head 的通用实现，
并补充 actual_update 特有的 delta 张量切片函数。

─────────────────────────────────────────────────────────────────────────────
转发函数（详见 metric.pre_importance.attn_head）：
  AttnHeadConfig           — num_heads / head_dim / hidden_size 配置数据类
  get_attn_head_config     — 从 model.config 提取注意力头配置（无 config → None）
  classify_attn_module     — 判断模块是否为 QKV/"out" 投影
  get_attn_modules         — 扫描模型，返回 {module_name: "qkv"/"out"}
  get_head_weight_view     — 按头切片权重矩阵（共享内存视图）
  get_head_bias_view       — 按头切片偏置向量（OUT 投影偏置返回 None）

actual_update 新增函数：
  compute_head_delta_l2    — 从 Δ参数张量字典计算各注意力头的 L2 范数
                             适用于 def1（绝对更新量）和 def3（路径长度步进 delta）
  compute_head_init_l2     — 从参数快照字典计算各注意力头的初始 L2 范数
                             适用于 def2（相对更新量）的分母
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Any, Dict, Optional

import torch

from metric.pre_importance.attn_head import (
    AttnHeadConfig,
    get_attn_head_config,
    classify_attn_module,
    get_attn_modules,
    get_head_weight_view,
    get_head_bias_view,
)

__all__ = [
    # 转发
    "AttnHeadConfig",
    "get_attn_head_config",
    "classify_attn_module",
    "get_attn_modules",
    "get_head_weight_view",
    "get_head_bias_view",
    # actual_update 新增
    "compute_head_delta_l2",
    "compute_head_init_l2",
]


def compute_head_delta_l2(
    delta_tensors: Dict[str, torch.Tensor],
    attn_cfg: AttnHeadConfig,
    attn_modules: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    从 Δ 参数张量字典计算各注意力模块每个头的 L2 范数。

    公式（头 h，模块 m）：
        head_score_h = sqrt( Σ_{p∈{weight,bias}} ||view_h(Δp)||_F² )

    其中 view_h 是对应头的切片（见 get_head_weight_view / get_head_bias_view）。

    适用场景：
        · def1 的 head_scores（Δθ 的头级别 L2）
        · def3 的单步 delta 贡献（步进路径长度的头级别累积）

    Args:
        delta_tensors: {param_name: Δp_tensor_cpu}
                       可以是训练前后的全局差值（def1），
                       也可以是单步参数差值（def3）
        attn_cfg:      注意力头配置
        attn_modules:  {module_name: "qkv"/"out"}

    Returns:
        {module_name: {"head_0": float, "head_1": float, ...}}
    """
    num_heads = attn_cfg.num_heads
    head_dim  = attn_cfg.head_dim
    result: Dict[str, Dict[str, float]] = {}

    for m_name, m_type in attn_modules.items():
        per_head = [0.0] * num_heads
        for suffix, use_bias_fn in (("weight", False), ("bias", True)):
            pn    = f"{m_name}.{suffix}"
            delta = delta_tensors.get(pn)
            if delta is None:
                continue
            for h in range(num_heads):
                view = (
                    get_head_bias_view(delta, m_type, h, head_dim)
                    if use_bias_fn
                    else get_head_weight_view(delta, m_type, h, head_dim)
                )
                if view is not None:
                    per_head[h] += view.pow(2).sum().item()

        result[m_name] = {f"head_{h}": per_head[h] ** 0.5 for h in range(num_heads)}

    return result


def compute_head_init_l2(
    theta0: Dict[str, torch.Tensor],
    attn_cfg: AttnHeadConfig,
    attn_modules: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    从初始参数快照计算各注意力模块每个头的 L2 范数。

    公式：
        head_init_norm_h = sqrt( Σ_{p∈{weight,bias}} ||view_h(p^(0))||_F² )

    适用场景：
        · def2 head_scores 的分母 ||θ_h^(0)||₂

    Args:
        theta0:       {param_name: p^(0)_tensor_cpu}，初始参数快照
        attn_cfg:     注意力头配置
        attn_modules: {module_name: "qkv"/"out"}

    Returns:
        {module_name: {"head_0": float, "head_1": float, ...}}
    """
    num_heads = attn_cfg.num_heads
    head_dim  = attn_cfg.head_dim
    result: Dict[str, Dict[str, float]] = {}

    for m_name, m_type in attn_modules.items():
        per_head = [0.0] * num_heads
        for suffix, use_bias_fn in (("weight", False), ("bias", True)):
            pn   = f"{m_name}.{suffix}"
            t0   = theta0.get(pn)
            if t0 is None:
                continue
            for h in range(num_heads):
                view = (
                    get_head_bias_view(t0, m_type, h, head_dim)
                    if use_bias_fn
                    else get_head_weight_view(t0, m_type, h, head_dim)
                )
                if view is not None:
                    per_head[h] += view.pow(2).sum().item()

        result[m_name] = {f"head_{h}": per_head[h] ** 0.5 for h in range(num_heads)}

    return result
