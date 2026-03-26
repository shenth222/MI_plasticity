"""
metric/training_gain/attn_head.py

注意力头级别（head-level）工具模块——转发 metric.actual_update.attn_head 的通用实现，
并补充 training_gain 特有的：
  · rollback_head_context   — 回滚单个注意力头参数的上下文管理器（def1/def2）
  · rollback_module_context — 回滚叶模块全部参数的上下文管理器（def1/def2）
  · compute_head_pi_dot     — 逐步累积路径积分的头级别内积（def3）

─────────────────────────────────────────────────────────────────────────────
转发函数（详见 metric.pre_importance.attn_head）：
  AttnHeadConfig           — num_heads / head_dim / hidden_size 配置数据类
  get_attn_head_config     — 从 model.config 提取注意力头配置（无 config → None）
  classify_attn_module     — 判断模块是否为 QKV/"out" 投影
  get_attn_modules         — 扫描模型，返回 {module_name: "qkv"/"out"}
  get_head_weight_view     — 按头切片权重矩阵（共享内存视图）
  get_head_bias_view       — 按头切片偏置向量（OUT 投影偏置返回 None）

training_gain 新增函数：
  rollback_head_context    — 临时回滚注意力头 h 的权重/偏置切片到 θ^(0)，
                             with 块退出时自动恢复（适用于 def1/def2 head 级别评估）
  rollback_module_context  — 临时回滚叶模块全部参数到 θ^(0)，
                             with 块退出时自动恢复（适用于 def1/def2 module 级别评估）
  compute_head_pi_dot      — 从 Δ 参数张量和梯度张量计算各注意力模块每个头的内积
                             适用于 def3 路径积分的头级别逐步累积
─────────────────────────────────────────────────────────────────────────────
"""

from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

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
    # training_gain 新增
    "rollback_head_context",
    "rollback_module_context",
    "compute_head_pi_dot",
]


# ---------------------------------------------------------------------------
# 上下文管理器：参数回滚（def1 / def2 使用）
# ---------------------------------------------------------------------------

@contextmanager
def rollback_module_context(
    model: torch.nn.Module,
    theta0: Dict[str, torch.Tensor],
    param_names: List[str],
) -> Generator[None, None, None]:
    """
    临时将叶模块的全部参数回滚到 θ^(0)，with 块退出时自动恢复。

    用于 def1/def2 的模块级别回滚评估：
        with rollback_module_context(model, theta0, param_names):
            result = eval_fn(model, device)
        # 退出后参数自动恢复到 θ^(T)

    Args:
        model:       当前模型（θ^(T) 状态）
        theta0:      初始参数快照 {param_name: tensor_cpu}
        param_names: 需要回滚的参数名列表（来自 group_params_by_module）

    注意：
        · param_names 应使用未包含 DDP 前缀的名称（通过 resolve_param_dict 获取）
        · theta0 中的张量会自动转换到目标参数的设备和数据类型
        · 若某参数在 theta0 中不存在，跳过该参数（不修改，不报错）
    """
    # 收集需要修改的参数张量并保存当前值
    param_dict: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        key = n[len("module."):] if n.startswith("module.") else n
        param_dict[key] = p

    saved: Dict[str, torch.Tensor] = {}
    for pn in param_names:
        p = param_dict.get(pn)
        t0 = theta0.get(pn)
        if p is None or t0 is None:
            continue
        saved[pn] = p.data.clone()
        p.data.copy_(t0.to(device=p.device, dtype=p.dtype))

    try:
        yield
    finally:
        # 恢复参数到 θ^(T)
        for pn, saved_data in saved.items():
            p = param_dict.get(pn)
            if p is not None:
                p.data.copy_(saved_data)


@contextmanager
def rollback_head_context(
    model: torch.nn.Module,
    theta0: Dict[str, torch.Tensor],
    m_name: str,
    m_type: str,
    head_idx: int,
    head_dim: int,
) -> Generator[None, None, None]:
    """
    临时将注意力模块 m 中头 head_idx 的参数切片回滚到 θ^(0)，with 块退出时自动恢复。

    切片策略（与 attn_head.py 一致）：
      qkv 投影：
        weight 切片: W[h*d : (h+1)*d, :]
        bias   切片: b[h*d : (h+1)*d]
      out 投影：
        weight 切片: W[:, h*d : (h+1)*d]
        bias  : 属于整个模块，不做头级别拆分（忽略）

    Args:
        model:    当前模型（θ^(T) 状态）
        theta0:   初始参数快照 {param_name: tensor_cpu}
        m_name:   注意力模块名称（如 "deberta.encoder.layer.0.attention.self.query_proj"）
        m_type:   "qkv" 或 "out"
        head_idx: 头编号（0-indexed）
        head_dim: 每头维度
    """
    # 找到模块
    named_mods: Dict[str, torch.nn.Module] = {}
    for n, m in model.named_modules():
        key = n[len("module."):] if n.startswith("module.") else n
        named_mods[key] = m

    module = named_mods.get(m_name)
    if module is None:
        yield
        return

    saved_slices: Dict[str, torch.Tensor] = {}

    # ── 回滚权重切片 ────────────────────────────────────────────────────────
    w = getattr(module, "weight", None)
    t0_w = theta0.get(f"{m_name}.weight")
    if w is not None and t0_w is not None:
        t0_w_dev = t0_w.to(device=w.device, dtype=w.dtype)
        w_view    = get_head_weight_view(w.data,      m_type, head_idx, head_dim)
        t0_w_view = get_head_weight_view(t0_w_dev,    m_type, head_idx, head_dim)
        saved_slices["weight"] = w_view.clone()
        w_view.copy_(t0_w_view)

    # ── 回滚偏置切片（仅 qkv 类型） ─────────────────────────────────────────
    b = getattr(module, "bias", None)
    t0_b = theta0.get(f"{m_name}.bias")
    if m_type == "qkv" and b is not None and t0_b is not None:
        t0_b_dev  = t0_b.to(device=b.device, dtype=b.dtype)
        b_view    = get_head_bias_view(b.data,     m_type, head_idx, head_dim)
        t0_b_view = get_head_bias_view(t0_b_dev,   m_type, head_idx, head_dim)
        if b_view is not None and t0_b_view is not None:
            saved_slices["bias"] = b_view.clone()
            b_view.copy_(t0_b_view)

    try:
        yield
    finally:
        # ── 恢复权重切片 ──────────────────────────────────────────────────────
        if "weight" in saved_slices:
            w = getattr(module, "weight", None)
            if w is not None:
                w_view = get_head_weight_view(w.data, m_type, head_idx, head_dim)
                w_view.copy_(saved_slices["weight"])

        # ── 恢复偏置切片 ──────────────────────────────────────────────────────
        if "bias" in saved_slices:
            b = getattr(module, "bias", None)
            if b is not None:
                b_view = get_head_bias_view(b.data, m_type, head_idx, head_dim)
                if b_view is not None:
                    b_view.copy_(saved_slices["bias"])


# ---------------------------------------------------------------------------
# 路径积分头级别内积（def3 使用）
# ---------------------------------------------------------------------------

def compute_head_pi_dot(
    delta_tensors: Dict[str, torch.Tensor],
    grad_tensors:  Dict[str, torch.Tensor],
    attn_cfg:      AttnHeadConfig,
    attn_modules:  Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    从步进 Δ 参数张量和梯度张量计算各注意力模块每个头的路径积分贡献。

    公式（头 h，模块 m，第 t 步）：
        G_h^(PI)_t = Σ_{p ∈ {weight, bias}} Σ_{i ∈ view_h(p)} g_{i,t} · Δθ_{i,t}

    适用场景：
        · def3 在 on_step_end 中累积各步的头级别贡献

    Args:
        delta_tensors: {param_name: Δθ_t_cpu}（步进参数变化，float32 CPU）
        grad_tensors:  {param_name: g_t_cpu}（步进梯度，float32 CPU）
        attn_cfg:      注意力头配置
        attn_modules:  {module_name: "qkv"/"out"}

    Returns:
        {module_name: {"head_0": float, "head_1": float, ...}}
        值为有符号内积；通常 ≤ 0（梯度下降时损失减小）
    """
    num_heads = attn_cfg.num_heads
    head_dim  = attn_cfg.head_dim
    result: Dict[str, Dict[str, float]] = {}

    for m_name, m_type in attn_modules.items():
        per_head = [0.0] * num_heads

        for suffix, use_bias_fn in (("weight", False), ("bias", True)):
            pn    = f"{m_name}.{suffix}"
            delta = delta_tensors.get(pn)
            grad  = grad_tensors.get(pn)
            if delta is None or grad is None:
                continue

            for h in range(num_heads):
                if use_bias_fn:
                    dv = get_head_bias_view(delta, m_type, h, head_dim)
                    gv = get_head_bias_view(grad,  m_type, h, head_dim)
                else:
                    dv = get_head_weight_view(delta, m_type, h, head_dim)
                    gv = get_head_weight_view(grad,  m_type, h, head_dim)

                if dv is not None and gv is not None:
                    per_head[h] += (gv.float() * dv.float()).sum().item()

        result[m_name] = {f"head_{h}": per_head[h] for h in range(num_heads)}

    return result
