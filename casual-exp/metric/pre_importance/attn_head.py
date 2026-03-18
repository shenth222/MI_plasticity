"""
metric/pre_importance/attn_head.py

注意力头级别（head-level）粒度的工具模块。
─────────────────────────────────────────────────────────────────────────────
背景：
  标准多头注意力中，Q/K/V 投影权重按头拼接存储：
    query_proj.weight ∈ [num_heads × head_dim, hidden_size]   (PyTorch Linear: [out, in])
    key_proj.weight   ∈ [num_heads × head_dim, hidden_size]
    value_proj.weight ∈ [num_heads × head_dim, hidden_size]
    output.dense.weight ∈ [hidden_size, num_heads × head_dim]  ← 输出投影

  本模块将上述参数按头维度切分，使各指标能以"头"为单位计算重要性。

切片策略：
  QKV 投影（"qkv" 类型）：
    头 h 的权重: weight[h×d : (h+1)×d, :]   (d = head_dim)
    头 h 的偏置: bias  [h×d : (h+1)×d]
  输出投影（"out" 类型）：
    头 h 的权重: weight[:, h×d : (h+1)×d]
    偏置: 属于整个模块（不做头级别拆分，忽略）

接口：
  get_attn_head_config(model)           → AttnHeadConfig | None
  classify_attn_module(module_name)     → "qkv" | "out" | None
  get_attn_modules(model, cfg)          → {module_name: "qkv"/"out"}
  get_head_weight_view(W, mtype, h, d)  → 头 h 对应的权重视图（共享内存）
  get_head_bias_view(b, mtype, h, d)    → 头 h 对应的偏置视图，或 None
  agg_head_scores_from_acc(...)         → 梯度类指标的头级别聚合
  compute_head_svd_scores(...)          → SVD 类指标的头级别计算
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 数据类：注意力头配置
# ---------------------------------------------------------------------------

@dataclass
class AttnHeadConfig:
    """从模型配置提取的注意力头基本参数。"""
    num_heads: int    # 多头注意力中头的数量
    head_dim:  int    # 每个头的维度（= hidden_size / num_heads）
    hidden_size: int  # 模型隐藏维度

    @property
    def attn_dim(self) -> int:
        """QKV 投影输出维度 = num_heads × head_dim"""
        return self.num_heads * self.head_dim


# ---------------------------------------------------------------------------
# 模式匹配：识别 Q/K/V 投影与输出投影
# ---------------------------------------------------------------------------

# QKV 投影的模块名关键字（命名规范参考 DeBERTa / BERT / GPT 等主流模型）
_QKV_KEYWORDS: Tuple[str, ...] = (
    "query_proj", "key_proj", "value_proj",   # DeBERTa / RoBERTa style
    "q_proj",     "k_proj",   "v_proj",       # LLaMA / GPT-NeoX style
    "c_attn",                                  # GPT-2（合并 QKV，仅供识别用）
)

# 输出投影的模块名关键字
_OUT_KEYWORDS: Tuple[str, ...] = (
    "attention.output.dense",                  # BERT / DeBERTa style
    "out_proj",                                # GPT-2 / OPT style
    "o_proj",                                  # LLaMA style
)


def classify_attn_module(module_name: str) -> Optional[str]:
    """
    判断模块是否为注意力相关线性层，并返回其切分类型。

    返回值：
        "qkv"  — Q/K/V 投影（权重按行 / 输出维度分头）
        "out"  — 输出投影（权重按列 / 输入维度分头）
        None   — 非注意力模块
    """
    # 通过模块路径的组成部分匹配 QKV，避免误匹配子串
    path_parts = set(module_name.split("."))
    for kw in _QKV_KEYWORDS:
        if kw in path_parts:
            return "qkv"

    # 输出投影通常是多级路径片段，用子串匹配
    for kw in _OUT_KEYWORDS:
        if kw in module_name:
            return "out"

    return None


# ---------------------------------------------------------------------------
# 扫描模型：获取所有注意力模块
# ---------------------------------------------------------------------------

def get_attn_head_config(model: nn.Module) -> Optional[AttnHeadConfig]:
    """
    从 model.config 提取注意力头配置。

    支持标准 HuggingFace 配置字段：
      config.num_attention_heads, config.hidden_size

    若模型无 config 或字段缺失，返回 None（功能安全降级）。
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    num_heads   = getattr(cfg, "num_attention_heads", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    if num_heads is None or hidden_size is None:
        return None
    return AttnHeadConfig(
        num_heads=int(num_heads),
        head_dim=int(hidden_size) // int(num_heads),
        hidden_size=int(hidden_size),
    )


def get_attn_modules(
    model: nn.Module,
    attn_cfg: AttnHeadConfig,
) -> Dict[str, str]:
    """
    扫描模型的所有命名模块，返回维度匹配的注意力线性层及其类型。

    维度匹配规则（防止误识别）：
      "qkv" — weight.shape[0] == num_heads × head_dim
      "out" — weight.shape[1] == num_heads × head_dim

    Returns:
        {module_name: "qkv" | "out"}，按模型遍历顺序排列。
    """
    expected_dim = attn_cfg.attn_dim
    result: Dict[str, str] = {}

    for module_name, module in model.named_modules():
        mtype = classify_attn_module(module_name)
        if mtype is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() < 2:
            continue
        W = weight.detach()
        if mtype == "qkv" and W.shape[0] == expected_dim:
            result[module_name] = mtype
        elif mtype == "out" and W.shape[1] == expected_dim:
            result[module_name] = mtype

    return result


# ---------------------------------------------------------------------------
# 头级别切片工具（返回视图，共享内存，支持原地操作）
# ---------------------------------------------------------------------------

def get_head_weight_view(
    W: torch.Tensor,
    module_type: str,
    h: int,
    head_dim: int,
) -> torch.Tensor:
    """
    返回头 h 对应的权重矩阵视图（与原张量共享内存）。

    注意：该函数返回的是视图，对其的原地操作会修改原参数，
    修改后需手动恢复（见 PerturbationImportance 中的用法）。
    """
    if module_type == "qkv":
        return W[h * head_dim: (h + 1) * head_dim, :]     # [head_dim, hidden]
    else:  # "out"
        return W[:, h * head_dim: (h + 1) * head_dim]     # [hidden, head_dim]


def get_head_bias_view(
    b: torch.Tensor,
    module_type: str,
    h: int,
    head_dim: int,
) -> Optional[torch.Tensor]:
    """
    返回头 h 对应的偏置视图，若该类型无头级别偏置则返回 None。

    仅 QKV 投影的偏置 [num_heads×head_dim] 可按头拆分；
    输出投影的偏置 [hidden_size] 属于整个模块，不做拆分。
    """
    if module_type == "qkv":
        return b[h * head_dim: (h + 1) * head_dim]        # [head_dim]
    else:
        return None   # 输出投影偏置不按头归属


# ---------------------------------------------------------------------------
# 梯度类指标（Fisher / Saliency）：从累积张量聚合头级别分数
# ---------------------------------------------------------------------------

def agg_head_scores_from_acc(
    param_acc: Dict[str, torch.Tensor],
    model: nn.Module,
    attn_cfg: AttnHeadConfig,
    attn_modules: Dict[str, str],
    count: int,
) -> Dict[str, Dict[str, float]]:
    """
    从每个参数的累积张量（逐元素统计）中，聚合出注意力头级别分数。

    适用于 Fisher（累积梯度平方）和 Saliency（累积梯度绝对值 / Taylor）。
    聚合方式：将头对应的张量切片 element-wise 求和后除以 count（取均值）。

    Args:
        param_acc:    {param_name: tensor（与参数同 shape 的累积量）}
        model:        模型（用于按名称查找模块）
        attn_cfg:     注意力头配置
        attn_modules: {module_name: "qkv"/"out"}（来自 get_attn_modules）
        count:        已累积的 batch 数量（用于计算均值）

    Returns:
        {module_name: {"head_0": float, "head_1": float, ...}}
    """
    num_heads = attn_cfg.num_heads
    head_dim  = attn_cfg.head_dim
    head_scores: Dict[str, Dict[str, float]] = {}

    for module_name, module_type in attn_modules.items():
        per_head = [0.0] * num_heads

        for suffix, is_bias in (("weight", False), ("bias", True)):
            pname = f"{module_name}.{suffix}"
            acc = param_acc.get(pname)
            if acc is None:
                continue

            for h in range(num_heads):
                if is_bias:
                    view = get_head_bias_view(acc, module_type, h, head_dim)
                else:
                    view = get_head_weight_view(acc, module_type, h, head_dim)

                if view is None:
                    continue

                per_head[h] += (view.sum() / count).item()

        head_scores[module_name] = {f"head_{h}": per_head[h] for h in range(num_heads)}

    return head_scores


# ---------------------------------------------------------------------------
# SVD 类指标（SingularValue / SpectralEntropy）：计算头级别 SVD
# ---------------------------------------------------------------------------

def compute_head_svd_scores(
    model: nn.Module,
    attn_cfg: AttnHeadConfig,
    attn_modules: Dict[str, str],
    metric_fn: Callable[[torch.Tensor], Any],
) -> Dict[str, Dict[str, Any]]:
    """
    对每个注意力头的权重子矩阵应用 SVD 类指标函数，返回头级别结果。

    Args:
        model:        模型
        attn_cfg:     注意力头配置
        attn_modules: {module_name: "qkv"/"out"}
        metric_fn:    接受 2D float 张量（一个头的权重切片），
                      返回 float 或 dict；内部可能抛出异常会被捕获并跳过。

    Returns:
        {module_name: {"head_0": result, "head_1": result, ...}}
    """
    num_heads = attn_cfg.num_heads
    head_dim  = attn_cfg.head_dim
    head_scores: Dict[str, Dict[str, Any]] = {}

    named_mods = dict(model.named_modules())

    for module_name, module_type in attn_modules.items():
        module_obj = named_mods.get(module_name)
        if module_obj is None:
            continue
        weight = getattr(module_obj, "weight", None)
        if weight is None:
            continue

        W = weight.detach().float()
        per_head: Dict[str, Any] = {}

        for h in range(num_heads):
            W_h = get_head_weight_view(W, module_type, h, head_dim)
            try:
                per_head[f"head_{h}"] = metric_fn(W_h)
            except Exception as e:
                print(f"  [attn_head] metric_fn failed for {module_name} head_{h}: {e}")

        head_scores[module_name] = per_head

    return head_scores
