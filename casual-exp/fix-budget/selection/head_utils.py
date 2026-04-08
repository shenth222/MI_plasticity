"""
fix-budget/selection/head_utils.py

注意力头级别操作的通用工具函数。
包含：
  - 从模块名称提取层标识符
  - 将各模块 head_scores 聚合为「概念头」维度分数
  - 对概念头排序
  - 根据预算选择头
  - 可读性强的排序/选择结果打印
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any

import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 概念头 = (layer_key, head_idx)
#   layer_key : 提取自模块名，代表同一注意力层内所有 Q/K/V/O 的公共前缀
#               例：deberta.encoder.layer.0.attention
#   head_idx  : 该层中头的下标（0-based）
# ─────────────────────────────────────────────────────────────────────────────

ConceptualHead = Tuple[str, int]  # (layer_key, head_idx)
RankedHead     = Tuple[str, int, float]  # (layer_key, head_idx, score)


def get_layer_key(module_name: str) -> str:
    """
    从模块名称中提取注意力层标识符（直到并包含 "attention" 为止）。

    例：
      "deberta.encoder.layer.0.attention.self.query_proj" → "deberta.encoder.layer.0.attention"
      "deberta.encoder.layer.0.attention.output.dense"   → "deberta.encoder.layer.0.attention"
    """
    match = re.match(r'^(.+?\.attention)(?:\.|$)', module_name)
    if match:
        return match.group(1)
    # 回退：去掉最后一级作为「层 key」
    parts = module_name.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else module_name


def extract_scalar_from_head_value(value: Any, sub_key: Optional[str] = None) -> Optional[float]:
    """
    将 head_scores 中的单个头值提取为标量 float。

    支持两种情况：
      1. 值已为 float/int → 直接返回
      2. 值为 dict（如 singular_value / spectral_entropy 的头分数）
         → 若提供 sub_key 则取对应字段；否则取第一个数值字段
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if sub_key is not None and sub_key in value:
            v = value[sub_key]
            if isinstance(v, (int, float)):
                return float(v)
        # 取第一个 float 字段
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def build_conceptual_head_scores(
    head_scores: Dict[str, Dict[str, Any]],
    agg: str = "mean",
    sub_key: Optional[str] = None,
) -> Dict[ConceptualHead, float]:
    """
    将各模块的头级别分数聚合为「概念头」(layer_key, head_idx) 维度的分数。

    概念头 = 同一注意力层中相同头下标对应的所有 Q/K/V/O 模块参数切片。
    聚合后每个概念头有一个统一分数，用于跨层排序与预算选择。

    Args:
        head_scores : {module_name: {"head_0": float|dict, ...}}
                      来自 metric 结果中的 head_scores 字段。
        agg         : 聚合方式，"mean" 或 "sum"。
        sub_key     : 当头值为 dict 时，指定取哪个子字段（如 "nuclear_norm"）。
                      默认为 None（自动取第一个数值字段）。

    Returns:
        {(layer_key, head_idx): score, ...}
    """
    accumulator: Dict[ConceptualHead, List[float]] = {}

    for module_name, per_head in head_scores.items():
        layer_key = get_layer_key(module_name)
        for head_key, value in per_head.items():
            # head_key 格式为 "head_0", "head_1", ...
            try:
                head_idx = int(head_key.split("_")[1])
            except (IndexError, ValueError):
                continue

            score = extract_scalar_from_head_value(value, sub_key)
            if score is None:
                continue

            key: ConceptualHead = (layer_key, head_idx)
            accumulator.setdefault(key, []).append(score)

    if agg == "sum":
        return {k: sum(v) for k, v in accumulator.items()}
    else:  # mean
        return {k: sum(v) / len(v) for k, v in accumulator.items()}


def rank_conceptual_heads(
    scores: Dict[ConceptualHead, float],
    descending: bool = True,
) -> List[RankedHead]:
    """
    对概念头按分数排序。

    Args:
        scores     : {(layer_key, head_idx): score}
        descending : True → 分数高的排前（选择重要性强 / 响应强的头）

    Returns:
        [(layer_key, head_idx, score), ...] 按排名从好到差排序
    """
    return sorted(
        [(lk, hi, sc) for (lk, hi), sc in scores.items()],
        key=lambda x: x[2],
        reverse=descending,
    )


def select_heads_by_rank(
    ranked: List[RankedHead],
    budget_count: int,
) -> Set[ConceptualHead]:
    """
    从排序后的头列表中选择前 budget_count 个，返回选中集合。

    Returns:
        {(layer_key, head_idx), ...}
    """
    return {(lk, hi) for lk, hi, _ in ranked[:budget_count]}


def get_all_conceptual_heads_from_model(model: nn.Module) -> List[ConceptualHead]:
    """
    从模型中枚举所有概念头，返回 (layer_key, head_idx) 列表（已排序）。

    依赖 metric.pre_importance.attn_head 中的工具，无需任何数据。
    """
    from metric.pre_importance.attn_head import get_attn_head_config, get_attn_modules

    attn_cfg = get_attn_head_config(model)
    if attn_cfg is None:
        raise RuntimeError("[head_utils] 无法从模型 config 中提取注意力头配置。")

    attn_mods = get_attn_modules(model, attn_cfg)
    heads: Set[ConceptualHead] = set()
    for m_name in attn_mods:
        lk = get_layer_key(m_name)
        for h in range(attn_cfg.num_heads):
            heads.add((lk, h))

    return sorted(heads)


def print_head_selection_table(
    ranked: List[RankedHead],
    selected_set: Set[ConceptualHead],
    strategy: str,
    metric_name: str,
    budget_count: int,
    step: int = 0,
    top_n: Optional[int] = 50,
) -> None:
    """
    以可读性强的表格形式打印头排序与选择结果。

    Args:
        ranked       : 按排名排序的 [(layer_key, head_idx, score), ...]
        selected_set : 被选中的头集合 {(layer_key, head_idx), ...}
        strategy     : 选择策略名称
        metric_name  : 指标名称
        budget_count : 预算头数
        step         : 当前训练步数
        top_n        : 仅显示前 top_n 行（None 则显示全部；默认 50）
    """
    total = len(ranked)
    budget_pct = budget_count / total * 100 if total > 0 else 0.0

    W = 90
    print(f"\n{'─'*W}")
    print(f"  【头选择结果】  Step={step}  策略={strategy}  指标={metric_name}")
    print(f"  总头数={total}  预算={budget_count}  ({budget_pct:.1f}%)")
    print(f"{'─'*W}")
    print(f"{'排名':>5}  {'层 Key':52}  {'头':>4}  {'分数':>14}  {'已选':>4}")
    print(f"{'─'*W}")

    rows = ranked if top_n is None else ranked[:top_n]
    for i, (layer_key, head_idx, score) in enumerate(rows):
        mark = "✓" if (layer_key, head_idx) in selected_set else " "
        short_lk = layer_key[-52:] if len(layer_key) > 52 else layer_key
        print(f"{i+1:>5}  {short_lk:52}  {head_idx:>4}  {score:>14.6f}  {mark:>4}")

    if top_n is not None and top_n < total:
        print(f"  ... (仅显示前 {top_n} / 共 {total} 行)")

    print(f"{'─'*W}")

    # 层级汇总
    layer_summary: Dict[str, Dict[str, int]] = {}
    for layer_key, head_idx, _ in ranked:
        entry = layer_summary.setdefault(layer_key, {"total": 0, "selected": 0})
        entry["total"] += 1
        if (layer_key, head_idx) in selected_set:
            entry["selected"] += 1

    print(f"\n  层级选择汇总:")
    for lk in sorted(layer_summary.keys()):
        info = layer_summary[lk]
        sel, tot = info["selected"], info["total"]
        bar = "█" * sel + "░" * (tot - sel)
        short_lk = lk[-52:] if len(lk) > 52 else lk
        print(f"    {short_lk:52}  [{bar}]  {sel}/{tot}")
    print()
