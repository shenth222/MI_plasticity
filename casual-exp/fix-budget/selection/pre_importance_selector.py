"""
fix-budget/selection/pre_importance_selector.py

策略②：基于训练前重要性（pre_importance）的头选择。

支持的指标（与 metric/pre_importance/ 完全对应）：
  fisher          — Fisher 型重要性，单一变体
  saliency        — 梯度敏感度，含 grad_norm / taylor 两种子变体，同时支持
  perturbation    — 扰动敏感度，单一变体
  singular_value  — 奇异值，含 nuclear_norm / top{k}_sum 两种子变体，同时支持
  spectral_entropy — 谱熵，单一变体（spectral_entropy 字段）

参数说明（CLI 示例）：
  --pre_importance_metric fisher
  --pre_importance_metric saliency            # 自动返回 grad_norm 和 taylor 两个变体
  --pre_importance_metric singular_value      # 自动返回 nuclear_norm 和 topk_sum 两个变体
  --pre_importance_num_batches 32             # fisher/saliency/perturbation 的 batch 数
  --pre_importance_top_k 32                   # singular_value 的截断奇异值数
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from metric.pre_importance.runner import REGISTRY

from .base import HeadSelector


class PreImportanceSelector(HeadSelector):
    """
    基于训练前重要性指标的头选择器。

    支持所有 pre_importance 指标，并自动处理多子变体情形。

    Args:
        metric        : 指标名（"fisher" / "saliency" / "perturbation" /
                         "singular_value" / "spectral_entropy"）
        metric_kwargs : 传递给 compute() 的额外超参（如 num_batches、top_k 等）
    """

    def __init__(
        self,
        metric: str = "fisher",
        metric_kwargs: Optional[Dict] = None,
    ):
        if metric not in REGISTRY:
            raise ValueError(
                f"[PreImportanceSelector] 未知指标 '{metric}'，"
                f"可用: {sorted(REGISTRY)}"
            )
        self._metric_name   = metric
        self._metric_obj    = REGISTRY[metric]()
        self._metric_kwargs = metric_kwargs or {}

    @property
    def selector_name(self) -> str:
        return f"pre_importance/{self._metric_name}"

    def compute_head_scores(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        计算 head_scores，自动处理多子变体。

        Returns:
            {variant_name: {module_name: {"head_0": float|dict, ...}}}

        变体命名规则：
          fisher          → {"fisher": {...}}
          saliency        → {"saliency/grad_norm": {...}, "saliency/taylor": {...}}
          perturbation    → {"perturbation": {...}}
          singular_value  → {"singular_value/nuclear_norm": {...},
                             "singular_value/top{k}_sum": {...}}
          spectral_entropy → {"spectral_entropy": {...}}
        """
        kw = dict(self._metric_kwargs)
        kw.update(kwargs)
        kw["head_granularity"] = True  # 必须启用头粒度

        dl = dataloader if self._metric_obj.needs_data else None
        result = self._metric_obj.compute(model, dl, device, **kw)

        return self._extract_variants(result, kw)

    def _extract_variants(self, result: Dict, kw: Dict) -> Dict[str, Dict]:
        """从 compute() 结果中提取各子变体的 head_scores。"""
        metric = self._metric_name

        # ── saliency：grad_norm + taylor 两子变体 ────────────────────────────
        if metric == "saliency":
            out: Dict[str, Dict] = {}
            for variant in ("grad_norm", "taylor"):
                hs = result.get(variant, {}).get("head_scores")
                if hs:
                    out[f"saliency/{variant}"] = hs
            if not out:
                raise RuntimeError(
                    "[PreImportanceSelector] saliency 未返回 head_scores，"
                    "请确保模型具有有效注意力头配置。"
                )
            return out

        # ── singular_value：nuclear_norm + top{k}_sum 两子变体 ────────────
        if metric == "singular_value":
            hs = result.get("head_scores")
            if not hs:
                raise RuntimeError(
                    "[PreImportanceSelector] singular_value 未返回 head_scores。"
                )
            top_k = result.get("top_k", kw.get("top_k", 32))
            # head 值为 dict，分拆成两个 float 变体
            nuclear_hs = self._flatten_head_scores(hs, "nuclear_norm")
            topk_hs    = self._flatten_head_scores(hs, f"top{top_k}_sum")
            return {
                "singular_value/nuclear_norm": nuclear_hs,
                f"singular_value/top{top_k}_sum": topk_hs,
            }

        # ── spectral_entropy：头值为 dict，取 spectral_entropy 字段 ─────────
        if metric == "spectral_entropy":
            hs = result.get("head_scores")
            if not hs:
                raise RuntimeError(
                    "[PreImportanceSelector] spectral_entropy 未返回 head_scores。"
                )
            return {
                "spectral_entropy": self._flatten_head_scores(hs, "spectral_entropy")
            }

        # ── fisher / perturbation：直接返回 head_scores ────────────────────
        hs = result.get("head_scores")
        if hs is None:
            raise RuntimeError(
                f"[PreImportanceSelector] '{metric}' 未返回 head_scores，"
                "请确认 head_granularity=True 且模型具有有效注意力头配置。"
            )
        return {metric: hs}

    @staticmethod
    def _flatten_head_scores(
        head_scores: Dict[str, Dict[str, Any]],
        sub_key: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        将 head_scores 中值为 dict 的条目扁平化：提取指定子字段的 float 值。

        输入：  {module_name: {"head_0": {"nuclear_norm": 0.5, ...}, ...}}
        输出：  {module_name: {"head_0": 0.5, ...}}
        """
        out: Dict[str, Dict[str, float]] = {}
        for m_name, per_head in head_scores.items():
            flat: Dict[str, float] = {}
            for h_key, val in per_head.items():
                if isinstance(val, dict) and sub_key in val:
                    v = val[sub_key]
                    if isinstance(v, (int, float)):
                        flat[h_key] = float(v)
                elif isinstance(val, (int, float)):
                    flat[h_key] = float(val)
            out[m_name] = flat
        return out
