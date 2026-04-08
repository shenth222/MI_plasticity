"""
fix-budget/selection/update_response_selector.py

策略③：基于更新响应（update_response）的头选择。

支持的指标（与 metric/update_response/ 完全对应）：
  def1  短程试跑更新量      — 训练前运行，探针 AdamW 步数，支持重新选择
  def2  梯度-曲率归一化     — 训练前运行，共享 backward，支持重新选择
  def3  累积早期梯度范数    — 训练中收集（需特殊处理，见下方说明）
  def4  梯度信噪比 Ppred    — 训练前运行，共享 backward，支持重新选择

def3 的特殊性：
  def3 通过真实训练步骤的梯度钩子累积分数，无法单独调用 compute()，
  需要在训练中嵌入 EarlyGradNormCallback 来收集。主训练脚本会在
  T_early 步后读取收集结果，调用 set_def3_scores() 注入，然后触发选择。
  def3 不支持周期性重新选择（supports_reselect() 返回 False）。
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from metric.update_response.runner import REGISTRY, PRE_TRAINING_METRICS

from .base import HeadSelector

# def3 是训练中指标，其余均为训练前指标
_SUPPORTS_RESELECT = PRE_TRAINING_METRICS  # {"def1", "def2", "def4"}


class UpdateResponseSelector(HeadSelector):
    """
    基于更新响应指标的头选择器。

    def1/def2/def4 可在训练前/重新选择时直接调用 compute()；
    def3 需通过 set_def3_scores() 注入训练中收集的分数后才能调用 select()。

    Args:
        metric        : 指标名（"def1" / "def2" / "def3" / "def4"）
        metric_kwargs : 传递给 compute() 的额外超参
                        def1 → probe_steps, probe_lr
                        def2/def4 → num_batches, epsilon
                        def3 → 不适用（分数由 callback 注入）
    """

    def __init__(
        self,
        metric: str = "def2",
        metric_kwargs: Optional[Dict] = None,
    ):
        if metric not in REGISTRY:
            raise ValueError(
                f"[UpdateResponseSelector] 未知指标 '{metric}'，"
                f"可用: {sorted(REGISTRY)}"
            )
        self._metric_name   = metric
        self._metric_kwargs = metric_kwargs or {}
        self._is_def3       = (metric == "def3")

        if not self._is_def3:
            self._metric_obj = REGISTRY[metric]()
        else:
            self._metric_obj = None

        # def3 分数占位（由外部 callback 通过 set_def3_scores 注入）
        self._def3_head_scores: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def selector_name(self) -> str:
        return f"update_response/{self._metric_name}"

    def supports_reselect(self) -> bool:
        """def3 不支持重新选择，其余支持。"""
        return self._metric_name in _SUPPORTS_RESELECT

    def set_def3_scores(
        self,
        head_scores: Dict[str, Dict[str, float]],
    ) -> None:
        """
        由主训练脚本在 EarlyGradNormCallback 完成后调用，注入 def3 head_scores。

        head_scores 格式：{module_name: {"head_0": float, ...}}
        （与 EarlyGradNormCallback._save() 保存的 "head_scores" 字段一致）
        """
        if not self._is_def3:
            raise RuntimeError(
                "[UpdateResponseSelector] set_def3_scores 仅用于 def3 指标。"
            )
        self._def3_head_scores = head_scores
        print(
            f"[UpdateResponseSelector] def3 分数已注入，"
            f"共 {len(head_scores)} 个模块。"
        )

    def compute_head_scores(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        计算头级别分数。

        def3：直接返回已注入的 head_scores（不进行任何计算）。
        其余：调用对应 metric 的 compute()。

        Returns:
            {metric_name: {module_name: {"head_0": float, ...}}}
        """
        if self._is_def3:
            if self._def3_head_scores is None:
                raise RuntimeError(
                    "[UpdateResponseSelector] def3 分数尚未注入。"
                    "请先等待 EarlyGradNormCallback 完成收集，"
                    "然后调用 set_def3_scores()。"
                )
            return {self._metric_name: self._def3_head_scores}

        kw = dict(self._metric_kwargs)
        kw.update(kwargs)
        kw["head_granularity"] = True

        result = self._metric_obj.compute(model, dataloader, device, **kw)

        hs = result.get("head_scores")
        if hs is None:
            raise RuntimeError(
                f"[UpdateResponseSelector] '{self._metric_name}' 未返回 head_scores，"
                "请确认 head_granularity=True 且模型具有有效注意力头配置。"
            )
        return {self._metric_name: hs}
