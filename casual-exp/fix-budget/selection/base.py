"""
fix-budget/selection/base.py

HeadSelection 数据类与 HeadSelector 抽象基类。

HeadSelection：
    单次头选择操作的完整结果，包含全量排序列表、选中集合与元数据，
    支持 JSON 保存和控制台表格打印，便于后续分析和扩展更复杂的策略。

HeadSelector（ABC）：
    所有选择策略的统一接口，子类只需实现 compute_head_scores()。
    select() 方法自动完成 聚合 → 排序 → 选择 → 封装 HeadSelection 的全流程。
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader

from .head_utils import (
    ConceptualHead,
    RankedHead,
    build_conceptual_head_scores,
    print_head_selection_table,
    rank_conceptual_heads,
    select_heads_by_rank,
)


@dataclass
class HeadSelection:
    """
    单次头选择操作的完整结果。

    字段：
        ranked_heads : 按排名排序的头列表 [(layer_key, head_idx, score), ...]
                       其中第 0 个为最优（得分最高）的头
        selected_set : 被选中的头集合 {(layer_key, head_idx), ...}
        strategy     : 选择策略名称（如 "random" / "pre_importance/fisher" 等）
        metric_name  : 具体指标变体名称（如 "fisher" / "saliency/grad_norm"）
        budget_count : 本次选择的预算头数
        total_heads  : 模型中概念头总数
        step         : 触发本次选择的训练步数（0 代表训练前初始选择）
        extra        : 额外元信息（如 metric 超参数等）

    使用方式：
        sel.print_table()          # 控制台打印排序/选择表格
        sel.to_dict()              # 序列化为可 JSON 保存的字典
        sel.save("path/to/dir")    # 保存到 JSON 文件
    """

    ranked_heads: List[RankedHead]          # [(layer_key, head_idx, score), ...]
    selected_set: Set[ConceptualHead]       # {(layer_key, head_idx), ...}
    strategy:     str
    metric_name:  str
    budget_count: int
    total_heads:  int
    step:         int = 0
    extra:        Dict[str, Any] = field(default_factory=dict)

    def print_table(self, top_n: Optional[int] = 50) -> None:
        """打印排序与选择结果表格（默认显示前 50 行）。"""
        print_head_selection_table(
            ranked=self.ranked_heads,
            selected_set=self.selected_set,
            strategy=self.strategy,
            metric_name=self.metric_name,
            budget_count=self.budget_count,
            step=self.step,
            top_n=top_n,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 保存的字典。"""
        return {
            "step":         self.step,
            "strategy":     self.strategy,
            "metric_name":  self.metric_name,
            "budget_count": self.budget_count,
            "total_heads":  self.total_heads,
            "ranked_heads": [
                {
                    "rank":      i + 1,
                    "layer_key": lk,
                    "head_idx":  hi,
                    "score":     sc,
                    "selected":  (lk, hi) in self.selected_set,
                }
                for i, (lk, hi, sc) in enumerate(self.ranked_heads)
            ],
            "selected_heads": [
                {"layer_key": lk, "head_idx": hi}
                for lk, hi in sorted(self.selected_set)
            ],
            **self.extra,
        }

    def save(self, save_dir: str) -> str:
        """
        保存到 JSON 文件，文件名含 step 信息，返回保存路径。

        文件名格式：selection_step{step:06d}.json
        """
        os.makedirs(save_dir, exist_ok=True)
        fname = f"selection_step{self.step:06d}.json"
        fpath = os.path.join(save_dir, fname)
        with open(fpath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return fpath

    def summary_str(self) -> str:
        """返回简短的单行汇总字符串，便于日志输出。"""
        return (
            f"strategy={self.strategy}  metric={self.metric_name}  "
            f"budget={self.budget_count}/{self.total_heads}  step={self.step}"
        )


class HeadSelector(ABC):
    """
    注意力头选择器抽象基类。

    子类需实现：
      1. `selector_name` 属性 — 策略名称字符串
      2. `compute_head_scores()` — 返回 {variant_name: {module_name: {"head_0": float|dict, ...}}}

    `select()` 方法自动完成：
      聚合（build_conceptual_head_scores）→ 排序（rank_conceptual_heads）
      → 选择（select_heads_by_rank）→ 封装（HeadSelection）
    """

    @property
    @abstractmethod
    def selector_name(self) -> str:
        """选择器名称，用于日志和文件命名。"""
        ...

    @abstractmethod
    def compute_head_scores(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        计算并返回各变体的头级别分数。

        Returns:
            {
              variant_name: {
                module_name: {"head_0": float | dict, "head_1": ..., ...}
              }
            }

        当指标无子变体时，variant_name 即为指标名（如 "fisher"）；
        有子变体时（如 saliency），返回多个条目（如 "saliency/grad_norm", "saliency/taylor"）。
        当头值为 dict（如 singular_value / spectral_entropy）时，
        需同时返回以各子字段命名的多个变体，或由调用方通过 sub_key 指定。
        """
        ...

    def select(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        budget_count: int,
        step: int = 0,
        agg: str = "mean",
        sub_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, "HeadSelection"]:
        """
        计算分数、排序、选择，返回各变体的 HeadSelection 结果。

        Args:
            model        : 待评估的模型
            dataloader   : 训练数据 DataLoader（无需数据的指标可为 None）
            device       : 计算设备
            budget_count : 选择的头数量
            step         : 当前训练步（用于记录）
            agg          : 聚合方式 "mean" | "sum"
            sub_key      : 当头值为 dict 时，指定取哪个子字段
            **kwargs     : 传递给 compute_head_scores 的额外参数

        Returns:
            {variant_name: HeadSelection}，每个变体独立排序和选择
        """
        all_scores = self.compute_head_scores(model, dataloader, device, **kwargs)
        results: Dict[str, HeadSelection] = {}

        for variant_name, head_scores in all_scores.items():
            conceptual_scores = build_conceptual_head_scores(
                head_scores, agg=agg, sub_key=sub_key
            )
            ranked = rank_conceptual_heads(conceptual_scores, descending=True)
            selected = select_heads_by_rank(ranked, budget_count)

            results[variant_name] = HeadSelection(
                ranked_heads=ranked,
                selected_set=selected,
                strategy=self.selector_name,
                metric_name=variant_name,
                budget_count=budget_count,
                total_heads=len(ranked),
                step=step,
            )

        return results

    def supports_reselect(self) -> bool:
        """
        是否支持在训练中途重新选择。

        大多数策略支持，def3（训练中累积梯度）不支持。
        子类可覆盖此方法返回 False。
        """
        return True
