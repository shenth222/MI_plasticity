"""
metric/pre_importance/runner.py

PreImportanceRunner：训练前重要性计算的统一组合入口。

─────────────────────────────────────────────────────────────────────────────
支持的指标：
  fisher          定义 1 – Fisher 型重要性
  saliency        定义 2 – 梯度敏感度（grad_norm + taylor 两种子变体）
  perturbation    定义 3 – 扰动敏感度
  singular_value  定义 4 – 奇异值（核范数 + 前 k 截断两种子变体）
  spectral_entropy 定义 5 – 谱熵

支持粒度（granularity）：
  module（默认）— 叶模块级别
  head           — 额外输出注意力头级别（叠加在 module 之上，不替代）
                   要求模型具有标准 HuggingFace config（含 num_attention_heads）

使用示例（独立脚本）：
─────────────────────────────────────────────────────────────────────────────
    # 仅模块级
    runner = PreImportanceRunner(
        metrics=["fisher", "singular_value"],
        metric_kwargs={"fisher": {"num_batches": 16}},
    )

    # 同时启用头级别
    runner = PreImportanceRunner(
        metrics=["fisher", "spectral_entropy"],
        head_granularity=True,
    )

    results = runner.run(model, dataloader, device, save_dir="outputs/pre_importance")

使用示例（从命令行字符串构造）：
─────────────────────────────────────────────────────────────────────────────
    runner = PreImportanceRunner.from_str(
        "fisher,saliency,spectral_entropy",
        head_granularity=True,
    )

嵌入训练代码（最小侵入）：
─────────────────────────────────────────────────────────────────────────────
    if args.pre_importance:
        from metric.pre_importance.runner import PreImportanceRunner
        runner = PreImportanceRunner.from_str(
            args.pre_importance,
            head_granularity=args.pre_importance_head_granularity,
        )
        runner.run(model, train_dataloader, device,
                   save_dir=os.path.join(args.out_dir, "pre_importance"))
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .fisher import FisherImportance
from .saliency import SaliencyImportance
from .perturbation import PerturbationImportance
from .singular_value import SingularValueImportance
from .spectral_entropy import SpectralEntropyImportance

# 全局注册表：{指标名 → 类}
REGISTRY = {
    "fisher":            FisherImportance,
    "saliency":          SaliencyImportance,
    "perturbation":      PerturbationImportance,
    "singular_value":    SingularValueImportance,
    "spectral_entropy":  SpectralEntropyImportance,
}


class PreImportanceRunner:
    """
    组合运行器：选择一种或多种训练前重要性定义，统一计算并独立保存结果。

    每种指标对应独立的 JSON 文件，互不干扰；
    需要数据的指标（Fisher/Saliency/Perturbation）共用同一个 DataLoader，
    不需要数据的指标（SingularValue/SpectralEntropy）直接在参数上计算。

    Args:
        metrics:          指标名列表，顺序不影响结果
        metric_kwargs:    各指标的超参数字典，格式为
                          {"fisher": {"num_batches": 16}, "perturbation": {"num_batches": 4}}
        head_granularity: 若为 True，所有指标额外计算注意力头级别分数；
                          结果附加在各 JSON 的 "head_scores" 字段中，不影响 "module_scores"。
    """

    def __init__(
        self,
        metrics: List[str],
        metric_kwargs: Optional[Dict[str, Dict]] = None,
        head_granularity: bool = False,
    ):
        unknown = set(metrics) - set(REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. Available: {sorted(REGISTRY)}"
            )
        self.metrics: Dict[str, Any] = {
            name: REGISTRY[name]() for name in metrics
        }
        self.metric_kwargs: Dict[str, Dict] = metric_kwargs or {}
        self.head_granularity: bool = head_granularity

    @classmethod
    def from_str(
        cls,
        metrics_str: str,
        metric_kwargs: Optional[Dict[str, Dict]] = None,
        head_granularity: bool = False,
    ) -> "PreImportanceRunner":
        """
        从逗号分隔的字符串构造，便于命令行传参。

        例：PreImportanceRunner.from_str(
               "fisher,singular_value,spectral_entropy",
               head_granularity=True,
            )
        """
        names = [m.strip() for m in metrics_str.split(",") if m.strip()]
        return cls(metrics=names, metric_kwargs=metric_kwargs,
                   head_granularity=head_granularity)

    def run(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        save_dir: str,
    ) -> Dict[str, Any]:
        """
        依次运行所有选定指标，将各自结果独立保存到 save_dir。

        Args:
            model:      待评估模型（θ₀，微调前）
            dataloader: 训练集 DataLoader；无数据需求的指标会自动跳过
            device:     计算设备
            save_dir:   结果保存目录（每个指标保存一个 JSON）

        Returns:
            {metric_name: scores_dict}，与各指标 compute() 返回值一致
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        all_results: Dict[str, Any] = {}

        gran_tag = " [+head]" if self.head_granularity else ""

        for name, metric in self.metrics.items():
            print(f"\n{'='*60}")
            print(f"[PreImportanceRunner] Computing: {name}{gran_tag}")
            print(f"{'='*60}")

            kw = dict(self.metric_kwargs.get(name, {}))
            if self.head_granularity:
                kw["head_granularity"] = True

            dl = dataloader if metric.needs_data else None
            scores = metric.compute(model, dl, device, **kw)
            metric.save(scores, save_dir)
            all_results[name] = scores

        print(f"\n[PreImportanceRunner] All {len(self.metrics)} metric(s) done.")
        print(f"[PreImportanceRunner] Results saved to: {save_dir}")
        return all_results

    @property
    def available_metrics(self) -> List[str]:
        return sorted(REGISTRY.keys())
