"""
metric/update_response/runner.py

UpdateResponseRunner：更新响应预测的统一组合入口。

─────────────────────────────────────────────────────────────────────────────
支持的指标：
  def1  短程试跑更新量     — 训练前运行（手动 AdamW 探针）
  def2  梯度-曲率归一化    — 训练前运行（梯度 + Fisher 共享一次 backward）
  def3  累积早期梯度范数   — 训练中收集（作为 TrainerCallback 嵌入）
  def4  梯度信噪比 Ppred   — 训练前运行（元素级 SNR 估计）

计算时机分类：
  PRE_TRAINING_METRICS  = {def1, def2, def4} — 通过 run_pre() 运行
  IN_TRAINING_METRICS   = {def3}              — 通过 make_training_callback() 获取回调

─────────────────────────────────────────────────────────────────────────────
使用示例（组合运行，嵌入训练代码）：
─────────────────────────────────────────────────────────────────────────────

    # 训练脚本（最小侵入）：
    runner = UpdateResponseRunner.from_str(
        "def1,def2,def3,def4",
        metric_kwargs={
            "def1": {"probe_steps": 20, "probe_lr": args.lr},
            "def2": {"num_batches": 32},
            "def3": {"T_early": 100},
            "def4": {"num_batches": 32},
        },
    )

    # ── 嵌入点 1：训练前（θ₀ 保存后，Trainer 创建前）
    runner.run_pre(model, train_dataloader, device,
                   save_dir=os.path.join(args.out_dir, "update_response"))

    # ── 嵌入点 2：注册 Trainer callback（def3 需要，其余指标无影响）
    ur_callbacks = runner.make_training_callbacks(model,
                       save_dir=os.path.join(args.out_dir, "update_response"))
    trainer = Trainer(..., callbacks=[..., *ur_callbacks])

─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .def1_probe_delta   import ProbeDeltaMetric
from .def2_grad_curvature import GradCurvatureMetric
from .def3_early_grad_norm import EarlyGradNormMetric
from .def4_ppred          import PpredMetric

# 全局注册表：{指标名 → 类}
REGISTRY: Dict[str, type] = {
    "def1": ProbeDeltaMetric,
    "def2": GradCurvatureMetric,
    "def3": EarlyGradNormMetric,
    "def4": PpredMetric,
}

PRE_TRAINING_METRICS  = {"def1", "def2", "def4"}
IN_TRAINING_METRICS   = {"def3"}


class UpdateResponseRunner:
    """
    组合运行器：选择一种或多种更新响应预测定义，统一计算并独立保存结果。

    · 每种指标对应独立的 JSON 文件（{name}.json），互不干扰。
    · 训练前指标（def1/def2/def4）共用同一个 DataLoader，
      各自独立运行 compute() 并调用 save()。
    · 训练中指标（def3）返回 TrainerCallback，注册到 Trainer 后自动运行。

    Args:
        metrics:       指标名列表，可任意组合，顺序不影响结果
        metric_kwargs: 各指标的超参数字典，格式为
                       {"def1": {"probe_steps": 20}, "def3": {"T_early": 100}}
    """

    def __init__(
        self,
        metrics: List[str],
        metric_kwargs: Optional[Dict[str, Dict]] = None,
    ):
        unknown = set(metrics) - set(REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. "
                f"Available: {sorted(REGISTRY)}"
            )

        self._pre_metrics: Dict[str, Any] = {
            name: REGISTRY[name]()
            for name in metrics
            if name in PRE_TRAINING_METRICS
        }
        self._in_metrics: Dict[str, Any] = {
            name: REGISTRY[name]()
            for name in metrics
            if name in IN_TRAINING_METRICS
        }
        self.metric_kwargs: Dict[str, Dict] = metric_kwargs or {}

    @classmethod
    def from_str(
        cls,
        metrics_str: str,
        metric_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> "UpdateResponseRunner":
        """
        从逗号分隔的字符串构造，便于命令行传参。

        例：
            UpdateResponseRunner.from_str(
                "def1,def2,def4",
                metric_kwargs={"def1": {"probe_steps": 20}},
            )
        """
        names = [m.strip() for m in metrics_str.split(",") if m.strip()]
        return cls(metrics=names, metric_kwargs=metric_kwargs)

    # ------------------------------------------------------------------
    # 训练前运行（def1, def2, def4）
    # ------------------------------------------------------------------

    def run_pre(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        save_dir: str,
    ) -> Dict[str, Any]:
        """
        依次运行所有训练前指标，将各自结果独立保存到 save_dir。

        Args:
            model:      待评估模型（θ₀，微调前；def1 会临时修改但会自动恢复）
            dataloader: 训练集 DataLoader（def1/def2/def4 均需要）
            device:     计算设备
            save_dir:   结果保存目录（每个指标保存一个 JSON）

        Returns:
            {metric_name: scores_dict}
        """
        if not self._pre_metrics:
            return {}

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        all_results: Dict[str, Any] = {}

        for name, metric in self._pre_metrics.items():
            print(f"\n{'='*60}")
            print(f"[UpdateResponseRunner] Computing pre-training: {name}")
            print(f"{'='*60}")

            kw = dict(self.metric_kwargs.get(name, {}))
            scores = metric.compute(model, dataloader, device, **kw)
            metric.save(scores, save_dir)
            all_results[name] = scores

        print(f"\n[UpdateResponseRunner] Pre-training metrics done"
              f" ({len(self._pre_metrics)} metric(s)).")
        return all_results

    # ------------------------------------------------------------------
    # 训练中回调（def3）
    # ------------------------------------------------------------------

    def make_training_callbacks(
        self,
        model: torch.nn.Module,
        save_dir: str,
    ) -> List:
        """
        为所有训练中指标创建 TrainerCallback 列表。

        Args:
            model:    未经 DDP 包装的原始模型（必须在 Trainer 创建前传入）
            save_dir: 结果保存目录

        Returns:
            TrainerCallback 列表（若无训练中指标则返回空列表）

        使用方式：
            ur_cbs = runner.make_training_callbacks(model, save_dir)
            trainer = Trainer(..., callbacks=[..., *ur_cbs])
        """
        callbacks = []
        for name, metric in self._in_metrics.items():
            kw = dict(self.metric_kwargs.get(name, {}))
            cb = metric.make_callback(model=model, save_dir=save_dir, **kw)
            callbacks.append(cb)
            print(f"[UpdateResponseRunner] Registered in-training callback: {name}")
        return callbacks

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def available_metrics(self) -> List[str]:
        return sorted(REGISTRY.keys())

    @property
    def selected_pre_metrics(self) -> List[str]:
        return list(self._pre_metrics.keys())

    @property
    def selected_in_metrics(self) -> List[str]:
        return list(self._in_metrics.keys())
