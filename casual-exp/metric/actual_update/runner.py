"""
metric/actual_update/runner.py

ActualUpdateRunner：实际更新量 U_m 的统一组合入口。

─────────────────────────────────────────────────────────────────────────────
支持的指标（可任意组合）：

  def1  绝对更新量      U_m = ||Δθ_m||_2（变体A）和 ||ΔW_m||_F（变体B）
                        训练结束后通过 on_train_end 计算，无需侵入训练循环
  def2  相对更新量      U_m^rel = ||Δθ_m||_2 / (||θ_m^(0)||_2 + ε)
                        训练结束后通过 on_train_end 计算
  def3  累计路径长度    U_m^path = Σ_t ||θ_m^(t) - θ_m^(t-1)||_2
                        训练中通过 on_step_end 逐步累积

所有指标均通过 TrainerCallback 嵌入训练，每种指标独立保存一个 JSON 文件。

头级别（head_granularity=True）：
  对 Q/K/V 投影和输出投影，额外输出 head_scores 字典。
  head_granularity 支持全局设置（from_str/构造函数）或通过 metric_kwargs 单独覆盖。

─────────────────────────────────────────────────────────────────────────────
嵌入训练代码（最小侵入，仅需 3 行）：
─────────────────────────────────────────────────────────────────────────────

    # ① 在 Trainer 创建前（model 尚未被 DDP 包装时）：
    from metric.actual_update.runner import ActualUpdateRunner
    au_runner = ActualUpdateRunner.from_str(
        args.actual_update,                          # e.g. "def1,def2,def3"
        metric_kwargs={"def3": {"log_every": 1}},
        head_granularity=args.au_head_granularity,   # True 则输出头级别分数
    )
    au_callbacks = au_runner.make_callbacks(
        model,
        save_dir=os.path.join(args.out_dir, "actual_update"),
    )

    # ② 注册 callbacks（与其他 callbacks 并列）：
    trainer = Trainer(..., callbacks=[eval_callback, *au_callbacks])

─────────────────────────────────────────────────────────────────────────────
独立使用（不依赖 Trainer）：
─────────────────────────────────────────────────────────────────────────────

    from metric.actual_update.base import snapshot_params
    from metric.actual_update.def1_absolute import AbsoluteUpdateMetric
    from metric.actual_update.def2_relative import RelativeUpdateMetric

    theta0 = snapshot_params(model)        # 训练前快照
    ... (训练) ...
    result1 = AbsoluteUpdateMetric().compute(theta0, model, device, head_granularity=True)
    result2 = RelativeUpdateMetric().compute(theta0, model, device, head_granularity=True)

─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .def1_absolute    import AbsoluteUpdateMetric
from .def2_relative    import RelativeUpdateMetric
from .def3_path_length import PathLengthMetric

# 全局注册表：{指标名 → 类}
REGISTRY: Dict[str, type] = {
    "def1": AbsoluteUpdateMetric,
    "def2": RelativeUpdateMetric,
    "def3": PathLengthMetric,
}


class ActualUpdateRunner:
    """
    组合运行器：选择一种或多种实际更新量定义，统一管理 callbacks。

    · 每种指标对应独立的 JSON 文件（{name}.json），互不干扰。
    · 所有指标均通过 TrainerCallback 嵌入训练：
        - def1/def2：在 on_train_end 时计算（零训练中开销）
        - def3：在 on_step_end 时逐步累积（每步一次 GPU→CPU 参数拷贝）
    · head_granularity 为全局开关，同时可在 metric_kwargs 中单独覆盖各指标的值。

    Args:
        metrics:          指标名列表，可任意组合，顺序不影响结果
        metric_kwargs:    各指标的超参数字典，格式为
                          {"def3": {"log_every": 10}, "def2": {"epsilon": 1e-6}}
                          每个指标的 head_granularity 可在此处单独覆盖全局设置
        head_granularity: 是否为所有指标开启注意力头级别粒度计算（默认 False）
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
        self._metrics: Dict[str, Any] = {
            name: REGISTRY[name]() for name in metrics
        }
        self._head_granularity = head_granularity
        self.metric_kwargs: Dict[str, Dict] = metric_kwargs or {}

    @classmethod
    def from_str(
        cls,
        metrics_str: str,
        metric_kwargs: Optional[Dict[str, Dict]] = None,
        head_granularity: bool = False,
    ) -> "ActualUpdateRunner":
        """
        从逗号分隔的字符串构造，便于命令行传参。

        例：
            ActualUpdateRunner.from_str(
                "def1,def2,def3",
                metric_kwargs={"def3": {"log_every": 10}},
                head_granularity=True,
            )
        """
        names = [m.strip() for m in metrics_str.split(",") if m.strip()]
        return cls(metrics=names, metric_kwargs=metric_kwargs,
                   head_granularity=head_granularity)

    def make_callbacks(
        self,
        model: torch.nn.Module,
        save_dir: str,
    ) -> List:
        """
        为所有选定指标创建 TrainerCallback 列表。

        ⚠️  必须在 Trainer 创建前调用：此时 model 尚未被 DDP 包装，
            def1/def2/def3 均在此刻立即记录 θ^(0) 快照。

        head_granularity 的优先级：
            metric_kwargs 中显式指定的 head_granularity > 构造函数的全局设置。

        Args:
            model:    未经 DDP 包装的原始模型
            save_dir: 结果保存目录（每个指标保存一个 JSON）

        Returns:
            TrainerCallback 列表，直接追加到 Trainer 的 callbacks 参数中

        使用方式：
            au_cbs  = runner.make_callbacks(model, save_dir)
            trainer = Trainer(..., callbacks=[..., *au_cbs])
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        callbacks = []

        for name, metric in self._metrics.items():
            kw = dict(self.metric_kwargs.get(name, {}))
            # 若 metric_kwargs 未单独指定 head_granularity，使用全局默认值
            kw.setdefault("head_granularity", self._head_granularity)
            cb = metric.make_callback(model=model, save_dir=save_dir, **kw)
            callbacks.append(cb)
            head_info = " [+head]" if kw.get("head_granularity") else ""
            print(f"[ActualUpdateRunner] 已注册 callback: {name}{head_info}")

        print(
            f"[ActualUpdateRunner] 共 {len(callbacks)} 个 callback，"
            f"结果将保存至: {save_dir}"
        )
        return callbacks

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def available_metrics(self) -> List[str]:
        return sorted(REGISTRY.keys())

    @property
    def selected_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    @property
    def head_granularity(self) -> bool:
        return self._head_granularity
