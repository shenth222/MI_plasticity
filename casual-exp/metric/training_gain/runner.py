"""
metric/training_gain/runner.py

TrainingGainRunner：训练收益 G_m 的统一组合入口。

─────────────────────────────────────────────────────────────────────────────
支持的指标（可任意组合）：

  def1  回滚 + val loss 变化
          G_m^(loss) = L_val(θ^(T)[m←θ^(0)]) − L_val(θ^(T))
          训练结束后通过 on_train_end 逐模块回滚并评估
  def2  回滚 + val accuracy 变化
          G_m^(acc) = Acc_val(θ^(T)[m←θ^(0)]) − Acc_val(θ^(T))
          与 def1 共用一次前向传播（同一个 RollbackCallback），分别保存文件
  def3  路径积分
          G_m^(PI) = Σ_t ∇_{θ_m}L(θ^(t)) · Δθ_{m,t}
          训练中通过梯度 hook + on_step_end 逐步累积

合并规则（自动优化计算量）：
  · def1 和 def2 同时选中时，创建一个 RollbackCallback（compute_loss=True, compute_acc=True），
    共用一次前向传播，分别保存 def1_rollback_loss.json 和 def2_rollback_acc.json。
  · 单独选 def1 或 def2 时，仅计算对应指标，节省计算量。
  · def3 独立创建 PathIntegralCallback，与 def1/def2 互不干扰。

─────────────────────────────────────────────────────────────────────────────
嵌入训练代码（最小侵入，仅需 ~5 行核心代码）：
─────────────────────────────────────────────────────────────────────────────

    # ① 在 Trainer 创建前（model 尚未被 DDP 包装时）：
    from metric.training_gain.runner import TrainingGainRunner
    from metric.training_gain.def12_rollback import build_glue_eval_fn

    eval_fn = build_glue_eval_fn(
        tokenizer, task_name=args.task, dataset_path=args.dataset_path,
        max_length=args.max_len, batch_size=args.bsz,
    )
    gm_runner = TrainingGainRunner.from_str(
        args.training_gain,                          # e.g. "def1,def2,def3"
        head_granularity=args.gm_head_granularity,
    )
    gm_callbacks = gm_runner.make_callbacks(
        model, save_dir=os.path.join(args.out_dir, "training_gain"),
        eval_fn=eval_fn,
        metric_kwargs={"def3": {"log_every": args.gm_log_every}},
    )

    # ② 注册 callbacks：
    trainer = Trainer(..., callbacks=[eval_callback, *gm_callbacks])

─────────────────────────────────────────────────────────────────────────────
独立使用（不依赖 Trainer，离线计算）：
─────────────────────────────────────────────────────────────────────────────

    from metric.training_gain.base import snapshot_params
    from metric.training_gain.def12_rollback import (
        build_glue_eval_fn, compute_rollback, _save_def1, _save_def2,
        RollbackGainMetric,
    )

    theta0   = snapshot_params(model_at_theta0)
    eval_fn  = build_glue_eval_fn(tokenizer, "mnli", "/path/to/glue")
    # 加载 theta_T 模型...
    result   = RollbackGainMetric().compute(theta0, model_T, eval_fn, device)
    metric.save(result, save_dir, compute_loss=True, compute_acc=True)

─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .base import EvalResult
from .def12_rollback import RollbackGainMetric
from .def3_path_integral import PathIntegralGainMetric

# 全局注册表：{指标名 → 类}
REGISTRY: Dict[str, type] = {
    "def1": RollbackGainMetric,
    "def2": RollbackGainMetric,
    "def3": PathIntegralGainMetric,
}


class TrainingGainRunner:
    """
    训练收益 G_m 组合运行器：选择一种或多种定义，统一管理 callbacks。

    · 每种指标独立保存 JSON 文件，互不干扰。
    · def1 和 def2 同时选中时，自动合并为一个 RollbackCallback（共用前向），
      节省计算量（相比分别计算减少约 50% 评估次数）。
    · def3 独立创建 PathIntegralCallback，与 def1/def2 并行运行。
    · head_granularity 为全局开关，也可在 metric_kwargs 中单独覆盖。

    Args:
        metrics:          指标名列表，可为 ["def1"], ["def2"], ["def3"],
                          ["def1", "def2"], ["def1", "def2", "def3"] 等任意组合
        head_granularity: 全局注意力头级别粒度开关（默认 False）
    """

    def __init__(
        self,
        metrics:          List[str],
        head_granularity: bool = False,
    ):
        unknown = set(metrics) - set(REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. Available: {sorted(REGISTRY)}"
            )
        self._metrics         = list(dict.fromkeys(metrics))  # 去重，保留顺序
        self._head_granularity = head_granularity

    @classmethod
    def from_str(
        cls,
        metrics_str:      str,
        head_granularity: bool = False,
    ) -> "TrainingGainRunner":
        """
        从逗号分隔的字符串构造，便于命令行传参。

        例：
            TrainingGainRunner.from_str("def1,def2,def3", head_granularity=True)
            TrainingGainRunner.from_str("def3")
        """
        names = [m.strip() for m in metrics_str.split(",") if m.strip()]
        return cls(metrics=names, head_granularity=head_granularity)

    def make_callbacks(
        self,
        model:         torch.nn.Module,
        save_dir:      str,
        eval_fn:       Optional[Callable[[torch.nn.Module, torch.device], EvalResult]] = None,
        metric_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> List:
        """
        为所有选定指标创建 TrainerCallback 列表。

        ⚠️  必须在 Trainer 创建前调用：
            此时 model 尚未被 DDP 包装，def1/def2 立即记录 θ^(0) 快照，
            def3 注册梯度 hook。

        合并逻辑：
            def1 和 def2 同时存在时，创建一个 RollbackCallback，compute_loss=True, compute_acc=True，
            既节省计算量，又分别保存 def1 和 def2 文件。

        Args:
            model:         未经 DDP 包装的原始模型
            save_dir:      结果保存目录（每个指标保存独立 JSON 文件）
            eval_fn:       验证集评估函数，def1/def2 必须提供
                           （由 build_glue_eval_fn 构建）
            metric_kwargs: 各指标的超参覆盖字典，格式：
                           {"def3": {"log_every": 10}, "def1": {"module_names": [...]}}

        Returns:
            TrainerCallback 列表，直接追加到 Trainer 的 callbacks 参数中
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        metric_kwargs = metric_kwargs or {}
        callbacks = []

        use_def1 = "def1" in self._metrics
        use_def2 = "def2" in self._metrics
        use_def3 = "def3" in self._metrics

        # ── def1 / def2：回滚指标（自动合并） ─────────────────────────────
        if use_def1 or use_def2:
            if eval_fn is None:
                raise ValueError(
                    "def1/def2 需要 eval_fn，请使用 build_glue_eval_fn() 构建后传入。"
                )
            metric = RollbackGainMetric()
            # 合并 def1/def2 的 metric_kwargs（def1 优先，def2 补充）
            kw: Dict[str, Any] = {}
            kw.update(metric_kwargs.get("def2", {}))  # def2 先
            kw.update(metric_kwargs.get("def1", {}))  # def1 覆盖（通常两者参数相同）
            kw.setdefault("head_granularity", self._head_granularity)

            cb = metric.make_callback(
                model=model,
                save_dir=save_dir,
                eval_fn=eval_fn,
                compute_loss=use_def1,
                compute_acc=use_def2,
                **kw,
            )
            callbacks.append(cb)
            combined = "+".join(
                [x for x, flag in [("def1", use_def1), ("def2", use_def2)] if flag]
            )
            head_info = " [+head]" if kw.get("head_granularity") else ""
            print(f"[TrainingGainRunner] 已注册 callback: {combined}{head_info}"
                  f"（共用一次前向传播）")

        # ── def3：路径积分 ───────────────────────────────────────────────────
        if use_def3:
            kw3: Dict[str, Any] = dict(metric_kwargs.get("def3", {}))
            kw3.setdefault("head_granularity", self._head_granularity)
            metric3 = PathIntegralGainMetric()
            cb3     = metric3.make_callback(model=model, save_dir=save_dir, **kw3)
            callbacks.append(cb3)
            head_info = " [+head]" if kw3.get("head_granularity") else ""
            print(f"[TrainingGainRunner] 已注册 callback: def3{head_info}")

        print(
            f"[TrainingGainRunner] 共 {len(callbacks)} 个 callback，"
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
        return list(self._metrics)

    @property
    def head_granularity(self) -> bool:
        return self._head_granularity
