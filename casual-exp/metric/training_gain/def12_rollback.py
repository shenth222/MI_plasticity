"""
metric/training_gain/def12_rollback.py

定义一：回滚模块/头参数，测试 Validation Loss 变化
定义二：回滚模块/头参数，测试 Validation Accuracy 变化
─────────────────────────────────────────────────────────────────────────────
两种定义合并实现（共用一次前向传播），分别保存结果文件：
  def1_rollback_loss.json
  def2_rollback_acc.json

公式：
  模块级别（m = 叶模块）：
      G_m^(loss) = L_val(θ^(T)[m ← θ_m^(0)]) − L_val(θ^(T))
      G_m^(acc)  = Acc_val(θ^(T)[m ← θ_m^(0)]) − Acc_val(θ^(T))

  头级别（head_granularity=True，h = 注意力头）：
      G_h^(loss) = L_val(θ^(T)[h ← θ_h^(0)]) − L_val(θ^(T))
      G_h^(acc)  = Acc_val(θ^(T)[h ← θ_h^(0)]) − Acc_val(θ^(T))

符号含义：
  · θ^(T)[m ← θ_m^(0)]：将模块 m 参数替换为初始值，其余保持训练后状态
  · G_m^(loss) > 0：回滚后 loss 升高 → 模块 m 的训练有正贡献
  · G_m^(acc)  < 0：回滚后 acc 降低  → 模块 m 的训练有正贡献

eval_fn 接口：
  eval_fn: Callable[[nn.Module, torch.device], EvalResult]
  · 验证集数据在 build_glue_eval_fn 内预先加载，eval_fn 调用时直接推理
  · build_glue_eval_fn 一次性加载，后续重复调用无 IO 开销

─────────────────────────────────────────────────────────────────────────────
保存格式：

def1_rollback_loss.json
{
  "baseline_loss":  float,
  "module_scores":  {module_name: G_m_loss, ...},
  "head_scores":    {module_name: {"head_0": G_h_loss, ...}, ...},  # head_granularity=True
  "num_modules_computed": int,
  "num_attn_modules_computed": int                                   # head_granularity=True
}

def2_rollback_acc.json
{
  "primary_metric":         str,    # 主指标名，如 "accuracy"
  "baseline_acc":           float,
  "baseline_all_metrics":   {metric_name: float, ...},
  "module_scores":          {module_name: G_m_acc, ...},   # 主指标变化
  "module_all_metric_scores": {module_name: {metric_name: delta, ...}, ...},
  "head_scores":            {module_name: {"head_0": G_h_acc, ...}, ...},  # head_granularity=True
  "num_modules_computed":   int
}
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .base import (
    EvalResult,
    RollbackMetric,
    TrainingGainBase,
    group_params_by_module,
    resolve_named_modules,
    resolve_param_dict,
    snapshot_params,
)
from .attn_head import (
    AttnHeadConfig,
    get_attn_head_config,
    get_attn_modules,
    rollback_head_context,
    rollback_module_context,
)


# ---------------------------------------------------------------------------
# 验证集评估函数构建（GLUE 任务）
# ---------------------------------------------------------------------------

def build_glue_eval_fn(
    tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    dataset_path: str,
    max_length: int = 256,
    batch_size: int = 32,
) -> Callable[[nn.Module, torch.device], EvalResult]:
    """
    构建 GLUE 任务的验证集评估函数。

    数据集仅加载一次，返回的 eval_fn 可多次调用（无额外 IO 开销）。
    计算内容：平均交叉熵 loss + 任务主指标（如 accuracy）及全部指标。

    Args:
        tokenizer:    分词器
        task_name:    GLUE 任务名，如 "mnli", "rte"
        dataset_path: 本地 GLUE 数据集根目录
        max_length:   最大序列长度
        batch_size:   评估 batch 大小

    Returns:
        eval_fn: (model, device) → EvalResult
    """
    from utils.evaluate import (
        GLUE_TASK_CONFIGS,
        load_glue_eval_dataset,
        get_compute_metrics_fn,
        _collate_fn,
    )

    task = task_name.lower()
    cfg  = GLUE_TASK_CONFIGS[task]
    is_regression      = cfg["is_regression"]
    primary_metric_key = cfg["metric_names"][0]

    data     = load_glue_eval_dataset(task_name, tokenizer, dataset_path, max_length)
    datasets = data["datasets"]
    compute_metrics_fn = get_compute_metrics_fn(task_name)

    def eval_fn(model: nn.Module, device: torch.device) -> EvalResult:
        model.eval()
        total_loss   = 0.0
        total_batches = 0
        all_preds    = []
        all_labels   = []

        with torch.no_grad():
            for split_name, dataset in datasets.items():
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda b: _collate_fn(b, tokenizer),
                )
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    if out.loss is not None:
                        total_loss    += out.loss.item()
                        total_batches += 1
                    if is_regression:
                        all_preds.append(out.logits.squeeze(-1).cpu().float().numpy())
                    else:
                        all_preds.append(out.logits.cpu().float().numpy())
                    all_labels.append(batch["labels"].cpu().numpy())

        avg_loss = total_loss / max(total_batches, 1)
        preds    = np.concatenate(all_preds,  axis=0)
        labels   = np.concatenate(all_labels, axis=0)
        metrics  = compute_metrics_fn((preds, labels))

        # MNLI：evaluate_glue 为 matched/mismatched 输出 accuracy_matched/mismatched，
        # 同时合并为 "accuracy" 作为主指标；若仅有一个 split，直接取 "accuracy"
        primary_val = metrics.get(primary_metric_key, metrics.get("accuracy", 0.0))

        return EvalResult(
            avg_loss=avg_loss,
            primary_metric_name=primary_metric_key,
            primary_metric_value=primary_val,
            all_metrics=metrics,
        )

    return eval_fn


# ---------------------------------------------------------------------------
# 核心计算函数（纯函数，独立可测）
# ---------------------------------------------------------------------------

def compute_rollback(
    theta0:           Dict[str, torch.Tensor],
    model:            nn.Module,
    eval_fn:          Callable[[nn.Module, torch.device], EvalResult],
    device:           torch.device,
    module_groups:    Dict[str, List[str]],
    head_granularity: bool = False,
    compute_loss:     bool = True,
    compute_acc:      bool = True,
    module_names:     Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    回滚参数并评估核心计算函数。

    计算流程：
      1. 计算 baseline（无回滚的验证集 loss 和 acc）
      2. 逐模块回滚，评估，计算 G_m^(loss) 和 G_m^(acc)
      3. （可选）逐注意力头回滚，评估，计算 G_h^(loss) 和 G_h^(acc)

    Args:
        theta0:           微调前参数快照 {param_name: tensor_cpu}
        model:            当前模型（θ^(T) 状态）
        eval_fn:          验证集评估函数 (model, device) → EvalResult
        device:           计算设备
        module_groups:    {module_name: [param_name, ...]}（叶模块分组）
        head_granularity: 是否额外计算注意力头级别分数
        compute_loss:     是否计算 val loss 变化（def1）
        compute_acc:      是否计算 val acc 变化（def2）
        module_names:     若指定，仅计算这些模块（None 表示全部叶模块）

    Returns:
        包含 baseline 信息、module_scores、head_scores（可选）的完整字典
    """
    model = model.to(device)

    # ── 1. 计算 baseline ──────────────────────────────────────────────────
    print("[rollback] 计算 baseline（无回滚）...")
    baseline = eval_fn(model, device)
    print(f"  baseline loss={baseline.avg_loss:.4f}  "
          f"{baseline.primary_metric_name}={baseline.primary_metric_value:.4f}")

    # ── 2. 模块级别回滚 ───────────────────────────────────────────────────
    names_to_compute = module_names if module_names is not None else list(module_groups.keys())
    total = len(names_to_compute)

    module_scores_loss: Dict[str, float] = {}
    module_scores_acc:  Dict[str, float] = {}
    module_all_metrics: Dict[str, Dict[str, float]] = {}

    print(f"[rollback] 开始模块级别回滚，共 {total} 个模块...")
    for i, m_name in enumerate(names_to_compute, 1):
        param_names = module_groups.get(m_name, [])
        if not param_names:
            continue

        with rollback_module_context(model, theta0, param_names):
            res = eval_fn(model, device)

        if compute_loss:
            module_scores_loss[m_name] = res.avg_loss - baseline.avg_loss
        if compute_acc:
            module_scores_acc[m_name] = res.primary_metric_value - baseline.primary_metric_value
            module_all_metrics[m_name] = {
                k: res.all_metrics.get(k, 0.0) - baseline.all_metrics.get(k, 0.0)
                for k in baseline.all_metrics
            }

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] {m_name}  "
                  + (f"ΔL={module_scores_loss.get(m_name, 0.0):+.4f}  " if compute_loss else "")
                  + (f"ΔAcc={module_scores_acc.get(m_name, 0.0):+.4f}" if compute_acc else ""))

    # ── 3. 头级别回滚 ─────────────────────────────────────────────────────
    head_scores_loss: Dict[str, Dict[str, float]] = {}
    head_scores_acc:  Dict[str, Dict[str, float]] = {}
    num_attn_mods_computed = 0

    if head_granularity:
        attn_cfg = get_attn_head_config(model)
        if attn_cfg is None:
            print("[rollback] head_granularity=True 但模型无 config，跳过头级别计算")
        else:
            attn_mods = get_attn_modules(model, attn_cfg)
            num_attn_mods_computed = len(attn_mods)
            print(f"[rollback] 开始头级别回滚，"
                  f"{len(attn_mods)} 个注意力模块 × {attn_cfg.num_heads} 头 = "
                  f"{len(attn_mods) * attn_cfg.num_heads} 次评估...")

            for mi, (m_name, m_type) in enumerate(attn_mods.items(), 1):
                head_scores_loss[m_name] = {}
                head_scores_acc[m_name]  = {}

                for h in range(attn_cfg.num_heads):
                    with rollback_head_context(
                        model, theta0, m_name, m_type, h, attn_cfg.head_dim
                    ):
                        res = eval_fn(model, device)

                    if compute_loss:
                        head_scores_loss[m_name][f"head_{h}"] = (
                            res.avg_loss - baseline.avg_loss
                        )
                    if compute_acc:
                        head_scores_acc[m_name][f"head_{h}"] = (
                            res.primary_metric_value - baseline.primary_metric_value
                        )

                print(f"  [{mi}/{len(attn_mods)}] {m_name} 完成"
                      f"（{attn_cfg.num_heads} 头）")

    result: Dict[str, Any] = {
        "baseline_loss":          baseline.avg_loss,
        "baseline_acc":           baseline.primary_metric_value,
        "baseline_all_metrics":   baseline.all_metrics,
        "primary_metric":         baseline.primary_metric_name,
        "module_scores_loss":     module_scores_loss,
        "module_scores_acc":      module_scores_acc,
        "module_all_metric_scores": module_all_metrics,
        "head_scores_loss":       head_scores_loss if head_granularity else {},
        "head_scores_acc":        head_scores_acc  if head_granularity else {},
        "num_modules_computed":   len(names_to_compute),
        "num_attn_modules_computed": num_attn_mods_computed,
    }
    return result


# ---------------------------------------------------------------------------
# JSON 保存辅助函数
# ---------------------------------------------------------------------------

def _save_def1(scores: Dict[str, Any], save_dir: str) -> Path:
    """将 compute_rollback 结果中的 def1（loss 变化）部分保存为 JSON。"""
    d: Dict[str, Any] = {
        "baseline_loss":         scores["baseline_loss"],
        "module_scores":         scores["module_scores_loss"],
        "num_modules_computed":  scores["num_modules_computed"],
    }
    if scores.get("head_scores_loss"):
        d["head_scores"]              = scores["head_scores_loss"]
        d["num_attn_modules_computed"] = scores["num_attn_modules_computed"]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "def1_rollback_loss.json"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[def1_rollback_loss] Saved → {path}")
    return path


def _save_def2(scores: Dict[str, Any], save_dir: str) -> Path:
    """将 compute_rollback 结果中的 def2（accuracy 变化）部分保存为 JSON。"""
    d: Dict[str, Any] = {
        "primary_metric":           scores["primary_metric"],
        "baseline_acc":             scores["baseline_acc"],
        "baseline_all_metrics":     scores["baseline_all_metrics"],
        "module_scores":            scores["module_scores_acc"],
        "module_all_metric_scores": scores["module_all_metric_scores"],
        "num_modules_computed":     scores["num_modules_computed"],
    }
    if scores.get("head_scores_acc"):
        d["head_scores"]              = scores["head_scores_acc"]
        d["num_attn_modules_computed"] = scores["num_attn_modules_computed"]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "def2_rollback_acc.json"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[def2_rollback_acc] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class RollbackCallback(TrainerCallback):
    """
    在 Trainer 创建前立即记录 θ^(0) 快照，
    训练结束时（on_train_end）逐模块/头回滚并评估，保存 JSON。

    设计：
      · def1 和 def2 共用一次前向传播（同一次回滚评估同时记录 loss 和 acc）
      · 若仅需 def1，设 compute_acc=False；仅需 def2，设 compute_loss=False
      · 通过 compute_loss / compute_acc 控制保存哪些文件
    """

    def __init__(
        self,
        model:            nn.Module,
        eval_fn:          Callable[[nn.Module, torch.device], EvalResult],
        save_dir:         str,
        head_granularity: bool = False,
        compute_loss:     bool = True,
        compute_acc:      bool = True,
        module_names:     Optional[List[str]] = None,
    ):
        """
        Args:
            model:            未经 DDP 包装的原始模型
            eval_fn:          验证集评估函数，由 build_glue_eval_fn 构建
            save_dir:         结果保存目录
            head_granularity: 是否额外计算注意力头级别分数
            compute_loss:     是否计算 val loss 变化（def1）
            compute_acc:      是否计算 val acc 变化（def2）
            module_names:     若指定，仅计算这些模块（None = 全部叶模块）
        """
        self._eval_fn          = eval_fn
        self._save_dir         = save_dir
        self._head_granularity = head_granularity
        self._compute_loss     = compute_loss
        self._compute_acc      = compute_acc
        self._module_names     = module_names
        self._module_groups    = group_params_by_module(model)
        self._model_ref        = model  # 保留原始 model 引用，用于提取 attn config

        # 立即记录 θ^(0) 快照（必须在 DDP 包装前）
        self._theta0 = snapshot_params(model)

        defs_str = "+".join(
            [x for x, flag in [("def1_loss", compute_loss), ("def2_acc", compute_acc)] if flag]
        )
        print(
            f"[RollbackCallback] θ^(0) 已快照（{len(self._theta0)} 个参数）；"
            f"计算: {defs_str}"
            + ("；含头级别粒度" if head_granularity else "")
        )

    def on_train_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        model:   nn.Module = None,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero or model is None:
            return control

        device = next(model.parameters()).device
        scores = compute_rollback(
            theta0=self._theta0,
            model=model,
            eval_fn=self._eval_fn,
            device=device,
            module_groups=self._module_groups,
            head_granularity=self._head_granularity,
            compute_loss=self._compute_loss,
            compute_acc=self._compute_acc,
            module_names=self._module_names,
        )

        if self._compute_loss:
            _save_def1(scores, self._save_dir)
        if self._compute_acc:
            _save_def2(scores, self._save_dir)

        return control


# ---------------------------------------------------------------------------
# RollbackGainMetric 包装类
# ---------------------------------------------------------------------------

class RollbackGainMetric(RollbackMetric):
    """
    回滚训练收益（定义一和定义二）——RollbackMetric 包装类。

    同时计算（共用一次前向）：
      · module_scores_loss  G_m^(loss)（def1）
      · module_scores_acc   G_m^(acc) （def2）
      · head_scores_loss    头级别 G_h^(loss)（仅 head_granularity=True）
      · head_scores_acc     头级别 G_h^(acc) （仅 head_granularity=True）

    可通过 compute_loss / compute_acc 控制仅保存 def1 或 def2。
    """

    name = "rollback"  # 注：实际文件名为 def1_rollback_loss.json / def2_rollback_acc.json

    def compute(
        self,
        theta0:           Dict[str, torch.Tensor],
        model:            nn.Module,
        eval_fn:          Callable[[nn.Module, torch.device], EvalResult],
        device:           torch.device,
        head_granularity: bool = False,
        compute_loss:     bool = True,
        compute_acc:      bool = True,
        module_names:     Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        独立测试接口。

        Args:
            theta0:           微调前参数快照
            model:            微调后的模型（θ^(T) 状态）
            eval_fn:          验证集评估函数
            device:           计算设备
            head_granularity: 是否额外计算注意力头级别分数
            compute_loss:     是否计算 val loss 变化（def1）
            compute_acc:      是否计算 val acc 变化（def2）
            module_names:     若指定，仅计算这些模块
        """
        module_groups = group_params_by_module(model)
        return compute_rollback(
            theta0=theta0,
            model=model,
            eval_fn=eval_fn,
            device=device,
            module_groups=module_groups,
            head_granularity=head_granularity,
            compute_loss=compute_loss,
            compute_acc=compute_acc,
            module_names=module_names,
        )

    def save(
        self,
        scores:      Dict[str, Any],
        save_dir:    str,
        compute_loss: bool = True,
        compute_acc:  bool = True,
    ) -> Dict[str, Path]:
        """
        将 compute() 的结果分别保存为 def1_rollback_loss.json 和/或 def2_rollback_acc.json。

        Returns:
            {"def1": Path, "def2": Path}（仅包含实际保存的文件）
        """
        saved: Dict[str, Path] = {}
        if compute_loss:
            saved["def1"] = _save_def1(scores, save_dir)
        if compute_acc:
            saved["def2"] = _save_def2(scores, save_dir)
        return saved

    def make_callback(
        self,
        model:            nn.Module,
        save_dir:         str,
        eval_fn:          Callable[[nn.Module, torch.device], EvalResult] = None,
        head_granularity: bool = False,
        compute_loss:     bool = True,
        compute_acc:      bool = True,
        module_names:     Optional[List[str]] = None,
        **kwargs,
    ) -> RollbackCallback:
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:            未经 DDP 包装的原始模型
            save_dir:         结果保存目录
            eval_fn:          验证集评估函数（由 build_glue_eval_fn 构建）
            head_granularity: 是否额外计算注意力头级别分数
            compute_loss:     是否计算 val loss 变化（def1）
            compute_acc:      是否计算 val acc 变化（def2）
            module_names:     若指定，仅计算这些模块
        """
        if eval_fn is None:
            raise ValueError(
                "eval_fn 不能为 None，请使用 build_glue_eval_fn() 构建后传入。"
            )
        return RollbackCallback(
            model=model,
            eval_fn=eval_fn,
            save_dir=save_dir,
            head_granularity=head_granularity,
            compute_loss=compute_loss,
            compute_acc=compute_acc,
            module_names=module_names,
        )
