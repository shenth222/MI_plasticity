"""
fix-budget/train/finetune_fixed_budget.py

固定预算微调：在注意力头粒度限制可训练参数数量。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心特性：
  1. 固定预算：全模型共 m 个注意力头，训练时仅激活 budget 个头的梯度。
  2. 三种选择策略：
       random          — 随机选择
       pre_importance  — 按训练前重要性指标排序（fisher/saliency/perturbation/
                          singular_value/spectral_entropy）
       update_response — 按更新响应指标排序（def1/def2/def3/def4）
  3. 每 epoch 评估 MNLI matched/mismatched 准确率。
  4. 支持每隔 n 步重新排序选择，并保存每次选择记录（JSON）。
  5. 非注意力参数（FFN / LayerNorm / embeddings 等）全程冻结。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用示例（直接运行）：
  python fix-budget/train/finetune_fixed_budget.py \
      --task MNLI --bsz 64 --epochs 10 \
      --budget_ratio 0.1 \
      --selection_strategy random \
      --out_dir fix-budget/outputs/random_r01_seed1

  # pre_importance / fisher
  python fix-budget/train/finetune_fixed_budget.py \
      --task MNLI --bsz 64 \
      --budget_ratio 0.1 \
      --selection_strategy pre_importance \
      --pre_importance_metric fisher \
      --pre_importance_num_batches 32 \
      --reselect_every 500 \
      --out_dir fix-budget/outputs/fisher_r01_resel500

  # update_response / def2
  python fix-budget/train/finetune_fixed_budget.py \
      --task MNLI --bsz 64 \
      --budget_ratio 0.1 \
      --selection_strategy update_response \
      --ur_metric def2 \
      --ur_num_batches 32 \
      --out_dir fix-budget/outputs/def2_r01

  # update_response / def3（特殊训练流：前 T_early 步全参，之后施加预算）
  python fix-budget/train/finetune_fixed_budget.py \
      --task MNLI --bsz 64 \
      --budget_ratio 0.1 \
      --selection_strategy update_response \
      --ur_metric def3 \
      --ur_T_early 200 \
      --out_dir fix-budget/outputs/def3_r01

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def3 说明：
  def3 使用真实训练步骤的梯度来估计更新响应，属于训练中指标。
  训练流程分两阶段：
    阶段 1（前 T_early 步）：所有注意力头均激活，EarlyGradNormCallback
                               在梯度钩子中累积各头的梯度范数之和。
    阶段 2（T_early 步后）：根据累积的头级别分数施加预算选择，
                              关闭非选中头的梯度，继续训练。
  def3 不支持重新选择（--reselect_every 对 def3 无效）。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import wandb
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ── 路径设置 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR     = Path(__file__).parent            # fix-budget/train/
_FIX_BUDGET_DIR = _SCRIPT_DIR.parent               # fix-budget/
_PROJECT_ROOT   = _FIX_BUDGET_DIR.parent           # casual-exp/

# 将项目根和 fix-budget/ 加入搜索路径
for _p in [str(_PROJECT_ROOT), str(_FIX_BUDGET_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 项目内导入 ───────────────────────────────────────────────────────────────
from data.glue import load_glue_dataset
from utils.evaluate import evaluate_glue
from metric.update_response.def3_early_grad_norm import EarlyGradNormCallback

from selection.base import HeadSelection, HeadSelector
from selection.gradient_masker import GradientMasker
from selection.head_utils import get_all_conceptual_heads_from_model
from selection.random_selector import RandomSelector
from selection.pre_importance_selector import PreImportanceSelector
from selection.update_response_selector import UpdateResponseSelector


# ─────────────────────────────────────────────────────────────────────────────
# 评估 Callback（与 baseline 一致）
# ─────────────────────────────────────────────────────────────────────────────

class GlueEvalCallback(TrainerCallback):
    """每个 epoch 结束时在验证集上评估，追踪并保存最优模型。"""

    def __init__(
        self,
        tokenizer,
        task: str,
        dataset_path: str,
        max_length: int,
        batch_size: int,
        metric_for_best_model: str,
        out_dir: str,
    ):
        self.tokenizer             = tokenizer
        self.task                  = task
        self.dataset_path          = dataset_path
        self.max_length            = max_length
        self.batch_size            = batch_size
        self.metric_for_best_model = metric_for_best_model
        self.best_metric           = None
        self.best_ckpt             = os.path.join(out_dir, "ckpt_best")
        self._epoch_results: List[Dict] = []

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero:
            return control
        if model is None:
            return control

        device = next(model.parameters()).device
        results = evaluate_glue(
            model=model,
            tokenizer=self.tokenizer,
            task_name=self.task,
            dataset_path=self.dataset_path,
            max_length=self.max_length,
            batch_size=self.batch_size,
            device=device,
        )

        # 仅在主进程上打日志
        if wandb.run is not None:
            log_dict = {f"eval/{k}": v for k, v in results.items()}
            wandb.log(log_dict, step=state.global_step)

        epoch_log = {"epoch": state.epoch, "step": state.global_step, **results}
        self._epoch_results.append(epoch_log)
        print(f"\n[Eval] epoch={state.epoch:.1f}  step={state.global_step}  {results}")

        key = self.metric_for_best_model
        val = results.get(key)
        if val is not None and (self.best_metric is None or val > self.best_metric):
            self.best_metric = val
            model.save_pretrained(self.best_ckpt)
            self.tokenizer.save_pretrained(self.best_ckpt)
            print(f"[Eval] New best {key}={val:.4f}, saved → {self.best_ckpt}")

        model.train()
        return control

    def save_history(self, save_dir: str) -> None:
        """将完整评估历史保存到 JSON。"""
        fpath = os.path.join(save_dir, "eval_history.json")
        with open(fpath, "w") as f:
            json.dump(self._epoch_results, f, indent=2)
        print(f"[GlueEvalCallback] 评估历史已保存 → {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 固定预算核心 Callback
# ─────────────────────────────────────────────────────────────────────────────

class FixedBudgetCallback(TrainerCallback):
    """
    固定预算训练核心 Callback。

    职责：
      1. 管理 GradientMasker，控制哪些注意力头的梯度被保留。
      2. 若 reselect_every > 0，则每隔 reselect_every 步重新计算指标并更新遮蔽。
      3. 记录每次选择结果（JSON + 控制台表格打印）。

    与 def3 集成：
      使用 def3 时，masker 初始为 None（所有注意力头均活跃），
      待 Def3PhaseCallback 在 T_early 步后调用 apply_initial_selection() 方能激活遮蔽。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        selector: HeadSelector,
        dataloader: DataLoader,
        budget_count: int,
        reselect_every: int,
        save_dir: str,
        device: torch.device,
    ):
        self._model          = model
        self._selector       = selector
        self._dataloader     = dataloader
        self._budget_count   = budget_count
        self._reselect_every = reselect_every
        self._save_dir       = save_dir
        self._device         = device

        self._masker: Optional[GradientMasker]            = None
        self._selection_log: List[Dict[str, Any]]         = []
        self._active_selections: Dict[str, HeadSelection] = {}
        self._current_selected_set: Optional[Set]         = None   # 断点续跑用

        os.makedirs(os.path.join(save_dir, "selections"), exist_ok=True)

    def apply_initial_selection(
        self,
        selections: Dict[str, HeadSelection],
    ) -> None:
        """
        根据给定的选择结果初始化/更新 GradientMasker。

        对多变体情形（如 saliency 有 grad_norm + taylor）：
          取第一个变体的选中集合作为激活的梯度遮蔽方案。
        """
        if not selections:
            print("[FixedBudgetCallback] 警告：selections 为空，跳过初始化。")
            return

        # 打印所有变体的排序结果
        for variant_name, sel in selections.items():
            print(f"\n[FixedBudgetCallback] 变体: {variant_name}")
            sel.print_table()
            fpath = sel.save(os.path.join(self._save_dir, "selections"))
            self._selection_log.append(sel.to_dict())
            print(f"[FixedBudgetCallback] 选择记录已保存 → {fpath}")

        # 使用第一个变体驱动梯度遮蔽
        primary_variant = next(iter(selections))
        primary_sel     = selections[primary_variant]
        self._active_selections    = selections
        self._current_selected_set = primary_sel.selected_set  # 断点续跑：记录当前选择

        if self._masker is None:
            self._masker = GradientMasker(self._model, primary_sel.selected_set)
        else:
            self._masker.update_selection(primary_sel.selected_set)

        print(
            f"[FixedBudgetCallback] 初始选择完成。"
            f" 主变体={primary_variant}  {primary_sel.summary_str()}"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # 周期性重新选择
        if (
            self._reselect_every > 0
            and self._masker is not None          # masker 已初始化（排除 def3 等待阶段）
            and state.global_step > 0
            and state.global_step % self._reselect_every == 0
        ):
            print(
                f"\n[FixedBudgetCallback] Step={state.global_step}: "
                f"触发周期性重新选择 (reselect_every={self._reselect_every})..."
            )
            self._do_reselect(step=state.global_step)

        return control

    def _do_reselect(self, step: int) -> None:
        """执行重新选择：计算 metric → 更新 mask。"""
        if not self._selector.supports_reselect():
            print(
                f"[FixedBudgetCallback] {self._selector.selector_name} "
                f"不支持重新选择，跳过。"
            )
            return

        print(f"[FixedBudgetCallback] 计算 {self._selector.selector_name} 分数...")
        self._model.eval()
        try:
            selections = self._selector.select(
                model=self._model,
                dataloader=self._dataloader,
                device=self._device,
                budget_count=self._budget_count,
                step=step,
            )
        finally:
            self._model.train()

        self.apply_initial_selection(selections)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # 清理梯度 hooks
        if self._masker:
            self._masker.remove_hooks()

        # 保存完整选择日志
        log_path = os.path.join(self._save_dir, "selection_log.json")
        with open(log_path, "w") as f:
            json.dump(self._selection_log, f, indent=2)
        print(f"[FixedBudgetCallback] 完整选择日志已保存 → {log_path}")

        return control


# ─────────────────────────────────────────────────────────────────────────────
# def3 专用 Callback：前 T_early 步收集梯度，之后激活预算限制
# ─────────────────────────────────────────────────────────────────────────────

class Def3PhaseCallback(TrainerCallback):
    """
    def3（累积早期梯度范数）训练流专用 Callback。

    阶段 1（前 T_early 步）：
      - 不施加预算遮蔽，所有注意力头均活跃（GradientMasker.masker is None）。
      - 与 EarlyGradNormCallback 配合收集各头的梯度范数。
    阶段 2（第 T_early 步开始）：
      - 从 EarlyGradNormCallback 的内部状态读取 head_scores。
      - 调用 UpdateResponseSelector.set_def3_scores() 注入分数。
      - 调用 FixedBudgetCallback.apply_initial_selection() 激活预算遮蔽。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        selector: UpdateResponseSelector,
        budget_callback: FixedBudgetCallback,
        early_grad_cb: EarlyGradNormCallback,
        T_early: int,
    ):
        self._model          = model
        self._selector       = selector
        self._budget_cb      = budget_callback
        self._early_grad_cb  = early_grad_cb
        self._T_early        = T_early
        self._triggered      = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if self._triggered:
            return control

        # EarlyGradNormCallback 在 step == T_early 时完成
        if state.global_step >= self._T_early and self._early_grad_cb._done:
            self._triggered = True
            print(
                f"\n[Def3PhaseCallback] Step={state.global_step}: "
                f"def3 收集完成，开始施加预算选择..."
            )

            # 提取 head_scores（{module_name: {"head_h": float, ...}}）
            cb = self._early_grad_cb
            if cb._attn_cfg is not None and cb._head_acc:
                head_scores = {
                    m_name: {f"head_{h}": v for h, v in heads.items()}
                    for m_name, heads in cb._head_acc.items()
                }
            else:
                print(
                    "[Def3PhaseCallback] 警告：EarlyGradNormCallback 未收集到头级别分数，"
                    "请确认 head_granularity=True。退出 def3 选择。"
                )
                return control

            # 注入分数 → 选择 → 激活遮蔽
            self._selector.set_def3_scores(head_scores)
            device = next(self._model.parameters()).device
            self._model.eval()
            try:
                selections = self._selector.select(
                    model=self._model,
                    dataloader=None,
                    device=device,
                    budget_count=self._budget_cb._budget_count,
                    step=state.global_step,
                )
            finally:
                self._model.train()

            self._budget_cb.apply_initial_selection(selections)
            print(
                f"[Def3PhaseCallback] 预算遮蔽已激活，"
                f"阶段 2 正式开始（剩余训练步骤：预算限制生效）。"
            )

        return control


# ─────────────────────────────────────────────────────────────────────────────
# 断点续跑：额外状态随检查点一同保存的 Callback
# ─────────────────────────────────────────────────────────────────────────────

class ExtraStateCheckpointCallback(TrainerCallback):
    """
    在每次 Trainer 保存检查点时，同步将额外训练状态写入同一目录，
    以支持断点续跑时完整恢复训练现场。

    保存内容（training_extra_state.json）：
      - global_step / epoch          训练进度
      - selection_log                完整头选择记录
      - eval_epoch_results           逐 epoch 评估历史
      - best_metric                  当前最优评估指标值
      - active_head_set              当前活跃头集合（GradientMasker 状态）
      - def3_triggered               def3 阶段是否已切换
      - def3_head_acc                def3 阶段 1 累积梯度数据（未完成时保存）
      - def3_steps_collected         def3 已收集步数
    """

    def __init__(
        self,
        budget_callback: "FixedBudgetCallback",
        eval_callback: "GlueEvalCallback",
        def3_phase_cb: Optional["Def3PhaseCallback"] = None,
        early_grad_cb=None,
    ):
        self._budget_cb     = budget_callback
        self._eval_cb       = eval_callback
        self._def3_phase_cb = def3_phase_cb
        self._early_grad_cb = early_grad_cb

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 序列化活跃头集合 Set[Tuple[str, int]] → List[[str, int]]
        active_head_set = None
        if self._budget_cb._current_selected_set is not None:
            active_head_set = [
                [lk, hi] for lk, hi in self._budget_cb._current_selected_set
            ]

        # def3 阶段状态
        def3_triggered       = None
        def3_head_acc        = None
        def3_steps_collected = None
        if self._def3_phase_cb is not None:
            def3_triggered = self._def3_phase_cb._triggered
        if self._early_grad_cb is not None:
            ecb = self._early_grad_cb
            def3_steps_collected = ecb._steps_collected
            if not ecb._done:
                # 阶段 1 尚未完成，保存累积梯度数据以便续跑后继续收集
                def3_head_acc = {
                    m: {str(h): float(v) for h, v in heads.items()}
                    for m, heads in ecb._head_acc.items()
                }

        extra_state = {
            "global_step":          state.global_step,
            "epoch":                state.epoch,
            "selection_log":        self._budget_cb._selection_log,
            "eval_epoch_results":   self._eval_cb._epoch_results,
            "best_metric":          self._eval_cb.best_metric,
            "active_head_set":      active_head_set,
            "def3_triggered":       def3_triggered,
            "def3_head_acc":        def3_head_acc,
            "def3_steps_collected": def3_steps_collected,
        }

        fpath = os.path.join(ckpt_dir, "training_extra_state.json")
        with open(fpath, "w") as f:
            json.dump(extra_state, f, indent=2)
        print(f"[ExtraStateCheckpoint] 额外训练状态已保存 → {fpath}")

        return control


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def compute_budget_count(model: torch.nn.Module, budget_ratio: float) -> int:
    """
    根据预算比例计算实际选择的概念头数量。

    Args:
        model        : 完整模型
        budget_ratio : 预算比例（如 0.1 代表选 10% 的概念头）

    Returns:
        budget_count（向上取整，最小为 1）
    """
    all_heads = get_all_conceptual_heads_from_model(model)
    total     = len(all_heads)
    count     = max(1, round(total * budget_ratio))
    print(
        f"[budget] 总概念头数 m={total}，"
        f"预算比例={budget_ratio:.2f}，"
        f"实际预算 budget_count={count}"
    )
    return count


def build_selector(args) -> HeadSelector:
    """根据命令行参数构建对应的头选择器。"""
    strategy = args.selection_strategy

    if strategy == "random":
        return RandomSelector(seed=args.seed)

    if strategy == "pre_importance":
        metric = args.pre_importance_metric
        metric_kwargs: Dict[str, Any] = {}
        if metric in ("fisher", "saliency", "perturbation"):
            metric_kwargs["num_batches"] = args.pre_importance_num_batches
        if metric == "singular_value":
            metric_kwargs["top_k"] = args.pre_importance_top_k
        return PreImportanceSelector(metric=metric, metric_kwargs=metric_kwargs)

    if strategy == "update_response":
        metric = args.ur_metric
        metric_kwargs = {}
        if metric == "def1":
            metric_kwargs["probe_steps"] = args.ur_probe_steps
            metric_kwargs["probe_lr"]    = args.ur_probe_lr or args.lr
        if metric in ("def2", "def4"):
            metric_kwargs["num_batches"] = args.ur_num_batches
            metric_kwargs["epsilon"]     = args.ur_epsilon
        # def3 超参由 EarlyGradNormCallback 处理，不在此设置
        return UpdateResponseSelector(metric=metric, metric_kwargs=metric_kwargs)

    raise ValueError(
        f"未知选择策略: '{strategy}'。"
        f"可选: random / pre_importance / update_response"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="固定预算注意力头微调（GLUE 任务）"
    )

    # ── 通用训练参数（与 baseline 保持一致） ────────────────────────────────
    ap.add_argument("--task",         type=str, required=True,
                    help="GLUE 任务名（如 MNLI / RTE / MRPC）")
    ap.add_argument("--model_name",   type=str,
                    default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir",      type=str, required=True,
                    help="输出目录（checkpoints / 选择记录等）")
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH",
                                           "/data1/shenth/datasets/glue"))
    ap.add_argument("--seed",         type=int, default=1)
    ap.add_argument("--max_len",      type=int, default=256)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--lr_scheduler_type", type=str, default="linear")
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--epochs",       type=float, default=10.0)
    ap.add_argument("--bsz",          type=int, default=64)

    # ── 预算参数 ─────────────────────────────────────────────────────────────
    ap.add_argument("--budget_ratio", type=float, default=0.1,
                    help="预算比例，如 0.1 代表选 10%% 的概念注意力头")
    ap.add_argument("--budget_count", type=int, default=None,
                    help="直接指定预算头数（与 budget_ratio 二选一；此参数优先）")

    # ── 选择策略 ─────────────────────────────────────────────────────────────
    ap.add_argument("--selection_strategy", type=str, default="random",
                    choices=["random", "pre_importance", "update_response"],
                    help="头选择策略")
    ap.add_argument("--reselect_every", type=int, default=0,
                    help="每隔多少训练步重新排序选择（0 = 不重新选择）")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "sum"],
                    help="将各模块头分数聚合为概念头分数的方式")

    # ── pre_importance 参数 ──────────────────────────────────────────────────
    ap.add_argument("--pre_importance_metric", type=str, default="fisher",
                    choices=["fisher", "saliency", "perturbation",
                             "singular_value", "spectral_entropy"])
    ap.add_argument("--pre_importance_num_batches", type=int, default=32,
                    help="fisher/saliency/perturbation 的 Monte Carlo batch 数")
    ap.add_argument("--pre_importance_top_k", type=int, default=32,
                    help="singular_value 的截断奇异值数")

    # ── update_response 参数 ─────────────────────────────────────────────────
    ap.add_argument("--ur_metric", type=str, default="def2",
                    choices=["def1", "def2", "def3", "def4"])
    ap.add_argument("--ur_num_batches",  type=int, default=32,
                    help="def2/def4 的 Monte Carlo batch 数")
    ap.add_argument("--ur_probe_steps",  type=int, default=20,
                    help="def1 的探针训练步数")
    ap.add_argument("--ur_probe_lr",     type=float, default=None,
                    help="def1 的探针 LR（默认同主训练 LR）")
    ap.add_argument("--ur_epsilon",      type=float, default=1e-8,
                    help="def2/def4 的数值稳定项")
    ap.add_argument("--ur_T_early",      type=int, default=200,
                    help="def3 的梯度累积步数（前 T_early 步全参训练）")

    # ── 断点续跑参数 ─────────────────────────────────────────────────────────
    ap.add_argument("--save_steps", type=int, default=500,
                    help="每隔多少训练步保存一次检查点（用于断点续跑，默认 500）")
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="最多保留多少个 Trainer 检查点，旧检查点自动删除（默认 3）")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="指定检查点目录路径以恢复训练；"
                         "留空时自动在 out_dir/trainer_out/ 中查找最新检查点")

    args = ap.parse_args()

    # ── 初始化 ───────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "Fixed-budget"
    torch.manual_seed(args.seed)

    # ── 断点续跑：自动检测或使用用户指定检查点 ──────────────────────────────
    resume_ckpt  = args.resume_from_checkpoint
    resume_state: Dict[str, Any] = {}
    _trainer_out = os.path.join(args.out_dir, "trainer_out")

    if resume_ckpt is None and os.path.isdir(_trainer_out):
        _ckpt_names = sorted(
            [d for d in os.listdir(_trainer_out) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if _ckpt_names:
            resume_ckpt = os.path.join(_trainer_out, _ckpt_names[-1])

    if resume_ckpt:
        _extra_file = os.path.join(resume_ckpt, "training_extra_state.json")
        if os.path.isfile(_extra_file):
            with open(_extra_file) as _f:
                resume_state = json.load(_f)
            print(
                f"[main] ✔ 检测到断点，从 step={resume_state.get('global_step')} "
                f"(epoch≈{resume_state.get('epoch', 0):.2f}) 继续训练。\n"
                f"       检查点路径: {resume_ckpt}"
            )
        else:
            print(
                f"[main] 检测到检查点 {resume_ckpt}，"
                f"但未找到额外状态文件（training_extra_state.json），"
                f"将仅恢复模型权重和优化器状态。"
            )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds  = load_glue_dataset(args.task, tok, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=ds["num_labels"]
    )

    # 保存 θ₀（续跑时跳过，避免覆盖）
    ckpt_init = os.path.join(args.out_dir, "ckpt_init")
    if not os.path.isdir(ckpt_init):
        model.save_pretrained(ckpt_init)
        tok.save_pretrained(ckpt_init)
        print(f"[main] θ₀ 已保存 → {ckpt_init}")
    else:
        print(f"[main] θ₀ 已存在，跳过保存 → {ckpt_init}")

    # ── 计算预算头数 ──────────────────────────────────────────────────────────
    if args.budget_count is not None:
        budget_count = args.budget_count
        print(f"[main] 使用指定预算数 budget_count={budget_count}")
    else:
        budget_count = compute_budget_count(model, args.budget_ratio)

    # ── 构建选择器 ────────────────────────────────────────────────────────────
    selector = build_selector(args)
    print(f"[main] 选择策略: {selector.selector_name}")

    # ── 判断是否使用 def3 特殊流程 ────────────────────────────────────────────
    is_def3 = (
        args.selection_strategy == "update_response"
        and args.ur_metric == "def3"
    )

    # ── 设备选择 ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ── 精度选择 ──────────────────────────────────────────────────────────────
    use_bf16 = use_fp16 = False
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16

    # ── 构建训练数据 DataLoader（供 metric 计算使用） ──────────────────────────
    train_dataloader = DataLoader(
        ds["train"],
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=ds["collate_fn"],
    )

    # ═════════════════════════════════════════════════════════════════════════
    # 非 def3 流程：在训练前完成一次性初始选择（续跑时从检查点恢复，跳过计算）
    # ═════════════════════════════════════════════════════════════════════════
    initial_selections: Optional[Dict] = None
    if not is_def3:
        if resume_state.get("active_head_set") is not None:
            print(
                f"[main] 续跑模式：跳过初始头选择计算，"
                f"将从检查点恢复选择状态（活跃头数: "
                f"{len(resume_state['active_head_set'])}）。"
            )
            # initial_selections 保持 None，稍后由恢复逻辑直接重建 masker
        else:
            print(f"\n[main] 计算初始头选择（{selector.selector_name}）...")
            model.eval()
            try:
                initial_selections = selector.select(
                    model=model,
                    dataloader=train_dataloader,
                    device=device,
                    budget_count=budget_count,
                    step=0,
                    agg=args.agg,
                )
            finally:
                model.train()

    # 无论何种策略，冻结非注意力参数（在 Trainer 创建前调用）
    GradientMasker.freeze_non_attn_params(model)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir                  = os.path.join(args.out_dir, "trainer_out"),
        per_device_train_batch_size = args.bsz,
        per_device_eval_batch_size  = args.bsz,
        learning_rate               = args.lr,
        lr_scheduler_type           = args.lr_scheduler_type,
        warmup_ratio                = args.warmup_ratio,
        num_train_epochs            = args.epochs,
        eval_strategy               = "no",
        save_strategy               = "steps",          # 断点续跑：按步保存
        save_steps                  = args.save_steps,
        save_total_limit            = args.save_total_limit,
        seed                        = args.seed,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        report_to=["wandb"],
        run_name=f"FFT-{args.task}-seed{args.seed}-lr{args.lr}-budget{args.budget_ratio}-{args.selection_strategy}-{selector.selector_name}",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
    )

    # ── 构建 Callbacks ────────────────────────────────────────────────────────
    eval_callback = GlueEvalCallback(
        tokenizer             = tok,
        task                  = args.task,
        dataset_path          = args.dataset_path,
        max_length            = args.max_len,
        batch_size            = args.bsz,
        metric_for_best_model = ds["metric_for_best_model"],
        out_dir               = args.out_dir,
    )

    budget_callback = FixedBudgetCallback(
        model          = model,
        selector       = selector,
        dataloader     = train_dataloader,
        budget_count   = budget_count,
        reselect_every = args.reselect_every,
        save_dir       = args.out_dir,
        device         = device,
    )

    callbacks = [eval_callback, budget_callback]

    # ── def3 特殊处理：注册 EarlyGradNormCallback + Def3PhaseCallback ─────────
    early_grad_cb: Optional[EarlyGradNormCallback] = None
    def3_phase_cb: Optional[Def3PhaseCallback]     = None
    if is_def3:
        early_grad_cb = EarlyGradNormCallback(
            model=model,
            T_early=args.ur_T_early,
            save_dir=os.path.join(args.out_dir, "def3_scores"),
            head_granularity=True,
        )
        def3_phase_cb = Def3PhaseCallback(
            model=model,
            selector=selector,         # type: ignore[arg-type]
            budget_callback=budget_callback,
            early_grad_cb=early_grad_cb,
            T_early=args.ur_T_early,
        )
        callbacks.extend([early_grad_cb, def3_phase_cb])
        print(
            f"[main] def3 模式：前 {args.ur_T_early} 步全参训练（梯度收集），"
            f"之后施加预算选择。"
        )

    # ── 断点续跑：注册额外状态保存 Callback ──────────────────────────────────
    extra_state_cb = ExtraStateCheckpointCallback(
        budget_callback = budget_callback,
        eval_callback   = eval_callback,
        def3_phase_cb   = def3_phase_cb,
        early_grad_cb   = early_grad_cb,
    )
    callbacks.append(extra_state_cb)

    # ── 创建 Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        args          = train_args,
        train_dataset = ds["train"],
        tokenizer     = tok,
        callbacks     = callbacks,
    )

    # 非 def3：在 Trainer 启动前应用初始选择
    if not is_def3 and initial_selections is not None:
        budget_callback.apply_initial_selection(initial_selections)

    # ── 断点续跑：恢复额外训练状态 ────────────────────────────────────────────
    if resume_state:
        # 恢复评估历史与最优指标
        eval_callback._epoch_results = resume_state.get("eval_epoch_results", [])
        eval_callback.best_metric    = resume_state.get("best_metric")

        # 恢复选择日志
        budget_callback._selection_log = resume_state.get("selection_log", [])

        # 恢复活跃头集合，重建 GradientMasker（hooks 需在 trainer.train() 前注册）
        saved_head_list = resume_state.get("active_head_set")
        if saved_head_list is not None:
            restored_set = {(lk, int(hi)) for lk, hi in saved_head_list}
            budget_callback._current_selected_set = restored_set
            budget_callback._masker = GradientMasker(model, restored_set)
            print(
                f"[main] 已恢复梯度遮蔽钩子，活跃头数: {len(restored_set)}"
            )

        # def3 专属状态恢复
        if is_def3 and early_grad_cb is not None and def3_phase_cb is not None:
            saved_triggered      = resume_state.get("def3_triggered")
            saved_head_acc       = resume_state.get("def3_head_acc")
            saved_steps_collected = resume_state.get("def3_steps_collected", 0) or 0

            if saved_triggered is not None:
                def3_phase_cb._triggered = saved_triggered

            if saved_head_list is not None:
                # 阶段 1 已完成：直接标记 EarlyGradNormCallback 为结束状态，移除钩子
                early_grad_cb._done            = True
                early_grad_cb._step            = early_grad_cb.T_early
                early_grad_cb._steps_collected = saved_steps_collected
                early_grad_cb._remove_hooks()
                print(
                    f"[main] def3 阶段 1 已完成（共收集 {saved_steps_collected} 步），"
                    f"钩子已卸载，梯度遮蔽已激活。"
                )
            elif saved_head_acc is not None:
                # 阶段 1 未完成：恢复累积梯度数据，续跑后继续收集
                early_grad_cb._head_acc = {
                    m: {int(h): v for h, v in heads.items()}
                    for m, heads in saved_head_acc.items()
                }
                early_grad_cb._step            = saved_steps_collected
                early_grad_cb._steps_collected = saved_steps_collected
                print(
                    f"[main] def3 阶段 1 未完成，已恢复累积梯度数据"
                    f"（已收集 {saved_steps_collected}/{args.ur_T_early} 步），"
                    f"续跑后将继续收集。"
                )

    # ── 训练 ──────────────────────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── 清理 & 保存 ───────────────────────────────────────────────────────────
    eval_callback.save_history(args.out_dir)

    ckpt_final = os.path.join(args.out_dir, "ckpt_final")
    best_ckpt  = eval_callback.best_ckpt
    if os.path.isdir(best_ckpt):
        best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
        best_model.save_pretrained(ckpt_final)
        tok.save_pretrained(ckpt_final)
        print(f"[main] θ₁ (best) 已保存 → {ckpt_final}（来自 {best_ckpt}）")
    else:
        trainer.model.save_pretrained(ckpt_final)
        tok.save_pretrained(ckpt_final)
        print(f"[main] θ₁ (final) 已保存 → {ckpt_final}")

    # 保存运行配置
    run_cfg = vars(args)
    run_cfg["budget_count_actual"] = budget_count
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"\n[main] 训练完成。所有结果保存在 → {args.out_dir}")


if __name__ == "__main__":
    main()
