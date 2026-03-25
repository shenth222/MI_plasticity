# score/actual_update/finetune_glue_U_m.py
"""
在 baseline 训练代码基础上，以最小侵入方式嵌入「实际更新量 U_m」三种定义。

嵌入点说明（仅 1 处，Trainer 创建前）：
  runner.make_callbacks(model, save_dir=...)
    └─ def1 AbsoluteUpdateCallback — 立即快照 θ^(0)，on_train_end 计算 ||Δθ_m||_2
    └─ def2 RelativeUpdateCallback — 立即快照 θ^(0)，on_train_end 计算相对更新量
    └─ def3 PathLengthCallback     — 立即快照 θ^(0)，on_step_end 逐步累积路径长度

与其他 score 脚本的区别：
  · 无训练前独立计算阶段（def1/def2 依赖 θ^(T)，def3 依赖逐步轨迹），
    所有指标均以 TrainerCallback 形式嵌入，零侵入训练循环本身。
  · make_callbacks 必须在 Trainer 创建前（model 未被 DDP 包装时）调用，
    以确保 θ^(0) 快照的参数名与 named_parameters() 完全一致。

结果保存路径：{out_dir}/actual_update/{def1_absolute|def2_relative|def3_path_length}.json
"""

import os
import json
import argparse

import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

from data.glue import load_glue_dataset
from utils.evaluate import evaluate_glue


# ---------------------------------------------------------------------------
# GlueEvalCallback（与 baseline 完全一致）
# ---------------------------------------------------------------------------

class GlueEvalCallback(TrainerCallback):
    """每个 epoch 结束时评估并保存验证集最优模型。"""

    def __init__(self, tokenizer, task, dataset_path, max_length, batch_size,
                 metric_for_best_model, out_dir):
        self.tokenizer             = tokenizer
        self.task                  = task
        self.dataset_path          = dataset_path
        self.max_length            = max_length
        self.batch_size            = batch_size
        self.metric_for_best_model = metric_for_best_model
        self.best_metric           = None
        self.best_ckpt             = os.path.join(out_dir, "ckpt_best")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return control

        device  = next(model.parameters()).device
        results = evaluate_glue(
            model=model, tokenizer=self.tokenizer, task_name=self.task,
            dataset_path=self.dataset_path, max_length=self.max_length,
            batch_size=self.batch_size, device=device,
        )
        if wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in results.items()},
                      step=state.global_step)
        print(f"\n[Eval] epoch={state.epoch:.1f}  {results}")

        key = self.metric_for_best_model
        val = results.get(key)
        if val is not None and (self.best_metric is None or val > self.best_metric):
            self.best_metric = val
            model.save_pretrained(self.best_ckpt)
            self.tokenizer.save_pretrained(self.best_ckpt)
            print(f"[Eval] New best {key}={val:.4f}, saved to {self.best_ckpt}")

        model.train()
        return control


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    kwargs      = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    ap = argparse.ArgumentParser()

    # ── 基础训练参数（与 baseline 完全一致）──────────────────────────────────
    ap.add_argument("--task",         type=str,   required=True,
                    help="GLUE 任务名，如 MNLI / RTE / MRPC")
    ap.add_argument("--model_name",   type=str,   default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir",      type=str,   required=True)
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue"))
    ap.add_argument("--seed",         type=int,   default=1)
    ap.add_argument("--max_len",      type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--lr_scheduler_type", type=str, default="linear")
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--epochs",       type=float, default=3.0)
    ap.add_argument("--bsz",          type=int,   default=16)

    # ── 实际更新量 U_m（嵌入点参数）────────────────────────────────────────────
    ap.add_argument(
        "--actual_update", type=str, default="def1,def2,def3",
        help=(
            "实际更新量定义，逗号分隔，空字符串则不计算。\n"
            "可选：def1, def2, def3\n"
            "  def1 — 绝对更新量  ||Δθ_m||₂（变体A全参数L2）和 ||ΔW_m||_F（变体B仅weight）\n"
            "  def2 — 相对更新量  ||Δθ_m||₂ / (||θ_m^(0)||₂ + ε)\n"
            "  def3 — 路径长度    Σ_t ||θ_m^(t) − θ_m^(t−1)||₂\n"
            "示例：--actual_update def1,def2,def3"
        ),
    )
    ap.add_argument(
        "--au_log_every", type=int, default=1,
        help=(
            "def3（路径长度）计算频率：每隔多少 optimizer step 计算一次步进变化量。\n"
            "默认 1（每步精确计算）；大模型或步数极多时建议设为 10 或 100（近似，省计算）。\n"
            "注：log_every > 1 时路径长度为近似值，但计算成本降低 log_every 倍。"
        ),
    )
    ap.add_argument(
        "--au_epsilon", type=float, default=1e-8,
        help="def2 数值稳定项 ε，防止除以零（默认 1e-8）。",
    )
    ap.add_argument(
        "--au_head_granularity", action="store_true", default=False,
        help=(
            "开启注意力头级别粒度计算。\n"
            "对 def1/def2：在 module_scores 基础上额外计算 head_scores，\n"
            "  按注意力头维度切分 Δθ（Q/K/V 投影和输出投影），独立评估各头更新量。\n"
            "对 def3：每步 on_step_end 中按头切片步进 delta 并累积头路径长度，\n"
            "  峰值内存 ≈ 1 份参数快照（常驻 CPU） + 步进 delta 张量（处理后即释放）。\n"
            "要求模型具有标准 HuggingFace config（num_attention_heads / hidden_size）。\n"
            "若模型无 config，自动安全降级（不崩溃，结果不含 head_scores）。"
        ),
    )
    # ──────────────────────────────────────────────────────────────────────────

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "casual-exp-U_m"
    torch.manual_seed(args.seed)

    tok   = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds    = load_glue_dataset(args.task, tok, max_len=args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=ds["num_labels"]
    )

    # 保存 θ₀
    ckpt_init = os.path.join(args.out_dir, "ckpt_init")
    model.save_pretrained(ckpt_init)
    tok.save_pretrained(ckpt_init)

    accelerator.wait_for_everyone()

    # 混精度选择
    use_bf16 = use_fp16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True

    train_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "trainer_out"),
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        eval_strategy="no",
        save_strategy="no",
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=["wandb"],
        run_name=f"FFT-U_m-{args.task}-seed{args.seed}-lr{args.lr}",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
    )

    eval_callback = GlueEvalCallback(
        tokenizer=tok, task=args.task, dataset_path=args.dataset_path,
        max_length=args.max_len, batch_size=args.bsz,
        metric_for_best_model=ds["metric_for_best_model"], out_dir=args.out_dir,
    )
    all_callbacks = [eval_callback]

    # ── 嵌入点：创建 U_m callbacks（Trainer 创建前，model 尚未被 DDP 包装）────
    #
    # 此处是唯一嵌入点，仅 3 行核心代码：
    #   1. 构造 runner（解析指标名称）
    #   2. make_callbacks()：立即快照 θ^(0)，返回 callback 列表
    #   3. 追加到 all_callbacks
    #
    # · 仅在主进程创建（避免非主进程的内存开销），
    #   callback 内部通过 state.is_world_process_zero 保证只有主进程保存结果。
    # · def1/def2：on_train_end 计算 θ^(T) - θ^(0)，零训练中开销。
    # · def3：on_step_end 逐步累积 ||θ^(t) - θ^(t-1)||₂，
    #   内存开销约 1 份参数快照（CPU，DeBERTa-v3-base bf16 ≈ 350MB）。
    if args.actual_update.strip() and accelerator.is_main_process:
        from metric.actual_update.runner import ActualUpdateRunner
        _au_runner = ActualUpdateRunner.from_str(
            args.actual_update,
            metric_kwargs={
                "def2": {"epsilon": args.au_epsilon},
                "def3": {"log_every": args.au_log_every},
            },
            head_granularity=args.au_head_granularity,
        )
        _au_callbacks = _au_runner.make_callbacks(
            model,
            save_dir=os.path.join(args.out_dir, "actual_update"),
        )
        all_callbacks.extend(_au_callbacks)
        head_info = " [+头级别]" if args.au_head_granularity else ""
        print(
            f"[U_m] 已注册 {len(_au_callbacks)} 个 callback："
            f" {_au_runner.selected_metrics}{head_info}"
        )
    # ──────────────────────────────────────────────────────────────────────────

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        tokenizer=tok,
        callbacks=all_callbacks,
    )

    trainer.train()

    # 保存 θ₁
    ckpt_final = os.path.join(args.out_dir, "ckpt_final")
    best_ckpt  = eval_callback.best_ckpt
    if os.path.isdir(best_ckpt):
        best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
        best_model.save_pretrained(ckpt_final)
        tok.save_pretrained(ckpt_final)
        print(f"[Save] θ₁ (best) saved to {ckpt_final}")
    else:
        trainer.model.save_pretrained(ckpt_final)
        tok.save_pretrained(ckpt_final)
        print(f"[Save] θ₁ (final) saved to {ckpt_final}")

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
