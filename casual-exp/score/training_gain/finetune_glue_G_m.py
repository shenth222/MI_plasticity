# score/training_gain/finetune_glue_G_m.py
"""
在 baseline 训练代码基础上，以最小侵入方式嵌入「训练收益 def3（路径积分）」。

─────────────────────────────────────────────────────────────────────────────
定义三（路径积分）：
    G_m^(PI) = Σ_{t=1}^{T} ∇_{θ_m} L(θ^(t)) · Δθ_{m,t}

嵌入点（Trainer 创建前，model 尚未被 DDP 包装）：
    PathIntegralCallback：注册梯度 hook，在每个 optimizer step 的 on_step_end
    累积 g·Δθ，训练结束后（on_train_end）保存结果。

嵌入仅需 ~4 行核心代码（见 "嵌入点" 注释块）。

结果保存路径：
    {out_dir}/training_gain/def3_path_integral.json
    字段：module_scores, param_scores, steps_collected[, head_scores]

─────────────────────────────────────────────────────────────────────────────
def1/def2（回滚收益）说明：
    · def1/def2 无需训练，只需 θ^(0) 和 θ^(T) 两个检查点及验证集。
    · 请使用独立离线脚本：score/training_gain/eval_rollback.py
      python -m score.training_gain.eval_rollback \\
          --theta0_path {out_dir}/ckpt_init \\
          --thetaT_path {out_dir}/ckpt_best \\
          --task MNLI --dataset_path ...
─────────────────────────────────────────────────────────────────────────────
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
    kwargs      = InitProcessGroupKwargs(timeout=timedelta(hours=4))
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

    # ── 训练收益 def3（路径积分）参数 ────────────────────────────────────────
    ap.add_argument(
        "--gm_head_granularity", action="store_true", default=False,
        help=(
            "开启注意力头级别粒度计算。\n"
            "每步额外按头切片累积路径积分内积，内存开销约增加 1 份参数快照。\n"
            "结果中额外包含 head_scores: {module: {head_0: float, ...}}"
        ),
    )
    ap.add_argument(
        "--gm_log_every", type=int, default=1,
        help=(
            "路径积分计算频率：每隔多少 optimizer step 计算一次梯度·Δθ。\n"
            "默认 1（每步精确）；MNLI 等大任务建议设为 10 或 50（近似，省计算）。"
        ),
    )
    # ──────────────────────────────────────────────────────────────────────────

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "casual-exp-G_m"
    torch.manual_seed(args.seed)

    tok   = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds    = load_glue_dataset(args.task, tok, max_len=args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=ds["num_labels"]
    )

    # 保存 θ₀（供事后离线运行 eval_rollback.py 使用）
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
        run_name=f"FFT-Gm-{args.task}-seed{args.seed}-lr{args.lr}",
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

    # ── 嵌入点：创建 def3 PathIntegralCallback（Trainer 创建前）──────────────
    #
    # 嵌入仅需 4 行核心代码，不影响训练流程本身：
    #   1. 构造 PathIntegralGainMetric
    #   2. make_callback()：注册梯度 hook，记录初始 prev_params
    #   3. 追加到 all_callbacks
    #
    # · 仅在主进程创建，避免重复保存。
    # · on_step_begin 清空梯度缓冲；on_step_end 累积 g·Δθ（轻量，每步 ~1 次内积）。
    # · 支持梯度累积（gradient_accumulation_steps > 1）：
    #   hook 跨 substep 累积，on_step_end 使用一次 optimizer step 后的参数差。
    if accelerator.is_main_process:
        from metric.training_gain.def3_path_integral import PathIntegralGainMetric

        _pi_metric = PathIntegralGainMetric()
        _pi_cb     = _pi_metric.make_callback(
            model=model,
            save_dir=os.path.join(args.out_dir, "training_gain"),
            log_every=args.gm_log_every,
            head_granularity=args.gm_head_granularity,
        )
        all_callbacks.append(_pi_cb)
        head_info = " [+头级别]" if args.gm_head_granularity else ""
        print(
            f"[G_m def3] PathIntegralCallback 已注册{head_info}，"
            f"log_every={args.gm_log_every}"
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

    # 保存 θ₁（供事后离线运行 eval_rollback.py 使用）
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
