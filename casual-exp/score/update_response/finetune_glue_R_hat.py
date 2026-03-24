# score/update_response/finetune_glue_R_hat.py
"""
在 baseline 训练代码基础上，以最小侵入方式嵌入「更新响应预测」四种定义。

嵌入点总结：
  嵌入点 1（训练前，θ₀ 保存后，Trainer 创建前）:
      runner.run_pre(model, train_dl, device, save_dir=...)
      计算 def1 / def2 / def4（若选中）

  嵌入点 2（Trainer callbacks 注册时）:
      callbacks=[..., *ur_callbacks]
      计算 def3（若选中，在训练中自动收集并保存）

结果保存路径：{out_dir}/update_response/{defX_name}.json
"""
import os, json, argparse
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import DataLoader
from data.glue import load_glue_dataset
from utils.evaluate import evaluate_glue
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


class GlueEvalCallback(TrainerCallback):
    """每个 epoch 结束时评估并保存最优模型（与 baseline 完全一致）。"""

    def __init__(self, tokenizer, task, dataset_path, max_length, batch_size,
                 metric_for_best_model, out_dir):
        self.tokenizer = tokenizer
        self.task = task
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.metric_for_best_model = metric_for_best_model
        self.best_metric = None
        self.best_ckpt = os.path.join(out_dir, "ckpt_best")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if model is None:
            return control

        device = next(model.parameters()).device
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


def main():
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    ap = argparse.ArgumentParser()
    # ── 基础训练参数（与 baseline 完全一致）──────────────────────────────────
    ap.add_argument("--task",         type=str, required=True)
    ap.add_argument("--model_name",   type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir",      type=str, required=True)
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue"))
    ap.add_argument("--seed",         type=int,   default=1)
    ap.add_argument("--max_len",      type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--lr_scheduler_type", type=str, default="linear")
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--epochs",       type=float, default=3.0)
    ap.add_argument("--bsz",          type=int,   default=16)

    # ── 训练前重要性（pre_importance，与 finetune_glue_I_pre.py 一致）────────
    ap.add_argument("--pre_importance", type=str, default="",
                    help="逗号分隔，如 fisher,singular_value（空则不计算）")
    ap.add_argument("--pre_importance_batches",         type=int, default=32)
    ap.add_argument("--pre_importance_perturb_batches", type=int, default=4)
    ap.add_argument("--pre_importance_head_granularity", action="store_true", default=False)

    # ── 更新响应预测（update_response，新增嵌入点）──────────────────────────
    ap.add_argument(
        "--update_response", type=str, default="",
        help=(
            "更新响应预测指标，逗号分隔，空字符串则不计算。\n"
            "可选：def1, def2, def3, def4\n"
            "  def1 — 短程试跑参数位移  ‖θ(t₀)−θ(0)‖₂\n"
            "  def2 — 梯度-曲率归一化   E[‖g‖] / √(Fisher+ε)\n"
            "  def3 — 累积早期梯度范数  Σ‖g(t)‖（嵌入训练中）\n"
            "  def4 — 梯度信噪比 Ppred  E[|g|]²/E[g²]\n"
            "示例：--update_response def1,def2,def3,def4"
        ),
    )
    ap.add_argument("--ur_probe_steps",   type=int,   default=20,
                    help="def1：探针训练步数（建议 10–50）")
    ap.add_argument("--ur_probe_lr",      type=float, default=None,
                    help="def1：探针 LR（默认等于主训练 --lr）")
    ap.add_argument("--ur_num_batches",   type=int,   default=32,
                    help="def2/def4：Monte Carlo 估计 batch 数（建议 16–64）")
    ap.add_argument("--ur_T_early",       type=int,   default=100,
                    help="def3：累积步数 T_early（建议 50–200）")
    ap.add_argument("--ur_epsilon",       type=float, default=1e-8,
                    help="def2/def4：数值稳定项 ε")
    ap.add_argument("--ur_head_granularity", action="store_true", default=False,
                    help=(
                        "开启注意力头级别粒度计算。\n"
                        "对 def1/def2/def4 按头拆分参数，对 def3 按头累积梯度范数。\n"
                        "要求模型具有标准 HuggingFace config（num_attention_heads）。"
                    ))
    # ──────────────────────────────────────────────────────────────────────────
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "casual-exp-R_hat"
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds  = load_glue_dataset(args.task, tok, max_len=args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=ds["num_labels"]
    )

    # 保存 θ₀
    ckpt_init = os.path.join(args.out_dir, "ckpt_init")
    model.save_pretrained(ckpt_init)
    tok.save_pretrained(ckpt_init)

    # ── 嵌入点：训练前计算（pre_importance + update_response def1/def2/def4）──
    if accelerator.is_main_process:
        _imp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(_imp_device)

        _imp_dl = DataLoader(
            ds["train"], batch_size=args.bsz, shuffle=True,
            collate_fn=ds["collate_fn"],
        )

        # ── pre_importance（已有功能，保留）──────────────────────────────────
        if args.pre_importance.strip():
            from metric.pre_importance.runner import PreImportanceRunner
            print(f"\n[PreImportance] 开始计算训练前重要性: {args.pre_importance}")
            _pi_runner = PreImportanceRunner.from_str(
                args.pre_importance,
                metric_kwargs={
                    "fisher":       {"num_batches": args.pre_importance_batches},
                    "saliency":     {"num_batches": args.pre_importance_batches},
                    "perturbation": {"num_batches": args.pre_importance_perturb_batches},
                },
                head_granularity=args.pre_importance_head_granularity,
            )
            _pi_runner.run(
                model, _imp_dl, _imp_device,
                save_dir=os.path.join(args.out_dir, "pre_importance"),
            )
            print("[PreImportance] 训练前重要性计算完毕\n")

        # ── update_response 训练前部分（def1 / def2 / def4）──────────────────
        if args.update_response.strip():
            from metric.update_response.runner import UpdateResponseRunner

            _ur_probe_lr = args.ur_probe_lr if args.ur_probe_lr is not None else args.lr
            _ur_runner = UpdateResponseRunner.from_str(
                args.update_response,
                head_granularity=args.ur_head_granularity,
                metric_kwargs={
                    "def1": {
                        "probe_steps":  args.ur_probe_steps,
                        "probe_lr":     _ur_probe_lr,
                    },
                    "def2": {
                        "num_batches": args.ur_num_batches,
                        "epsilon":     args.ur_epsilon,
                    },
                    "def3": {
                        "T_early": args.ur_T_early,
                    },
                    "def4": {
                        "num_batches": args.ur_num_batches,
                        "epsilon":     args.ur_epsilon,
                    },
                },
            )

            _ur_save_dir = os.path.join(args.out_dir, "update_response")
            if _ur_runner.selected_pre_metrics:
                print(f"\n[UpdateResponse] 开始计算训练前指标: "
                      f"{_ur_runner.selected_pre_metrics}")
                _ur_runner.run_pre(model, _imp_dl, _imp_device, save_dir=_ur_save_dir)
                print("[UpdateResponse] 训练前指标计算完毕\n")

        model.to("cpu")

    accelerator.wait_for_everyone()
    # ──────────────────────────────────────────────────────────────────────────

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
        run_name=f"FFT-R_hat-{args.task}-seed{args.seed}-lr{args.lr}",
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

    # ── 嵌入点 2：注册 def3 训练中回调（最小侵入：仅追加到 callbacks 列表）──
    if args.update_response.strip() and accelerator.is_main_process:
        _ur_save_dir = os.path.join(args.out_dir, "update_response")
        _ur_callbacks = _ur_runner.make_training_callbacks(model, save_dir=_ur_save_dir)
        all_callbacks.extend(_ur_callbacks)
        if _ur_callbacks:
            print(f"[UpdateResponse] 已注册训练中回调: "
                  f"{_ur_runner.selected_in_metrics}")
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
