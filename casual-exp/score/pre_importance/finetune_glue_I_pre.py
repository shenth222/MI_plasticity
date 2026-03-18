# src/train/finetune_glue.py
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
from data.glue import load_glue_dataset
from utils.evaluate import evaluate_glue
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

class GlueEvalCallback(TrainerCallback):
    """每个 epoch 结束时调用 utils.evaluate.evaluate_glue 进行评估，
    并将指标记录到 wandb；同时追踪并保存验证集最优模型。"""

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
        # 非主进程直接跳过，避免重复评估 / 重复保存 / wandb.log 报错
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
    ap.add_argument("--task", type=str, required=True)  # MNLI / RTE
    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_scheduler_type", type=str, default="linear")
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bsz", type=int, default=16)

    # ── 训练前重要性计算（最小侵入嵌入点）──────────────────────────────────
    ap.add_argument(
        "--pre_importance", type=str, default="",
        help=(
            "训练前重要性指标，逗号分隔，空字符串则不计算。\n"
            "可选：fisher, saliency, perturbation, singular_value, spectral_entropy\n"
            "示例：--pre_importance fisher,singular_value,spectral_entropy"
        ),
    )
    ap.add_argument(
        "--pre_importance_batches", type=int, default=32,
        help="梯度类指标（fisher/saliency）使用的 mini-batch 数量",
    )
    ap.add_argument(
        "--pre_importance_perturb_batches", type=int, default=4,
        help="扰动类指标使用的 mini-batch 数量（开销较大，建议 2-8）",
    )
    ap.add_argument(
        "--pre_importance_head_granularity", action="store_true", default=False,
        help=(
            "开启注意力头级别粒度计算。\n"
            "效果：在 module_scores 基础上额外计算并保存 head_scores，\n"
            "      对每个注意力 Q/K/V / 输出投影按头维度切分，独立评估各头重要性。\n"
            "要求：模型具有标准 HuggingFace config（num_attention_heads、hidden_size）。\n"
            "开销：Fisher/Saliency 无额外前向传播；Perturbation 会增加\n"
            "      O(num_attn_modules × num_heads) 次前向传播。"
        ),
    )
    # ──────────────────────────────────────────────────────────────────────────
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = "casual-exp-I_pre"
    # os.environ["NCCL_TIMEOUT"] = "1800"  # 30分钟，单位秒
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 启用阻塞等待，配合 timeout
    torch.manual_seed(args.seed)
    # wandb.init(project="casual-exp", name=f"Debug-FFT-baseline-{args.task}-seed{args.seed}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=ds["num_labels"])

    # Save θ0
    ckpt_init = os.path.join(args.out_dir, "ckpt_init")
    model.save_pretrained(ckpt_init)
    tok.save_pretrained(ckpt_init)

    # ── 训练前重要性计算（最小侵入嵌入点）──────────────────────────────────
    # 在 θ0 保存后、Trainer 创建前执行，此时模型为纯 nn.Module，无 DDP 包装。
    # 仅在主进程（local_rank==0 或无分布式）计算重要性
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.pre_importance.strip() and accelerator.is_main_process:
        from torch.utils.data import DataLoader
        from metric.pre_importance.runner import PreImportanceRunner

        print(f"\n[PreImportance] 开始计算训练前重要性: {args.pre_importance}")
        _imp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(_imp_device)

        _imp_dl = DataLoader(
            ds["train"],
            batch_size=args.bsz,
            shuffle=True,
            collate_fn=ds["collate_fn"],
        )
        _runner = PreImportanceRunner.from_str(
            args.pre_importance,
            metric_kwargs={
                "fisher":       {"num_batches": args.pre_importance_batches},
                "saliency":     {"num_batches": args.pre_importance_batches},
                "perturbation": {"num_batches": args.pre_importance_perturb_batches},
            },
            head_granularity=args.pre_importance_head_granularity,
        )
        _runner.run(
            model, _imp_dl, _imp_device,
            save_dir=os.path.join(args.out_dir, "pre_importance"),
        )
        model.to("cpu")   # 交还给 Trainer 统一管理设备
        print("[PreImportance] 训练前重要性计算完毕\n")
    # 等待所有进程同步，避免提前进入训练（尤其在某些多节点情况下重要）
    # if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    #     pass
        # import torch.distributed as dist
        # if not dist.is_initialized():
        #     dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        # dist.barrier()
    accelerator.wait_for_everyone()
    # ──────────────────────────────────────────────────────────────────────────

    # Auto-select precision: bf16 > fp16 > fp32
    use_bf16 = False
    use_fp16 = False
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
        run_name=f"FFT-baseline-{args.task}-seed{args.seed}-lr{args.lr}",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True
    )

    eval_callback = GlueEvalCallback(
        tokenizer=tok,
        task=args.task,
        dataset_path=args.dataset_path,
        max_length=args.max_len,
        batch_size=args.bsz,
        metric_for_best_model=ds["metric_for_best_model"],
        out_dir=args.out_dir,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        tokenizer=tok,
        callbacks=[eval_callback],
    )

    trainer.train()

    # Save θ1: 优先使用 callback 保存的最优模型，否则使用训练结束时的模型
    ckpt_final = os.path.join(args.out_dir, "ckpt_final")
    best_ckpt = eval_callback.best_ckpt
    if os.path.isdir(best_ckpt):
        best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
        best_model.save_pretrained(ckpt_final)
        tok.save_pretrained(ckpt_final)
        print(f"[Save] θ1 (best) saved to {ckpt_final} (from {best_ckpt})")
    else:
        trainer.model.save_pretrained(ckpt_final)
        tok.save_pretrained(ckpt_final)
        print(f"[Save] θ1 (final) saved to {ckpt_final}")

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    main()
