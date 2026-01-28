# src/train/finetune_glue_lora.py
import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from src.data.glue import load_glue_dataset
import wandb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)  # MNLI / RTE
    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)  # LoRA通常使用更高的学习率
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bsz", type=int, default=16)
    # LoRA参数
    ap.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    ap.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    ap.add_argument("--lora_target_modules", type=str, default="query_proj,key_proj,value_proj,dense", 
                    help="逗号分隔的目标模块名称")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    wandb.init(project="minimal-exp-lora", name=f"{args.task}-lora-seed{args.seed}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)

    # 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=ds["num_labels"])

    # 保存初始模型（θ0）- 保存基础模型用于后续对比
    ckpt_init = os.path.join(args.out_dir, "ckpt_init")
    model.save_pretrained(ckpt_init)
    tok.save_pretrained(ckpt_init)

    # 配置LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=ds["metric_for_best_model"],
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=["wandb"],
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        compute_metrics=ds["compute_metrics"],
        tokenizer=tok,
    )

    trainer.train()

    # 保存LoRA权重（θ1）
    ckpt_final = os.path.join(args.out_dir, "ckpt_final")
    model.save_pretrained(ckpt_final)
    tok.save_pretrained(ckpt_final)

    # 保存配置
    run_config = vars(args)
    run_config["method"] = "LoRA"
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

if __name__ == "__main__":
    main()
