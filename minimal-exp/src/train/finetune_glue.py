# src/train/finetune_glue.py
import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.data.glue import load_glue_dataset
import wandb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)  # MNLI / RTE
    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bsz", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    wandb.init(project="minimal-exp", name=f"{args.task}-seed{args.seed}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=ds["num_labels"])

    # Save θ0
    ckpt_init = os.path.join(args.out_dir, "ckpt_init")
    model.save_pretrained(ckpt_init)
    tok.save_pretrained(ckpt_init)

    # Auto-select precision: bf16 > fp16 > fp32
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        # Check if bf16 is supported
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True

    # 增加 logging_steps 保证 loss 日志能被 Trainer 定期记录到 wandb
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
        logging_strategy="steps",    # 确保 log 在步骤执行
        logging_steps=10,            # 每 10 步记录一次 loss 到 wandb
        logging_first_step=True      # 确保一开始就有 loss log
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

    # Save θ1 (best)
    ckpt_final = os.path.join(args.out_dir, "ckpt_final")
    trainer.model.save_pretrained(ckpt_final)
    tok.save_pretrained(ckpt_final)

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    main()
