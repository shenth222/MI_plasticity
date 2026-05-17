import argparse
import csv
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

from ipd_lora import (
    NEVER_UPDATE_INTERVAL,
    apply_module_early_stopping,
    apply_update_frequency_mask,
    build_calibration_split,
    collect_module_rows,
    compute_importance_scores,
    compute_plasticity_scores,
    count_parameters,
    inject_ipd_lora,
    update_quadrants_and_budget,
)

try:
    import evaluate
except Exception:  # pragma: no cover - optional dependency fallback
    evaluate = None
try:
    import wandb
except Exception:  # pragma: no cover - optional dependency fallback
    wandb = None


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="IPD-LoRA training for GLUE with DeBERTa-v3-base.")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None, help="Local dataset path for load_from_disk.")
    parser.add_argument("--train_file", type=str, default=None, help="Local train file (csv/json/jsonl).")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="Local validation file (csv/json/jsonl)."
    )
    parser.add_argument("--local_train_split", type=str, default="train")
    parser.add_argument("--local_eval_split", type=str, default="validation")
    parser.add_argument("--text_column1", type=str, default=None)
    parser.add_argument("--text_column2", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--initial_active_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--score_interval", type=int, default=100)
    parser.add_argument("--warmup_steps_for_ipd", type=int, default=100)
    parser.add_argument("--calibration_size", type=int, default=256)
    parser.add_argument("--total_rank_budget", type=int, default=96)
    parser.add_argument("--beta_I", type=float, default=0.9)
    parser.add_argument("--beta_P", type=float, default=0.9)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--calibration_max_batches", type=int, default=16)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_i_tolerance", type=float, default=1e-4)
    parser.add_argument("--report_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ipd-lora")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, row: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def init_csv(path: str, fieldnames: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_csv(path: str, row: Dict, fieldnames: List[str]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def get_eval_split(task_name: str) -> str:
    return "validation_matched" if task_name == "mnli" else "validation"


def _choose_text_keys(args, task: str, train_raw):
    if args.text_column1 is not None:
        if args.text_column1 not in train_raw.column_names:
            raise ValueError(f"text_column1={args.text_column1} not found in dataset columns.")
        if args.text_column2 is not None and args.text_column2 not in train_raw.column_names:
            raise ValueError(f"text_column2={args.text_column2} not found in dataset columns.")
        return args.text_column1, args.text_column2

    if task in GLUE_TASK_TO_KEYS:
        return GLUE_TASK_TO_KEYS[task]

    # For generic local datasets, auto-detect text columns.
    candidates = [c for c in train_raw.column_names if c != args.label_column]
    if len(candidates) < 1:
        raise ValueError("Cannot infer text columns from local dataset.")
    if len(candidates) == 1:
        return candidates[0], None
    return candidates[0], candidates[1]


def load_raw_datasets(args):
    task = args.task_name.lower()

    if args.dataset_path:
        raw = load_from_disk(args.dataset_path)
        return raw

    if args.train_file or args.validation_file:
        if not args.train_file or not args.validation_file:
            raise ValueError("When using local files, both --train_file and --validation_file are required.")
        ext = os.path.splitext(args.train_file)[1].lower()
        if ext == ".jsonl":
            dataset_loader = "json"
        elif ext in [".json", ".csv"]:
            dataset_loader = ext.lstrip(".")
        else:
            raise ValueError(f"Unsupported local file extension: {ext}")
        raw = load_dataset(dataset_loader, data_files={"train": args.train_file, "validation": args.validation_file})
        return raw

    dataset_name = args.dataset_name
    dataset_config = args.dataset_config_name
    if dataset_name == "glue":
        config = dataset_config if dataset_config is not None else task
        return load_dataset("glue", config)
    if dataset_config is not None:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def infer_num_labels(train_raw, label_column: str) -> int:
    feat = train_raw.features.get(label_column, None)
    if feat is not None and hasattr(feat, "num_classes") and feat.num_classes is not None:
        return int(feat.num_classes)
    labels = train_raw[label_column]
    unique = len(set(labels))
    if unique < 2:
        raise ValueError("num_labels inferred < 2. Please check label column.")
    return unique


def prepare_datasets(args, tokenizer):
    task = args.task_name.lower()
    raw = load_raw_datasets(args)
    train_split = "train" if "train" in raw else args.local_train_split
    eval_split = (
        get_eval_split(task)
        if (args.dataset_name == "glue" and args.dataset_path is None and args.train_file is None)
        else args.local_eval_split
    )
    if train_split not in raw:
        raise ValueError(f"Train split '{train_split}' not found in dataset.")
    if eval_split not in raw:
        raise ValueError(f"Eval split '{eval_split}' not found in dataset.")

    train_raw_base = raw[train_split]
    eval_raw = raw[eval_split]
    sentence1_key, sentence2_key = _choose_text_keys(args, task, train_raw_base)
    if args.label_column not in train_raw_base.column_names:
        raise ValueError(f"label_column={args.label_column} not found in dataset columns.")

    train_raw, calib_raw = build_calibration_split(
        train_raw_base, calibration_size=args.calibration_size, seed=args.seed
    )

    def preprocess(examples):
        if sentence2_key is None:
            toks = tokenizer(examples[sentence1_key], truncation=True, max_length=args.max_length)
        else:
            toks = tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                truncation=True,
                max_length=args.max_length,
            )
        toks["labels"] = examples[args.label_column]
        return toks

    train_ds = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
    calib_ds = calib_raw.map(preprocess, batched=True, remove_columns=calib_raw.column_names)
    eval_ds = eval_raw.map(preprocess, batched=True, remove_columns=eval_raw.column_names)
    return train_ds, calib_ds, eval_ds, raw, train_split


def freeze_backbone_except_lora_and_classifier(model):
    for p in model.parameters():
        p.requires_grad = False

    # Keep classification head trainable.
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True

    # Keep LoRA trainable.
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


@torch.no_grad()
def evaluate_model(model, dataloader, device, task_name: str, use_glue_metric: bool = True):
    model.eval()
    metric = evaluate.load("glue", task_name) if (evaluate is not None and use_glue_metric) else None
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_refs = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = float(outputs.loss.item())
        logits = outputs.logits
        if task_name == "stsb":
            preds = logits.squeeze(-1)
            preds_cpu = preds.detach().cpu().numpy()
            refs_cpu = batch["labels"].detach().cpu().numpy()
        else:
            preds = torch.argmax(logits, dim=-1)
            preds_cpu = preds.detach().cpu().numpy()
            refs_cpu = batch["labels"].detach().cpu().numpy()
        all_preds.append(preds_cpu)
        all_refs.append(refs_cpu)
        if metric is not None:
            metric.add_batch(predictions=preds.detach().cpu(), references=batch["labels"].detach().cpu())
        bs = int(batch["labels"].shape[0])
        total_loss += loss * bs
        total_n += bs

    preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    refs_np = np.concatenate(all_refs) if all_refs else np.array([])
    if metric is not None:
        scores = metric.compute()
    else:
        if task_name == "stsb":
            corr = pearsonr(preds_np, refs_np)[0] if len(preds_np) > 1 else 0.0
            scores = {"pearson": float(corr)}
        else:
            acc_fallback = float((preds_np == refs_np).mean()) if len(preds_np) > 0 else 0.0
            scores = {"accuracy": acc_fallback}
    avg_loss = total_loss / max(total_n, 1)
    if "accuracy" in scores:
        acc = float(scores["accuracy"])
    elif "pearson" in scores:
        acc = float(scores["pearson"])
    else:
        # Fallback for tasks that might not expose accuracy directly.
        acc = float(next(iter(scores.values())))
    return avg_loss, acc, scores


def active_rank_stats(lora_module_dict):
    active_total_rank = sum(int(m.active_rank) for m in lora_module_dict.values())
    active_module_count = sum(int(m.active_rank > 0) for m in lora_module_dict.values())
    frozen_module_count = sum(int(m.frozen_by_early_stop) for m in lora_module_dict.values())
    return active_total_rank, active_module_count, frozen_module_count


def maybe_save_checkpoint(args, model, tokenizer, global_step):
    if args.save_steps <= 0:
        return
    if global_step % args.save_steps != 0:
        return
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    ensure_dir(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def main():
    args = parse_args()
    task_name = args.task_name.lower()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    use_wandb = bool(args.report_to_wandb and args.wandb_mode != "disabled")
    if use_wandb and wandb is None:
        print("[warn] report_to_wandb is enabled but wandb is not installed. Logging falls back to local files only.")
        use_wandb = False
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=args.output_dir,
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_ds, calib_ds, eval_ds, raw, train_split = prepare_datasets(args, tokenizer)
    num_labels = infer_num_labels(raw[train_split], args.label_column)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    )

    lora_module_dict = inject_ipd_lora(
        model=model,
        target_modules=["query_proj", "value_proj"],
        max_rank=args.max_lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        initial_active_rank=args.initial_active_rank,
        verbose=True,
    )
    if len(lora_module_dict) == 0:
        raise RuntimeError("No LoRA module injected. Please verify target module names for this model.")
    freeze_backbone_except_lora_and_classifier(model)
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    calib_loader = DataLoader(
        calib_ds,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    optimizer_grouped = [
        {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.num_train_epochs
    lr_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    training_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    module_scores_path = os.path.join(args.output_dir, "module_scores.jsonl")
    rank_history_path = os.path.join(args.output_dir, "rank_history.csv")
    quadrant_history_path = os.path.join(args.output_dir, "quadrant_history.csv")
    init_csv(rank_history_path, ["step", "module_name", "active_rank"])
    init_csv(quadrant_history_path, ["step", "module_name", "quadrant"])

    best_eval_accuracy = -1e9
    best_eval_loss = 1e9
    final_eval_accuracy = 0.0
    final_eval_loss = 0.0
    last_eval_accuracy = None
    last_eval_loss = None
    running_loss = 0.0
    running_steps = 0
    global_step = 0
    rank_total_history: List[int] = []

    # Warmup phase for IPD policy: all modules update every step, no dynamic reallocation.
    for module in lora_module_dict.values():
        module.active_rank = min(args.initial_active_rank, args.max_lora_rank)
        module.target_rank = module.active_rank
        module.update_interval = 1
        module.quadrant = "warmup"

    active_rank_choices = [c for c in [0, 1, 2, 4, 8, 16] if c <= args.max_lora_rank]
    if args.max_lora_rank not in active_rank_choices:
        active_rank_choices.append(args.max_lora_rank)
    active_rank_choices = sorted(set(active_rank_choices))

    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            running_loss += float(loss.item())
            running_steps += 1

            do_scoring = (
                global_step > args.warmup_steps_for_ipd
                and args.score_interval > 0
                and global_step % args.score_interval == 0
            )
            if do_scoring:
                # P measures AdamW-aware expected marginal gain for each module.
                compute_plasticity_scores(
                    lora_module_dict=lora_module_dict,
                    optimizer=optimizer,
                    beta_P=args.beta_P,
                )
                # I uses forward ablation on calibration set to evaluate retained value.
                compute_importance_scores(
                    model=model,
                    lora_module_dict=lora_module_dict,
                    calibration_dataloader=calib_loader,
                    device=device,
                    beta_I=args.beta_I,
                    max_batches=args.calibration_max_batches,
                )

                update_quadrants_and_budget(
                    lora_module_dict=lora_module_dict,
                    total_rank_budget=args.total_rank_budget,
                    active_rank_choices=active_rank_choices,
                )
                apply_module_early_stopping(
                    lora_module_dict=lora_module_dict,
                    patience=args.early_stop_patience,
                    p_low_threshold=-0.5,
                    i_tolerance=args.early_stop_i_tolerance,
                )
                module_rows = collect_module_rows(lora_module_dict, global_step)
                for row in module_rows:
                    write_jsonl(module_scores_path, row)
                    append_csv(
                        rank_history_path,
                        {
                            "step": int(global_step),
                            "module_name": row["module_name"],
                            "active_rank": int(row["active_rank"]),
                        },
                        ["step", "module_name", "active_rank"],
                    )
                    append_csv(
                        quadrant_history_path,
                        {
                            "step": int(global_step),
                            "module_name": row["module_name"],
                            "quadrant": row["quadrant"],
                        },
                        ["step", "module_name", "quadrant"],
                    )

            # Keep forward active but selectively disable gradient updates by schedule.
            apply_update_frequency_mask(lora_module_dict=lora_module_dict, global_step=global_step)
            optimizer.step()
            scheduler.step()

            active_total_rank, active_module_count, frozen_module_count = active_rank_stats(lora_module_dict)
            rank_total_history.append(active_total_rank)

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_train_loss = running_loss / max(running_steps, 1)
                lr = float(scheduler.get_last_lr()[0])
                write_jsonl(
                    training_log_path,
                    {
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "train_loss": float(avg_train_loss),
                        "eval_loss": None if last_eval_loss is None else float(last_eval_loss),
                        "eval_accuracy": None if last_eval_accuracy is None else float(last_eval_accuracy),
                        "learning_rate": lr,
                        "active_total_rank": int(active_total_rank),
                        "active_module_count": int(active_module_count),
                        "frozen_module_count": int(frozen_module_count),
                    },
                )
                print(
                    f"[train] step={global_step} epoch={epoch} loss={avg_train_loss:.4f} "
                    f"lr={lr:.3e} active_rank={active_total_rank} frozen={frozen_module_count}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": float(avg_train_loss),
                            "train/lr": lr,
                            "train/active_total_rank": int(active_total_rank),
                            "train/active_module_count": int(active_module_count),
                            "train/frozen_module_count": int(frozen_module_count),
                            "epoch": int(epoch),
                        },
                        step=global_step,
                    )
                running_loss = 0.0
                running_steps = 0

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                use_glue_metric = (
                    args.dataset_name == "glue" and args.dataset_path is None and args.train_file is None
                )
                eval_loss, eval_accuracy, _ = evaluate_model(
                    model, eval_loader, device, task_name, use_glue_metric=use_glue_metric
                )
                last_eval_loss, last_eval_accuracy = eval_loss, eval_accuracy
                write_jsonl(
                    training_log_path,
                    {
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "train_loss": None,
                        "eval_loss": float(eval_loss),
                        "eval_accuracy": float(eval_accuracy),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "active_total_rank": int(active_total_rank),
                        "active_module_count": int(active_module_count),
                        "frozen_module_count": int(frozen_module_count),
                    },
                )
                print(f"[eval] step={global_step} eval_loss={eval_loss:.4f} eval_acc={eval_accuracy:.4f}")
                if use_wandb:
                    wandb.log(
                        {
                            "eval/loss": float(eval_loss),
                            "eval/accuracy": float(eval_accuracy),
                            "eval/best_accuracy": float(max(best_eval_accuracy, eval_accuracy)),
                        },
                        step=global_step,
                    )
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    best_eval_loss = eval_loss
                    best_dir = os.path.join(args.output_dir, "best_model")
                    ensure_dir(best_dir)
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                maybe_save_checkpoint(args, model, tokenizer, global_step)

    use_glue_metric = args.dataset_name == "glue" and args.dataset_path is None and args.train_file is None
    final_eval_loss, final_eval_accuracy, _ = evaluate_model(
        model, eval_loader, device, task_name, use_glue_metric=use_glue_metric
    )
    final_dir = os.path.join(args.output_dir, "final_model")
    ensure_dir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    total_params, trainable_params, trainable_ratio = count_parameters(model)
    final_active_total_rank, _, _ = active_rank_stats(lora_module_dict)
    rank_mean = float(np.mean(rank_total_history)) if len(rank_total_history) > 0 else 0.0

    results = {
        "task_name": task_name,
        "seed": int(args.seed),
        "best_eval_accuracy": float(best_eval_accuracy),
        "best_eval_loss": float(best_eval_loss),
        "final_eval_accuracy": float(final_eval_accuracy),
        "final_eval_loss": float(final_eval_loss),
        "total_active_rank_mean": rank_mean,
        "total_active_rank_final": int(final_active_total_rank),
        "trainable_params_final": int(trainable_params),
        "total_params": int(total_params),
        "trainable_param_ratio": float(trainable_ratio),
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if use_wandb:
        wandb.log(
            {
                "final/loss": float(final_eval_loss),
                "final/accuracy": float(final_eval_accuracy),
                "final/best_accuracy": float(best_eval_accuracy),
                "final/active_total_rank": int(final_active_total_rank),
            },
            step=global_step,
        )
        wandb.finish()

    print("[done] Training finished.")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal] {type(e).__name__}: {e}")
        raise
