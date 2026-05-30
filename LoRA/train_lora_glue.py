#!/usr/bin/env python3
"""
使用 LoRA 微调 DeBERTa-v3-base（本地模型与本地 GLUE 数据集）。

核心能力：
1) 仅在 attention 的 query/value 投影上注入 LoRA；
2) rank 可选 2/4/8/16；
3) 训练与评估指标（loss、accuracy、F1、MCC、Pearson 等）记录到 W&B；
4) 统计并记录可训练参数量与占比。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import wandb
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


# GLUE 各任务的输入字段定义
GLUE_TASK_TO_KEYS: Dict[str, Tuple[str, Union[str, None]]] = {
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


@dataclass
class ParamStats:
    trainable_params: int
    total_params: int
    trainable_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetuning for local GLUE datasets.")
    parser.add_argument("--model_path", type=str, required=True, help="本地 DeBERTa-v3-base 模型路径")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="本地 GLUE 数据根路径（优先 load_dataset(path, task)；也兼容 load_from_disk 目录）",
    )
    parser.add_argument("--task_name", type=str, required=True, choices=sorted(GLUE_TASK_TO_KEYS.keys()))
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=0, help="仅 steps 策略下生效，0 表示忽略")
    parser.add_argument("--save_steps", type=int, default=0, help="仅 steps 策略下生效，0 表示忽略")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--rank", type=int, default=8, choices=[2, 4, 8, 16], help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--wandb_project", type=str, default="lora-glue-local")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def ensure_path_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 路径不存在: {path}")


def load_local_dataset(dataset_path: str, task_name: str) -> DatasetDict:
    """
    参考 evaluate.py 的实现：
    1) 优先按 GLUE 根目录加载：load_dataset(dataset_path, task_name)
    2) 若失败，回退到 load_from_disk（兼容已保存的 DatasetDict）
    """
    dataset = None
    load_errors = []

    # 优先按 GLUE 本地数据根路径加载，适配 dataset_path=/.../glue + task_name=rte 这种用法
    try:
        dataset = load_dataset(dataset_path, task_name)
    except Exception as e:
        load_errors.append(f"load_dataset(path, task) 失败: {repr(e)}")
        # STS-B 在一些实现中使用 sts_b 名称
        if task_name == "stsb":
            try:
                dataset = load_dataset(dataset_path, "sts_b")
            except Exception as e2:
                load_errors.append(f"load_dataset(path, sts_b) 失败: {repr(e2)}")

    # 回退：如果用户传入的是 load_from_disk 导出的目录，也可以直接加载
    if dataset is None:
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            load_errors.append(f"load_from_disk 失败: {repr(e)}")
            raise RuntimeError(
                "无法从本地路径加载数据集。请确认 dataset_path 为 GLUE 数据根目录"
                "（可被 load_dataset(path, task) 读取）或是 load_from_disk 导出的目录。\n"
                + "\n".join(load_errors)
            )

    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"数据集必须是 DatasetDict，当前类型: {type(dataset)}")

    required_splits = ["train", "validation"]
    if task_name == "mnli":
        required_splits = ["train", "validation_matched", "validation_mismatched"]

    missing = [split for split in required_splits if split not in dataset]
    if missing:
        raise ValueError(f"数据集中缺少必要 split: {missing}，当前可用: {list(dataset.keys())}")
    return dataset


def build_metrics_fn(task_name: str, is_regression: bool):
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if is_regression:
            # stsb 是回归任务，输出单值分数
            preds = np.squeeze(predictions)
            rmse = mean_squared_error(labels, preds, squared=False)
            pearson_val = pearsonr(labels, preds)[0] if len(np.unique(labels)) > 1 else 0.0
            spearman_val = spearmanr(labels, preds).correlation if len(np.unique(labels)) > 1 else 0.0
            pearson_val = float(0.0 if np.isnan(pearson_val) else pearson_val)
            spearman_val = float(0.0 if np.isnan(spearman_val) else spearman_val)
            return {
                "pearson": pearson_val,
                "spearmanr": spearman_val,
                "rmse": float(rmse),
                "combined_score": float((pearson_val + spearman_val) / 2.0),
            }

        preds = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, preds)
        metrics: Dict[str, float] = {"accuracy": float(accuracy)}

        # 按 GLUE 任务追加关键指标，确保 W&B 中能看到准确率及任务特有指标
        if task_name == "cola":
            metrics["matthews_correlation"] = float(matthews_corrcoef(labels, preds))
        elif task_name in {"mrpc", "qqp"}:
            metrics["f1"] = float(f1_score(labels, preds))

        if "f1" in metrics:
            metrics["combined_score"] = float((metrics["accuracy"] + metrics["f1"]) / 2.0)
        elif "matthews_correlation" in metrics:
            metrics["combined_score"] = float((metrics["accuracy"] + metrics["matthews_correlation"]) / 2.0)
        return metrics

    return compute_metrics


def compute_trainable_ratio(model) -> ParamStats:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params > 0 else 0.0
    return ParamStats(
        trainable_params=int(trainable_params),
        total_params=int(total_params),
        trainable_ratio=float(ratio),
    )


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("fp16 与 bf16 不能同时开启")

    ensure_path_exists(args.model_path, "model_path")
    ensure_path_exists(args.dataset_path, "dataset_path")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    raw_datasets = load_local_dataset(args.dataset_path, args.task_name)
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[args.task_name]
    is_regression = args.task_name == "stsb"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    train_label_feature = raw_datasets["train"].features["label"]

    if is_regression:
        num_labels = 1
        label2id = None
        id2label = None
    else:
        # 尽量从 ClassLabel 读取标签定义，无法读取时回退到数据中动态推断
        if hasattr(train_label_feature, "names") and train_label_feature.names:
            label_list = train_label_feature.names
        else:
            label_list = sorted(set(raw_datasets["train"]["label"]))
        num_labels = len(label_list)
        label2id = {str(label): i for i, label in enumerate(label_list)}
        id2label = {i: str(label) for i, label in enumerate(label_list)}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        local_files_only=True,
        label2id=label2id,
        id2label=id2label,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_proj", "value_proj", "key_proj", "dense", "intermediate.dense", "output.dense"],  # 仅微调 query/value
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    param_stats = compute_trainable_ratio(model)
    print(
        f"Trainable params: {param_stats.trainable_params:,} / "
        f"{param_stats.total_params:,} ({param_stats.trainable_ratio:.4%})"
    )

    wandb.log(
        {
            "meta/trainable_params": param_stats.trainable_params,
            "meta/total_params": param_stats.total_params,
            "meta/trainable_ratio": param_stats.trainable_ratio,
        },
        step=0,
    )

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length=args.max_length)
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=args.max_length,
        )

    # 统一预处理所有 split，保持训练/验证管线一致
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = tokenized_datasets["train"]
    if args.task_name == "mnli":
        # MNLI 同时跟踪 matched/mismatched 两个验证集，Trainer 会分别打日志到 W&B
        eval_dataset = {
            "matched": tokenized_datasets["validation_matched"],
            "mismatched": tokenized_datasets["validation_mismatched"],
        }
        metric_for_best_model = "eval_matched_accuracy"
    else:
        eval_dataset = tokenized_datasets["validation"]
        metric_for_best_model = "eval_pearson" if is_regression else "eval_accuracy"

    eval_steps = args.eval_steps if args.evaluation_strategy == "steps" and args.eval_steps > 0 else None
    save_steps = args.save_steps if args.save_strategy == "steps" and args.save_steps > 0 else None
    load_best_model = args.evaluation_strategy != "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best_model,
        metric_for_best_model=metric_for_best_model if load_best_model else None,
        greater_is_better=True if load_best_model else None,
        report_to=["wandb"],
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        remove_unused_columns=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_metrics_fn(args.task_name, is_regression),
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 训练后再做一次完整评估，确保最终指标落库并写入 W&B
    final_eval_metrics = trainer.evaluate()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", final_eval_metrics)
    trainer.save_metrics("eval", final_eval_metrics)
    trainer.save_state()

    merged_summary = {
        "task_name": args.task_name,
        "is_regression": is_regression,
        "parameter_stats": asdict(param_stats),
        "train_metrics": train_result.metrics,
        "eval_metrics": final_eval_metrics,
    }

    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)

    wandb_run.summary["trainable_params"] = param_stats.trainable_params
    wandb_run.summary["total_params"] = param_stats.total_params
    wandb_run.summary["trainable_ratio"] = param_stats.trainable_ratio
    wandb_run.finish()


if __name__ == "__main__":
    main()
