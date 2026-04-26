# src/data/glue.py
# Copied from minimal-exp/src/data/glue.py to keep IP-DEF self-contained.
import torch
from datasets import load_dataset
from typing import Dict, Any, List
from functools import partial
import os

GLUE_TASK_CONFIGS = {
    "MNLI": {
        "dataset_name": "glue",
        "subset_name": "mnli",
        "train_split": "train",
        "eval_split": "validation_matched",
        "sentence1_key": "premise",
        "sentence2_key": "hypothesis",
        "label_key": "label",
        "num_labels": 3,
        "metric_for_best_model": "accuracy",
    },
    "RTE": {
        "dataset_name": "glue",
        "subset_name": "rte",
        "train_split": "train",
        "eval_split": "validation",
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "label_key": "label",
        "num_labels": 2,
        "metric_for_best_model": "accuracy",
    },
}


def preprocess_function(examples, tokenizer, s1_key, s2_key, max_len, label_key):
    encoded = tokenizer(
        examples[s1_key],
        examples[s2_key],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    encoded["labels"] = examples[label_key]
    return encoded


def load_glue_dataset(task: str, tokenizer, max_len: int = 256) -> Dict[str, Any]:
    task = task.upper()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Task {task} not supported. Available: {list(GLUE_TASK_CONFIGS.keys())}")
    cfg = GLUE_TASK_CONFIGS[task]

    local_glue_path = os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue")
    if local_glue_path and os.path.exists(local_glue_path):
        raw_datasets = load_dataset(local_glue_path, cfg["subset_name"])
    else:
        raw_datasets = load_dataset(cfg["dataset_name"], cfg["subset_name"])

    train_ds = raw_datasets[cfg["train_split"]]
    eval_ds = raw_datasets[cfg["eval_split"]]

    s1_key = cfg["sentence1_key"]
    s2_key = cfg["sentence2_key"]
    label_key = cfg["label_key"]

    preproc = partial(
        preprocess_function,
        tokenizer=tokenizer,
        s1_key=s1_key,
        s2_key=s2_key,
        max_len=max_len,
        label_key=label_key,
    )

    train_tokenized = train_ds.map(
        preproc, batched=True, num_proc=8,
        remove_columns=train_ds.column_names, desc="Tokenizing train",
    )
    eval_tokenized = eval_ds.map(
        preproc, batched=True, num_proc=8,
        remove_columns=eval_ds.column_names, desc="Tokenizing eval",
    )

    train_tokenized.set_format("torch")
    eval_tokenized.set_format("torch")
    eval_raw = eval_tokenized

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        to_pad = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch[0]:
            to_pad["token_type_ids"] = [item["token_type_ids"] for item in batch]
        padded = tokenizer.pad(to_pad, padding=True, return_tensors="pt")
        result = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
        }
        if "token_type_ids" in padded:
            result["token_type_ids"] = padded["token_type_ids"]
        return result

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        acc = (predictions == labels).mean()
        return {"accuracy": float(acc)}

    return {
        "train": train_tokenized,
        "eval": eval_tokenized,
        "eval_raw": eval_raw,
        "collate_fn": collate_fn,
        "num_labels": cfg["num_labels"],
        "metric_for_best_model": cfg["metric_for_best_model"],
        "compute_metrics": compute_metrics,
    }
