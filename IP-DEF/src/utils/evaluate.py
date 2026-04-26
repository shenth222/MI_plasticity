"""
GLUE evaluation utility for IP-DEF.

Ported from `casual-exp/utils/evaluate.py` (logic kept aligned with the official
HuggingFace `run_glue.py`):

- Loads the eval split(s) from a local GLUE dataset directory via `load_dataset(path, task)`.
- For MNLI, evaluates both `validation_matched` and `validation_mismatched`.
- Returns the canonical metric dict per task (accuracy / accuracy_matched / f1 / etc.).

This module deliberately has no IP-DEF-specific logic so it can be reused
elsewhere.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# ---------------------------- task configs (run_glue.py compatible) ----------

GLUE_TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cola": {"sentence1_key": "sentence", "sentence2_key": None, "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["matthews_correlation"]},
    "sst2": {"sentence1_key": "sentence", "sentence2_key": None, "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["accuracy"]},
    "mrpc": {"sentence1_key": "sentence1", "sentence2_key": "sentence2", "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["accuracy", "f1"]},
    "stsb": {"sentence1_key": "sentence1", "sentence2_key": "sentence2", "label_key": "label",
              "num_labels": 1, "is_regression": True, "metric_names": ["pearson", "spearman"]},
    "qqp":  {"sentence1_key": "question1", "sentence2_key": "question2", "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["accuracy", "f1"]},
    "mnli": {"sentence1_key": "premise", "sentence2_key": "hypothesis", "label_key": "label",
              "num_labels": 3, "is_regression": False, "metric_names": ["accuracy"],
              "eval_splits": ["validation_matched", "validation_mismatched"]},
    "qnli": {"sentence1_key": "question", "sentence2_key": "sentence", "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["accuracy"]},
    "rte":  {"sentence1_key": "sentence1", "sentence2_key": "sentence2", "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["accuracy"]},
    "wnli": {"sentence1_key": "sentence1", "sentence2_key": "sentence2", "label_key": "label",
              "num_labels": 2, "is_regression": False, "metric_names": ["accuracy"]},
}


# --------------------------------- metric helpers ---------------------------

def _simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def _acc_and_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score
    return {
        "accuracy": _simple_accuracy(preds, labels),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
    }


def _pearson_spearman(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    return {
        "pearson":  float(pearsonr(preds, labels)[0])  if len(preds) > 1 else 0.0,
        "spearman": float(spearmanr(preds, labels)[0]) if len(preds) > 1 else 0.0,
    }


def _matthews_correlation(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import matthews_corrcoef
    return {"matthews_correlation": float(matthews_corrcoef(labels, preds))}


def get_compute_metrics_fn(task_name: str) -> Callable[[Any], Dict[str, float]]:
    task = task_name.lower()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(GLUE_TASK_CONFIGS.keys())}")
    cfg = GLUE_TASK_CONFIGS[task]
    metric_names = cfg["metric_names"]
    is_regression = cfg["is_regression"]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if is_regression:
            preds = np.squeeze(predictions.astype(np.float32))
            labels_ = np.squeeze(labels.astype(np.float32)) if labels.dtype != np.float32 else np.squeeze(labels)
        else:
            preds = np.argmax(predictions, axis=-1)
            labels_ = np.squeeze(labels.astype(np.int64)) if labels.dtype != np.int64 else np.squeeze(labels)

        if "matthews_correlation" in metric_names:
            return _matthews_correlation(preds, labels_)
        if "pearson" in metric_names or "spearman" in metric_names:
            return _pearson_spearman(preds, labels_)
        if "f1" in metric_names:
            return _acc_and_f1(preds, labels_)
        return {"accuracy": _simple_accuracy(preds, labels_)}

    return compute_metrics


# -------------------------------- dataset loading ---------------------------

def _tokenize_examples(examples, tokenizer, s1_key, s2_key, label_key, max_length):
    if s2_key:
        encoded = tokenizer(examples[s1_key], examples[s2_key],
                            truncation=True, max_length=max_length, padding=False)
    else:
        encoded = tokenizer(examples[s1_key],
                            truncation=True, max_length=max_length, padding=False)
    encoded["labels"] = examples[label_key]
    return encoded


def load_glue_eval_dataset(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    max_length: int = 256,
    eval_splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    task = task_name.lower()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(GLUE_TASK_CONFIGS.keys())}")
    cfg = GLUE_TASK_CONFIGS[task]
    s1_key, s2_key = cfg["sentence1_key"], cfg["sentence2_key"]
    label_key = cfg["label_key"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if eval_splits is not None:
        splits_to_load = eval_splits
    elif task == "mnli":
        splits_to_load = cfg.get("eval_splits", ["validation_matched", "validation_mismatched"])
    else:
        splits_to_load = ["validation"]

    try:
        raw = load_dataset(dataset_path, task)
    except Exception:
        if task == "stsb":
            raw = load_dataset(dataset_path, "sts_b")
        else:
            raise

    def tokenize_fn(ex):
        return _tokenize_examples(ex, tokenizer, s1_key, s2_key, label_key, max_length)

    datasets = {}
    for split in splits_to_load:
        if split not in raw:
            continue
        ds = raw[split]
        tokenized = ds.map(
            tokenize_fn, batched=True, remove_columns=ds.column_names,
            desc=f"Tokenize {split}",
        )
        tokenized.set_format("torch")
        datasets[split] = tokenized

    return {"datasets": datasets, "config": cfg, "task": task}


# ----------------------------------- evaluation -----------------------------

def _collate(batch, tokenizer):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = torch.stack([
        b["labels"] if isinstance(b["labels"], torch.Tensor) else torch.tensor(b["labels"])
        for b in batch
    ])
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True, return_tensors="pt",
    )
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": labels,
    }


@torch.no_grad()
def _evaluate_one_split(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device: torch.device,
    is_regression: bool,
    compute_metrics_fn: Callable,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=lambda b: _collate(b, tokenizer), shuffle=False,
    )
    all_preds, all_labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        if is_regression:
            all_preds.append(out.logits.squeeze(-1).cpu().float().numpy())
        else:
            all_preds.append(out.logits.cpu().float().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return compute_metrics_fn((preds, labels))


def evaluate_glue(
    model: Union[PreTrainedModel, str],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    task_name: str = "mnli",
    dataset_path: str = "",
    max_length: int = 256,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    eval_splits: Optional[List[str]] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Evaluate a sequence-classification model on a GLUE subtask.

    Args mirror `casual-exp/utils/evaluate.py`. Adds optional ``autocast_dtype``
    so callers can run eval in bf16 to match training precision.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    task = task_name.lower()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(GLUE_TASK_CONFIGS.keys())}")
    cfg = GLUE_TASK_CONFIGS[task]

    if isinstance(model, str):
        model_path = model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = cfg["num_labels"]
        if cfg.get("is_regression"):
            config.problem_type = "regression"
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    if tokenizer is None:
        raise ValueError("tokenizer must be provided when model is a PreTrainedModel instance")
    model = model.to(device)

    if not dataset_path:
        raise ValueError("dataset_path is required")

    is_regression = cfg["is_regression"]
    compute_metrics_fn = get_compute_metrics_fn(task_name)

    data = load_glue_eval_dataset(
        task_name, tokenizer, dataset_path, max_length=max_length, eval_splits=eval_splits,
    )
    datasets = data["datasets"]
    if not datasets:
        raise FileNotFoundError(f"No eval splits found for task {task} under {dataset_path}")

    results: Dict[str, Any] = {}
    for split_name, dataset in datasets.items():
        metrics = _evaluate_one_split(
            model, dataset, tokenizer, batch_size, device, is_regression, compute_metrics_fn,
            autocast_dtype=autocast_dtype,
        )
        if task == "mnli":
            if "mismatched" in split_name:
                results["accuracy_mismatched"] = metrics["accuracy"]
            elif "matched" in split_name:
                results["accuracy_matched"] = metrics["accuracy"]
            if "accuracy_matched" in results and "accuracy_mismatched" in results:
                results["accuracy"] = (results["accuracy_matched"] + results["accuracy_mismatched"]) / 2.0
            elif "accuracy_matched" in results:
                results["accuracy"] = results["accuracy_matched"]
        else:
            for k, v in metrics.items():
                results[k] = v
    return results
