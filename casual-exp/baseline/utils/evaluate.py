"""
GLUE 评估接口：基于 Hugging Face 官方 run_glue 逻辑，
对 DeBERTa v3 等模型在 GLUE 子任务上做准确率/指标测试。
支持 MNLI 的 matched / mismatched 双路评估。

为什么不直接复用官方 run_glue.py？
--------------------------------
官方脚本: examples/pytorch/text-classification/run_glue.py

差异概要:
1) 使用方式
   - 官方: CLI 脚本，HfArgumentParser + Trainer，需传 --do_eval、--model_name_or_path 等，
     评估时通过 Trainer.evaluate() 在单个 eval split 上跑。
   - 本模块: 函数接口 evaluate_glue(model, tokenizer, task_name, dataset_path)，
     便于在代码里传入「已加载的模型对象」和「本地数据路径」，无需起子进程或改脚本参数。

2) 数据来源
   - 官方: 默认 load_dataset("nyu-mll/glue", task_name)；也可用 dataset_name/config 或本地 csv/json 文件。
   - 本模块: 仅支持「本地数据集路径」dataset_path（load_dataset(dataset_path, task)），
     满足“任务数据集本地路径”的固定入参需求。

3) MNLI 评估
   - 官方: do_eval 时只评估 validation_matched（eval_dataset = raw_datasets["validation_matched"]），
     不跑 validation_mismatched。
   - 本模块: 默认同时评估 matched 与 mismatched，返回 accuracy_matched / accuracy_mismatched（及 accuracy=matched）。

4) 指标实现
   - 官方: evaluate.load("glue", task_name) + metric.compute(...)，并算 combined_score。
   - 本模块: 自实现 accuracy / f1 / pearson / spearman / matthews_correlation，与 run_glue 的 glue 指标一致，
     不依赖 evaluate 库，便于离线或版本可控。

若需与官方完全一致（含 evaluate 库、只 matched）：可直接调用官方脚本，例如:
  python run_glue.py --model_name_or_path <path> --task_name mnli --do_eval \\
    (不传 --do_train 即仅评估；数据默认从 nyu-mll/glue 拉取)
"""

from __future__ import annotations

import os
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


# ---------- 与 run_glue 一致的 GLUE 任务配置 ----------
# 参考: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py

GLUE_TASK_CONFIGS = {
    "cola": {
        "sentence1_key": "sentence",
        "sentence2_key": None,
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["matthews_correlation"],
    },
    "sst2": {
        "sentence1_key": "sentence",
        "sentence2_key": None,
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["accuracy"],
    },
    "mrpc": {
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["accuracy", "f1"],
    },
    "stsb": {
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "label_key": "label",
        "num_labels": 1,
        "is_regression": True,
        "metric_names": ["pearson", "spearman"],
    },
    "qqp": {
        "sentence1_key": "question1",
        "sentence2_key": "question2",
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["accuracy", "f1"],
    },
    "mnli": {
        "sentence1_key": "premise",
        "sentence2_key": "hypothesis",
        "label_key": "label",
        "num_labels": 3,
        "is_regression": False,
        "metric_names": ["accuracy"],
        "eval_splits": ["validation_matched", "validation_mismatched"],
    },
    "qnli": {
        "sentence1_key": "question",
        "sentence2_key": "sentence",
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["accuracy"],
    },
    "rte": {
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["accuracy"],
    },
    "wnli": {
        "sentence1_key": "sentence1",
        "sentence2_key": "sentence2",
        "label_key": "label",
        "num_labels": 2,
        "is_regression": False,
        "metric_names": ["accuracy"],
    },
}


def _simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def _acc_and_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    acc = _simple_accuracy(preds, labels)
    # F1: binary, use pos_label=1
    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    return {"accuracy": acc, "f1": f1}


def _pearson_spearman(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    pearson = float(pearsonr(preds, labels)[0]) if len(preds) > 1 else 0.0
    spearman = float(spearmanr(preds, labels)[0]) if len(preds) > 1 else 0.0
    return {"pearson": pearson, "spearman": spearman}


def _matthews_correlation(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import matthews_corrcoef
    mcc = float(matthews_corrcoef(labels, preds))
    return {"matthews_correlation": mcc}


def get_compute_metrics_fn(
    task_name: str,
) -> callable:
    """返回与 run_glue 一致的 compute_metrics 函数。"""
    task = task_name.lower()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(GLUE_TASK_CONFIGS.keys())}")

    cfg = GLUE_TASK_CONFIGS[task]
    metric_names = cfg["metric_names"]
    is_regression = cfg["is_regression"]

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        if is_regression:
            preds = np.squeeze(predictions.astype(np.float32))
            labels = np.squeeze(labels.astype(np.float32)) if labels.dtype != np.float32 else np.squeeze(labels)
        else:
            preds = np.argmax(predictions, axis=-1)
            labels = np.squeeze(labels.astype(np.int64)) if labels.dtype != np.int64 else np.squeeze(labels)

        if "matthews_correlation" in metric_names:
            return _matthews_correlation(preds, labels)
        if "pearson" in metric_names or "spearman" in metric_names:
            return _pearson_spearman(preds, labels)
        if "f1" in metric_names:
            return _acc_and_f1(preds, labels)
        return {"accuracy": _simple_accuracy(preds, labels)}

    return compute_metrics


def _tokenize_examples(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    s1_key: str,
    s2_key: Optional[str],
    label_key: str,
    max_length: int,
    is_regression: bool,
) -> Dict[str, Any]:
    if s2_key:
        encoded = tokenizer(
            examples[s1_key],
            examples[s2_key],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    else:
        encoded = tokenizer(
            examples[s1_key],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    encoded["labels"] = examples[label_key]
    return encoded


def load_glue_eval_dataset(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    max_length: int = 256,
    eval_splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    从本地路径加载 GLUE 评估集并做 tokenize。
    dataset_path: 本地 GLUE 数据集根目录（与 load_dataset(dataset_path, subset) 兼容的结构）。
    eval_splits: 若指定，则只加载这些 split（如 MNLI 可传 ["validation_matched", "validation_mismatched"]）。
    返回 dict:
      - "datasets": { split_name: Dataset } 的 tokenized 数据集
      - "config": GLUE_TASK_CONFIGS[task]
      - "collate": 用于 DataLoader 的 collate 函数（如需）
    """
    task = task_name.lower()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(GLUE_TASK_CONFIGS.keys())}")

    cfg = GLUE_TASK_CONFIGS[task]
    s1_key = cfg["sentence1_key"]
    s2_key = cfg["sentence2_key"]
    label_key = cfg["label_key"]
    is_regression = cfg["is_regression"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # 确定要加载的 eval splits
    if eval_splits is not None:
        splits_to_load = eval_splits
    elif task == "mnli":
        splits_to_load = cfg.get("eval_splits", ["validation_matched", "validation_mismatched"])
    else:
        splits_to_load = ["validation"]

    # 本地 path：需为 HF load_dataset 可识别的数据根（如含 dataset script 的目录）
    # 或使用 dataset_path="glue" 从 Hub 加载（不传 trust_remote_code 避免告警）
    try:
        raw = load_dataset(dataset_path, task)
    except Exception:
        if task == "stsb":
            raw = load_dataset(dataset_path, "sts_b")
        else:
            raise

    tokenize_fn = lambda ex: _tokenize_examples(
        ex, tokenizer, s1_key, s2_key, label_key, max_length, is_regression
    )

    datasets = {}
    for split in splits_to_load:
        if split not in raw:
            continue
        ds = raw[split]
        tokenized = ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=ds.column_names,
            desc=f"Tokenize {split}",
        )
        tokenized.set_format("torch")
        datasets[split] = tokenized

    return {
        "datasets": datasets,
        "config": cfg,
        "task": task,
    }


def _collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizerBase, label_key: str = "labels") -> Dict[str, torch.Tensor]:
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = torch.stack([b[label_key] if isinstance(b[label_key], torch.Tensor) else torch.tensor(b[label_key]) for b in batch])
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": labels,
    }


def _evaluate_one_split(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device: torch.device,
    is_regression: bool,
    compute_metrics_fn: callable,
) -> Dict[str, float]:
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda b: _collate_fn(b, tokenizer),
        shuffle=False,
    )
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
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
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
    eval_splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    在 GLUE 子任务上评估模型（与 Hugging Face run_glue 逻辑一致）。

    参数:
        model: 已加载的序列分类模型（或 checkpoint 路径，与 tokenizer 一起从路径加载）。
        tokenizer: 分词器；若 model 为路径则可与 model 一起从同路径加载。
        task_name: GLUE 子任务名，如 "mnli", "rte", "sst2", "cola", "mrpc", "qqp", "stsb", "qnli", "wnli"。
        dataset_path: 任务数据集本地路径（GLUE 格式根目录，供 load_dataset(path, config) 使用）。
        max_length: 最大序列长度。
        batch_size: 评估 batch 大小。
        device: 设备；默认自动选择。
        eval_splits: 要评估的 split 列表；MNLI 默认会评估 validation_matched 与 validation_mismatched。

    返回:
        字典，包含该任务所需指标：
        - 仅准确率任务: {"accuracy": float}
        - MRPC/QQP: {"accuracy": float, "f1": float}
        - CoLA: {"matthews_correlation": float}
        - STS-B: {"pearson": float, "spearman": float}
        - MNLI: {"accuracy_matched": float, "accuracy_mismatched": float}，以及可选 "accuracy"（matched 作为主指标）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = task_name.lower()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(GLUE_TASK_CONFIGS.keys())}")
    cfg = GLUE_TASK_CONFIGS[task]

    # 支持传入路径：按任务设置 num_labels / problem_type，避免标签越界 (t >= 0 && t < n_classes)
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
        task_name,
        tokenizer,
        dataset_path,
        max_length=max_length,
        eval_splits=eval_splits,
    )
    datasets = data["datasets"]
    if not datasets:
        raise FileNotFoundError(f"No eval splits found for task {task} under {dataset_path}")

    results: Dict[str, Any] = {}
    for split_name, dataset in datasets.items():
        metrics = _evaluate_one_split(
            model,
            dataset,
            tokenizer,
            batch_size,
            device,
            is_regression,
            compute_metrics_fn,
        )
        if task == "mnli":
            if "mismatched" in split_name:
                results["accuracy_mismatched"] = metrics["accuracy"]
            elif "matched" in split_name:
                results["accuracy_matched"] = metrics["accuracy"]
            # 主指标用 matched 作为 accuracy
            if "accuracy_matched" and "accuracy_mismatched" in results:
                results["accuracy"] = (results["accuracy_matched"]+results["accuracy_mismatched"])/2
            elif "accuracy_matched" in results:
                results["accuracy"] = results["accuracy_matched"]
        else:
            for k, v in metrics.items():
                results[k] = v
    return results


# ---------- 使用示例 ----------
# 依赖: pip install transformers datasets torch scikit-learn scipy
#
# 示例 1：传入模型对象与本地数据集路径
#   from transformers import AutoTokenizer, AutoModelForSequenceClassification
#   from utils.evaluate import evaluate_glue
#   model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=3)
#   tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
#   metrics = evaluate_glue(model, tokenizer, task_name="mnli", dataset_path="/path/to/glue")
#   # metrics -> {"accuracy": ..., "accuracy_matched": ..., "accuracy_mismatched": ...}
#
# 示例 2：仅准确率任务（如 RTE）
#   metrics = evaluate_glue(model, tokenizer, task_name="rte", dataset_path="/path/to/glue")
#   # metrics -> {"accuracy": ...}
#
# 示例 3：传入 checkpoint 路径（tokenizer 可省略，与 model 同路径加载）
#   metrics = evaluate_glue("path/to/ckpt", task_name="mnli", dataset_path="/path/to/glue")
