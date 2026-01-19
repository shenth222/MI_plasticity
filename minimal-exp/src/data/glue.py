# src/data/glue.py
import torch
from datasets import load_dataset
from typing import Dict, Any, List

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


def load_glue_dataset(task: str, tokenizer, max_len: int = 256) -> Dict[str, Any]:
    """
    Load and preprocess GLUE dataset for the given task.
    
    Returns a dict with:
    - train: tokenized HF dataset for training
    - eval: tokenized HF dataset for evaluation
    - eval_raw: raw eval dataset that returns dict items (for Subset + DataLoader)
    - collate_fn: function to collate batches
    - num_labels: int
    - metric_for_best_model: str
    - compute_metrics: function
    """
    task = task.upper()
    if task not in GLUE_TASK_CONFIGS:
        raise ValueError(f"Task {task} not supported. Available: {list(GLUE_TASK_CONFIGS.keys())}")
    
    cfg = GLUE_TASK_CONFIGS[task]
    
    # Load dataset from HuggingFace
    # raw_datasets = load_dataset(cfg["dataset_name"], cfg["subset_name"])
    raw_datasets = load_dataset("/data1/shenth/datasets/glue", "mnli")
    train_ds = raw_datasets[cfg["train_split"]]
    eval_ds = raw_datasets[cfg["eval_split"]]
    
    s1_key = cfg["sentence1_key"]
    s2_key = cfg["sentence2_key"]
    label_key = cfg["label_key"]
    
    # Tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples[s1_key],
            examples[s2_key],
            truncation=True,
            max_length=max_len,
            padding=False,  # dynamic padding in collate_fn
        )
    
    # Tokenize
    print("Tokenizing train")
    train_tokenized = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train"
    )
    train_tokenized = train_tokenized.add_column("labels", train_ds[label_key])
    
    print("Tokenizing eval")
    eval_tokenized = eval_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval"
    )
    eval_tokenized = eval_tokenized.add_column("labels", eval_ds[label_key])
    
    print("Setting format to torch tensors")
    # Set format to torch tensors
    train_tokenized.set_format("torch")
    eval_tokenized.set_format("torch")
    
    # For eval_raw, we keep the tokenized dataset but ensure it returns dict
    # (Subset + DataLoader will use this)
    eval_raw = eval_tokenized
    
    # Collate function
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of tokenized examples into padded tensors.
        """
        # batch is a list of dicts with keys: input_ids, attention_mask, labels, (token_type_ids)
        # We pad input_ids and attention_mask to the max length in the batch
        
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        
        # Pad using tokenizer
        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
        }
        
        # Add token_type_ids if present
        if "token_type_ids" in batch[0]:
            token_type_ids = [item["token_type_ids"] for item in batch]
            padded_tt = tokenizer.pad(
                {"token_type_ids": token_type_ids},
                padding=True,
                return_tensors="pt"
            )
            result["token_type_ids"] = padded_tt["token_type_ids"]
        
        return result
    
    # Compute metrics function
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
