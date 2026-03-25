"""
Data utilities for MNLI activation patching experiments.
Reuses data loading logic from casual-exp/data/glue.py.
"""
import os
import sys
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any, List
from functools import partial


GLUE_DATA_PATH = os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue")


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


def load_mnli(tokenizer, max_len: int = 256, split: str = "validation_matched"):
    """Load MNLI dataset and return tokenized HF Dataset."""
    local_path = GLUE_DATA_PATH
    if local_path and os.path.exists(local_path):
        raw = load_dataset(local_path, "mnli")[split]
    else:
        raw = load_dataset("glue", "mnli")[split]

    preproc = partial(
        preprocess_function,
        tokenizer=tokenizer,
        s1_key="premise",
        s2_key="hypothesis",
        max_len=max_len,
        label_key="label",
    )
    tokenized = raw.map(preproc, batched=True, num_proc=4, remove_columns=raw.column_names,
                        desc=f"Tokenizing MNLI {split}")
    tokenized.set_format("torch")
    return tokenized


def load_mnli_train(tokenizer, max_len: int = 256):
    return load_mnli(tokenizer, max_len, split="train")


def collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate function."""
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


def make_dataloader(dataset, tokenizer, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    """Build a DataLoader with dynamic padding."""
    from functools import partial as _partial
    cfn = _partial(collate_fn, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=cfn)
