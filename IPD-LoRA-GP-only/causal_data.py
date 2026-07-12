"""Data + generative evaluation for causal-LM tasks (commonsense / gsm8k).

Shared by train_causal.py and train_adalora.py so both use the exact same
prompts, tokenization, and answer scoring. Kept separate from the DeBERTa/GLUE
pipeline in train_ipd_lora.py so that path is untouched.

Datasets (local):
  commonsense: /data/shenth/datasets/commonsense
    - merged_commonsense_train.json  (147k Alpaca-style train examples)
    - formatted/<subtask>/<subtask>_validation.json  (per-subtask eval)
      subtasks: arc_c arc_e boolq hella obqa piqa siqa wino
    - answer field is one of true/false | answerN | endingN | solutionN | optionN
    - output field is always "the correct answer is <answer>"
  gsm8k: /data/shenth/datasets/gsm8k/main/*.parquet
    - answer ends with "#### <number>"
"""

import glob
import json
import os
import re
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

COMMONSENSE_SUBTASKS = ["arc_c", "arc_e", "boolq", "hella", "obqa", "piqa", "siqa", "wino"]

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

# Candidate answer tokens for commonsense multiple-choice extraction.
_CS_ANSWER_RE = re.compile(r"(answer\d+|ending\d+|solution\d+|option\d+|true|false)", re.IGNORECASE)
_GSM_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(instruction: str, input_text: str = "") -> str:
    instr = instruction if not input_text else f"{instruction}\n{input_text}"
    return PROMPT_TEMPLATE.format(instruction=instr)


# ---------------- loading raw examples ----------------

def load_commonsense_train(root: str, max_samples: Optional[int] = None) -> List[Dict]:
    path = os.path.join(root, "commonsense", "merged_commonsense_train.json")
    data = _read_json(path)
    examples = []
    for d in data:
        examples.append({
            "prompt": build_prompt(d["instruction"], d.get("input", "")),
            "target": d["output"].strip(),
            "answer": str(d.get("answer", "")).strip(),
            "task": "commonsense",
        })
    if max_samples and len(examples) > max_samples:
        examples = examples[:max_samples]
    return examples


def load_commonsense_eval(root: str, subtasks: Optional[List[str]] = None,
                          per_task: Optional[int] = None) -> List[Dict]:
    subtasks = subtasks or COMMONSENSE_SUBTASKS
    examples = []
    for st in subtasks:
        files = glob.glob(os.path.join(root, "commonsense", "formatted", st, "*_validation.json"))
        if not files:
            continue
        data = _read_json(files[0])
        if per_task:
            data = data[:per_task]
        for d in data:
            examples.append({
                "prompt": build_prompt(d["instruction"], d.get("input", "")),
                "target": d["output"].strip(),
                "answer": str(d.get("answer", "")).strip(),
                "subtask": st,
                "task": "commonsense",
            })
    return examples


def load_gsm8k(root: str, split: str, max_samples: Optional[int] = None) -> List[Dict]:
    import pandas as pd
    pat = os.path.join(root, "gsm8k", "main", f"{split}-*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no gsm8k parquet for split={split} at {pat}")
    df = pd.read_parquet(files[0])
    examples = []
    for _, row in df.iterrows():
        ans = str(row["answer"]).strip()
        examples.append({
            "prompt": build_prompt(
                f"{row['question']}\nAnswer the question. End with '#### <number>'.", ""),
            "target": ans,
            "answer": extract_gsm_gold(ans),
            "task": "gsm8k",
        })
    if max_samples and len(examples) > max_samples:
        examples = examples[:max_samples]
    return examples


def load_task(root: str, task: str, split: str, max_samples: Optional[int] = None,
              subtasks: Optional[List[str]] = None, per_task: Optional[int] = None) -> List[Dict]:
    task = task.lower()
    if task == "commonsense":
        if split == "train":
            return load_commonsense_train(root, max_samples=max_samples)
        return load_commonsense_eval(root, subtasks=subtasks, per_task=per_task)
    if task == "gsm8k":
        gsplit = "train" if split == "train" else "test"
        return load_gsm8k(root, gsplit, max_samples=max_samples)
    raise ValueError(f"unknown causal task: {task}")


# ---------------- answer extraction / scoring ----------------

def extract_gsm_gold(answer_text: str) -> str:
    """Gold number after '####'."""
    if "####" in answer_text:
        tail = answer_text.split("####")[-1]
        m = _GSM_NUM_RE.search(tail)
        if m:
            return m.group(0).replace(",", "")
    m = _GSM_NUM_RE.findall(answer_text)
    return m[-1].replace(",", "") if m else ""


def extract_prediction(generated: str, task: str) -> str:
    if task == "gsm8k":
        if "####" in generated:
            tail = generated.split("####")[-1]
            m = _GSM_NUM_RE.search(tail)
            if m:
                return m.group(0).replace(",", "")
        nums = _GSM_NUM_RE.findall(generated)
        return nums[-1].replace(",", "") if nums else ""
    # commonsense: first candidate token in the generated text
    m = _CS_ANSWER_RE.search(generated)
    return m.group(0).lower() if m else ""


def is_correct(pred: str, gold: str, task: str) -> bool:
    if not pred:
        return False
    if task == "gsm8k":
        try:
            return abs(float(pred) - float(gold)) < 1e-4
        except ValueError:
            return pred == gold
    return pred.lower() == gold.lower()


# ---------------- torch Dataset for training ----------------

class CausalLMDataset(Dataset):
    """Tokenizes prompt+target; masks prompt tokens to -100 in labels."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = self.tok(ex["prompt"], add_special_tokens=True)["input_ids"]
        target_ids = self.tok(ex["target"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tok.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        return {"input_ids": input_ids, "labels": labels}


class CausalCollator:
    """Right-pads input_ids/labels for teacher-forced training."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        self.tok = tokenizer

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def generate_accuracy(model, tokenizer, examples: List[Dict], device, task: str,
                      max_new_tokens: int = 32, batch_size: int = 8,
                      max_samples: Optional[int] = None) -> Dict[str, float]:
    """Left-padded batched generation + answer matching. Returns overall and
    per-subtask accuracy (for commonsense)."""
    if max_samples:
        examples = examples[:max_samples]
    was_training = model.training
    model.eval()
    # decoder-only generation requires left padding
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    use_cache_prev = getattr(model.config, "use_cache", None)
    try:
        model.config.use_cache = True
    except Exception:
        pass

    correct, total = 0, 0
    per_task_hit: Dict[str, List[int]] = {}
    for i in range(0, len(examples), batch_size):
        chunk = examples[i: i + batch_size]
        prompts = [c["prompt"] for c in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for c, txt in zip(chunk, texts):
            pred = extract_prediction(txt, task)
            ok = int(is_correct(pred, c["answer"], task))
            correct += ok
            total += 1
            st = c.get("subtask", task)
            per_task_hit.setdefault(st, []).append(ok)

    tokenizer.padding_side = orig_side
    if use_cache_prev is not None:
        model.config.use_cache = use_cache_prev
    if was_training:
        model.train()

    result = {"accuracy": correct / max(total, 1), "n": total}
    for st, hits in per_task_hit.items():
        result[f"acc_{st}"] = sum(hits) / max(len(hits), 1)
    return result
