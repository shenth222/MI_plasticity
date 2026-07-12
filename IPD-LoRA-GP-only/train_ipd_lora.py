import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, mean_squared_error
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

from ipd_lora import (
    _avg_loss_over_loader,
    allocate_rank_by_score,
    amp_autocast,
    apply_update_frequency_mask,
    collect_module_rows,
    compute_pretrain_gradient_importance,
    compute_probing_goodput,
    compute_proxy_goodput,
    count_effective_trainable_parameters,
    count_parameters,
    finalize_goodput_stats,
    inject_ipd_lora,
    set_amp,
    set_uniform_rank,
    update_goodput_rank_allocation,
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
    parser = argparse.ArgumentParser(description="GP-only dynamic-rank LoRA training for GLUE with DeBERTa-v3-base.")
    parser.add_argument(
        "--method",
        type=str,
        default="goodput",
        choices=["goodput", "lora", "gora"],
        help=(
            "Rank-allocation method on the allocation-timing axis: "
            "goodput=online dynamic reallocation; lora=static uniform rank; "
            "gora=one-shot pre-training gradient-importance allocation."
        ),
    )
    parser.add_argument(
        "--gora_num_batches",
        type=int,
        default=8,
        help="gora: number of batches used to accumulate pre-training gradient importance.",
    )
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
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision (autocast) for train/eval/scoring/probing forwards.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable transformer gradient checkpointing to reduce activation memory at some speed cost.",
    )
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
    parser.add_argument("--calibration_resample_stride", type=int, default=9973)
    parser.add_argument("--total_rank_budget", type=int, default=0)
    parser.add_argument("--target_rank", type=int, default=4)
    # ----- Module Learning Goodput (GP-only LoRA) -----
    parser.add_argument(
        "--goodput_method",
        type=str,
        default="probing",
        choices=["probing", "proxy"],
        help="probing=one-step causal probe, proxy=low-cost gradient-share proxy; neither uses I/P.",
    )
    parser.add_argument(
        "--goodput_every_n_scoring",
        type=int,
        default=1,
        help="Estimate goodput every N scoring events (the E window in scoring-event units).",
    )
    parser.add_argument("--beta_G", type=float, default=0.9, help="EMA smoothing for goodput.")
    parser.add_argument(
        "--goodput_probe_max_batches",
        type=int,
        default=8,
        help="probing: max calibration batches used per probing loss evaluation.",
    )
    parser.add_argument(
        "--goodput_probe_use_adam",
        action="store_true",
        help="Use Adam preconditioned direction in probing/proxy goodput.",
    )
    parser.add_argument(
        "--goodput_min_rank",
        type=int,
        default=1,
        help="r_min: guaranteed floor rank per module in goodput-only allocation.",
    )
    parser.add_argument("--tfinal_steps", type=int, default=0)
    parser.add_argument("--tfinal_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--calibration_max_batches", type=int, default=16)
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
        load_errors = []
        try:
            # 对齐 LoRA/AdaLoRA：本地 GLUE 根目录优先使用 load_dataset(path, task)
            return load_dataset(args.dataset_path, task)
        except Exception as e:
            load_errors.append(f"load_dataset(path, task) failed: {repr(e)}")
            if task == "stsb":
                try:
                    return load_dataset(args.dataset_path, "sts_b")
                except Exception as e2:
                    load_errors.append(f"load_dataset(path, sts_b) failed: {repr(e2)}")
        try:
            # 兼容 load_from_disk 导出的 DatasetDict
            return load_from_disk(args.dataset_path)
        except Exception as e:
            load_errors.append(f"load_from_disk failed: {repr(e)}")
            raise RuntimeError(
                f"Failed to load dataset from dataset_path={args.dataset_path}\n" + "\n".join(load_errors)
            )

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
    if dataset_config is None and os.path.exists(dataset_name):
        # 兼容把本地路径错误地传给 dataset_name 的场景
        load_errors = []
        try:
            return load_dataset(dataset_name, task)
        except Exception as e:
            load_errors.append(f"load_dataset(dataset_name_path, task) failed: {repr(e)}")
            if task == "stsb":
                try:
                    return load_dataset(dataset_name, "sts_b")
                except Exception as e2:
                    load_errors.append(f"load_dataset(dataset_name_path, sts_b) failed: {repr(e2)}")
        try:
            return load_from_disk(dataset_name)
        except Exception as e:
            load_errors.append(f"load_from_disk(dataset_name_path) failed: {repr(e)}")
            raise RuntimeError(
                f"Failed to load local dataset from dataset_name path={dataset_name}\n"
                + "\n".join(load_errors)
            )
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
    if train_split not in raw:
        raise ValueError(f"Train split '{train_split}' not found in dataset.")

    has_mnli_dual_eval = (
        task == "mnli" and "validation_matched" in raw and "validation_mismatched" in raw
    )
    if has_mnli_dual_eval:
        eval_raw_dict = {
            "matched": raw["validation_matched"],
            "mismatched": raw["validation_mismatched"],
        }
    else:
        eval_split = args.local_eval_split
        if eval_split not in raw and "validation" in raw:
            eval_split = "validation"
        if eval_split not in raw:
            raise ValueError(f"Eval split '{eval_split}' not found in dataset.")
        eval_raw_dict = {"validation": raw[eval_split]}

    train_raw_base = raw[train_split]
    sentence1_key, sentence2_key = _choose_text_keys(args, task, train_raw_base)
    if args.label_column not in train_raw_base.column_names:
        raise ValueError(f"label_column={args.label_column} not found in dataset columns.")

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
        labels = examples[args.label_column]
        if task == "stsb":
            # STS-B is a regression task; keep labels as floats for MSE loss.
            labels = [float(x) for x in labels]
        toks["labels"] = labels
        return toks

    train_ds = train_raw_base.map(preprocess, batched=True, remove_columns=train_raw_base.column_names)
    eval_ds_dict = {
        split_name: ds.map(preprocess, batched=True, remove_columns=ds.column_names)
        for split_name, ds in eval_raw_dict.items()
    }
    return train_ds, eval_ds_dict, raw, train_split


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


# GLUE-standard primary metric per task (used for best-checkpoint selection and
# reporting). Tasks not listed default to accuracy.
PRIMARY_METRIC_BY_TASK = {
    "cola": ["matthews_correlation"],
    "mrpc": ["f1"],
    "qqp": ["f1"],
    "stsb": ["pearson", "spearmanr", "combined_score"],
}


def select_primary_metric(task_name: str, scores: Dict[str, float]) -> float:
    """Pick the GLUE-standard primary metric for a task from a scores dict."""
    if not scores:
        return 0.0
    preferred = PRIMARY_METRIC_BY_TASK.get((task_name or "").lower(), ["accuracy"])
    for key in preferred:
        if key in scores:
            return float(scores[key])
    for key in ("accuracy", "pearson", "matthews_correlation", "f1"):
        if key in scores:
            return float(scores[key])
    return float(next(iter(scores.values())))


@torch.no_grad()
def evaluate_model(model, dataloader, device, task_name: str, use_glue_metric: bool = True):
    model.eval()
    metric = evaluate.load("glue", task_name) if (evaluate is not None and use_glue_metric) else None
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_refs = []
    eval_start = time.perf_counter()
    num_eval_steps = 0
    for batch in dataloader:
        num_eval_steps += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_autocast(device):
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
            pearson_val = pearsonr(preds_np, refs_np)[0] if len(preds_np) > 1 else 0.0
            spearman_val = spearmanr(preds_np, refs_np).correlation if len(preds_np) > 1 else 0.0
            pearson_val = float(0.0 if np.isnan(pearson_val) else pearson_val)
            spearman_val = float(0.0 if np.isnan(spearman_val) else spearman_val)
            rmse = mean_squared_error(refs_np, preds_np, squared=False) if len(preds_np) > 0 else 0.0
            scores = {
                "pearson": pearson_val,
                "spearmanr": spearman_val,
                "rmse": float(rmse),
                "combined_score": float((pearson_val + spearman_val) / 2.0),
            }
        else:
            acc_fallback = float((preds_np == refs_np).mean()) if len(preds_np) > 0 else 0.0
            scores = {"accuracy": acc_fallback}
            if task_name == "cola" and len(preds_np) > 0:
                scores["matthews_correlation"] = float(matthews_corrcoef(refs_np, preds_np))
                scores["combined_score"] = float((scores["accuracy"] + scores["matthews_correlation"]) / 2.0)
            elif task_name in {"mrpc", "qqp"} and len(preds_np) > 0:
                scores["f1"] = float(f1_score(refs_np, preds_np))
                scores["combined_score"] = float((scores["accuracy"] + scores["f1"]) / 2.0)
    avg_loss = total_loss / max(total_n, 1)
    acc = select_primary_metric(task_name, scores)
    runtime = float(max(time.perf_counter() - eval_start, 1e-12))
    samples_per_second = float(total_n / runtime)
    steps_per_second = float(num_eval_steps / runtime)
    perf = {
        "runtime": runtime,
        "samples_per_second": samples_per_second,
        "steps_per_second": steps_per_second,
    }
    return avg_loss, acc, scores, perf


def build_random_calibration_loader(
    train_ds,
    data_collator,
    batch_size: int,
    calibration_size: int,
    seed: int,
    step: int,
    stride: int,
):
    n_total = len(train_ds)
    if n_total <= 1:
        raise ValueError("Training dataset too small for calibration resampling.")
    n = int(max(1, min(calibration_size, n_total)))
    rng = np.random.default_rng(int(seed) + int(step) * int(max(1, stride)))
    indices = rng.choice(n_total, size=n, replace=False)
    subset = train_ds.select(indices.tolist())
    return DataLoader(
        subset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
    )


def evaluate_all_splits(model, eval_loaders, device, task_name: str, use_glue_metric: bool = True):
    split_results = {}
    for split_name, loader in eval_loaders.items():
        loss, primary_metric, scores, perf = evaluate_model(
            model, loader, device, task_name, use_glue_metric=use_glue_metric
        )
        split_results[split_name] = {
            "loss": float(loss),
            "primary_metric": float(primary_metric),
            "scores": {k: float(v) for k, v in scores.items()},
            "perf": {k: float(v) for k, v in perf.items()},
        }
    return split_results


def build_eval_wandb_payload(
    split_results: Dict[str, Dict],
    task_name: str,
    eval_loss: float,
) -> Dict[str, float]:
    payload: Dict[str, float] = {}

    # 与 LoRA/AdaLoRA 对齐：
    # - 非 MNLI: eval/accuracy, eval/f1, eval/loss, eval/runtime...
    # - MNLI: eval/matched_accuracy, eval/mismatched_accuracy, ...
    if task_name == "mnli":
        for split_name, split_row in split_results.items():
            payload[f"eval/{split_name}_loss"] = float(split_row["loss"])
            for perf_name, perf_val in split_row.get("perf", {}).items():
                payload[f"eval/{split_name}_{perf_name}"] = float(perf_val)
            for metric_name, metric_value in split_row["scores"].items():
                payload[f"eval/{split_name}_{metric_name}"] = float(metric_value)
        return payload

    # 单验证集任务使用不带 split 前缀的键名
    first_split = sorted(split_results.keys())[0] if split_results else "validation"
    split_row = split_results.get(first_split, {"scores": {}, "perf": {}})
    payload["eval/loss"] = float(eval_loss)
    for perf_name, perf_val in split_row.get("perf", {}).items():
        payload[f"eval/{perf_name}"] = float(perf_val)
    for metric_name, metric_value in split_row.get("scores", {}).items():
        payload[f"eval/{metric_name}"] = float(metric_value)
    return payload


def choose_primary_eval(split_results: Dict[str, Dict], task_name: str):
    if len(split_results) == 0:
        return "validation", 0.0, 0.0
    if task_name == "mnli" and "matched" in split_results:
        ref = split_results["matched"]
        return "matched", float(ref["primary_metric"]), float(ref["loss"])
    first_split = sorted(split_results.keys())[0]
    ref = split_results[first_split]
    return first_split, float(ref["primary_metric"]), float(ref["loss"])


def active_rank_stats(lora_module_dict):
    active_total_rank = sum(int(m.active_rank) for m in lora_module_dict.values())
    active_module_count = sum(int(m.active_rank > 0) for m in lora_module_dict.values())
    return active_total_rank, active_module_count, 0


def active_lora_cost(lora_module_dict) -> int:
    """Sum of active LoRA parameter cost = sum_m active_rank_m * (in + out).

    Used as a lightweight per-step FLOP proxy for FLOP-goodput.
    """
    return int(sum(int(m.cost) for m in lora_module_dict.values()))


def compute_global_goodput(
    cur_eval_loss: float,
    cur_rank_steps: int,
    cur_flops: float,
    cur_wall: float,
    prev_eval_loss,
    prev_rank_steps: int,
    prev_flops: float,
    prev_wall: float,
) -> Dict[str, float]:
    """Global goodput between two consecutive evaluation points.

    Goodput      = ΔValLoss / rank-step
    Time Goodput = ΔValLoss / second
    FLOP Goodput = ΔValLoss / FLOPs(proxy)
    where ΔValLoss = L_val(prev) - L_val(cur) (positive means improvement).
    """
    if prev_eval_loss is None:
        return {}
    d_loss = float(prev_eval_loss) - float(cur_eval_loss)
    d_rank = max(1, int(cur_rank_steps - prev_rank_steps))
    d_flops = max(1e-9, float(cur_flops - prev_flops))
    d_time = max(1e-9, float(cur_wall - prev_wall))
    return {
        "goodput/delta_val_loss": float(d_loss),
        "goodput/rank_step": float(d_loss / d_rank),
        "goodput/per_second": float(d_loss / d_time),
        "goodput/per_flop": float(d_loss / d_flops),
        "goodput/window_rank_steps": float(d_rank),
        "goodput/window_seconds": float(d_time),
        "goodput/window_flops": float(d_flops),
    }


def cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    """Detached CPU copy of the model state_dict.

    Keeps the best checkpoint in host RAM instead of duplicating the whole
    model on GPU (the previous deepcopy(model.state_dict()) held a full GPU
    copy). Zero training-time cost; only a small host-memory footprint.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def snapshot_ipd_runtime_state(lora_module_dict) -> Dict[str, Dict]:
    snapshot: Dict[str, Dict] = {}
    for name, module in lora_module_dict.items():
        snapshot[name] = {
            "active_rank": int(module.active_rank),
            "target_rank": int(module.target_rank),
            "update_interval": int(module.update_interval),
            "current_G": float(module.current_G),
            "ema_G": float(module.ema_G),
            "G_z": float(module.G_z),
            "G_rank": int(module.G_rank),
            "current_share": float(module.current_share),
        }
    return snapshot


def restore_ipd_runtime_state(lora_module_dict, snapshot: Dict[str, Dict]) -> None:
    if not snapshot:
        return
    for name, module in lora_module_dict.items():
        row = snapshot.get(name)
        if row is None:
            continue
        module.active_rank = int(row.get("active_rank", module.active_rank))
        module.target_rank = int(row.get("target_rank", module.target_rank))
        module.update_interval = int(row.get("update_interval", module.update_interval))
        module.current_G = float(row.get("current_G", module.current_G))
        module.ema_G = float(row.get("ema_G", module.ema_G))
        module.G_z = float(row.get("G_z", module.G_z))
        module.G_rank = int(row.get("G_rank", module.G_rank))
        module.current_share = float(row.get("current_share", module.current_share))


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

    if args.evaluation_strategy == "steps" and args.eval_steps <= 0:
        raise ValueError("evaluation_strategy=steps requires eval_steps > 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    if args.bf16 and not use_bf16:
        print("[warn] --bf16 requested but CUDA is unavailable; running in fp32.")
    set_amp(use_bf16, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_ds, eval_ds_dict, raw, train_split = prepare_datasets(args, tokenizer)
    # STS-B is regression: force a single output head so the model uses MSE loss.
    num_labels = 1 if task_name == "stsb" else infer_num_labels(raw[train_split], args.label_column)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    )
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            print("[memory] Enabled gradient checkpointing.")
        else:
            print("[warn] --gradient_checkpointing requested but this model does not support it.")

    lora_module_dict = inject_ipd_lora(
        model=model,
        target_modules=[
            "query_proj",
            "key_proj",
            "value_proj",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ],
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
    eval_loaders = {
        split_name: DataLoader(
            ds,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        for split_name, ds in eval_ds_dict.items()
    }

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
    init_csv(rank_history_path, ["step", "module_name", "active_rank"])

    best_eval_accuracy = -1e9
    best_eval_loss = 1e9
    best_model_state = None
    best_ipd_state = None
    final_eval_accuracy = 0.0
    final_eval_loss = 0.0
    last_eval_accuracy = None
    last_eval_loss = None
    running_loss = 0.0
    running_steps = 0
    global_step = 0
    rank_total_history: List[int] = []
    active_total_rank, active_module_count, frozen_module_count = active_rank_stats(lora_module_dict)

    # ----- Goodput bookkeeping -----
    goodput_event_idx = 0
    prev_goodput_calib_loss = None  # method A: previous-window calibration loss L_val(t-K)
    train_wall_start = time.perf_counter()
    cum_rank_steps = 0  # sum of active_total_rank over optimizer steps (rank-step budget consumed)
    cum_flop_proxy = 0.0  # proxy for cumulative LoRA training FLOPs
    # Global goodput is measured between consecutive evaluation points.
    prev_eval_loss_for_gp = None
    prev_eval_rank_steps = 0
    prev_eval_flops = 0.0
    prev_eval_wall = train_wall_start

    # Warmup phase for IPD policy: all modules update every step, no dynamic reallocation.
    for module in lora_module_dict.values():
        module.active_rank = max(1, min(args.initial_active_rank, args.max_lora_rank))
        module.target_rank = module.active_rank
        module.update_interval = 1

    # Rank adjustment granularity = 1.
    active_rank_choices = list(range(0, int(args.max_lora_rank) + 1))

    n_lora_modules = len(lora_module_dict)
    target_rank_budget = int(max(1, args.target_rank)) * int(max(1, n_lora_modules))
    if args.total_rank_budget > 0:
        # Keep final average rank <= target_rank.
        effective_rank_budget = min(int(args.total_rank_budget), target_rank_budget)
    else:
        effective_rank_budget = target_rank_budget

    tfinal_steps = int(args.tfinal_steps) if args.tfinal_steps > 0 else int(args.tfinal_ratio * num_training_steps)
    tfinal_steps = max(0, min(tfinal_steps, num_training_steps))
    rank_adapt_end_step = max(0, num_training_steps - tfinal_steps)
    score_event_idx = 0
    planned_scoring_events = 0
    if args.score_interval > 0 and rank_adapt_end_step > args.warmup_steps_for_ipd:
        for step_id in range(args.score_interval, rank_adapt_end_step + 1, args.score_interval):
            if step_id > args.warmup_steps_for_ipd:
                planned_scoring_events += 1
    planned_goodput_events = 0
    if planned_scoring_events > 0:
        planned_goodput_events = planned_scoring_events // max(1, int(args.goodput_every_n_scoring))

    print(
        "[coverage-plan] "
        f"method={args.method}, "
        f"scoring_events={planned_scoring_events if args.method == 'goodput' else 0}, "
        f"goodput_events~={planned_goodput_events if args.method == 'goodput' else 0}, "
        f"modules={len(lora_module_dict)}"
    )

    # ----- Method dispatch on the allocation-timing axis -----
    # goodput: online warmup-then-reallocate (handled inside the training loop).
    # lora:    static uniform rank (no scoring, fixed budget per module).
    # gora:    one-shot pre-training gradient-importance allocation, then frozen.
    uniform_rank = max(1, int(round(effective_rank_budget / max(1, n_lora_modules))))
    if args.method == "lora":
        set_uniform_rank(
            lora_module_dict,
            rank=uniform_rank,
            active_rank_choices=active_rank_choices,
            max_rank=int(args.max_lora_rank),
        )
        print(f"[method=lora] static uniform rank={uniform_rank} for all {n_lora_modules} modules.")
    elif args.method == "gora":
        importance = compute_pretrain_gradient_importance(
            model=model,
            lora_module_dict=lora_module_dict,
            dataloader=train_loader,
            device=device,
            num_batches=int(args.gora_num_batches),
        )
        gora_alloc = allocate_rank_by_score(
            lora_module_dict=lora_module_dict,
            score_map=importance,
            total_rank_budget=effective_rank_budget,
            active_rank_choices=active_rank_choices,
            r_min=int(args.goodput_min_rank),
            max_rank=int(args.max_lora_rank),
        )
        gora_total = sum(gora_alloc.values())
        print(
            f"[method=gora] one-shot allocation over {args.gora_num_batches} batches; "
            f"total_rank={gora_total}/{effective_rank_budget}."
        )
        module_rows = collect_module_rows(lora_module_dict, global_step)
        for row in module_rows:
            write_jsonl(module_scores_path, row)
            append_csv(
                rank_history_path,
                {"step": 0, "module_name": row["module_name"], "active_rank": int(row["active_rank"])},
                ["step", "module_name", "active_rank"],
            )

    num_batches_per_epoch = max(1, len(train_loader))

    for epoch in range(args.num_train_epochs):
        model.train()
        for step_in_epoch, batch in enumerate(train_loader, start=1):
            global_step += 1
            epoch_float = float(epoch + step_in_epoch / num_batches_per_epoch)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device):
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            running_loss += float(loss.item())
            running_steps += 1

            do_scoring = (
                args.method == "goodput"
                and global_step > args.warmup_steps_for_ipd
                and args.score_interval > 0
                and global_step % args.score_interval == 0
                and global_step <= rank_adapt_end_step
            )
            if do_scoring:
                score_event_idx += 1
                calib_loader = build_random_calibration_loader(
                    train_ds=train_ds,
                    data_collator=data_collator,
                    batch_size=args.per_device_eval_batch_size,
                    calibration_size=args.calibration_size,
                    seed=args.seed,
                    step=global_step,
                    stride=args.calibration_resample_stride,
                )

                # ----- Module Learning Goodput estimation -----
                # GP-only does not compute Importance or Plasticity. Rank
                # allocation below consumes only ema_G.
                is_goodput_event = (
                    args.goodput_every_n_scoring <= 1
                    or score_event_idx % int(args.goodput_every_n_scoring) == 0
                )
                if is_goodput_event:
                    goodput_event_idx += 1
                    if args.goodput_method == "probing":
                        compute_probing_goodput(
                            model=model,
                            lora_module_dict=lora_module_dict,
                            eval_dataloader=calib_loader,
                            device=device,
                            learning_rate=float(scheduler.get_last_lr()[0]),
                            beta_G=args.beta_G,
                            max_batches=int(args.goodput_probe_max_batches),
                            module_subset_names=None,
                            optimizer=optimizer,
                            use_adam_direction=bool(args.goodput_probe_use_adam),
                        )
                    else:
                        model.eval()
                        cur_calib_loss = _avg_loss_over_loader(
                            model, calib_loader, device, max_batches=args.calibration_max_batches
                        )
                        if prev_goodput_calib_loss is None:
                            delta_val_loss = 0.0
                        else:
                            delta_val_loss = float(prev_goodput_calib_loss - cur_calib_loss)
                        prev_goodput_calib_loss = float(cur_calib_loss)
                        interval_steps = int(args.score_interval) * int(max(1, args.goodput_every_n_scoring))
                        compute_proxy_goodput(
                            lora_module_dict=lora_module_dict,
                            delta_val_loss=delta_val_loss,
                            interval_steps=interval_steps,
                            learning_rate=float(scheduler.get_last_lr()[0]),
                            optimizer=optimizer,
                            use_adam_direction=bool(args.goodput_probe_use_adam),
                            beta_G=args.beta_G,
                        )
                    model.train()
                else:
                    finalize_goodput_stats(lora_module_dict)

                update_goodput_rank_allocation(
                    lora_module_dict=lora_module_dict,
                    total_rank_budget=effective_rank_budget,
                    active_rank_choices=active_rank_choices,
                    r_min=int(args.goodput_min_rank),
                    max_rank=int(args.max_lora_rank),
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
                # Scoring internally switches to eval mode; always restore train mode.
                model.train()

            # Keep forward active but selectively disable gradient updates by schedule.
            apply_update_frequency_mask(lora_module_dict=lora_module_dict, global_step=global_step)
            optimizer.step()
            scheduler.step()

            active_total_rank, active_module_count, frozen_module_count = active_rank_stats(lora_module_dict)
            rank_total_history.append(active_total_rank)
            cum_rank_steps += int(active_total_rank)
            cum_flop_proxy += float(active_lora_cost(lora_module_dict))

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_train_loss = running_loss / max(running_steps, 1)
                lr = float(scheduler.get_last_lr()[0])
                write_jsonl(
                    training_log_path,
                    {
                        "step": int(global_step),
                        "epoch": float(epoch_float),
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
                    f"[train] step={global_step} epoch={epoch_float:.2f} loss={avg_train_loss:.4f} "
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
                            "train/epoch": float(epoch_float),
                            "epoch": float(epoch_float),
                        },
                        step=global_step,
                    )
                running_loss = 0.0
                running_steps = 0

            should_eval_on_steps = (
                args.evaluation_strategy == "steps"
                and args.eval_steps > 0
                and global_step % args.eval_steps == 0
            )
            if should_eval_on_steps:
                use_glue_metric = task_name in GLUE_TASK_TO_KEYS
                split_results = evaluate_all_splits(
                    model=model,
                    eval_loaders=eval_loaders,
                    device=device,
                    task_name=task_name,
                    use_glue_metric=use_glue_metric,
                )
                primary_split, eval_accuracy, eval_loss = choose_primary_eval(split_results, task_name=task_name)
                last_eval_loss, last_eval_accuracy = eval_loss, eval_accuracy
                gp_now = time.perf_counter()
                gp_payload = compute_global_goodput(
                    cur_eval_loss=eval_loss,
                    cur_rank_steps=cum_rank_steps,
                    cur_flops=cum_flop_proxy,
                    cur_wall=gp_now,
                    prev_eval_loss=prev_eval_loss_for_gp,
                    prev_rank_steps=prev_eval_rank_steps,
                    prev_flops=prev_eval_flops,
                    prev_wall=prev_eval_wall,
                )
                prev_eval_loss_for_gp = float(eval_loss)
                prev_eval_rank_steps = int(cum_rank_steps)
                prev_eval_flops = float(cum_flop_proxy)
                prev_eval_wall = float(gp_now)
                write_jsonl(
                    training_log_path,
                    {
                        "step": int(global_step),
                        "epoch": float(epoch_float),
                        "train_loss": None,
                        "eval_loss": float(eval_loss),
                        "eval_accuracy": float(eval_accuracy),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "active_total_rank": int(active_total_rank),
                        "active_module_count": int(active_module_count),
                        "frozen_module_count": int(frozen_module_count),
                        "eval_primary_split": primary_split,
                        "eval_split_results": split_results,
                        "global_goodput": gp_payload,
                    },
                )
                print(
                    f"[eval] step={global_step} epoch={epoch_float:.2f} primary_split={primary_split} "
                    f"eval_loss={eval_loss:.4f} eval_metric={eval_accuracy:.4f}"
                )
                if use_wandb:
                    eval_log_payload = build_eval_wandb_payload(
                        split_results=split_results,
                        task_name=task_name,
                        eval_loss=eval_loss,
                    )
                    eval_log_payload["epoch"] = float(epoch_float)
                    eval_log_payload.update(gp_payload)
                    wandb.log(
                        eval_log_payload,
                        step=global_step,
                    )
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    best_eval_loss = eval_loss
                    best_model_state = cpu_state_dict(model)
                    best_ipd_state = deepcopy(snapshot_ipd_runtime_state(lora_module_dict))
                    best_dir = os.path.join(args.output_dir, "best_model")
                    ensure_dir(best_dir)
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    with open(os.path.join(best_dir, "ipd_runtime_state.json"), "w", encoding="utf-8") as f:
                        json.dump(best_ipd_state, f, ensure_ascii=False, indent=2)
                maybe_save_checkpoint(args, model, tokenizer, global_step)

        if args.evaluation_strategy == "epoch":
            eval_epoch_float = float(epoch + 1.0)
            active_total_rank, active_module_count, frozen_module_count = active_rank_stats(lora_module_dict)
            use_glue_metric = task_name in GLUE_TASK_TO_KEYS
            split_results = evaluate_all_splits(
                model=model,
                eval_loaders=eval_loaders,
                device=device,
                task_name=task_name,
                use_glue_metric=use_glue_metric,
            )
            primary_split, eval_accuracy, eval_loss = choose_primary_eval(split_results, task_name=task_name)
            last_eval_loss, last_eval_accuracy = eval_loss, eval_accuracy
            gp_now = time.perf_counter()
            gp_payload = compute_global_goodput(
                cur_eval_loss=eval_loss,
                cur_rank_steps=cum_rank_steps,
                cur_flops=cum_flop_proxy,
                cur_wall=gp_now,
                prev_eval_loss=prev_eval_loss_for_gp,
                prev_rank_steps=prev_eval_rank_steps,
                prev_flops=prev_eval_flops,
                prev_wall=prev_eval_wall,
            )
            prev_eval_loss_for_gp = float(eval_loss)
            prev_eval_rank_steps = int(cum_rank_steps)
            prev_eval_flops = float(cum_flop_proxy)
            prev_eval_wall = float(gp_now)
            write_jsonl(
                training_log_path,
                {
                    "step": int(global_step),
                    "epoch": float(eval_epoch_float),
                    "train_loss": None,
                    "eval_loss": float(eval_loss),
                    "eval_accuracy": float(eval_accuracy),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "active_total_rank": int(active_total_rank),
                    "active_module_count": int(active_module_count),
                    "frozen_module_count": int(frozen_module_count),
                    "eval_primary_split": primary_split,
                    "eval_split_results": split_results,
                    "global_goodput": gp_payload,
                },
            )
            print(
                f"[eval] step={global_step} epoch={eval_epoch_float:.2f} primary_split={primary_split} "
                f"eval_loss={eval_loss:.4f} eval_metric={eval_accuracy:.4f}"
            )
            if use_wandb:
                eval_log_payload = build_eval_wandb_payload(
                    split_results=split_results,
                    task_name=task_name,
                    eval_loss=eval_loss,
                )
                eval_log_payload["epoch"] = float(eval_epoch_float)
                eval_log_payload.update(gp_payload)
                wandb.log(
                    eval_log_payload,
                    step=global_step,
                )
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                best_eval_loss = eval_loss
                best_model_state = cpu_state_dict(model)
                best_ipd_state = deepcopy(snapshot_ipd_runtime_state(lora_module_dict))
                best_dir = os.path.join(args.output_dir, "best_model")
                ensure_dir(best_dir)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "ipd_runtime_state.json"), "w", encoding="utf-8") as f:
                    json.dump(best_ipd_state, f, ensure_ascii=False, indent=2)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_ipd_state is not None:
        restore_ipd_runtime_state(lora_module_dict=lora_module_dict, snapshot=best_ipd_state)

    use_glue_metric = task_name in GLUE_TASK_TO_KEYS
    final_split_results = evaluate_all_splits(
        model=model,
        eval_loaders=eval_loaders,
        device=device,
        task_name=task_name,
        use_glue_metric=use_glue_metric,
    )
    final_primary_split, final_eval_accuracy, final_eval_loss = choose_primary_eval(
        final_split_results, task_name=task_name
    )
    final_dir = os.path.join(args.output_dir, "final_model")
    ensure_dir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "ipd_runtime_state.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_ipd_runtime_state(lora_module_dict), f, ensure_ascii=False, indent=2)

    total_params, trainable_params, trainable_ratio = count_parameters(model)
    (
        effective_trainable_params,
        effective_lora_params,
        non_lora_trainable_params,
    ) = count_effective_trainable_parameters(
        model=model,
        lora_module_dict=lora_module_dict,
    )
    final_active_total_rank, _, _ = active_rank_stats(lora_module_dict)
    rank_mean = float(np.mean(rank_total_history)) if len(rank_total_history) > 0 else 0.0

    results = {
        "task_name": task_name,
        "seed": int(args.seed),
        "best_eval_primary_metric": float(best_eval_accuracy),
        "best_eval_accuracy": float(best_eval_accuracy),
        "best_eval_loss": float(best_eval_loss),
        "final_eval_primary_metric": float(final_eval_accuracy),
        "final_eval_accuracy": float(final_eval_accuracy),
        "final_eval_loss": float(final_eval_loss),
        "final_eval_primary_split": final_primary_split,
        "final_eval_split_results": final_split_results,
        "total_active_rank_mean": rank_mean,
        "total_active_rank_final": int(final_active_total_rank),
        "method": str(args.method),
        "goodput_method": str(args.goodput_method) if args.method == "goodput" else None,
        "rank_allocation": str(args.method),
        "goodput_events": int(goodput_event_idx),
        "cum_rank_steps": int(cum_rank_steps),
        "cum_flop_proxy": float(cum_flop_proxy),
        "total_train_wall_seconds": float(time.perf_counter() - train_wall_start),
        "target_rank_budget": int(target_rank_budget),
        "effective_rank_budget": int(effective_rank_budget),
        "planned_scoring_events": int(planned_scoring_events),
        "planned_goodput_events": int(planned_goodput_events),
        "trainable_params_final": int(trainable_params),
        "effective_trainable_params_final": int(effective_trainable_params),
        "effective_lora_params_final": int(effective_lora_params),
        "non_lora_trainable_params_final": int(non_lora_trainable_params),
        "total_params": int(total_params),
        "trainable_param_ratio": float(trainable_ratio),
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if use_wandb:
        final_log_payload = {
            "final/active_total_rank": int(final_active_total_rank),
        }
        if task_name == "mnli":
            for split_name, split_row in final_split_results.items():
                final_log_payload[f"final/{split_name}_loss"] = float(split_row["loss"])
                for perf_name, perf_val in split_row.get("perf", {}).items():
                    final_log_payload[f"final/{split_name}_{perf_name}"] = float(perf_val)
                for metric_name, metric_value in split_row["scores"].items():
                    final_log_payload[f"final/{split_name}_{metric_name}"] = float(metric_value)
        else:
            final_log_payload["final/loss"] = float(final_eval_loss)
            first_split = sorted(final_split_results.keys())[0] if final_split_results else "validation"
            split_row = final_split_results.get(first_split, {"scores": {}, "perf": {}})
            for perf_name, perf_val in split_row.get("perf", {}).items():
                final_log_payload[f"final/{perf_name}"] = float(perf_val)
            for metric_name, metric_value in split_row.get("scores", {}).items():
                final_log_payload[f"final/{metric_name}"] = float(metric_value)
        wandb.log(
            final_log_payload,
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
