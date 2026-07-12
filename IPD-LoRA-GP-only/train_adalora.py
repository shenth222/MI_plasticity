"""AdaLoRA baseline on the unified GLUE harness.

This is the "training-time complex" end of the allocation-timing axis. It uses
the official peft AdaLoRA (SVD reparam + sensitivity-based budget reallocation)
but shares the *same* data pipeline, evaluation, metric, logging schema and rank
budget as the lora/gora/goodput methods in ``train_ipd_lora.py`` so results are
directly comparable.

Budget parity: peft AdaLoRA injects into the same 6 projections per layer
(72 modules for DeBERTa-v3-base) and prunes from ``init_r`` (= --max_lora_rank)
down to an average ``target_r`` (= --target_rank). The average converged rank
matches the fixed budget used by the other methods.
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

from ipd_lora import set_amp, amp_autocast
from train_ipd_lora import (
    GLUE_TASK_TO_KEYS,
    choose_primary_eval,
    compute_global_goodput,
    cpu_state_dict,
    ensure_dir,
    evaluate_all_splits,
    infer_num_labels,
    prepare_datasets,
    set_seed,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="AdaLoRA baseline for the unified GLUE harness.")
    # data / model (mirror train_ipd_lora so run_glue.sh can dispatch identically)
    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="glue")
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--local_train_split", type=str, default="train")
    p.add_argument("--local_eval_split", type=str, default="validation")
    p.add_argument("--text_column1", type=str, default=None)
    p.add_argument("--text_column2", type=str, default=None)
    p.add_argument("--label_column", type=str, default="label")
    p.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # budget (shared names; map to AdaLoRA init_r / target_r)
    p.add_argument("--target_rank", type=int, default=6, help="AdaLoRA target_r (avg converged rank).")
    p.add_argument("--max_lora_rank", type=int, default=12, help="AdaLoRA init_r (starting rank).")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # AdaLoRA schedule
    p.add_argument("--adalora_tinit_ratio", type=float, default=0.1)
    p.add_argument("--tfinal_ratio", type=float, default=0.15)
    p.add_argument("--adalora_deltaT", type=int, default=10)
    p.add_argument("--adalora_orth_reg_weight", type=float, default=0.5)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=100)
    # wandb (accepted for CLI parity; logging stays local by default)
    p.add_argument("--report_to_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ipd-lora-harness")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    return p.parse_args()


ADALORA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "output.dense", "intermediate.dense"]


def count_parameters(model):
    total, trainable = 0, 0
    for prm in model.parameters():
        n = prm.numel()
        total += n
        if prm.requires_grad:
            trainable += n
    return total, trainable, (trainable / total if total else 0.0)


def adalora_active_total_rank(model) -> int:
    """Sum of currently-active ranks across AdaLoRA layers.

    AdaLoRA prunes singular triplets by zeroing rows of lora_E, so a module's
    live rank is the count of nonzero lora_E entries. Summed over modules this is
    the true adaptation budget spent per step (it starts at init_r*n and shrinks
    to ~target_r*n), which is what the rank-step Goodput integral needs.
    """
    total = 0
    for _, m in model.named_modules():
        e_container = getattr(m, "lora_E", None)
        if e_container is None:
            continue
        # lora_E is a ParameterDict keyed by adapter name in peft.
        params = e_container.values() if hasattr(e_container, "values") else [e_container]
        for e in params:
            total += int((e.detach().abs() > 1e-12).sum().item())
    return total


def _normalize_module_name(name: str) -> str:
    """Strip peft wrapper prefixes so names match the other methods' logs."""
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def snapshot_adalora_runtime_state(model) -> Dict[str, Dict]:
    """Per-module active rank at checkpoint time (schema mirrors ipd_runtime_state)."""
    snapshot: Dict[str, Dict] = {}
    for name, m in model.named_modules():
        e_container = getattr(m, "lora_E", None)
        if e_container is None:
            continue
        params = e_container.values() if hasattr(e_container, "values") else [e_container]
        for e in params:
            active = int((e.detach().abs() > 1e-12).sum().item())
            mod_name = _normalize_module_name(name)
            snapshot[mod_name] = {
                "active_rank": active,
                "target_rank": active,
            }
    return snapshot


def save_adalora_checkpoint(model, tokenizer, out_dir: str) -> None:
    ensure_dir(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "ipd_runtime_state.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_adalora_runtime_state(model), f, ensure_ascii=False, indent=2)


def run_eval_and_log(args, model, eval_loaders, device, task_name, training_log_path, gp_state, epoch_float, global_step, active_total_rank=None):
    use_glue_metric = task_name in GLUE_TASK_TO_KEYS
    split_results = evaluate_all_splits(model, eval_loaders, device, task_name, use_glue_metric=use_glue_metric)
    primary_split, eval_acc, eval_loss = choose_primary_eval(split_results, task_name=task_name)
    now = time.perf_counter()
    gp_payload = compute_global_goodput(
        cur_eval_loss=eval_loss,
        cur_rank_steps=gp_state["cum_rank_steps"],
        cur_flops=gp_state["cum_flop_proxy"],
        cur_wall=now,
        prev_eval_loss=gp_state["prev_eval_loss"],
        prev_rank_steps=gp_state["prev_eval_rank_steps"],
        prev_flops=gp_state["prev_eval_flops"],
        prev_wall=gp_state["prev_eval_wall"],
    )
    gp_state.update(
        prev_eval_loss=float(eval_loss),
        prev_eval_rank_steps=int(gp_state["cum_rank_steps"]),
        prev_eval_flops=float(gp_state["cum_flop_proxy"]),
        prev_eval_wall=float(now),
    )
    write_jsonl(
        training_log_path,
        {
            "step": int(global_step),
            "epoch": float(epoch_float),
            "train_loss": None,
            "eval_loss": float(eval_loss),
            "eval_accuracy": float(eval_acc),
            "eval_primary_split": primary_split,
            "eval_split_results": split_results,
            "active_total_rank": int(active_total_rank) if active_total_rank is not None else None,
            "global_goodput": gp_payload,
        },
    )
    print(f"[eval] step={global_step} epoch={epoch_float:.2f} split={primary_split} loss={eval_loss:.4f} metric={eval_acc:.4f}")
    return split_results, primary_split, eval_acc, eval_loss


def main():
    args = parse_args()
    task_name = args.task_name.lower()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    set_amp(use_bf16, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_ds, eval_ds_dict, raw, train_split = prepare_datasets(args, tokenizer)
    # STS-B is regression: force a single output head so the model uses MSE loss.
    num_labels = 1 if task_name == "stsb" else infer_num_labels(raw[train_split], args.label_column)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_loaders = {
        name: DataLoader(ds, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        for name, ds in eval_ds_dict.items()
    }

    num_training_steps = len(train_loader) * args.num_train_epochs
    tinit = int(args.adalora_tinit_ratio * num_training_steps)
    tfinal = int(args.tfinal_ratio * num_training_steps)

    peft_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.target_rank,
        init_r=args.max_lora_rank,
        target_r=args.target_rank,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=max(1, args.adalora_deltaT),
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=args.adalora_orth_reg_weight,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=ADALORA_TARGET_MODULES,
        total_step=num_training_steps,
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    # Count the AdaLoRA layers actually injected (an AdaLoRA layer exposes lora_E).
    # Note: the "output.dense" suffix matches BOTH attention output and FFN output,
    # so a naive len(target_modules) * num_layers under-counts the modules.
    n_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_E"))
    if n_modules == 0:  # fallback to nominal count
        n_modules = len(ADALORA_TARGET_MODULES) * base_model.config.num_hidden_layers
    target_rank_budget = int(args.target_rank) * int(n_modules)
    print(f"[adalora] injected {n_modules} AdaLoRA layers; target_rank_budget={target_rank_budget}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    lr_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=num_training_steps)

    training_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    gp_state = {
        "cum_rank_steps": 0,
        "cum_flop_proxy": 0.0,
        "prev_eval_loss": None,
        "prev_eval_rank_steps": 0,
        "prev_eval_flops": 0.0,
        "prev_eval_wall": time.perf_counter(),
    }

    best_acc, best_loss, best_state = -1e9, 1e9, None
    train_wall_start = time.perf_counter()
    global_step = 0
    running_loss, running_steps = 0.0, 0
    num_batches_per_epoch = max(1, len(train_loader))

    for epoch in range(args.num_train_epochs):
        model.train()
        for step_in_epoch, batch in enumerate(train_loader, start=1):
            global_step += 1
            epoch_float = float(epoch + step_in_epoch / num_batches_per_epoch)
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_autocast(device):
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            # AdaLoRA budget update must run after step() and before zero_grad().
            try:
                model.base_model.update_and_allocate(global_step)
            except Exception as e:  # defensive: never let allocator kill training
                if global_step <= 1:
                    print(f"[warn] update_and_allocate failed at step {global_step}: {e}")
            optimizer.zero_grad(set_to_none=True)

            active_total_rank = adalora_active_total_rank(model)
            gp_state["cum_rank_steps"] += int(active_total_rank)
            gp_state["cum_flop_proxy"] += float(active_total_rank)
            running_loss += float(loss.item())
            running_steps += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg = running_loss / max(running_steps, 1)
                print(
                    f"[train] step={global_step} epoch={epoch_float:.2f} loss={avg:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} active_rank={active_total_rank}"
                )
                write_jsonl(
                    training_log_path,
                    {
                        "step": int(global_step),
                        "epoch": float(epoch_float),
                        "train_loss": float(avg),
                        "active_total_rank": int(active_total_rank),
                    },
                )
                running_loss, running_steps = 0.0, 0

            if args.evaluation_strategy == "steps" and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                _, _, eval_acc, eval_loss = run_eval_and_log(
                    args,
                    model,
                    eval_loaders,
                    device,
                    task_name,
                    training_log_path,
                    gp_state,
                    epoch_float,
                    global_step,
                    active_total_rank=active_total_rank,
                )
                model.train()
                if eval_acc > best_acc:
                    best_acc, best_loss, best_state = eval_acc, eval_loss, cpu_state_dict(model)
                    save_adalora_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best_model"))

        if args.evaluation_strategy == "epoch":
            active_total_rank = adalora_active_total_rank(model)
            _, _, eval_acc, eval_loss = run_eval_and_log(
                args,
                model,
                eval_loaders,
                device,
                task_name,
                training_log_path,
                gp_state,
                float(epoch + 1.0),
                global_step,
                active_total_rank=active_total_rank,
            )
            model.train()
            if eval_acc > best_acc:
                best_acc, best_loss, best_state = eval_acc, eval_loss, cpu_state_dict(model)
                save_adalora_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best_model"))

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    save_adalora_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final_model"))

    use_glue_metric = task_name in GLUE_TASK_TO_KEYS
    final_split_results = evaluate_all_splits(model, eval_loaders, device, task_name, use_glue_metric=use_glue_metric)
    final_primary_split, final_acc, final_loss = choose_primary_eval(final_split_results, task_name=task_name)

    total_params, trainable_params, trainable_ratio = count_parameters(model)
    results = {
        "task_name": task_name,
        "seed": int(args.seed),
        "best_eval_primary_metric": float(best_acc),
        "best_eval_accuracy": float(best_acc),
        "best_eval_loss": float(best_loss),
        "final_eval_primary_metric": float(final_acc),
        "final_eval_accuracy": float(final_acc),
        "final_eval_loss": float(final_loss),
        "final_eval_primary_split": final_primary_split,
        "final_eval_split_results": final_split_results,
        "method": "adalora",
        "goodput_method": None,
        "rank_allocation": "adalora",
        "adalora_init_r": int(args.max_lora_rank),
        "adalora_target_r": int(args.target_rank),
        "adalora_tinit": int(tinit),
        "adalora_tfinal": int(tfinal),
        "cum_rank_steps": int(gp_state["cum_rank_steps"]),
        "cum_flop_proxy": float(gp_state["cum_flop_proxy"]),
        "total_active_rank_final": int(adalora_active_total_rank(model)),
        "total_train_wall_seconds": float(time.perf_counter() - train_wall_start),
        "target_rank_budget": int(target_rank_budget),
        "effective_rank_budget": int(target_rank_budget),
        "trainable_params_final": int(trainable_params),
        "total_params": int(total_params),
        "trainable_param_ratio": float(trainable_ratio),
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[done] AdaLoRA training finished.")
    print(json.dumps({k: results[k] for k in ["best_eval_accuracy", "final_eval_accuracy", "target_rank_budget"]}, indent=2))


if __name__ == "__main__":
    main()
