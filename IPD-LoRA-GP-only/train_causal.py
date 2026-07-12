"""Causal-LM harness (LLaMA + QLoRA) for the unified rank-allocation study.

Mirrors train_ipd_lora.py's method dispatch (goodput / lora / gora) and logging
schema, but for AutoModelForCausalLM with 4-bit QLoRA on commonsense / gsm8k.
Reuses ipd_lora.py's allocation + goodput logic verbatim and several helpers
from train_ipd_lora.py. The DeBERTa/GLUE path in train_ipd_lora.py is untouched.

Checkpoints save ONLY the LoRA A/B tensors (+ ipd_runtime_state.json) as
model.safetensors, so scripts/effective_rank.py and e4_signal_validity.py work
unchanged.
"""

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

from ipd_lora import (
    LLAMA_LAYER_PATTERN,
    LLAMA_PROJECTION_PATTERNS,
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
from train_ipd_lora import (
    active_lora_cost,
    active_rank_stats,
    append_csv,
    compute_global_goodput,
    ensure_dir,
    freeze_backbone_except_lora_and_classifier,
    init_csv,
    set_seed,
    snapshot_ipd_runtime_state,
    write_jsonl,
)
from causal_data import (
    CausalCollator,
    CausalLMDataset,
    COMMONSENSE_SUBTASKS,
    generate_accuracy,
    load_task,
)

LLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description="Causal-LM QLoRA harness (goodput/lora/gora).")
    p.add_argument("--task_name", type=str, required=True, help="commonsense | gsm8k")
    p.add_argument("--dataset_root", type=str, default="/data/shenth/datasets")
    p.add_argument("--model_name_or_path", type=str, default="/data/shenth/models/llama/3.1-8b")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA 4-bit (nf4).")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=0, help="0 = all")
    p.add_argument("--eval_subset", type=int, default=0, help="0 = all eval; else cap per run")
    p.add_argument("--eval_per_task", type=int, default=0, help="commonsense: cap per subtask (0=all)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    # LoRA / budget
    p.add_argument("--max_lora_rank", type=int, default=32)
    p.add_argument("--initial_active_rank", type=int, default=16)
    p.add_argument("--target_rank", type=int, default=16)
    p.add_argument("--total_rank_budget", type=int, default=0)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # method
    p.add_argument("--method", type=str, default="goodput", choices=["goodput", "lora", "gora"])
    p.add_argument("--goodput_method", type=str, default="proxy", choices=["proxy", "probing"])
    p.add_argument("--goodput_min_rank", type=int, default=1)
    p.add_argument("--score_interval", type=int, default=100)
    p.add_argument("--warmup_steps_for_ipd", type=int, default=100)
    p.add_argument("--goodput_every_n_scoring", type=int, default=1)
    p.add_argument("--beta_G", type=float, default=0.9)
    p.add_argument("--calibration_size", type=int, default=128)
    p.add_argument("--calibration_max_batches", type=int, default=8)
    p.add_argument("--goodput_probe_max_batches", type=int, default=4)
    p.add_argument("--goodput_probe_use_adam", action="store_true")
    p.add_argument("--gora_num_batches", type=int, default=16)
    p.add_argument("--tfinal_ratio", type=float, default=0.15)
    p.add_argument("--tfinal_steps", type=int, default=0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=0, help="0 = eval each epoch only")
    return p.parse_args()


def build_calibration_loader(train_ds, collator, batch_size, calibration_size, seed, step, stride):
    n = len(train_ds)
    if n == 0:
        return None
    rng = np.random.default_rng(seed + (step // max(1, stride)))
    size = min(int(calibration_size), n)
    idx = rng.choice(n, size=size, replace=False).tolist()
    return DataLoader(Subset(train_ds, idx), batch_size=batch_size, shuffle=False, collate_fn=collator)


def save_lora_checkpoint(lora_module_dict, out_dir, runtime_state):
    ensure_dir(out_dir)
    tensors = {}
    for name, m in lora_module_dict.items():
        tensors[f"{name}.lora_A"] = m.lora_A.detach().to(torch.float32).cpu()
        tensors[f"{name}.lora_B"] = m.lora_B.detach().to(torch.float32).cpu()
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "ipd_runtime_state.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_state, f, ensure_ascii=False, indent=2)


def snapshot_lora_state(lora_module_dict) -> Dict[str, torch.Tensor]:
    return {
        name: (m.lora_A.detach().cpu().clone(), m.lora_B.detach().cpu().clone())
        for name, m in lora_module_dict.items()
    }


def restore_lora_state(lora_module_dict, state):
    for name, m in lora_module_dict.items():
        if name in state:
            a, b = state[name]
            with torch.no_grad():
                m.lora_A.copy_(a.to(m.lora_A.device, m.lora_A.dtype))
                m.lora_B.copy_(b.to(m.lora_B.device, m.lora_B.dtype))


def run_generative_eval(model, tokenizer, eval_examples, device, task_name, args):
    return generate_accuracy(
        model, tokenizer, eval_examples, device, task_name,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.per_device_eval_batch_size,
        max_samples=(args.eval_subset or None),
    )


def main():
    args = parse_args()
    task_name = args.task_name.lower()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_index = torch.cuda.current_device() if device.type == "cuda" else 0
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    set_amp(use_bf16, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- data -----
    train_examples = load_task(
        args.dataset_root, task_name, "train",
        max_samples=(args.max_train_samples or None),
    )
    eval_examples = load_task(
        args.dataset_root, task_name, "validation",
        subtasks=COMMONSENSE_SUBTASKS if task_name == "commonsense" else None,
        per_task=(args.eval_per_task or None),
    )
    train_ds = CausalLMDataset(train_examples, tokenizer, max_length=args.max_length)
    collator = CausalCollator(tokenizer)
    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=collator
    )
    print(f"[data] task={task_name} train={len(train_ds)} eval={len(eval_examples)}")

    # ----- model (QLoRA 4-bit) -----
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map={"": device_index} if quant_config is not None else None,
    )
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_module_dict = inject_ipd_lora(
        model=model,
        target_modules=LLAMA_TARGET_MODULES,
        max_rank=args.max_lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        initial_active_rank=args.initial_active_rank,
        verbose=False,
        projection_patterns=LLAMA_PROJECTION_PATTERNS,
        layer_pattern=LLAMA_LAYER_PATTERN,
    )
    if len(lora_module_dict) == 0:
        raise RuntimeError("No LoRA module injected; check LLaMA target module names.")
    freeze_backbone_except_lora_and_classifier(model)
    # move LoRA params onto the quantized model's device
    for m in lora_module_dict.values():
        m.lora_A.data = m.lora_A.data.to(device)
        m.lora_B.data = m.lora_B.data.to(device)
        m.lora_dropout.to(device)
    if quant_config is None:
        model.to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    print(f"[model] injected {len(lora_module_dict)} LoRA modules on {len(set(m.layer_index for m in lora_module_dict.values()))} layers")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // max(1, args.gradient_accumulation_steps))
    num_training_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    training_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    module_scores_path = os.path.join(args.output_dir, "module_scores.jsonl")
    rank_history_path = os.path.join(args.output_dir, "rank_history.csv")
    init_csv(rank_history_path, ["step", "module_name", "active_rank"])

    active_rank_choices = list(range(0, int(args.max_lora_rank) + 1))
    n_lora_modules = len(lora_module_dict)
    target_rank_budget = int(max(1, args.target_rank)) * int(n_lora_modules)
    effective_rank_budget = (
        min(int(args.total_rank_budget), target_rank_budget)
        if args.total_rank_budget > 0 else target_rank_budget
    )
    for m in lora_module_dict.values():
        m.active_rank = max(1, min(args.initial_active_rank, args.max_lora_rank))
        m.target_rank = m.active_rank
        m.update_interval = 1

    tfinal_steps = int(args.tfinal_steps) if args.tfinal_steps > 0 else int(args.tfinal_ratio * num_training_steps)
    rank_adapt_end_step = max(0, num_training_steps - max(0, tfinal_steps))

    # ----- method dispatch (allocation-timing axis) -----
    uniform_rank = max(1, int(round(effective_rank_budget / max(1, n_lora_modules))))
    if args.method == "lora":
        set_uniform_rank(lora_module_dict, rank=uniform_rank,
                         active_rank_choices=active_rank_choices, max_rank=int(args.max_lora_rank))
        print(f"[method=lora] uniform rank={uniform_rank} x {n_lora_modules} modules")
    elif args.method == "gora":
        # small-batch loader for the one-shot importance pass to bound memory
        gora_bs = max(1, min(2, args.per_device_train_batch_size))
        gora_loader = DataLoader(train_ds, shuffle=True, batch_size=gora_bs, collate_fn=collator)
        importance = compute_pretrain_gradient_importance(
            model=model, lora_module_dict=lora_module_dict, dataloader=gora_loader,
            device=device, num_batches=int(args.gora_num_batches))
        allocate_rank_by_score(
            lora_module_dict=lora_module_dict, score_map=importance,
            total_rank_budget=effective_rank_budget, active_rank_choices=active_rank_choices,
            r_min=int(args.goodput_min_rank), max_rank=int(args.max_lora_rank))
        for row in collect_module_rows(lora_module_dict, 0):
            write_jsonl(module_scores_path, row)
        print(f"[method=gora] one-shot allocation over {args.gora_num_batches} batches")

    # ----- training -----
    best_acc, best_loss, best_state = -1e9, 1e9, None
    global_step = 0
    micro_step = 0
    running_loss, running_steps = 0.0, 0
    cum_rank_steps, cum_flop_proxy = 0, 0.0
    prev_gp_calib_loss = None
    score_event_idx, goodput_event_idx = 0, 0
    train_wall_start = time.perf_counter()
    prev_eval_loss_gp, prev_eval_rank_steps, prev_eval_flops = None, 0, 0.0
    prev_eval_wall = train_wall_start
    last_eval_acc, last_eval_loss = None, None

    def do_eval(epoch_float):
        nonlocal best_acc, best_loss, best_state, last_eval_acc, last_eval_loss
        nonlocal prev_eval_loss_gp, prev_eval_rank_steps, prev_eval_flops, prev_eval_wall
        atr, amc, fmc = active_rank_stats(lora_module_dict)
        eval_res = run_generative_eval(model, tokenizer, eval_examples, device, task_name, args)
        # a small calibration loss for goodput bookkeeping (causal LM loss)
        calib_loader = build_calibration_loader(
            train_ds, collator, args.per_device_eval_batch_size,
            args.calibration_size, args.seed, global_step, 9973)
        eval_loss = _avg_loss_over_loader(model, calib_loader, device, max_batches=args.calibration_max_batches)
        model.train()
        eval_acc = float(eval_res["accuracy"])
        last_eval_acc, last_eval_loss = eval_acc, eval_loss
        gp_now = time.perf_counter()
        gp_payload = compute_global_goodput(
            cur_eval_loss=eval_loss, cur_rank_steps=cum_rank_steps, cur_flops=cum_flop_proxy,
            cur_wall=gp_now, prev_eval_loss=prev_eval_loss_gp, prev_rank_steps=prev_eval_rank_steps,
            prev_flops=prev_eval_flops, prev_wall=prev_eval_wall)
        prev_eval_loss_gp = float(eval_loss)
        prev_eval_rank_steps = int(cum_rank_steps)
        prev_eval_flops = float(cum_flop_proxy)
        prev_eval_wall = float(gp_now)
        write_jsonl(training_log_path, {
            "step": int(global_step), "epoch": float(epoch_float), "train_loss": None,
            "eval_loss": float(eval_loss), "eval_accuracy": eval_acc,
            "active_total_rank": int(atr), "eval_detail": eval_res,
            "global_goodput": gp_payload,
        })
        print(f"[eval] step={global_step} epoch={epoch_float:.2f} acc={eval_acc:.4f} loss={eval_loss:.4f} active_rank={atr}")
        if eval_acc > best_acc:
            best_acc, best_loss = eval_acc, eval_loss
            best_state = snapshot_lora_state(lora_module_dict)
            save_lora_checkpoint(lora_module_dict, os.path.join(args.output_dir, "best_model"),
                                 snapshot_ipd_runtime_state(lora_module_dict))

    accum = max(1, args.gradient_accumulation_steps)
    for epoch in range(args.num_train_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step_in_epoch, batch in enumerate(train_loader, start=1):
            micro_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_autocast(device):
                outputs = model(**batch)
                loss = outputs.loss / accum
            loss.backward()
            running_loss += float(loss.item()) * accum
            running_steps += 1
            if micro_step % accum != 0:
                continue
            global_step += 1
            epoch_float = float(epoch + step_in_epoch / max(1, len(train_loader)))

            do_scoring = (
                args.method == "goodput" and global_step > args.warmup_steps_for_ipd
                and args.score_interval > 0 and global_step % args.score_interval == 0
                and global_step <= rank_adapt_end_step
            )
            if do_scoring:
                score_event_idx += 1
                calib_loader = build_calibration_loader(
                    train_ds, collator, args.per_device_eval_batch_size,
                    args.calibration_size, args.seed, global_step, 9973)
                is_gp_event = (args.goodput_every_n_scoring <= 1
                               or score_event_idx % int(args.goodput_every_n_scoring) == 0)
                if is_gp_event:
                    goodput_event_idx += 1
                    if args.goodput_method == "probing":
                        compute_probing_goodput(
                            model=model, lora_module_dict=lora_module_dict, eval_dataloader=calib_loader,
                            device=device, learning_rate=float(scheduler.get_last_lr()[0]),
                            beta_G=args.beta_G, max_batches=int(args.goodput_probe_max_batches),
                            optimizer=optimizer, use_adam_direction=bool(args.goodput_probe_use_adam))
                    else:
                        model.eval()
                        cur_loss = _avg_loss_over_loader(model, calib_loader, device,
                                                         max_batches=args.calibration_max_batches)
                        delta = 0.0 if prev_gp_calib_loss is None else float(prev_gp_calib_loss - cur_loss)
                        prev_gp_calib_loss = float(cur_loss)
                        interval = int(args.score_interval) * int(max(1, args.goodput_every_n_scoring))
                        compute_proxy_goodput(
                            lora_module_dict=lora_module_dict, delta_val_loss=delta,
                            interval_steps=interval, learning_rate=float(scheduler.get_last_lr()[0]),
                            optimizer=optimizer, use_adam_direction=bool(args.goodput_probe_use_adam),
                            beta_G=args.beta_G)
                        model.train()
                else:
                    finalize_goodput_stats(lora_module_dict)
                update_goodput_rank_allocation(
                    lora_module_dict=lora_module_dict, total_rank_budget=effective_rank_budget,
                    active_rank_choices=active_rank_choices, r_min=int(args.goodput_min_rank),
                    max_rank=int(args.max_lora_rank))
                for row in collect_module_rows(lora_module_dict, global_step):
                    write_jsonl(module_scores_path, row)
                    append_csv(rank_history_path,
                               {"step": int(global_step), "module_name": row["module_name"],
                                "active_rank": int(row["active_rank"])},
                               ["step", "module_name", "active_rank"])
                model.train()

            apply_update_frequency_mask(lora_module_dict=lora_module_dict, global_step=global_step)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            atr, amc, fmc = active_rank_stats(lora_module_dict)
            cum_rank_steps += int(atr)
            cum_flop_proxy += float(active_lora_cost(lora_module_dict))

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg = running_loss / max(running_steps, 1)
                lr = float(scheduler.get_last_lr()[0])
                write_jsonl(training_log_path, {
                    "step": int(global_step), "epoch": float(epoch_float), "train_loss": float(avg),
                    "eval_loss": last_eval_loss, "eval_accuracy": last_eval_acc,
                    "learning_rate": lr, "active_total_rank": int(atr),
                })
                print(f"[train] step={global_step} epoch={epoch_float:.2f} loss={avg:.4f} lr={lr:.2e} active_rank={atr}")
                running_loss, running_steps = 0.0, 0

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                do_eval(epoch_float)

        do_eval(float(epoch + 1.0))

    if best_state is not None:
        restore_lora_state(lora_module_dict, best_state)

    final_res = run_generative_eval(model, tokenizer, eval_examples, device, task_name, args)
    final_acc = float(final_res["accuracy"])
    save_lora_checkpoint(lora_module_dict, os.path.join(args.output_dir, "final_model"),
                         snapshot_ipd_runtime_state(lora_module_dict))

    total_params, trainable_params, trainable_ratio = count_parameters(model)
    eff_trainable, eff_lora, non_lora = count_effective_trainable_parameters(
        model=model, lora_module_dict=lora_module_dict)
    final_atr, _, _ = active_rank_stats(lora_module_dict)
    results = {
        "task_name": task_name, "seed": int(args.seed),
        "best_eval_primary_metric": float(best_acc), "best_eval_accuracy": float(best_acc),
        "best_eval_loss": float(best_loss),
        "final_eval_primary_metric": final_acc, "final_eval_accuracy": final_acc,
        "final_eval_detail": final_res,
        "method": str(args.method),
        "goodput_method": str(args.goodput_method) if args.method == "goodput" else None,
        "rank_allocation": str(args.method),
        "goodput_events": int(goodput_event_idx),
        "cum_rank_steps": int(cum_rank_steps), "cum_flop_proxy": float(cum_flop_proxy),
        "total_active_rank_final": int(final_atr),
        "total_train_wall_seconds": float(time.perf_counter() - train_wall_start),
        "target_rank_budget": int(target_rank_budget),
        "effective_rank_budget": int(effective_rank_budget),
        "trainable_params_final": int(trainable_params),
        "effective_trainable_params_final": int(eff_trainable),
        "total_params": int(total_params), "trainable_param_ratio": float(trainable_ratio),
        "model_name_or_path": args.model_name_or_path,
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[done] causal training finished.")
    print(json.dumps({k: results[k] for k in ["best_eval_accuracy", "final_eval_accuracy", "target_rank_budget", "method"]}, indent=2))


if __name__ == "__main__":
    main()
