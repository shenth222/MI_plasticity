"""AdaLoRA baseline for causal-LM (LLaMA + QLoRA) — the "training-time complex"
end of the allocation-timing axis, mirroring train_causal.py's data/eval/logging.

Independent from train_adalora.py (which stays DeBERTa/GLUE-only) so neither
path can break the other. Reuses AdaLoRA rank bookkeeping helpers from
train_adalora.py and the shared causal data/eval from causal_data.py.
"""

import argparse
import json
import os
import time

import torch
from peft import AdaLoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

from ipd_lora import _avg_loss_over_loader, amp_autocast, set_amp
from train_ipd_lora import compute_global_goodput, ensure_dir, set_seed, write_jsonl
from train_adalora import adalora_active_total_rank, save_adalora_checkpoint
from causal_data import (
    COMMONSENSE_SUBTASKS,
    CausalCollator,
    CausalLMDataset,
    generate_accuracy,
    load_task,
)

LLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description="AdaLoRA causal-LM QLoRA baseline.")
    p.add_argument("--task_name", type=str, required=True, help="commonsense | gsm8k")
    p.add_argument("--dataset_root", type=str, default="/data/shenth/datasets")
    p.add_argument("--model_name_or_path", type=str, default="/data/shenth/models/llama/3.1-8b")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--eval_subset", type=int, default=0)
    p.add_argument("--eval_per_task", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    # budget (shared names -> AdaLoRA init_r / target_r)
    p.add_argument("--target_rank", type=int, default=16, help="AdaLoRA target_r (avg converged rank).")
    p.add_argument("--max_lora_rank", type=int, default=32, help="AdaLoRA init_r (start rank).")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--adalora_tinit_ratio", type=float, default=0.1)
    p.add_argument("--tfinal_ratio", type=float, default=0.15)
    p.add_argument("--adalora_deltaT", type=int, default=10)
    p.add_argument("--adalora_orth_reg_weight", type=float, default=0.5)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--calibration_size", type=int, default=128)
    p.add_argument("--calibration_max_batches", type=int, default=8)
    return p.parse_args()


def build_calibration_loader(train_ds, collator, batch_size, size, seed):
    import numpy as np
    n = len(train_ds)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(int(size), n), replace=False).tolist()
    return DataLoader(Subset(train_ds, idx), batch_size=batch_size, shuffle=False, collate_fn=collator)


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

    train_examples = load_task(args.dataset_root, task_name, "train",
                               max_samples=(args.max_train_samples or None))
    eval_examples = load_task(args.dataset_root, task_name, "validation",
                              subtasks=COMMONSENSE_SUBTASKS if task_name == "commonsense" else None,
                              per_task=(args.eval_per_task or None))
    train_ds = CausalLMDataset(train_examples, tokenizer, max_length=args.max_length)
    collator = CausalCollator(tokenizer)
    train_loader = DataLoader(train_ds, shuffle=True,
                              batch_size=args.per_device_train_batch_size, collate_fn=collator)
    print(f"[data] task={task_name} train={len(train_ds)} eval={len(eval_examples)}")

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map={"": device_index} if quant_config is not None else None)
    if quant_config is not None:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=args.gradient_checkpointing)

    accum = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = max(1, len(train_loader) // accum)
    num_training_steps = steps_per_epoch * args.num_train_epochs
    tinit = int(args.adalora_tinit_ratio * num_training_steps)
    tfinal = int(args.tfinal_ratio * num_training_steps)

    peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.target_rank, init_r=args.max_lora_rank, target_r=args.target_rank,
        tinit=tinit, tfinal=tfinal, deltaT=max(1, args.adalora_deltaT),
        beta1=0.85, beta2=0.85, orth_reg_weight=args.adalora_orth_reg_weight,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LLAMA_TARGET_MODULES, total_step=num_training_steps)
    model = get_peft_model(base_model, peft_config)
    if quant_config is None:
        model.to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.print_trainable_parameters()

    n_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_E"))
    target_rank_budget = int(args.target_rank) * int(max(1, n_modules))
    print(f"[adalora] injected {n_modules} AdaLoRA layers; target_rank_budget={target_rank_budget}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=int(args.warmup_ratio * num_training_steps),
                              num_training_steps=num_training_steps)

    training_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    gp_state = {"cum_rank_steps": 0, "cum_flop_proxy": 0.0, "prev_eval_loss": None,
                "prev_eval_rank_steps": 0, "prev_eval_flops": 0.0, "prev_eval_wall": time.perf_counter()}

    best_acc, best_loss = -1e9, 1e9
    train_wall_start = time.perf_counter()
    global_step, micro_step = 0, 0
    running_loss, running_steps = 0.0, 0
    last_eval_acc, last_eval_loss = None, None

    def do_eval(epoch_float):
        nonlocal best_acc, best_loss, last_eval_acc, last_eval_loss
        atr = adalora_active_total_rank(model)
        eval_res = generate_accuracy(model, tokenizer, eval_examples, device, task_name,
                                     max_new_tokens=args.max_new_tokens,
                                     batch_size=args.per_device_eval_batch_size,
                                     max_samples=(args.eval_subset or None))
        calib = build_calibration_loader(train_ds, collator, args.per_device_eval_batch_size,
                                         args.calibration_size, args.seed)
        eval_loss = _avg_loss_over_loader(model, calib, device, max_batches=args.calibration_max_batches)
        model.train()
        eval_acc = float(eval_res["accuracy"])
        last_eval_acc, last_eval_loss = eval_acc, eval_loss
        now = time.perf_counter()
        gp_payload = compute_global_goodput(
            cur_eval_loss=eval_loss, cur_rank_steps=gp_state["cum_rank_steps"],
            cur_flops=gp_state["cum_flop_proxy"], cur_wall=now,
            prev_eval_loss=gp_state["prev_eval_loss"], prev_rank_steps=gp_state["prev_eval_rank_steps"],
            prev_flops=gp_state["prev_eval_flops"], prev_wall=gp_state["prev_eval_wall"])
        gp_state.update(prev_eval_loss=float(eval_loss), prev_eval_rank_steps=int(gp_state["cum_rank_steps"]),
                        prev_eval_flops=float(gp_state["cum_flop_proxy"]), prev_eval_wall=float(now))
        write_jsonl(training_log_path, {
            "step": int(global_step), "epoch": float(epoch_float), "train_loss": None,
            "eval_loss": float(eval_loss), "eval_accuracy": eval_acc,
            "active_total_rank": int(atr), "eval_detail": eval_res, "global_goodput": gp_payload})
        print(f"[eval] step={global_step} epoch={epoch_float:.2f} acc={eval_acc:.4f} loss={eval_loss:.4f} active_rank={atr}")
        if eval_acc > best_acc:
            best_acc, best_loss = eval_acc, eval_loss
            save_adalora_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best_model"))

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
            optimizer.step()
            scheduler.step()
            try:
                model.base_model.update_and_allocate(global_step)
            except Exception as e:
                if global_step <= 1:
                    print(f"[warn] update_and_allocate failed at step {global_step}: {e}")
            optimizer.zero_grad(set_to_none=True)

            atr = adalora_active_total_rank(model)
            gp_state["cum_rank_steps"] += int(atr)
            gp_state["cum_flop_proxy"] += float(atr)

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg = running_loss / max(running_steps, 1)
                write_jsonl(training_log_path, {"step": int(global_step), "epoch": float(epoch_float),
                                                "train_loss": float(avg), "active_total_rank": int(atr)})
                print(f"[train] step={global_step} epoch={epoch_float:.2f} loss={avg:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} active_rank={atr}")
                running_loss, running_steps = 0.0, 0

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                do_eval(epoch_float)

        do_eval(float(epoch + 1.0))

    save_adalora_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final_model"))
    final_res = generate_accuracy(model, tokenizer, eval_examples, device, task_name,
                                  max_new_tokens=args.max_new_tokens,
                                  batch_size=args.per_device_eval_batch_size,
                                  max_samples=(args.eval_subset or None))
    final_acc = float(final_res["accuracy"])

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results = {
        "task_name": task_name, "seed": int(args.seed),
        "best_eval_primary_metric": float(best_acc), "best_eval_accuracy": float(best_acc),
        "best_eval_loss": float(best_loss),
        "final_eval_primary_metric": final_acc, "final_eval_accuracy": final_acc,
        "final_eval_detail": final_res, "method": "adalora", "goodput_method": None,
        "rank_allocation": "adalora", "adalora_init_r": int(args.max_lora_rank),
        "adalora_target_r": int(args.target_rank), "adalora_tinit": int(tinit), "adalora_tfinal": int(tfinal),
        "cum_rank_steps": int(gp_state["cum_rank_steps"]), "cum_flop_proxy": float(gp_state["cum_flop_proxy"]),
        "total_active_rank_final": int(adalora_active_total_rank(model)),
        "total_train_wall_seconds": float(time.perf_counter() - train_wall_start),
        "target_rank_budget": int(target_rank_budget), "effective_rank_budget": int(target_rank_budget),
        "trainable_params_final": int(trainable), "total_params": int(total),
        "trainable_param_ratio": float(trainable / total if total else 0.0),
        "model_name_or_path": args.model_name_or_path,
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[done] AdaLoRA causal training finished.")
    print(json.dumps({k: results[k] for k in ["best_eval_accuracy", "final_eval_accuracy", "target_rank_budget"]}, indent=2))


if __name__ == "__main__":
    main()
