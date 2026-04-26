# src/train/finetune_ipdef.py
"""
IP-DEF training entry point.

Custom training loop (needed for the per-head grad gating + scaling between
``loss.backward()`` and ``optimizer.step()`` and for the sparse importance
calibration forwards).

Wandb / evaluation features (parallel to ``casual-exp/baseline/train/finetune_glue.py``):

  * **Per-step train loss** is logged to wandb on every optimizer step
    (in addition to a windowed average printed every ``--log_every`` steps).
  * **Custom periodic evaluation**: every ``--eval_every_steps`` steps and at
    the end of every epoch, ``utils.evaluate.evaluate_glue`` is invoked to get
    canonical GLUE metrics (MNLI runs both matched / mismatched).
  * The best checkpoint w.r.t. the task's primary metric is saved to
    ``<out_dir>/ckpt_best`` and copied to ``ckpt_final`` at the end of training.

Example::

    python -m src.train.finetune_ipdef \
        --task MNLI \
        --model_name /data1/shenth/models/deberta/v3-base \
        --dataset_path /data1/shenth/datasets/glue \
        --out_dir outputs/IPDEF/MNLI/seed42 \
        --seed 42 --epochs 3 --bsz 32 --lr 1e-5 \
        --eval_every_steps 500 \
        --budget_ratio 0.3 --T0 300 --K_c 100 --K_I 100 --M 2 --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.glue import load_glue_dataset
from src.ip_def import IPDEFConfig, IPDEFController
from src.utils.evaluate import GLUE_TASK_CONFIGS, evaluate_glue


# ----------------------------- task -> primary metric / metric_for_best ----

# Primary metric used to select the best checkpoint, per task.
# Aligned with HF run_glue.py + casual-exp/utils/evaluate.py.
_TASK_PRIMARY_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "stsb": "pearson",
    "qqp":  "f1",
    "mnli": "accuracy",          # average of matched / mismatched, see evaluate_glue
    "qnli": "accuracy",
    "rte":  "accuracy",
    "wnli": "accuracy",
}


# ----------------------------------------------------- calibration utilities


class CalibrationBatchSampler:
    """Holds a fixed batch on device for sparse importance calibration."""

    def __init__(self, dataloader: DataLoader, device: torch.device):
        it = iter(dataloader)
        batch = next(it)
        self.batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def make_loss_fn(self, model, autocast_dtype=None):
        def _loss_fn():
            if autocast_dtype is not None and self.batch["input_ids"].is_cuda:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = model(**self.batch)
            else:
                out = model(**self.batch)
            return out.loss.detach()

        return _loss_fn


# ------------------------------------------------------------------- main


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_amp_dtype() -> Optional[torch.dtype]:
    """We only enable bf16 autocast (no GradScaler in this custom loop).
    fp16 would need a GradScaler to avoid grad underflow, which we skip here.
    """
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # ---- data / model
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue"),
                    help="local GLUE dataset root used by evaluate_glue")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--eval_bsz", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)

    # ---- optimization
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1)

    # ---- IP-DEF hyperparameters
    ap.add_argument("--budget_ratio", type=float, default=0.3, help="B")
    ap.add_argument("--beta_I", type=float, default=0.95)
    ap.add_argument("--beta_P", type=float, default=0.95)
    ap.add_argument("--T0", type=int, default=300, help="warmup steps")
    ap.add_argument("--K_c", type=int, default=100, help="reselect period")
    ap.add_argument("--K_I", type=int, default=100, help="calibration period")
    ap.add_argument("--M", type=int, default=2, help="min stay periods")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--r_min", type=float, default=0.5)
    ap.add_argument("--r_max", type=float, default=2.0)
    ap.add_argument("--lambda_calib", type=float, default=0.5)
    ap.add_argument("--calib_sample_ratio", type=float, default=0.10)
    ap.add_argument("--calib_group_size", type=int, default=4)

    # ---- runtime / logging
    ap.add_argument("--log_every", type=int, default=20,
                    help="print a windowed average every N optimizer steps "
                         "(per-step loss is always logged to wandb)")
    ap.add_argument("--eval_every_steps", type=int, default=500,
                    help="run evaluate_glue every N optimizer steps; "
                         "set 0 to evaluate only at end of each epoch")
    ap.add_argument("--eval_at_epoch_end", action="store_true", default=True,
                    help="(default true) also evaluate at end of every epoch")
    ap.add_argument("--no_eval_at_epoch_end", dest="eval_at_epoch_end",
                    action="store_false")
    ap.add_argument("--save_signals_every", type=int, default=500)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="IP-DEF")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)
    return ap.parse_args()


def _run_evaluation(
    model,
    tokenizer,
    args,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    primary_metric: str,
) -> Dict[str, float]:
    """Wraps evaluate_glue and returns {metric_name: value, ..., 'primary': ...}."""
    results = evaluate_glue(
        model=model,
        tokenizer=tokenizer,
        task_name=args.task.lower(),
        dataset_path=args.dataset_path,
        max_length=args.max_len,
        batch_size=args.eval_bsz,
        device=device,
        autocast_dtype=amp_dtype,
    )
    if primary_metric in results:
        results["primary"] = results[primary_metric]
    elif "accuracy" in results:
        results["primary"] = results["accuracy"]
    return results


def _save_best_ckpt(model, tokenizer, ckpt_dir: str) -> None:
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _select_amp_dtype()

    task_lc = args.task.lower()
    primary_metric = _TASK_PRIMARY_METRIC.get(task_lc, "accuracy")

    # ---------- model / data
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    ds = load_glue_dataset(args.task, tok, max_len=args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=ds["num_labels"]
    ).to(device)

    # save θ0
    init_dir = os.path.join(args.out_dir, "ckpt_init")
    model.save_pretrained(init_dir)
    tok.save_pretrained(init_dir)

    # dataloaders
    train_dl = DataLoader(
        ds["train"], batch_size=args.bsz, shuffle=True,
        collate_fn=ds["collate_fn"], num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    # calibration uses a fixed small in-domain batch (eval set, deterministic)
    calib_dl = DataLoader(
        ds["eval"], batch_size=args.bsz, shuffle=False,
        collate_fn=ds["collate_fn"], num_workers=0,
    )
    calib_holder = CalibrationBatchSampler(calib_dl, device)

    # ---------- IP-DEF controller
    cfg = IPDEFConfig(
        num_layers=len(model.deberta.encoder.layer),
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
        budget_ratio=args.budget_ratio,
        base_lr=args.lr,
        beta_I=args.beta_I,
        beta_P=args.beta_P,
        K_c=args.K_c,
        K_I=args.K_I,
        T_0=args.T0,
        alpha=args.alpha,
        r_min=args.r_min,
        r_max=args.r_max,
        M=args.M,
        lambda_calib=args.lambda_calib,
        calib_sample_ratio=args.calib_sample_ratio,
        calib_group_size=args.calib_group_size,
    )
    ctl = IPDEFController(model, cfg)

    # ---------- optimizer / scheduler
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params_decay, params_no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (params_no_decay if any(nd in n for nd in no_decay) else params_decay).append(p)
    optim_groups = [
        {"params": params_decay, "weight_decay": args.weight_decay},
        {"params": params_no_decay, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    total_steps = math.ceil(len(train_dl) / args.grad_accum) * int(math.ceil(args.epochs))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---------- wandb
    use_wandb = (not args.no_wandb) and _HAS_WANDB
    if use_wandb:
        run_name = args.run_name or f"{args.task}-IPDEF-B{args.budget_ratio}-seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    # ---------- training loop banner
    print(f"[IP-DEF] device={device} amp={amp_dtype} L={cfg.num_layers} H={cfg.num_heads}")
    print(f"[IP-DEF] budget B={args.budget_ratio}  k_active={ctl.k_active}/{cfg.num_layers*cfg.num_heads}")
    print(f"[IP-DEF] total_steps={total_steps}  warmup_T0={args.T0}  K_c={args.K_c}  K_I={args.K_I}")
    print(f"[IP-DEF] eval_every_steps={args.eval_every_steps}  primary_metric={primary_metric}")
    print(f"[IP-DEF] dataset_path={args.dataset_path}")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    train_log = []
    eval_log = []
    t0 = time.time()
    best_metric = -float("inf")
    best_step = -1
    best_ckpt_dir = os.path.join(args.out_dir, "ckpt_best")

    for epoch in range(int(math.ceil(args.epochs))):
        model.train()
        windowed_loss_sum, windowed_n = 0.0, 0
        for it, batch in enumerate(train_dl):
            if it % args.grad_accum == 0:
                ctl.begin_step()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(**batch)
                    loss = out.loss / args.grad_accum
            else:
                out = model(**batch)
                loss = out.loss / args.grad_accum

            loss.backward()

            # accumulate windowed (sentence-weighted) train loss for printing
            full_loss_val = float(loss.item()) * args.grad_accum
            n_in_micro = int(batch["labels"].shape[0])
            windowed_loss_sum += full_loss_val * n_in_micro
            windowed_n += n_in_micro

            if (it + 1) % args.grad_accum != 0:
                continue

            # ---------- IP-DEF block (between backward and step) ----------
            ctl.update_signals(loss=loss.detach())

            if ctl.should_calibrate():
                with torch.no_grad():
                    baseline = float(calib_holder.make_loss_fn(model, amp_dtype)().item())
                ctl.sparse_importance_calibration(
                    loss_fn=calib_holder.make_loss_fn(model, amp_dtype),
                    baseline_loss=baseline,
                )

            if ctl.should_reselect():
                ctl.update_active_set()

            ctl.apply_grad_control()
            # ---------------------------------------------------------------

            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # -------- per-step wandb logging --------
            if use_wandb:
                wandb.log(
                    {
                        "train/loss_step": full_loss_val,
                        "train/lr": scheduler.get_last_lr()[0],
                        "ipdef/warmup": int(ctl.in_warmup),
                        "ipdef/active_count": int(ctl.active.sum().item()),
                        "ipdef/I_hat_mean": float(ctl.I_hat.mean().item()),
                        "ipdef/P_hat_mean": float(ctl.P_hat.mean().item()),
                        "ipdef/scale_mean_active": float(
                            ctl.scale[ctl.active].mean().item()
                            if ctl.active.any() else 0.0
                        ),
                    },
                    step=global_step,
                )

            # -------- windowed print + train_log --------
            if global_step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = windowed_loss_sum / max(1, windowed_n)
                msg = (
                    f"[ep{epoch} step{global_step:>5d}] "
                    f"loss={avg_loss:.4f} lr={lr_now:.2e} "
                    f"warmup={int(ctl.in_warmup)} active={int(ctl.active.sum().item())}"
                )
                print(msg, flush=True)
                if use_wandb:
                    wandb.log({"train/loss_window": avg_loss}, step=global_step)
                train_log.append({"step": global_step, "loss": avg_loss, "lr": lr_now})
                windowed_loss_sum, windowed_n = 0.0, 0

            # -------- snapshot --------
            if args.save_signals_every and global_step % args.save_signals_every == 0:
                snap = ctl.snapshot()
                snap_path = os.path.join(args.out_dir, f"signals_step{global_step}.pt")
                torch.save(snap, snap_path)

            # -------- step-triggered evaluation --------
            if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                results = _run_evaluation(model, tok, args, device, amp_dtype, primary_metric)
                results_log = {"step": global_step, **results}
                eval_log.append(results_log)
                print(f"[eval @ step {global_step}] {results}", flush=True)

                if use_wandb:
                    wandb.log(
                        {f"eval/{k}": v for k, v in results.items()
                         if isinstance(v, (int, float))},
                        step=global_step,
                    )

                cur = results.get("primary", -float("inf"))
                if cur > best_metric:
                    best_metric = cur
                    best_step = global_step
                    _save_best_ckpt(model, tok, best_ckpt_dir)
                    print(f"[eval] new best {primary_metric}={cur:.4f} @ step {global_step}, "
                          f"saved -> {best_ckpt_dir}", flush=True)
                    if use_wandb:
                        wandb.log(
                            {"eval/best_primary": best_metric, "eval/best_step": best_step},
                            step=global_step,
                        )
                model.train()

        # -------- end-of-epoch evaluation --------
        if args.eval_at_epoch_end:
            results = _run_evaluation(model, tok, args, device, amp_dtype, primary_metric)
            results_log = {"step": global_step, "epoch": epoch, **results}
            eval_log.append(results_log)
            print(f"[eval @ end of epoch {epoch}] {results}", flush=True)

            if use_wandb:
                wandb.log(
                    {f"eval/{k}": v for k, v in results.items()
                     if isinstance(v, (int, float))},
                    step=global_step,
                )

            cur = results.get("primary", -float("inf"))
            if cur > best_metric:
                best_metric = cur
                best_step = global_step
                _save_best_ckpt(model, tok, best_ckpt_dir)
                print(f"[eval] new best {primary_metric}={cur:.4f} @ step {global_step}, "
                      f"saved -> {best_ckpt_dir}", flush=True)
                if use_wandb:
                    wandb.log(
                        {"eval/best_primary": best_metric, "eval/best_step": best_step},
                        step=global_step,
                    )
            model.train()

    # ---------- save final / best
    final_dir = os.path.join(args.out_dir, "ckpt_final")
    if os.path.isdir(best_ckpt_dir):
        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir, ignore_errors=True)
        shutil.copytree(best_ckpt_dir, final_dir)
        print(f"[save] θ1 (best, {primary_metric}={best_metric:.4f} @ step {best_step}) "
              f"copied to {final_dir}")
    else:
        model.save_pretrained(final_dir)
        tok.save_pretrained(final_dir)
        print(f"[save] θ1 (last) saved to {final_dir} (no eval ran)")

    snap = ctl.snapshot()
    torch.save(snap, os.path.join(args.out_dir, "signals_final.pt"))

    summary: Dict[str, Any] = {
        "task": args.task,
        "model_name": args.model_name,
        "method": "IP-DEF",
        "primary_metric": primary_metric,
        "best_primary": best_metric if math.isfinite(best_metric) else None,
        "best_step": best_step,
        "total_steps": global_step,
        "wallclock_seconds": time.time() - t0,
        "config": vars(args),
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    with open(os.path.join(args.out_dir, "eval_log.json"), "w") as f:
        json.dump(eval_log, f, indent=2)

    ctl.remove()
    if use_wandb:
        wandb.finish()
    print(f"[IP-DEF] done. best_{primary_metric}={best_metric:.4f} @ step {best_step}  "
          f"steps={global_step}  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
