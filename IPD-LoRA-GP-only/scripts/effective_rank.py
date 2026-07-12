#!/usr/bin/env python
"""Effective-rank utilization: a Pollux-style, cross-method "statistical
efficiency" term for LoRA budget allocation.

Pollux decomposes GOODPUT = system throughput x statistical efficiency, where
statistical efficiency = useful learning per unit of allocated resource. For
LoRA-rank allocation the natural, method-agnostic statistical-efficiency term is
how much of the *allocated rank a module actually uses*: rank that collapses onto
a couple of directions is wasted budget.

For each adapted module we form the update delta_W = B @ A (scale-invariant, so
lora_alpha is irrelevant to the spectrum shape), take singular values sigma_i,
and compute the effective rank (Roy & Vetterli 2007):

    p_i   = sigma_i / sum_j sigma_j
    erank = exp( -sum_i p_i * ln p_i )          in [1, r_alloc]
    util  = erank / r_alloc                       in (0, 1]

util -> 1 : the module spreads learning across all its allocated directions
            (budget well used).
util << 1 : the update is effectively low-rank inside its allocation (budget
            wasted); giving this module less rank would cost almost nothing.

Comparing util across methods answers: does any allocator place rank where the
model can actually use it? If all methods reach the same util, that is direct
mechanistic evidence that *which* modules get rank does not matter here.

Works for lora / gora / goodput (they share the IPDLoRALinear save: full
model.safetensors with .lora_A/.lora_B + ipd_runtime_state.json for per-module
active_rank). AdaLoRA runs did not save adapters, so they need a small re-run
with checkpointing to be included.

Usage:
  python scripts/effective_rank.py --task rte
  python scripts/effective_rank.py --task rte --outputs outputs/budget_r2
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev

import torch
from safetensors import safe_open

METHOD_ORDER = {"lora": 0, "gora": 1, "adalora": 2, "goodput": 3}


def _normalize_module_name(name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _find_weights_file(best_dir: str):
    for fname in ("model.safetensors", "adapter_model.safetensors"):
        path = os.path.join(best_dir, fname)
        if os.path.exists(path):
            return path
    return None


def _module_delta_w(f, mod_prefix: str, r_alloc=None):
    """Return (delta_W, r_used) for IPD-LoRA or AdaLoRA module tensors.

    IPD-LoRA / goodput / lora / gora store full max_rank A/B matrices but only
    the first ``active_rank`` rows/cols are live, so we MUST truncate to the
    allocated rank before forming delta_W = B @ A; otherwise erank can exceed the
    allocation and utilization goes above 1. AdaLoRA masks live rank via lora_E.
    """
    a_key = mod_prefix + ".lora_A"
    b_key = mod_prefix + ".lora_B"
    e_key = mod_prefix + ".lora_E"
    keys = set(f.keys())
    if a_key not in keys or b_key not in keys:
        return None, 0
    A = f.get_tensor(a_key)
    B = f.get_tensor(b_key)
    if e_key in keys:
        E = f.get_tensor(e_key).reshape(-1)
        idx = (E.abs() > 1e-12).nonzero(as_tuple=False).flatten()
        active = int(idx.numel())
        if active <= 0:
            return None, 0
        A_sub = A[idx]
        B_sub = B[:, idx]
        E_sub = E[idx]
        dW = B_sub @ (A_sub * E_sub[:, None])
        return dW, active
    max_r = min(A.shape[0], B.shape[1])
    r = max_r if r_alloc is None else min(int(r_alloc), max_r)
    if r <= 0:
        return None, 0
    dW = B[:, :r] @ A[:r, :]
    return dW, r


def analyze_run(run_dir):
    best_dir = os.path.join(run_dir, "best_model")
    st = _find_weights_file(best_dir)
    if st is None:
        return None
    rs_path = os.path.join(best_dir, "ipd_runtime_state.json")
    alloc = {}
    if os.path.exists(rs_path):
        rs = json.load(open(rs_path, "r", encoding="utf-8"))
        alloc = {k: int(v.get("active_rank", 0)) for k, v in rs.items()}

    eranks, utils, ralloc = [], [], []
    with safe_open(st, framework="pt") as f:
        keys = list(f.keys())
        mods = sorted({_normalize_module_name(k[:-len(".lora_A")]) for k in keys if k.endswith(".lora_A")})
        key_by_mod = {_normalize_module_name(k[:-len(".lora_A")]): k[:-len(".lora_A")] for k in keys if k.endswith(".lora_A")}
        for m in mods:
            prefix = key_by_mod.get(m, m)
            dW, r_used = _module_delta_w(f, prefix, r_alloc=alloc.get(m))
            if dW is None:
                continue
            r = r_used
            if r <= 0:
                continue
            er = effective_rank(dW)
            eranks.append(er)
            ralloc.append(r)
            utils.append(er / r if r > 0 else 0.0)
    if not eranks:
        return None
    return {
        "erank": mean(eranks),
        "r_alloc": mean(ralloc),
        "util": mean(utils),
        "n_modules": len(eranks),
    }


def effective_rank(delta_w):
    """erank via normalized-singular-value entropy (Roy & Vetterli)."""
    sv = torch.linalg.svdvals(delta_w.float())
    sv = sv[sv > 1e-12]
    if sv.numel() == 0:
        return 0.0
    p = sv / sv.sum()
    h = -(p * (p + 1e-30).log()).sum().item()
    return math.exp(h)


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), pstdev(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()

    task_dir = os.path.join(args.outputs, args.task)
    runs = defaultdict(list)
    for st in glob.glob(os.path.join(task_dir, "*", "seed*", "best_model", "*.safetensors")):
        run_dir = os.path.dirname(os.path.dirname(st))
        method = os.path.basename(os.path.dirname(run_dir))
        r = analyze_run(run_dir)
        if r is not None:
            runs[method].append(r)

    if not runs:
        print(f"[erank] no runs with best_model/model.safetensors under {task_dir}")
        print("        (adalora saved no adapter; re-run with checkpointing to include it)")
        return

    header = ["method", "n", "erank", "r_alloc", "utilization"]
    print(f"\n=== {args.task.upper()} effective-rank utilization (statistical efficiency) ===")
    print(" | ".join(header))
    print("-" * 70)
    rows = []
    for method in sorted(runs.keys(), key=lambda m: METHOD_ORDER.get(m, 99)):
        rs = runs[method]
        er_m, er_s = _agg([r["erank"] for r in rs])
        ra_m, _ = _agg([r["r_alloc"] for r in rs])
        ut_m, ut_s = _agg([r["util"] for r in rs])
        rows.append((method, len(rs), er_m, er_s, ra_m, ut_m, ut_s))
        print(" | ".join([
            method, str(len(rs)),
            f"{er_m:.2f}±{er_s:.2f}",
            f"{ra_m:.2f}",
            f"{ut_m:.3f}±{ut_s:.3f}",
        ]))

    out_md = os.path.join(task_dir, "effective_rank.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.task.upper()} 有效秩利用率（Pollux 式统计效率项）\n\n")
        f.write("erank = 归一化奇异值熵的指数（实际用到的秩）；utilization = erank / 分配秩。\n")
        f.write("util→1 分配的秩被充分使用；util≪1 秩被浪费（该模块少给点几乎无损）。\n\n")
        f.write("| method | n | erank | r_alloc | utilization |\n|---|---|---|---|---|\n")
        for method, n, er_m, er_s, ra_m, ut_m, ut_s in rows:
            f.write(f"| {method} | {n} | {er_m:.2f}±{er_s:.2f} | {ra_m:.2f} | {ut_m:.3f}±{ut_s:.3f} |\n")
    print(f"\n[erank] wrote {out_md}")


if __name__ == "__main__":
    main()
