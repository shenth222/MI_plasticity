"""Positive control A.2: phase-heterogeneous importance (online > one-shot).

Two training phases with DIFFERENT planted supports:
  Phase 1 (steps < switch): teacher A plants ΔW on group A (heads 0..K-1)
  Phase 2 (steps >= switch): teacher B plants ΔW on group B (heads K..2K-1)

Budget fits ONE group but not both. Therefore:
  - uniform: always spreads thin
  - gora: one-shot on phase-1 data → locks onto A → fails phase 2
  - goodput: online reallocation → should migrate A→B after the switch
  - oracle_switch: knows current phase support (upper bound)

IMPORTANT: r_min must be >= 1. With r_min=0, phase-1 zeros group B and probing
skips rank-0 modules forever (irreversible allocation trap). That trap is itself
a useful algorithmic finding; the migration demo uses r_min=1.

Primary metrics: phase2_mse, final_budget_on_B.
"""
import argparse
import copy
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipd_lora import (  # noqa: E402
    inject_ipd_lora,
    set_uniform_rank,
    allocate_rank_by_score,
    compute_pretrain_gradient_importance,
    compute_probing_goodput,
    update_goodput_rank_allocation,
)

SYNTH_PROJECTION_PATTERNS = [(r"blocks\.(\d+)\.proj$", "proj")]
SYNTH_LAYER_PATTERN = r"blocks\.(\d+)\."


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return F.gelu(self.proj(x))


class SynthNet(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()
        self.blocks = nn.ModuleList([Head(dim) for _ in range(n_layers)])

    def encode(self, x):
        return torch.stack([blk(x) for blk in self.blocks], dim=1)

    def forward(self, x, labels):
        pred = self.encode(x)
        return SimpleNamespace(loss=F.mse_loss(pred, labels), logits=pred)


def make_batches(x, y, bs):
    ds = TensorDataset(x, y)

    def collate(items):
        return {
            "x": torch.stack([it[0] for it in items]),
            "labels": torch.stack([it[1] for it in items]),
        }

    return DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate)


def plant_teacher(base_state, n_layers, dim, planted_idx, planted_rank, scale, device):
    teacher = SynthNet(n_layers, dim)
    teacher.load_state_dict(base_state)
    with torch.no_grad():
        for i in planted_idx:
            U = torch.randn(dim, planted_rank)
            V = torch.randn(planted_rank, dim)
            dW = (U @ V) * (scale / np.sqrt(planted_rank * dim))
            teacher.blocks[i].proj.weight.add_(dW)
    return teacher.to(device).eval()


@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["labels"].shape[0]
        tot += float(out.loss.item()) * bs
        n += bs
    return tot / max(1, n)


def build_student(base_state, n_layers, dim, max_rank, alpha, device):
    net = SynthNet(n_layers, dim)
    net.load_state_dict(base_state)
    net = net.to(device)
    lora = inject_ipd_lora(
        net, target_modules=["proj"], max_rank=max_rank, alpha=alpha, dropout=0.0,
        initial_active_rank=max_rank, verbose=False,
        projection_patterns=SYNTH_PROJECTION_PATTERNS, layer_pattern=SYNTH_LAYER_PATTERN,
    )
    for m in lora.values():
        m.to(device)
    return net, lora


def budget_frac(lora, names):
    alloc = {n: int(m.active_rank) for n, m in lora.items()}
    total = sum(alloc.values())
    return sum(alloc[n] for n in names) / max(1, total), alloc


def train(net, lora, loaders, device, args, method, names_a, names_b):
    choices = list(range(0, args.max_rank + 1))
    budget = args.budget
    n_mod = len(lora)
    switch = args.steps // 2
    r_min = int(args.r_min)

    if method == "uniform":
        set_uniform_rank(lora, rank=budget // n_mod, active_rank_choices=choices, max_rank=args.max_rank)
    elif method == "oracle_switch":
        for name, m in lora.items():
            m.active_rank = args.planted_rank if name in names_a else r_min
            m.target_rank = m.active_rank
    elif method == "gora":
        imp = compute_pretrain_gradient_importance(
            net, lora, loaders["train_a"], device, num_batches=args.gora_batches
        )
        allocate_rank_by_score(
            lora, imp, total_rank_budget=budget, active_rank_choices=choices,
            r_min=r_min, max_rank=args.max_rank,
        )
    elif method == "goodput":
        set_uniform_rank(lora, rank=budget // n_mod, active_rank_choices=choices, max_rank=args.max_rank)
    else:
        raise ValueError(method)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    mid_budget_a = mid_budget_b = None
    alloc_log = []
    step = 0
    net.train()
    it_a = iter(loaders["train_a"])
    it_b = iter(loaders["train_b"])

    def next_batch(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    while step < args.steps:
        phase2 = step >= switch
        if method == "oracle_switch" and step == switch:
            for name, m in lora.items():
                m.active_rank = args.planted_rank if name in names_b else r_min
                m.target_rank = m.active_rank

        if not phase2:
            batch, it_a = next_batch(it_a, loaders["train_a"])
        else:
            batch, it_b = next_batch(it_b, loaders["train_b"])

        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        out = net(**batch)
        out.loss.backward()

        if method == "goodput" and step > 0 and step % args.gp_interval == 0:
            cal = loaders["cal_b"] if phase2 else loaders["cal_a"]
            compute_probing_goodput(
                net, lora, cal, device, learning_rate=args.lr, max_batches=args.gp_eval_batches
            )
            update_goodput_rank_allocation(
                lora, total_rank_budget=budget, active_rank_choices=choices,
                r_min=r_min, max_rank=args.max_rank,
            )
            fa, _ = budget_frac(lora, names_a)
            fb, alloc = budget_frac(lora, names_b)
            alloc_log.append({"step": step, "budget_a": fa, "budget_b": fb, "alloc": alloc})
            net.train()

        opt.step()
        step += 1

        if step == switch:
            fa, _ = budget_frac(lora, names_a)
            fb, _ = budget_frac(lora, names_b)
            mid_budget_a, mid_budget_b = fa, fb

    mse_a = eval_mse(net, loaders["eval_a"], device)
    mse_b = eval_mse(net, loaders["eval_b"], device)
    fa, alloc = budget_frac(lora, names_a)
    fb, _ = budget_frac(lora, names_b)
    return {
        "method": method,
        "mse_phase1": mse_a,
        "mse_phase2": mse_b,
        "mse_avg": 0.5 * (mse_a + mse_b),
        "mid_budget_a": mid_budget_a,
        "mid_budget_b": mid_budget_b,
        "final_budget_a": fa,
        "final_budget_b": fb,
        "alloc": alloc,
        "alloc_log": alloc_log,
    }


def main():
    ap = argparse.ArgumentParser()
    # Default sized so one phase-group can take full planted_rank under r_min=1:
    # floor = r_min * M = 4; distributable = budget - 4 = 6; two heads get +3 → rank 4.
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--planted_rank", type=int, default=4)
    ap.add_argument("--n_planted", type=int, default=2)
    ap.add_argument("--max_rank", type=int, default=4)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--r_min", type=int, default=1)
    ap.add_argument("--alpha", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--gp_interval", type=int, default=40)
    ap.add_argument("--gp_eval_batches", type=int, default=2)
    ap.add_argument("--gora_batches", type=int, default=8)
    ap.add_argument("--planted_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["uniform", "gora", "goodput", "oracle_switch"])
    ap.add_argument("--out", default="experiments/results/a2.json")
    args = ap.parse_args()

    need = args.r_min * args.n_layers + args.n_planted * (args.planted_rank - args.r_min)
    assert args.budget >= need, (
        f"budget={args.budget} < {need} needed to fit one phase group under r_min={args.r_min}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = SynthNet(args.n_layers, args.dim)
    base_state = copy.deepcopy(base.state_dict())

    idx_a = list(range(0, args.n_planted))
    idx_b = list(range(args.n_planted, 2 * args.n_planted))
    names_a = [f"blocks.{i}.proj" for i in idx_a]
    names_b = [f"blocks.{i}.proj" for i in idx_b]

    teacher_a = plant_teacher(
        base_state, args.n_layers, args.dim, idx_a, args.planted_rank, args.planted_scale, device
    )
    teacher_b = plant_teacher(
        base_state, args.n_layers, args.dim, idx_b, args.planted_rank, args.planted_scale, device
    )

    x_tr = torch.randn(args.n_train, args.dim)
    x_ev = torch.randn(args.n_eval, args.dim)
    with torch.no_grad():
        y_tr_a = teacher_a.encode(x_tr.to(device)).cpu()
        y_tr_b = teacher_b.encode(x_tr.to(device)).cpu()
        y_ev_a = teacher_a.encode(x_ev.to(device)).cpu()
        y_ev_b = teacher_b.encode(x_ev.to(device)).cpu()

    loaders = {
        "train_a": make_batches(x_tr, y_tr_a, args.bs),
        "train_b": make_batches(x_tr, y_tr_b, args.bs),
        "eval_a": make_batches(x_ev, y_ev_a, args.bs),
        "eval_b": make_batches(x_ev, y_ev_b, args.bs),
        "cal_a": make_batches(x_tr[: args.bs * 2], y_tr_a[: args.bs * 2], args.bs),
        "cal_b": make_batches(x_tr[: args.bs * 2], y_tr_b[: args.bs * 2], args.bs),
    }

    results = {}
    for method in args.methods:
        torch.manual_seed(args.seed)
        net, lora = build_student(
            base_state, args.n_layers, args.dim, args.max_rank, args.alpha, device
        )
        r = train(net, lora, loaders, device, args, method, names_a, names_b)
        results[method] = r
        print(
            f"[{method:14s}] mse1={r['mse_phase1']:.3e} mse2={r['mse_phase2']:.3e} "
            f"avg={r['mse_avg']:.3e}  "
            f"mid(A/B)={r['mid_budget_a']}/{r['mid_budget_b']}  "
            f"final(A/B)={r['final_budget_a']:.2f}/{r['final_budget_b']:.2f}"
        )

    summary = {
        "config": vars(args),
        "names_a": names_a,
        "names_b": names_b,
        "mse_phase1": {m: results[m]["mse_phase1"] for m in results},
        "mse_phase2": {m: results[m]["mse_phase2"] for m in results},
        "mse_avg": {m: results[m]["mse_avg"] for m in results},
        "final_budget_a": {m: results[m]["final_budget_a"] for m in results},
        "final_budget_b": {m: results[m]["final_budget_b"] for m in results},
    }
    if "goodput" in results and "gora" in results:
        g2, o2 = results["goodput"]["mse_phase2"], results["gora"]["mse_phase2"]
        summary["phase2_ratio_gora_over_goodput"] = (o2 / g2) if g2 > 0 else None
        summary["goodput_migrated"] = results["goodput"]["final_budget_b"] >= 0.6

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("\n[A.2] summary:")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[A.2] wrote {args.out}")


if __name__ == "__main__":
    main()
