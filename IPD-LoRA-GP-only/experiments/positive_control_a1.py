"""Positive control A.1: planted low-rank recovery.

Constructs a controlled scenario where the module-importance signal is GENUINELY
predictive, so dynamic/importance-based rank allocation *should* beat uniform.
This validates the "positive half" of the signal-predictability criterion
(complementing the null results on GLUE/LLaMA).

Setup
-----
- A frozen synthetic deep net (residual GELU blocks). Each block has one Linear
  `proj` that we adapt with LoRA (reusing ipd_lora's IPDLoRALinear + allocators).
- A *teacher* = same frozen net but with a rank-`planted_rank` update ΔW planted
  on a KNOWN subset of blocks (the "planted support"); all other blocks unchanged.
- We train student LoRA (frozen base) to match teacher outputs (MSE).

Why allocation matters
-----------------------
A block on the planted support needs active_rank >= planted_rank to fit its ΔW
exactly. With a total budget B < M*planted_rank, uniform under-ranks the planted
blocks and cannot fit; concentrating the budget on the planted support fits
exactly. So: oracle < gora ~ goodput << uniform in final MSE (lower is better).

Methods (all budget-conserved, same base + data + seed)
  uniform  : static equal rank        (lora baseline)
  gora     : one-shot gradient importance -> allocate_rank_by_score
  goodput  : online probing goodput -> update_goodput_rank_allocation
  oracle   : full rank on planted support, 0 elsewhere (upper bound)
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
    """Frozen bank of INDEPENDENT parallel heads.

    Each head maps x -> its own output slot; the total loss is the sum of
    per-head MSEs. This decouples modules: head i's rank only affects output
    slot i, so a planted ΔW on head i can ONLY be fit by giving head i rank
    (no cross-head compensation). Importance is therefore exactly localized on
    the planted support -- the clean setting to test whether allocation helps.
    """

    def __init__(self, n_layers, dim):
        super().__init__()
        self.blocks = nn.ModuleList([Head(dim) for _ in range(n_layers)])

    def encode(self, x):
        return torch.stack([blk(x) for blk in self.blocks], dim=1)  # [B, M, dim]

    def forward(self, x, labels):
        pred = self.encode(x)
        loss = F.mse_loss(pred, labels)
        return SimpleNamespace(loss=loss, logits=pred)


def make_batches(x, y, bs):
    ds = TensorDataset(x, y)

    def collate(items):
        xb = torch.stack([it[0] for it in items])
        yb = torch.stack([it[1] for it in items])
        return {"x": xb, "labels": yb}

    return DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate)


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


def train(net, lora, train_loader, eval_loader, cal_loader, device, args, method, planted_names):
    choices = list(range(0, args.max_rank + 1))
    budget = args.budget
    n_mod = len(lora)

    if method == "uniform":
        set_uniform_rank(lora, rank=budget // n_mod, active_rank_choices=choices, max_rank=args.max_rank)
    elif method == "oracle":
        for name, m in lora.items():
            m.active_rank = args.planted_rank if name in planted_names else 0
            m.target_rank = m.active_rank
    elif method == "gora":
        imp = compute_pretrain_gradient_importance(net, lora, train_loader, device, num_batches=args.gora_batches)
        allocate_rank_by_score(lora, imp, total_rank_budget=budget, active_rank_choices=choices,
                               r_min=0, max_rank=args.max_rank)
    elif method == "goodput":
        set_uniform_rank(lora, rank=budget // n_mod, active_rank_choices=choices, max_rank=args.max_rank)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    goodput_log = []
    step = 0
    curve = []
    net.train()
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = net(**batch)
            out.loss.backward()

            if method == "goodput" and step > 0 and step % args.gp_interval == 0:
                # grads are populated -> probe (rolls back its own perturbations), then reallocate
                compute_probing_goodput(net, lora, cal_loader, device, learning_rate=args.lr, max_batches=args.gp_eval_batches)
                update_goodput_rank_allocation(lora, total_rank_budget=budget, active_rank_choices=choices,
                                               r_min=0, max_rank=args.max_rank)
                goodput_log.append({n: float(m.ema_G) for n, m in lora.items()})
                net.train()

            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                curve.append((step, eval_mse(net, eval_loader, device)))
                net.train()

    final_mse = eval_mse(net, eval_loader, device)
    alloc = {n: int(m.active_rank) for n, m in lora.items()}
    budget_on_planted = sum(alloc[n] for n in planted_names) / max(1, sum(alloc.values()))
    return {
        "method": method,
        "final_mse": final_mse,
        "alloc": alloc,
        "budget_on_planted": budget_on_planted,
        "curve": curve,
        "goodput_log": goodput_log,
    }


def signal_predictability(goodput_log, planted_names):
    """persistence = mean lag-1 cross-module autocorr of ema_G;
    planted_topk_recall = fraction of realloc events where planted modules are the top-|planted| by ema_G."""
    if len(goodput_log) < 2:
        return {"persistence": None, "planted_topk_recall": None}
    names = list(goodput_log[0].keys())
    mat = np.array([[ev[n] for n in names] for ev in goodput_log])  # [T, M]
    cors = []
    for t in range(mat.shape[0] - 1):
        a, b = mat[t], mat[t + 1]
        if a.std() > 1e-9 and b.std() > 1e-9:
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    persistence = float(np.mean(cors)) if cors else None
    k = len(planted_names)
    hits = 0
    for ev in goodput_log:
        top = set(sorted(names, key=lambda n: ev[n], reverse=True)[:k])
        hits += len(top & set(planted_names)) / max(1, k)
    return {"persistence": persistence, "planted_topk_recall": hits / len(goodput_log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--planted_rank", type=int, default=8)
    ap.add_argument("--n_planted", type=int, default=2)
    ap.add_argument("--max_rank", type=int, default=8)
    ap.add_argument("--budget", type=int, default=16, help="total sum of active ranks across modules")
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--gp_interval", type=int, default=50)
    ap.add_argument("--gp_eval_batches", type=int, default=2)
    ap.add_argument("--gora_batches", type=int, default=8)
    ap.add_argument("--planted_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["uniform", "gora", "goodput", "oracle"])
    ap.add_argument("--out", default="experiments/results/a1.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # frozen base
    base = SynthNet(args.n_layers, args.dim)
    base_state = copy.deepcopy(base.state_dict())

    # planted support + teacher (base + low-rank ΔW on planted blocks)
    planted_idx = list(range(0, args.n_planted))  # first n_planted blocks
    planted_names = [f"blocks.{i}.proj" for i in planted_idx]
    teacher = SynthNet(args.n_layers, args.dim)
    teacher.load_state_dict(base_state)
    with torch.no_grad():
        for i in planted_idx:
            U = torch.randn(args.dim, args.planted_rank)
            V = torch.randn(args.planted_rank, args.dim)
            dW = (U @ V) * (args.planted_scale / np.sqrt(args.planted_rank * args.dim))
            teacher.blocks[i].proj.weight.add_(dW)
    teacher = teacher.to(device).eval()

    # data
    x_tr = torch.randn(args.n_train, args.dim)
    x_ev = torch.randn(args.n_eval, args.dim)
    with torch.no_grad():
        y_tr = teacher.encode(x_tr.to(device)).cpu()
        y_ev = teacher.encode(x_ev.to(device)).cpu()
    train_loader = make_batches(x_tr, y_tr, args.bs)
    eval_loader = make_batches(x_ev, y_ev, args.bs)
    cal_loader = make_batches(x_tr[: args.bs * 2], y_tr[: args.bs * 2], args.bs)

    results = {}
    for method in args.methods:
        torch.manual_seed(args.seed)  # same LoRA init per method
        net, lora = build_student(base_state, args.n_layers, args.dim, args.max_rank, args.alpha, device)
        r = train(net, lora, train_loader, eval_loader, cal_loader, device, args, method, planted_names)
        if method == "goodput":
            r["predictability"] = signal_predictability(r["goodput_log"], planted_names)
        results[method] = r
        pred = r.get("predictability", {})
        print(f"[{method:8s}] final_mse={r['final_mse']:.4e}  budget_on_planted={r['budget_on_planted']:.2f}"
              + (f"  persist={pred.get('persistence')}  recall={pred.get('planted_topk_recall')}" if pred else ""))

    # headline: gap vs uniform (positive => dynamic helps)
    uni = results.get("uniform", {}).get("final_mse")
    summary = {"config": vars(args), "planted_names": planted_names,
               "final_mse": {m: results[m]["final_mse"] for m in results},
               "budget_on_planted": {m: results[m]["budget_on_planted"] for m in results}}
    if uni:
        summary["mse_ratio_to_uniform"] = {m: results[m]["final_mse"] / uni for m in results}
    if "goodput" in results:
        summary["goodput_predictability"] = results["goodput"].get("predictability")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("\n[A.1] summary:")
    print(json.dumps(summary, indent=2))
    print(f"[A.1] wrote {args.out}")


if __name__ == "__main__":
    main()
