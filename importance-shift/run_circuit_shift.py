"""
run_circuit_shift.py
====================
Compute training-induced circuit shift by comparing importance scores
between the pre-training checkpoint (θ0) and the fine-tuned model (θ1).

Circuit Shift Metrics
---------------------
For each component c:

  shift(c)        = I(c, θ1) - I(c, θ0)
  rel_shift(c)    = shift(c) / (|I(c, θ0)| + ε)
  cos_sim         = cosine similarity between score vectors of θ0 and θ1
  l2_dist         = ‖I(θ1) - I(θ0)‖₂ (attention heads / MLP separately)

Outputs (saved to --out_dir):
  circuit_shift.json           : per-component shift + rel_shift
  shift_attn.npy               : [L, H] attention head shift matrix
  shift_mlp.npy                : [L]    MLP shift vector
  summary.json                 : global metrics (cosine_sim, l2_dist, top movers)

Usage
-----
# Option A: supply pre-computed score JSON files
python -m importance_shift.run_circuit_shift \
    --theta0_scores outputs/theta0/attribution_scores.json \
    --theta1_scores outputs/theta1/attribution_scores.json \
    --out_dir outputs/circuit_shift

# Option B: run patching on the fly for both checkpoints
python -m importance_shift.run_circuit_shift \
    --theta0_ckpt /path/to/ckpt_init \
    --theta1_ckpt /path/to/ckpt_final \
    --out_dir outputs/circuit_shift \
    --num_batches 64
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_utils import load_mnli, make_dataloader
from patching import DeBERTaActHooker, compute_attribution_scores
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    ap = argparse.ArgumentParser(description="Training-induced circuit shift analysis")

    # Option A: pre-computed JSON
    ap.add_argument("--theta0_scores", type=str, default="",
                    help="Path to pre-computed θ0 attribution_scores.json")
    ap.add_argument("--theta1_scores", type=str, default="",
                    help="Path to pre-computed θ1 attribution_scores.json")

    # Option B: compute on the fly
    ap.add_argument("--theta0_ckpt", type=str, default="",
                    help="θ0 model checkpoint (ckpt_init)")
    ap.add_argument("--theta1_ckpt", type=str, default="",
                    help="θ1 model checkpoint (ckpt_final)")
    ap.add_argument("--num_batches", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--split", type=str, default="validation_matched")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue"))
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_patching(ckpt_path: str, args, device: torch.device) -> dict:
    """Run attribution patching for a model checkpoint, return scores dict."""
    print(f"\n  Loading model from {ckpt_path}")
    tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.eval()

    os.environ["GLUE_DATA_PATH"] = args.dataset_path
    dataset = load_mnli(tok, max_len=args.max_len, split=args.split)
    dl = make_dataloader(dataset, tok, batch_size=args.batch_size, shuffle=True)

    hooker = DeBERTaActHooker(
        model,
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
    )

    scores = compute_attribution_scores(
        model=model,
        dataloader=dl,
        hooker=hooker,
        device=device,
        num_batches=args.num_batches,
    )
    return scores


def _load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_shift(
    scores0: dict,
    scores1: dict,
    num_layers: int = 12,
    num_heads: int = 12,
) -> dict:
    """
    Compute training-induced circuit shift.

    Returns a dict:
      per_component : {name: {score0, score1, shift, rel_shift}}
      attn_shift    : np.ndarray [L, H]
      mlp_shift     : np.ndarray [L]
      summary       : {cosine_sim_attn, cosine_sim_mlp, l2_attn, l2_mlp,
                       top_gainers, top_losers}
    """
    eps = 1e-8
    per_component = {}
    all_names = list(scores0.keys())

    for name in all_names:
        s0 = scores0.get(name, 0.0)
        s1 = scores1.get(name, 0.0)
        shift = s1 - s0
        rel_shift = shift / (abs(s0) + eps)
        per_component[name] = {
            "score_theta0": s0,
            "score_theta1": s1,
            "shift": shift,
            "rel_shift": rel_shift,
        }

    # Attention matrix shifts
    attn_shift = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            key = f"L{l}_H{h}"
            attn_shift[l, h] = per_component.get(key, {}).get("shift", 0.0)

    # MLP vector shifts
    mlp_shift = np.array([
        per_component.get(f"L{l}_MLP", {}).get("shift", 0.0)
        for l in range(num_layers)
    ])

    # ── Summary statistics ─────────────────────────────────────────────────
    # Cosine similarity of score vectors (attn and MLP separately)
    head_names = [f"L{l}_H{h}" for l in range(num_layers) for h in range(num_heads)]
    mlp_names  = [f"L{l}_MLP" for l in range(num_layers)]

    def _cosine_sim(names):
        v0 = np.array([scores0.get(n, 0.0) for n in names])
        v1 = np.array([scores1.get(n, 0.0) for n in names])
        denom = np.linalg.norm(v0) * np.linalg.norm(v1)
        return float(np.dot(v0, v1) / (denom + eps))

    def _l2_dist(names):
        v0 = np.array([scores0.get(n, 0.0) for n in names])
        v1 = np.array([scores1.get(n, 0.0) for n in names])
        return float(np.linalg.norm(v1 - v0))

    # Top gainers / losers by absolute shift
    all_shifts = [(name, d["shift"]) for name, d in per_component.items()]
    all_shifts.sort(key=lambda x: x[1], reverse=True)
    top_gainers = [{"component": n, "shift": float(s)} for n, s in all_shifts[:20]]
    top_losers  = [{"component": n, "shift": float(s)} for n, s in all_shifts[-20:]]

    # Percentage of heads with positive vs negative shift
    head_shifts = [per_component[n]["shift"] for n in head_names if n in per_component]
    frac_positive = float(np.mean(np.array(head_shifts) > 0))

    summary = {
        "cosine_sim_attn": _cosine_sim(head_names),
        "cosine_sim_mlp":  _cosine_sim(mlp_names),
        "l2_dist_attn":    _l2_dist(head_names),
        "l2_dist_mlp":     _l2_dist(mlp_names),
        "mean_shift_attn": float(np.mean(attn_shift)),
        "mean_abs_shift_attn": float(np.mean(np.abs(attn_shift))),
        "mean_shift_mlp":  float(np.mean(mlp_shift)),
        "mean_abs_shift_mlp": float(np.mean(np.abs(mlp_shift))),
        "frac_heads_positive_shift": frac_positive,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
    }

    return {
        "per_component": per_component,
        "attn_shift": attn_shift,
        "mlp_shift": mlp_shift,
        "summary": summary,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[circuit_shift] device = {device}")

    # ── Load or compute scores ────────────────────────────────────────────
    if args.theta0_scores and args.theta1_scores:
        print("[circuit_shift] Loading pre-computed scores ...")
        scores0 = _load_scores(args.theta0_scores)
        scores1 = _load_scores(args.theta1_scores)
    elif args.theta0_ckpt and args.theta1_ckpt:
        print("[circuit_shift] Computing scores on-the-fly ...")
        print("── θ0 ──")
        scores0 = _run_patching(args.theta0_ckpt, args, device)
        print("── θ1 ──")
        scores1 = _run_patching(args.theta1_ckpt, args, device)
        # Cache them
        with open(os.path.join(args.out_dir, "theta0_attribution_scores.json"), "w") as f:
            json.dump(scores0, f, indent=2)
        with open(os.path.join(args.out_dir, "theta1_attribution_scores.json"), "w") as f:
            json.dump(scores1, f, indent=2)
    else:
        raise ValueError(
            "Provide either (--theta0_scores + --theta1_scores) "
            "or (--theta0_ckpt + --theta1_ckpt)"
        )

    # Infer model dimensions from score keys
    num_layers = max(int(k.split("_H")[0].replace("L", "")) for k in scores0 if "_H" in k) + 1
    num_heads  = max(int(k.split("_H")[1]) for k in scores0 if "_H" in k) + 1
    print(f"[circuit_shift] num_layers={num_layers}, num_heads={num_heads}")

    # ── Compute shift ─────────────────────────────────────────────────────
    print("\n[circuit_shift] Computing circuit shift ...")
    result = compute_shift(scores0, scores1, num_layers=num_layers, num_heads=num_heads)

    # ── Save outputs ──────────────────────────────────────────────────────
    # 1. Per-component JSON
    with open(os.path.join(args.out_dir, "circuit_shift.json"), "w") as f:
        json.dump(result["per_component"], f, indent=2)

    # 2. NumPy matrices
    np.save(os.path.join(args.out_dir, "shift_attn.npy"), result["attn_shift"])
    np.save(os.path.join(args.out_dir, "shift_mlp.npy"), result["mlp_shift"])

    # 3. Summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    s = result["summary"]
    print("\n" + "=" * 60)
    print("  Training-Induced Circuit Shift  —  Summary")
    print("=" * 60)
    print(f"  Cosine similarity (attn heads): {s['cosine_sim_attn']:.4f}")
    print(f"  Cosine similarity (MLP layers): {s['cosine_sim_mlp']:.4f}")
    print(f"  L2 distance       (attn heads): {s['l2_dist_attn']:.6f}")
    print(f"  L2 distance       (MLP layers): {s['l2_dist_mlp']:.6f}")
    print(f"  Mean shift        (attn heads): {s['mean_shift_attn']:+.6f}")
    print(f"  Mean |shift|      (attn heads): {s['mean_abs_shift_attn']:.6f}")
    print(f"  Mean shift        (MLP layers): {s['mean_shift_mlp']:+.6f}")
    print(f"  Fraction of heads with +shift : {s['frac_heads_positive_shift']:.2%}")
    print("\n  Top gainers (θ1 > θ0):")
    for item in s["top_gainers"][:10]:
        print(f"    {item['component']:12s}  +{item['shift']:+.6f}")
    print("\n  Top losers (θ1 < θ0):")
    for item in s["top_losers"][:10]:
        print(f"    {item['component']:12s}  {item['shift']:+.6f}")
    print("=" * 60)
    print(f"\n[circuit_shift] Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
