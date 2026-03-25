"""
run_patching.py
===============
Compute activation-patching causal importance scores for DeBERTa V3
on MNLI, for a given model checkpoint.

Outputs (saved to --out_dir):
  - attribution_scores.json   : {component_name: score} via gradient attribution
  - zero_ablation_scores.json : {component_name: score} via zero-ablation (optional)
  - scores_matrix_attn.npy    : [num_layers, num_heads] attention head scores
  - scores_matrix_mlp.npy     : [num_layers] MLP layer scores

Usage example:
  python -m importance_shift.run_patching \
      --model_path /data1/shenth/work/MI_plasticity/casual-exp/baseline/outputs/FFT/MNLI/seed42/lr1e-5/ckpt_init \
      --out_dir outputs/seed42_lr1e-5/theta0 \
      --num_batches 64 \
      --batch_size 32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure the project root (importance-shift/) is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_utils import load_mnli, make_dataloader
from patching import DeBERTaActHooker, compute_attribution_scores, compute_zero_ablation_scores


def parse_args():
    ap = argparse.ArgumentParser(description="Activation patching for DeBERTa on MNLI")
    ap.add_argument("--model_path", type=str, required=True,
                    help="Path to the model checkpoint (HuggingFace format)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to save results")
    ap.add_argument("--dataset_path", type=str,
                    default=os.environ.get("GLUE_DATA_PATH", "/data1/shenth/datasets/glue"))
    ap.add_argument("--split", type=str, default="validation_matched",
                    help="MNLI split to use (validation_matched / train)")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_batches", type=int, default=64,
                    help="Number of batches for attribution scoring")
    ap.add_argument("--zero_ablation_batches", type=int, default=0,
                    help="Batches for zero-ablation scoring (0 = skip, expensive)")
    ap.add_argument("--method", type=str, default="attribution",
                    choices=["attribution", "zero_ablation", "both"],
                    help="Scoring method")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_patching] device = {device}")

    # ── Model & Tokenizer ─────────────────────────────────────────────────
    print(f"[run_patching] loading model from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    num_layers = model.config.num_hidden_layers   # 12
    num_heads  = model.config.num_attention_heads  # 12
    print(f"[run_patching] model: {num_layers} layers × {num_heads} heads")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"[run_patching] loading MNLI ({args.split})")
    os.environ["GLUE_DATA_PATH"] = args.dataset_path
    dataset = load_mnli(tok, max_len=args.max_len, split=args.split)
    dataloader = make_dataloader(dataset, tok, batch_size=args.batch_size, shuffle=True)
    print(f"[run_patching] dataset size = {len(dataset)}, "
          f"batches = {len(dataloader)}")

    # ── Hooker ────────────────────────────────────────────────────────────
    hooker = DeBERTaActHooker(model, num_layers=num_layers, num_heads=num_heads)

    # ── Attribution Patching ──────────────────────────────────────────────
    if args.method in ("attribution", "both"):
        print(f"\n[run_patching] === Attribution Patching "
              f"(num_batches={args.num_batches}) ===")
        attr_scores = compute_attribution_scores(
            model=model,
            dataloader=dataloader,
            hooker=hooker,
            device=device,
            num_batches=args.num_batches,
        )
        # Save raw scores
        out_path = os.path.join(args.out_dir, "attribution_scores.json")
        with open(out_path, "w") as f:
            json.dump(attr_scores, f, indent=2)
        print(f"[run_patching] attribution scores saved → {out_path}")

        # Save matrix forms
        _save_matrices(attr_scores, hooker, args.out_dir, prefix="attribution")

    # ── Zero-Ablation ─────────────────────────────────────────────────────
    if args.method in ("zero_ablation", "both") and args.zero_ablation_batches > 0:
        print(f"\n[run_patching] === Zero-Ablation "
              f"(num_batches={args.zero_ablation_batches}) ===")
        za_scores = compute_zero_ablation_scores(
            model=model,
            dataloader=dataloader,
            hooker=hooker,
            device=device,
            num_batches=args.zero_ablation_batches,
        )
        out_path = os.path.join(args.out_dir, "zero_ablation_scores.json")
        with open(out_path, "w") as f:
            json.dump(za_scores, f, indent=2)
        print(f"[run_patching] zero-ablation scores saved → {out_path}")
        _save_matrices(za_scores, hooker, args.out_dir, prefix="zero_ablation")

    # ── Save config ───────────────────────────────────────────────────────
    cfg = vars(args)
    cfg["num_layers"] = num_layers
    cfg["num_heads"] = num_heads
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[run_patching] All results saved to {args.out_dir}")


def _save_matrices(scores: dict, hooker: DeBERTaActHooker, out_dir: str, prefix: str):
    """Save attention head scores [L, H] and MLP scores [L] as .npy arrays."""
    nl, nh = hooker.num_layers, hooker.num_heads

    # Attention heads matrix
    attn_mat = np.zeros((nl, nh))
    for l in range(nl):
        for h in range(nh):
            key = hooker.head_key(l, h)
            attn_mat[l, h] = scores.get(key, 0.0)
    np.save(os.path.join(out_dir, f"{prefix}_attn.npy"), attn_mat)

    # MLP vector
    mlp_vec = np.array([scores.get(hooker.mlp_key(l), 0.0) for l in range(nl)])
    np.save(os.path.join(out_dir, f"{prefix}_mlp.npy"), mlp_vec)

    print(f"[run_patching] saved {prefix}_attn.npy [{nl}×{nh}] and "
          f"{prefix}_mlp.npy [{nl}]")

    # Print top-10 attention heads
    flat = [(hooker.head_key(l, h), float(attn_mat[l, h]))
            for l in range(nl) for h in range(nh)]
    flat.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top-10 attention heads (by |score|):")
    for name, val in flat[:10]:
        print(f"    {name:12s} {val:+.6f}")


if __name__ == "__main__":
    main()
