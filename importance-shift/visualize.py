"""
visualize.py
============
Generate publication-quality plots for activation patching results and
training-induced circuit shift.

Plots produced
--------------
1. importance_heatmap.png    — [L×H] heatmap of attention head importance (θ0 or θ1)
2. mlp_bar.png               — [L] bar chart of MLP importance
3. shift_heatmap.png         — [L×H] heatmap of attention head shift (θ1 − θ0)
4. shift_mlp_bar.png         — [L] bar chart of MLP shift
5. theta0_vs_theta1.png      — scatter plot: θ0 vs θ1 component scores

Usage
-----
python -m importance_shift.visualize \
    --theta0_dir outputs/seed42_lr1e-5/theta0 \
    --theta1_dir outputs/seed42_lr1e-5/theta1 \
    --shift_dir  outputs/seed42_lr1e-5/circuit_shift \
    --out_dir    outputs/seed42_lr1e-5/figures \
    --prefix     attribution
"""

import os
import sys
import argparse
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        return plt, mcolors
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_importance_heatmap(
    attn_scores: np.ndarray,      # [L, H]
    out_path: str,
    title: str = "Attention Head Importance",
    cmap: str = "RdBu_r",
    vmax: float = None,
):
    plt, _ = _try_import_matplotlib()
    num_layers, num_heads = attn_scores.shape
    fig, ax = plt.subplots(figsize=(max(8, num_heads), max(5, num_layers * 0.55)))
    vabs = vmax or float(np.max(np.abs(attn_scores))) or 1e-6
    im = ax.imshow(attn_scores, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)], rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_mlp_bar(
    mlp_scores: np.ndarray,   # [L]
    out_path: str,
    title: str = "MLP Layer Importance",
    color_pos: str = "#e74c3c",
    color_neg: str = "#3498db",
):
    plt, _ = _try_import_matplotlib()
    num_layers = len(mlp_scores)
    colors = [color_pos if v >= 0 else color_neg for v in mlp_scores]
    fig, ax = plt.subplots(figsize=(6, max(4, num_layers * 0.4)))
    bars = ax.barh(range(num_layers), mlp_scores, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_side_by_side_heatmap(
    mat0: np.ndarray,   # [L, H]
    mat1: np.ndarray,   # [L, H]
    out_path: str,
    title0: str = "θ0 (pre-training)",
    title1: str = "θ1 (fine-tuned)",
    cmap: str = "RdBu_r",
):
    plt, _ = _try_import_matplotlib()
    num_layers, num_heads = mat0.shape
    vabs = max(np.max(np.abs(mat0)), np.max(np.abs(mat1))) or 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(max(14, num_heads * 2), max(5, num_layers * 0.55)))
    for ax, mat, title in zip(axes, [mat0, mat1], [title0, title1]):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)
        ax.set_xlabel("Head", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)], rotation=45, ha="right")
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scatter(
    scores0: dict,
    scores1: dict,
    out_path: str,
    title: str = "θ0 vs θ1 Component Scores",
    highlight_top_n: int = 10,
):
    plt, _ = _try_import_matplotlib()
    names = sorted(scores0.keys())
    x = np.array([scores0.get(n, 0.0) for n in names])
    y = np.array([scores1.get(n, 0.0) for n in names])

    # Separate heads and MLPs
    is_mlp  = np.array(["_MLP" in n for n in names])
    is_head = ~is_mlp

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x[is_head], y[is_head], s=15, alpha=0.5, label="Attn Heads",
               color="#2980b9", zorder=2)
    ax.scatter(x[is_mlp], y[is_mlp], s=40, alpha=0.8, label="MLP Layers",
               color="#e74c3c", zorder=3, marker="D")

    # y=x reference line
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")

    # Annotate top movers
    shift = np.abs(y - x)
    top_idx = np.argsort(shift)[-highlight_top_n:]
    for i in top_idx:
        ax.annotate(names[i], (x[i], y[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Score (θ0)", fontsize=12)
    ax.set_ylabel("Score (θ1)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_shift_with_significance(
    shift_mat: np.ndarray,   # [L, H]
    out_path: str,
    title: str = "Circuit Shift (θ1 − θ0)",
):
    plt, _ = _try_import_matplotlib()
    num_layers, num_heads = shift_mat.shape
    vabs = np.max(np.abs(shift_mat)) or 1e-6
    fig, ax = plt.subplots(figsize=(max(8, num_heads), max(5, num_layers * 0.6)))
    im = ax.imshow(shift_mat, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)

    # Annotate cells with values
    for l in range(num_layers):
        for h in range(num_heads):
            val = shift_mat[l, h]
            text_color = "white" if abs(val) > 0.5 * vabs else "black"
            ax.text(h, l, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)], rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Shift")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta0_dir", type=str, required=True,
                    help="Directory with θ0 patching results")
    ap.add_argument("--theta1_dir", type=str, required=True,
                    help="Directory with θ1 patching results")
    ap.add_argument("--shift_dir", type=str, default="",
                    help="Directory with circuit_shift results (optional if computed separately)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="attribution",
                    choices=["attribution", "zero_ablation"],
                    help="Score file prefix to use")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[visualize] Output directory: {args.out_dir}")

    # ── Load θ0 ──────────────────────────────────────────────────────────
    mat0 = np.load(os.path.join(args.theta0_dir, f"{args.prefix}_attn.npy"))
    mlp0 = np.load(os.path.join(args.theta0_dir, f"{args.prefix}_mlp.npy"))
    # run_patching.py saves as "{prefix}_scores.json"
    scores_file0 = os.path.join(args.theta0_dir, f"{args.prefix}_scores.json")
    if not os.path.exists(scores_file0):
        # fallback: attribution_scores.json
        scores_file0 = os.path.join(args.theta0_dir, f"{args.prefix}_scores.json".replace(
            f"{args.prefix}_scores", f"{args.prefix}_scores"))
        scores_file0 = os.path.join(args.theta0_dir, "attribution_scores.json")
    with open(scores_file0) as f:
        scores0 = json.load(f)

    # ── Load θ1 ──────────────────────────────────────────────────────────
    mat1 = np.load(os.path.join(args.theta1_dir, f"{args.prefix}_attn.npy"))
    mlp1 = np.load(os.path.join(args.theta1_dir, f"{args.prefix}_mlp.npy"))
    scores_file1 = os.path.join(args.theta1_dir, f"{args.prefix}_scores.json")
    if not os.path.exists(scores_file1):
        scores_file1 = os.path.join(args.theta1_dir, "attribution_scores.json")
    with open(scores_file1) as f:
        scores1 = json.load(f)

    # ── Load shift ────────────────────────────────────────────────────────
    if args.shift_dir and os.path.isdir(args.shift_dir):
        shift_attn = np.load(os.path.join(args.shift_dir, "shift_attn.npy"))
        shift_mlp  = np.load(os.path.join(args.shift_dir, "shift_mlp.npy"))
    else:
        shift_attn = mat1 - mat0
        shift_mlp  = mlp1 - mlp0

    # ── Generate plots ────────────────────────────────────────────────────
    print("[visualize] Generating plots ...")

    plot_importance_heatmap(
        mat0, os.path.join(args.out_dir, "theta0_importance_heatmap.png"),
        title="Attention Head Importance  (θ0, pre-training)"
    )
    plot_importance_heatmap(
        mat1, os.path.join(args.out_dir, "theta1_importance_heatmap.png"),
        title="Attention Head Importance  (θ1, fine-tuned)"
    )
    plot_side_by_side_heatmap(
        mat0, mat1,
        os.path.join(args.out_dir, "importance_comparison.png"),
        title0="θ0 (pre-training)", title1="θ1 (fine-tuned)"
    )
    plot_shift_with_significance(
        shift_attn,
        os.path.join(args.out_dir, "shift_heatmap.png"),
        title="Training-Induced Circuit Shift  (θ1 − θ0)"
    )
    plot_mlp_bar(
        mlp0, os.path.join(args.out_dir, "theta0_mlp_bar.png"),
        title="MLP Importance (θ0)"
    )
    plot_mlp_bar(
        mlp1, os.path.join(args.out_dir, "theta1_mlp_bar.png"),
        title="MLP Importance (θ1)"
    )
    plot_mlp_bar(
        shift_mlp, os.path.join(args.out_dir, "shift_mlp_bar.png"),
        title="MLP Layer Shift (θ1 − θ0)"
    )
    plot_scatter(
        scores0, scores1,
        os.path.join(args.out_dir, "theta0_vs_theta1_scatter.png"),
        title=f"Component Scores: θ0 vs θ1  ({args.prefix})"
    )

    print(f"\n[visualize] All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
