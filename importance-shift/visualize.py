"""
visualize.py
============
Generate publication-quality plots for activation patching results and
training-induced circuit shift.

Plots produced
--------------
1. importance_heatmap.png         — [L×H] heatmap of attention head importance (θ0 or θ1)
2. mlp_bar.png                    — [L] bar chart of MLP importance
3. shift_heatmap.png              — [L×H] heatmap of absolute shift (θ1 − θ0)
4. shift_mlp_bar.png              — [L] bar chart of MLP shift
5. theta0_vs_theta1.png           — scatter plot: θ0 vs θ1 component scores
6. rel_shift_head_heatmap.png     — [L×H] heatmap of relative shift per head (log-scaled)
7. rel_shift_module_heatmap.png   — [L×2] heatmap: attn-module vs MLP relative shift

Usage
-----
python visualize.py \
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


# ─────────────────────────────────────────────────────────────────────────────
# Relative-shift helpers
# ─────────────────────────────────────────────────────────────────────────────

def _signed_log1p(arr: np.ndarray) -> np.ndarray:
    """sign(x) · log(1 + |x|)  — symmetric log scale for large-range values."""
    return np.sign(arr) * np.log1p(np.abs(arr))


def _clip_percentile(arr: np.ndarray, pct: float = 2.0) -> np.ndarray:
    """Clip array to [pct, 100-pct] percentile range."""
    lo = np.percentile(arr, pct)
    hi = np.percentile(arr, 100 - pct)
    return np.clip(arr, lo, hi)


def _load_rel_shift_from_json(
    shift_json_path: str,
    num_layers: int,
    num_heads: int,
) -> tuple:
    """
    Fallback: extract rel_shift_attn [L,H] and rel_shift_mlp [L]
    directly from circuit_shift.json when .npy files are absent.
    """
    with open(shift_json_path) as f:
        data = json.load(f)
    attn = np.zeros((num_layers, num_heads))
    mlp  = np.zeros(num_layers)
    for l in range(num_layers):
        for h in range(num_heads):
            attn[l, h] = data.get(f"L{l}_H{h}", {}).get("rel_shift", 0.0)
        mlp[l] = data.get(f"L{l}_MLP", {}).get("rel_shift", 0.0)
    return attn, mlp


def plot_rel_shift_head_heatmap(
    rel_shift_attn: np.ndarray,   # [L, H]  raw rel_shift values
    out_path: str,
    title: str = "Relative Shift per Head  sign·log(1+|Δ_rel|)",
    clip_pct: float = 2.0,
):
    """
    [L × H] heatmap of per-head relative shift.

    Because rel_shift = (θ1−θ0) / (|θ0|+ε) can span several orders of magnitude,
    the values are first transformed with sign·log(1+|x|) and then percentile-clipped
    for display.  Raw values are annotated inside each cell.
    """
    plt, _ = _try_import_matplotlib()
    num_layers, num_heads = rel_shift_attn.shape

    # Transform for display
    display = _clip_percentile(_signed_log1p(rel_shift_attn), clip_pct)
    vabs = np.max(np.abs(display)) or 1e-6

    fig, ax = plt.subplots(figsize=(max(10, num_heads * 0.9), max(6, num_layers * 0.65)))
    im = ax.imshow(display, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)

    # Annotate with raw (not log) values — use 1-decimal-place scientific notation
    for l in range(num_layers):
        for h in range(num_heads):
            raw = rel_shift_attn[l, h]
            disp_val = display[l, h]
            text_color = "white" if abs(disp_val) > 0.55 * vabs else "black"
            # Short annotation: ×10 if large
            if abs(raw) >= 100:
                label = f"{raw:.0f}"
            elif abs(raw) >= 1:
                label = f"{raw:.1f}"
            else:
                label = f"{raw:.2f}"
            ax.text(h, l, label, ha="center", va="center",
                    fontsize=5.5, color=text_color)

    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)], rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("sign·log(1+|rel_shift|)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rel_shift_module_heatmap(
    rel_shift_attn: np.ndarray,   # [L, H]
    rel_shift_mlp: np.ndarray,    # [L]
    out_path: str,
    title: str = "Relative Shift by Module  sign·log(1+|Δ_rel|)",
    clip_pct: float = 2.0,
    agg: str = "mean",            # 'mean' | 'median' | 'max'
):
    """
    [L × 2] heatmap: column 0 = attention-module aggregated rel_shift,
                     column 1 = MLP rel_shift.

    Attention heads within each layer are aggregated by `agg` (default: mean).
    Values are log-transformed and percentile-clipped for display.
    """
    plt, _ = _try_import_matplotlib()
    num_layers = rel_shift_attn.shape[0]

    # Aggregate head rel_shifts per layer
    if agg == "mean":
        attn_agg = rel_shift_attn.mean(axis=1)          # [L]
    elif agg == "median":
        attn_agg = np.median(rel_shift_attn, axis=1)
    elif agg == "max":
        # max by absolute value, preserving sign
        idx = np.argmax(np.abs(rel_shift_attn), axis=1)
        attn_agg = rel_shift_attn[np.arange(num_layers), idx]
    else:
        attn_agg = rel_shift_attn.mean(axis=1)

    # Stack into [L, 2] matrix
    module_mat = np.stack([attn_agg, rel_shift_mlp], axis=1)   # [L, 2]

    # Log-transform + clip
    display = _clip_percentile(_signed_log1p(module_mat), clip_pct)
    vabs = np.max(np.abs(display)) or 1e-6

    fig, ax = plt.subplots(figsize=(5, max(6, num_layers * 0.65)))
    im = ax.imshow(display, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)

    # Annotate
    col_labels = [f"Attn\n({agg})", "MLP"]
    for l in range(num_layers):
        for c in range(2):
            raw = module_mat[l, c]
            disp_val = display[l, c]
            text_color = "white" if abs(disp_val) > 0.55 * vabs else "black"
            if abs(raw) >= 100:
                label = f"{raw:.0f}"
            elif abs(raw) >= 1:
                label = f"{raw:.1f}"
            else:
                label = f"{raw:.2f}"
            ax.text(c, l, label, ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xlabel("Module", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in range(num_layers)])
    cbar = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("sign·log(1+|rel_shift|)", fontsize=9)
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

    # ── Infer dimensions from loaded matrices ────────────────────────────
    num_layers, num_heads = mat0.shape

    # ── Load absolute shift ───────────────────────────────────────────────
    if args.shift_dir and os.path.isdir(args.shift_dir):
        shift_attn = np.load(os.path.join(args.shift_dir, "shift_attn.npy"))
        shift_mlp  = np.load(os.path.join(args.shift_dir, "shift_mlp.npy"))
    else:
        shift_attn = mat1 - mat0
        shift_mlp  = mlp1 - mlp0

    # ── Load relative shift ───────────────────────────────────────────────
    # Priority: .npy → circuit_shift.json → compute on the fly
    rel_shift_attn = None
    rel_shift_mlp  = None
    if args.shift_dir and os.path.isdir(args.shift_dir):
        npy_attn = os.path.join(args.shift_dir, "rel_shift_attn.npy")
        npy_mlp  = os.path.join(args.shift_dir, "rel_shift_mlp.npy")
        json_path = os.path.join(args.shift_dir, "circuit_shift.json")

        if os.path.exists(npy_attn) and os.path.exists(npy_mlp):
            # Best case: npy already saved by run_circuit_shift.py
            rel_shift_attn = np.load(npy_attn)
            rel_shift_mlp  = np.load(npy_mlp)
            print("[visualize] rel_shift loaded from .npy files")
        elif os.path.exists(json_path):
            # Fallback: extract from circuit_shift.json and save npy for future use
            print("[visualize] rel_shift .npy not found — extracting from circuit_shift.json ...")
            rel_shift_attn, rel_shift_mlp = _load_rel_shift_from_json(
                json_path, num_layers, num_heads
            )
            np.save(npy_attn, rel_shift_attn)
            np.save(npy_mlp,  rel_shift_mlp)
            print(f"[visualize] saved rel_shift_attn.npy and rel_shift_mlp.npy to {args.shift_dir}")

    if rel_shift_attn is None:
        # Last resort: compute from score dicts
        print("[visualize] computing rel_shift on the fly from score dicts ...")
        eps = 1e-8
        rel_shift_attn = np.zeros((num_layers, num_heads))
        rel_shift_mlp  = np.zeros(num_layers)
        for l in range(num_layers):
            for h in range(num_heads):
                k = f"L{l}_H{h}"
                s0 = scores0.get(k, 0.0); s1 = scores1.get(k, 0.0)
                rel_shift_attn[l, h] = (s1 - s0) / (abs(s0) + eps)
            k = f"L{l}_MLP"
            s0 = scores0.get(k, 0.0); s1 = scores1.get(k, 0.0)
            rel_shift_mlp[l] = (s1 - s0) / (abs(s0) + eps)

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

    # ── NEW: Relative-shift plots ─────────────────────────────────────────
    plot_rel_shift_head_heatmap(
        rel_shift_attn,
        os.path.join(args.out_dir, "rel_shift_head_heatmap.png"),
        title="Relative Shift per Head  (θ1−θ0)/|θ0|  [sign·log scale]",
    )
    plot_rel_shift_module_heatmap(
        rel_shift_attn, rel_shift_mlp,
        os.path.join(args.out_dir, "rel_shift_module_heatmap_mean.png"),
        title="Relative Shift by Module  [mean-agg, sign·log scale]",
        agg="mean",
    )
    plot_rel_shift_module_heatmap(
        rel_shift_attn, rel_shift_mlp,
        os.path.join(args.out_dir, "rel_shift_module_heatmap_max.png"),
        title="Relative Shift by Module  [max|head|-agg, sign·log scale]",
        agg="max",
    )

    print(f"\n[visualize] All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
