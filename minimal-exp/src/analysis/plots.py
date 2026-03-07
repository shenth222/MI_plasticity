# src/analysis/plots.py
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use a clean style without seaborn
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9


def load_csv(path: str):
    """Load heads.csv into numpy arrays."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data


def load_cases(path: str):
    """Load cases.json."""
    with open(path, "r") as f:
        return json.load(f)


def load_stats(path: str):
    """Load stats.json."""
    with open(path, "r") as f:
        return json.load(f)


def plot_I_vs_U(data, cases, out_path):
    """
    Scatter plot: I_pre vs Urel (or U).
    Mark important-but-static and plastic-but-unimportant cases with different markers.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    I_pre = data['I_pre']
    Urel = data['Urel']
    
    # Get case indices
    ibs_indices = set()
    pbu_indices = set()
    
    for c in cases["important_but_static"]:
        layer, head = c["layer"], c["head"]
        # Find index in data
        idx = np.where((data['layer'] == layer) & (data['head'] == head))[0]
        if len(idx) > 0:
            ibs_indices.add(idx[0])
    
    for c in cases["plastic_but_unimportant"]:
        layer, head = c["layer"], c["head"]
        idx = np.where((data['layer'] == layer) & (data['head'] == head))[0]
        if len(idx) > 0:
            pbu_indices.add(idx[0])
    
    # All other points
    normal_mask = np.ones(len(data), dtype=bool)
    for idx in ibs_indices | pbu_indices: 
        normal_mask[idx] = False
    
    # Plot normal points
    ax.scatter(I_pre[normal_mask], Urel[normal_mask], s=20, alpha=0.6, marker='o', label='Normal')
    
    # Plot important-but-static
    if len(ibs_indices) > 0:
        ibs_idx = np.array(list(ibs_indices))
        ax.scatter(I_pre[ibs_idx], Urel[ibs_idx], s=50, alpha=0.8, marker='^', label='Important-but-static')
    
    # Plot plastic-but-unimportant
    if len(pbu_indices) > 0:
        pbu_idx = np.array(list(pbu_indices))
        ax.scatter(I_pre[pbu_idx], Urel[pbu_idx], s=50, alpha=0.8, marker='s', label='Plastic-but-unimportant')
    
    ax.set_xlabel('Importance (I_pre)')
    ax.set_ylabel('Relative Update (Urel)')
    ax.set_title('Importance vs. Update Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_I_vs_G(data, cases, out_path):
    """
    Scatter plot: I_pre vs G (gradient magnitude).
    Mark cases similarly.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    I_pre = data['I_pre']
    G = data['G']
    
    # Filter out NaN in G
    valid_mask = ~np.isnan(G)
    I_pre = I_pre[valid_mask]
    G = G[valid_mask]
    data_valid = data[valid_mask]
    
    # Get case indices
    ibs_indices = set()
    pbu_indices = set()
    
    for c in cases["important_but_static"]:
        layer, head = c["layer"], c["head"]
        idx = np.where((data_valid['layer'] == layer) & (data_valid['head'] == head))[0]
        if len(idx) > 0:
            ibs_indices.add(idx[0])
    
    for c in cases["plastic_but_unimportant"]:
        layer, head = c["layer"], c["head"]
        idx = np.where((data_valid['layer'] == layer) & (data_valid['head'] == head))[0]
        if len(idx) > 0:
            pbu_indices.add(idx[0])
    
    # All other points
    normal_mask = np.ones(len(data_valid), dtype=bool)
    for idx in ibs_indices | pbu_indices:
        normal_mask[idx] = False
    
    # Plot
    ax.scatter(I_pre[normal_mask], G[normal_mask], s=20, alpha=0.6, marker='o', label='Normal')
    
    if len(ibs_indices) > 0:
        ibs_idx = np.array(list(ibs_indices))
        ax.scatter(I_pre[ibs_idx], G[ibs_idx], s=50, alpha=0.8, marker='^', label='Important-but-static')
    
    if len(pbu_indices) > 0:
        pbu_idx = np.array(list(pbu_indices))
        ax.scatter(I_pre[pbu_idx], G[pbu_idx], s=50, alpha=0.8, marker='s', label='Plastic-but-unimportant')
    
    ax.set_xlabel('Importance (I_pre)')
    ax.set_ylabel('Gradient Magnitude (G)')
    ax.set_title('Importance vs. Gradient (Plasticity Proxy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_stats(stats, out_path):
    """
    Bar plot or text plot showing Spearman rho and top-k overlap.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    metrics = ['Spearman ρ(I,U)', f'Top-{stats["topk"]} overlap']
    values = [stats['spearman_rho_Ipre_U'], stats['topk_overlap_Ipre_U']]
    
    ax.bar(metrics, values, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Correlation & Overlap Metrics')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text labels on bars
    for i, (m, v) in enumerate(zip(metrics, values)):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_I_pre_vs_I_post(data, out_path):
    """
    Scatter plot: I_pre vs I_post (post-training importance).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    I_pre = data['I_pre']
    I_post = data['I_post']

    valid_mask = ~np.isnan(I_post)
    I_pre = I_pre[valid_mask]
    I_post = I_post[valid_mask]

    ax.scatter(I_pre, I_post, s=20, alpha=0.6, marker='o')

    if len(I_pre) > 0:
        vmin = float(min(I_pre.min(), I_post.min()))
        vmax = float(max(I_pre.max(), I_post.max()))
        ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', color='gray', linewidth=1, alpha=0.8)

    ax.set_xlabel('Importance (I_pre)')
    ax.set_ylabel('Importance (I_post)')
    ax.set_title('Importance: Pre vs Post')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def _quantile_bins(arr, n_buckets):
    """Return bin edges for n_buckets quantile-based bins (excluding min/max as first/last edge)."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < n_buckets:
        n_buckets = max(1, len(valid))
    percentiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def plot_bucket_orthogonality(data, out_path_prefix, P_col="Urel", n_buckets=10, use_violin=True):
    """
    分桶图：反驳“相关性低是噪声/尺度问题”，说明 I 与 P 是正交维度。

    1) I_pre 分桶 → 每桶内画 P 的分布（箱线图/小提琴图）
       - 若同一 I 桶内 P 跨度很大，或高 I 桶里也有低 P、低 I 桶里也有高 P → 支持正交
    2) P 分桶 → 每桶内画 I_pre 的分布（同上）
       - 若同一 P 桶内 I 跨度很大 → 同样支持正交

    P_col: 塑性指标列名，'Urel' | 'U' | 'G'
    """
    I_pre = np.asarray(data["I_pre"], dtype=float)
    P = np.asarray(data[P_col], dtype=float)

    valid = ~np.isnan(I_pre) & ~np.isnan(P)
    I_pre = I_pre[valid]
    P = P[valid]
    if len(I_pre) < n_buckets:
        print(f"Skip bucket plot: too few valid points ({len(I_pre)} < {n_buckets})")
        return

    bin_edges_I = _quantile_bins(I_pre, n_buckets)
    bin_edges_P = _quantile_bins(P, n_buckets)
    bucket_I = np.digitize(I_pre, bin_edges_I[1:-1])  # 1..n_buckets
    bucket_P = np.digitize(P, bin_edges_P[1:-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def _plot_buckets(ax, by_bucket, ylabel, xlabel_bucket, title, use_violin, bucket_labels_full):
        """Plot only non-empty buckets to avoid violinplot/boxplot failing on empty arrays."""
        non_empty = [(i, by_bucket[i]) for i in range(n_buckets) if len(by_bucket[i]) > 0]
        if not non_empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        indices, data = zip(*non_empty)
        positions = np.arange(1, len(data) + 1)
        labels = [bucket_labels_full[i] for i in indices]
        if use_violin:
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.7)
        else:
            bp = ax.boxplot(data, positions=positions, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel_bucket)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    bucket_labels_full = [str(i + 1) for i in range(n_buckets)]
    P_by_bucket = [P[bucket_I == (i + 1)] for i in range(n_buckets)]
    I_by_bucket = [I_pre[bucket_P == (i + 1)] for i in range(n_buckets)]

    # (1) I_pre 分桶 → P 的分布
    _plot_buckets(
        axes[0],
        P_by_bucket,
        f"Plasticity ({P_col})",
        "I_pre bucket (1=lowest, 10=highest)",
        f"P ({P_col}) within each I_pre bucket",
        use_violin,
        bucket_labels_full,
    )

    # (2) P 分桶 → I_pre 的分布
    _plot_buckets(
        axes[1],
        I_by_bucket,
        "Importance (I_pre)",
        f"{P_col} bucket (1=lowest, 10=highest)",
        f"I_pre within each {P_col} bucket",
        use_violin,
        bucket_labels_full,
    )

    plt.suptitle("Orthogonality: same bucket still spans full range of the other dimension", fontsize=11)
    plt.tight_layout()
    out_path = f"{out_path_prefix}_bucket_orthogonality.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_Ipost_correlations(stats, out_path):
    """
    Bar plot: Spearman correlations between I_post and {U, Urel, G, F, Ppred}.
    """
    keys = [
        ("U", "spearman_rho_Ipost_U"),
        ("Urel", "spearman_rho_Ipost_Urel"),
        ("G", "spearman_rho_Ipost_G"),
        ("F", "spearman_rho_Ipost_F"),
        ("Ppred", "spearman_rho_Ipost_Ppred"),
    ]
    labels = [k[0] for k in keys]
    values = [stats.get(k[1], float("nan")) for k in keys]

    # Drop NaNs (e.g., when I_post missing)
    labels_f = []
    values_f = []
    for l, v in zip(labels, values):
        if not np.isnan(v):
            labels_f.append(l)
            values_f.append(v)

    if len(values_f) == 0:
        print(f"Skip {out_path}: no valid I_post correlations")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels_f, values_f, alpha=0.7)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('I_post vs. {U, Urel, G, F, Ppred}')
    ax.set_ylim([-1.0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    for i, (m, v) in enumerate(zip(labels_f, values_f)):
        ax.text(i, v + (0.02 if v >= 0 else -0.06), f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g. outputs/MNLI/seed1")
    ap.add_argument("--P_col", type=str, default="Urel", choices=["U", "Urel", "G"],
                    help="Plasticity column for bucket plot (default: Urel)")
    ap.add_argument("--n_buckets", type=int, default=10, help="Number of quantile buckets (default: 10)")
    ap.add_argument("--boxplot", action="store_true", help="Use boxplot instead of violin for bucket plot")
    args = ap.parse_args()

    csv_path = os.path.join(args.exp_dir, "heads.csv")
    cases_path = os.path.join(args.exp_dir, "cases.json")
    stats_path = os.path.join(args.exp_dir, "stats.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"Missing {cases_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing {stats_path}")

    data = load_csv(csv_path)
    cases = load_cases(cases_path)
    stats = load_stats(stats_path)

    # Generate plots
    plot_I_vs_U(data, cases, os.path.join(args.exp_dir, "fig_I_vs_U.png"))
    plot_I_vs_G(data, cases, os.path.join(args.exp_dir, "fig_I_vs_G.png"))
    plot_stats(stats, os.path.join(args.exp_dir, "fig_stats.png"))
    plot_I_pre_vs_I_post(data, os.path.join(args.exp_dir, "fig_Ipre_vs_Ipost.png"))
    plot_Ipost_correlations(stats, os.path.join(args.exp_dir, "fig_Ipost_corrs.png"))

    # 分桶图：反驳“相关性低=噪声/尺度”，说明 I 与 P 正交
    out_prefix = os.path.join(args.exp_dir, "fig")
    plot_bucket_orthogonality(
        data,
        out_prefix,
        P_col=args.P_col,
        n_buckets=args.n_buckets,
        use_violin=not args.boxplot,
    )


if __name__ == "__main__":
    main()
