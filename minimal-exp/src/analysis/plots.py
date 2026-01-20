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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g. outputs/MNLI/seed1")
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


if __name__ == "__main__":
    main()
