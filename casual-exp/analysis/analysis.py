#!/usr/bin/env python3
"""
Score table comprehensive analysis for MI_plasticity experiment.

Analyzes:
1. Within-metric consistency (pairwise correlations, rank agreement)
2. Cross-metric correlations (Pearson, Spearman, Kendall)
3. Top-k ranking overlap (Jaccard / intersection) at k=10,20,30
4. Binned mean plots
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from itertools import combinations

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SCORE_CSV  = os.path.join(BASE_DIR, "../score/score_table.csv")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
TBL_DIR    = os.path.join(BASE_DIR, "tables")

os.makedirs(os.path.join(FIG_DIR, "within_metric"), exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, "cross_metric"),  exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, "ranking"),       exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, "binned_mean"),   exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Metric group definitions
# ─────────────────────────────────────────────────────────────────
METRIC_GROUPS = {
    "U_m": {
        "cols": ["U_m | def1_abs", "U_m | def2_rel", "U_m | def3_path"],
        "labels": ["def1_abs", "def2_rel", "def3_path"],
        "descriptions": {
            "def1_abs":  "参数绝对变化量 ‖Δθ_m‖₂（训练后 vs 初始）",
            "def2_rel":  "相对变化量 ‖Δθ_m‖₂ / (‖θ_m^(0)‖₂ + ε)",
            "def3_path": "训练路径长度 Σ_t ‖θ_m^(t) − θ_m^(t-1)‖₂",
        },
        # 计算成本从低到高
        "cost_order": ["def1_abs", "def2_rel", "def3_path"],
        "cost_note": {
            "def1_abs":  "极低——仅需保存初始参数，训练结束后一次计算",
            "def2_rel":  "极低——与def1_abs相同，多一次除法",
            "def3_path": "低——需在每训练步 hook 参数增量并累加",
        },
    },
    "I_pre": {
        "cols": [
            "I_pre | fisher",
            "I_pre | saliency_grad_norm",
            "I_pre | saliency_taylor",
            "I_pre | perturbation",
            "I_pre | sv_nuclear_norm",
            "I_pre | sv_top32_sum",
            "I_pre | sv_max_sv",
            "I_pre | sv_min_sv",
            "I_pre | se_spectral_entropy",
            "I_pre | se_raw_entropy",
        ],
        "labels": [
            "fisher", "saliency_grad_norm", "saliency_taylor",
            "perturbation",
            "sv_nuclear_norm", "sv_top32_sum", "sv_max_sv", "sv_min_sv",
            "se_spectral_entropy", "se_raw_entropy",
        ],
        "descriptions": {
            "fisher":              "对角Fisher信息 Σ E[(∂L/∂θ_i)²]，多batch蒙特卡洛估计后模块求和",
            "saliency_grad_norm":  "梯度L2范数 ‖∇_{θ_m}L‖₂（多batch均值）",
            "saliency_taylor":     "Taylor显著性 Σ |θ_i · g_i|（参数×梯度，多batch均值）",
            "perturbation":        "扰动重要性 E[L(θ+ε_m) − L(θ)]（仅对模块加噪声）",
            "sv_nuclear_norm":     "权重矩阵核范数 Σ_j σ_j（所有奇异值之和，需SVD）",
            "sv_top32_sum":        "前32个奇异值之和 Σ_{j≤32} σ_j",
            "sv_max_sv":           "最大奇异值 σ_max",
            "sv_min_sv":           "最小奇异值 σ_min",
            "se_spectral_entropy": "归一化谱熵 H(σ)/log(r)，度量奇异值分布均匀度",
            "se_raw_entropy":      "原始谱熵 H(σ) = −Σ s_i log(s_i)",
        },
        # 从低到高
        "cost_order": [
            "sv_nuclear_norm", "sv_top32_sum", "sv_max_sv", "sv_min_sv",
            "se_spectral_entropy", "se_raw_entropy",
            "saliency_grad_norm", "saliency_taylor",
            "fisher",
            "perturbation",
        ],
        "cost_note": {
            "sv_nuclear_norm":    "极低——仅SVD权重矩阵，无需梯度，O(m·n·min(m,n))",
            "sv_top32_sum":       "极低——同sv_nuclear_norm，只取前32个奇异值",
            "sv_max_sv":          "极低——同sv_nuclear_norm，仅取最大奇异值",
            "sv_min_sv":          "极低——同sv_nuclear_norm，仅取最小奇异值",
            "se_spectral_entropy":"极低——同sv_nuclear_norm，计算熵即可",
            "se_raw_entropy":     "极低——同se_spectral_entropy",
            "saliency_grad_norm": "低——1次反向传播",
            "saliency_taylor":    "低——1次反向传播 + 元素级乘法",
            "fisher":             "中——多batch蒙特卡洛反向传播，O(B×N_batch)",
            "perturbation":       "高——多次前向传播（每次对不同模块加噪声），O(N_module×N_batch)",
        },
    },
    "G_m": {
        "cols": [
            "G_m | def1_rollback_loss",
            "G_m | def2_rollback_acc",
            "G_m | def3_path_integral",
        ],
        "labels": ["def1_rollback_loss", "def2_rollback_acc", "def3_path_integral"],
        "descriptions": {
            "def1_rollback_loss":  "将模块回滚到θ^(0)后验证集损失增量 ΔL_val",
            "def2_rollback_acc":   "将模块回滚到θ^(0)后验证集准确率变化量 ΔAcc_val",
            "def3_path_integral":  "梯度-参数增量内积路径积分 Σ_t ∇_{θ_m}L^(t)·(θ_m^(t)−θ_m^(t-1))",
        },
        "cost_order": ["def3_path_integral", "def1_rollback_loss", "def2_rollback_acc"],
        "cost_note": {
            "def3_path_integral":  "中——需在训练中hook梯度和参数增量并逐步累积内积",
            "def1_rollback_loss":  "高——训练后逐模块回滚并在验证集前向推理（O(N_module×N_val)）",
            "def2_rollback_acc":   "高——同def1，额外计算准确率，成本相近",
        },
    },
    "R_hat": {
        "cols": [
            "R_hat | def1_probe_delta",
            "R_hat | def2_grad_curvature",
            "R_hat | def3_early_grad_norm",
            "R_hat | def4_ppred",
        ],
        "labels": ["def1_probe_delta", "def2_grad_curvature", "def3_early_grad_norm", "def4_ppred"],
        "descriptions": {
            "def1_probe_delta":    "探测训练后参数变化量 ‖θ_m^(t₀)−θ_m^(0)‖₂（探测步骤后恢复参数）",
            "def2_grad_curvature": "梯度曲率代理 E[‖g_m‖] / √(E[‖g_m‖²]+ε)",
            "def3_early_grad_norm":"早期T步训练梯度范数累积 Σ_{t=1}^{T_early} ‖g_m^(t)‖₂",
            "def4_ppred":         "元素级预测性 mean_i[E[|g_i|]² / (E[g_i²]+ε)]",
        },
        "cost_order": ["def2_grad_curvature", "def4_ppred", "def3_early_grad_norm", "def1_probe_delta"],
        "cost_note": {
            "def2_grad_curvature": "低——从已有梯度统计量（一阶/二阶矩）计算，无额外开销",
            "def4_ppred":         "低——元素级梯度统计量，额外内存但无额外推理",
            "def3_early_grad_norm":"低-中——仅需前T_early步hook梯度范数",
            "def1_probe_delta":   "高——需额外运行完整探测微调流程（若干步AdamW+参数恢复）",
        },
    },
}

# 为 Markdown 报告展示所有定义的成本排序（总体）
GLOBAL_COST_RANK = {
    "极低（无梯度，仅权重分析）": [
        "I_pre | sv_nuclear_norm", "I_pre | sv_top32_sum",
        "I_pre | sv_max_sv", "I_pre | sv_min_sv",
        "I_pre | se_spectral_entropy", "I_pre | se_raw_entropy",
        "U_m | def1_abs", "U_m | def2_rel",
    ],
    "低（1次反向传播或步内hook）": [
        "I_pre | saliency_grad_norm", "I_pre | saliency_taylor",
        "U_m | def3_path",
        "R_hat | def2_grad_curvature", "R_hat | def4_ppred",
        "R_hat | def3_early_grad_norm",
        "G_m | def3_path_integral",
    ],
    "中（多次前向/反向或验证集推理）": [
        "I_pre | fisher",
        "G_m | def1_rollback_loss", "G_m | def2_rollback_acc",
    ],
    "高（需额外训练/探测流程）": [
        "I_pre | perturbation",
        "R_hat | def1_probe_delta",
    ],
}

# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────
def corr_trio(s1, s2):
    """Return (pearson_r, spearman_r, kendall_tau) for two Series, drop NaN pairs."""
    df_tmp = pd.concat([s1, s2], axis=1).dropna()
    if len(df_tmp) < 5:
        return (np.nan, np.nan, np.nan)
    x, y = df_tmp.iloc[:, 0].values, df_tmp.iloc[:, 1].values
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y)[0]
    kt = kendalltau(x, y)[0]
    return pr, sr, kt


def top_k_metrics(series, k):
    """Return set of indices in the top-k (largest) values."""
    return set(series.dropna().nlargest(k).index)


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def overlap_at_k(s1, s2, k):
    a = top_k_metrics(s1, k)
    b = top_k_metrics(s2, k)
    inter = len(a & b)
    return inter, jaccard(a, b)


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# 1. Load & filter data
# ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(SCORE_CSV)

    # Remove layer == -1
    df = df[df["layer"] != -1]

    # Remove norm-related rows (LayerNorm modules)
    norm_mask = (
        df["module"].str.contains("LayerNorm", case=False, na=False)
        | df["module"].str.contains("layer_norm", case=False, na=False)
        | df["submodule"].str.contains("LayerNorm", case=False, na=False)
        | df["submodule"].str.contains("layer_norm", case=False, na=False)
    )
    df = df[~norm_mask].reset_index(drop=True)

    print(f"[Data] Loaded {len(df)} rows after filtering (no layer=-1, no Norm modules)")
    print(f"       Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. Within-metric consistency analysis
# ─────────────────────────────────────────────────────────────────
def within_metric_analysis(df):
    """
    For each metric group, compute pairwise (Pearson, Spearman, Kendall) correlations
    and produce scatter-matrix plots.
    """
    print("\n[1] Within-metric consistency analysis ...")
    records = []

    for metric, info in METRIC_GROUPS.items():
        cols   = info["cols"]
        labels = info["labels"]
        sub_df = df[cols].copy()
        sub_df.columns = labels

        n = len(labels)
        # ── Pairwise correlation table
        rows = []
        for i, j in combinations(range(n), 2):
            c1, c2 = labels[i], labels[j]
            pr, sr, kt = corr_trio(sub_df[c1], sub_df[c2])
            rows.append({
                "metric": metric, "def_A": c1, "def_B": c2,
                "pearson": pr, "spearman": sr, "kendall": kt,
                "n_valid": int(pd.concat([sub_df[c1], sub_df[c2]], axis=1).dropna().shape[0]),
            })
            records.append(rows[-1])

        tbl = pd.DataFrame(rows)
        tbl.to_csv(os.path.join(TBL_DIR, f"within_{metric}_corr.csv"), index=False)

        # ── Scatter-matrix plot
        valid_cols = [c for c in labels if sub_df[c].notna().sum() > 5]
        if len(valid_cols) < 2:
            continue
        sub_valid = sub_df[valid_cols].dropna(how="all")

        n_valid = len(valid_cols)
        fig, axes = plt.subplots(n_valid, n_valid,
                                 figsize=(3.5 * n_valid, 3.2 * n_valid))
        if n_valid == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        for i, ci in enumerate(valid_cols):
            for j, cj in enumerate(valid_cols):
                ax = axes[i, j]
                if i == j:
                    vals = sub_valid[ci].dropna().values
                    ax.hist(vals, bins=20, color="#4C72B0", alpha=0.75)
                    ax.set_xlabel(ci, fontsize=8)
                    ax.set_ylabel("Count", fontsize=7)
                else:
                    pair = pd.concat(
                        [sub_valid[ci].rename("x"), sub_valid[cj].rename("y")], axis=1
                    ).dropna()
                    pr, sr, kt = corr_trio(sub_valid[ci], sub_valid[cj])
                    ax.scatter(pair["y"], pair["x"], alpha=0.5, s=18, color="#4C72B0")
                    ax.set_xlabel(cj, fontsize=8)
                    ax.set_ylabel(ci, fontsize=8)
                    ax.set_title(f"r={pr:.2f}  ρ={sr:.2f}  τ={kt:.2f}",
                                 fontsize=8, color="#333")
                ax.tick_params(labelsize=7)

        fig.suptitle(f"Within-metric scatter matrix: {metric}", fontsize=13, y=1.01)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR, "within_metric", f"{metric}_scatter_matrix.png"))

        # ── Heatmap (Spearman)
        sp_mat = pd.DataFrame(np.eye(len(valid_cols)), index=valid_cols, columns=valid_cols)
        for i, ci in enumerate(valid_cols):
            for j, cj in enumerate(valid_cols):
                if i != j:
                    _, sr, _ = corr_trio(sub_valid[ci], sub_valid[cj])
                    sp_mat.loc[ci, cj] = sr

        fig2, ax2 = plt.subplots(figsize=(max(5, len(valid_cols) * 1.2 + 1),
                                          max(4, len(valid_cols) * 1.1 + 1)))
        sns.heatmap(sp_mat.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
                    vmin=-1, vmax=1, ax=ax2, square=True,
                    annot_kws={"size": 9}, linewidths=0.4)
        ax2.set_title(f"Spearman ρ within {metric}", fontsize=12)
        ax2.tick_params(axis="x", rotation=45, labelsize=8)
        ax2.tick_params(axis="y", rotation=0, labelsize=8)
        fig2.tight_layout()
        save_fig(fig2, os.path.join(FIG_DIR, "within_metric", f"{metric}_spearman_heatmap.png"))

    all_records = pd.DataFrame(records)
    all_records.to_csv(os.path.join(TBL_DIR, "within_metric_all_corr.csv"), index=False)
    print(f"  → Table: tables/within_metric_all_corr.csv")
    return all_records


# ─────────────────────────────────────────────────────────────────
# 3. Cross-metric correlations
# ─────────────────────────────────────────────────────────────────
def cross_metric_analysis(df):
    """Pairwise correlations between ALL metric columns (Pearson, Spearman, Kendall)."""
    print("\n[2] Cross-metric correlation analysis ...")

    all_cols = []
    short_labels = {}
    for metric, info in METRIC_GROUPS.items():
        for col, lab in zip(info["cols"], info["labels"]):
            all_cols.append(col)
            short_labels[col] = f"{metric}|{lab}"

    sub = df[all_cols].copy()
    sub.columns = [short_labels[c] for c in all_cols]

    # Build correlation matrices
    pearson_mat  = pd.DataFrame(np.nan, index=sub.columns, columns=sub.columns)
    spearman_mat = pd.DataFrame(np.nan, index=sub.columns, columns=sub.columns)
    kendall_mat  = pd.DataFrame(np.nan, index=sub.columns, columns=sub.columns)

    records = []
    for ci in sub.columns:
        for cj in sub.columns:
            pr, sr, kt = corr_trio(sub[ci], sub[cj])
            pearson_mat.loc[ci, cj]  = pr
            spearman_mat.loc[ci, cj] = sr
            kendall_mat.loc[ci, cj]  = kt
            if ci < cj:
                records.append({"col_A": ci, "col_B": cj,
                                 "pearson": pr, "spearman": sr, "kendall": kt})

    pd.DataFrame(records).to_csv(os.path.join(TBL_DIR, "cross_metric_pairwise_corr.csv"), index=False)

    # Save matrices
    pearson_mat.to_csv(os.path.join(TBL_DIR, "cross_metric_pearson_matrix.csv"))
    spearman_mat.to_csv(os.path.join(TBL_DIR, "cross_metric_spearman_matrix.csv"))
    kendall_mat.to_csv(os.path.join(TBL_DIR, "cross_metric_kendall_matrix.csv"))

    # Color annotation masks for metric groups
    group_colors = {"U_m": "#1f77b4", "I_pre": "#ff7f0e", "G_m": "#2ca02c", "R_hat": "#d62728"}
    col_to_group = {}
    for metric, info in METRIC_GROUPS.items():
        for col, lab in zip(info["cols"], info["labels"]):
            col_to_group[short_labels[col]] = metric

    # ── Plot all three heatmaps
    for name, mat in [("pearson", pearson_mat), ("spearman", spearman_mat), ("kendall", kendall_mat)]:
        fig, ax = plt.subplots(figsize=(len(sub.columns) * 0.9 + 2,
                                        len(sub.columns) * 0.85 + 2))
        cmap = "RdBu_r"
        vabs = 1.0 if name != "kendall" else 0.7
        sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap=cmap,
                    vmin=-vabs, vmax=vabs, ax=ax, square=True,
                    annot_kws={"size": 6.5}, linewidths=0.3)
        ax.set_title(f"Cross-metric {name.capitalize()} correlation", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", rotation=0, labelsize=7.5)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR, "cross_metric", f"cross_{name}_heatmap.png"))

    # ── Clustermap (Spearman)
    fill_mat = spearman_mat.astype(float).fillna(0)
    col_palette = [group_colors[col_to_group[c]] for c in fill_mat.columns]
    try:
        cg = sns.clustermap(fill_mat, cmap="RdBu_r", vmin=-1, vmax=1,
                             figsize=(len(sub.columns) * 0.9 + 3, len(sub.columns) * 0.85 + 3),
                             annot=True, fmt=".2f",
                             annot_kws={"size": 6},
                             col_colors=[col_palette],
                             row_colors=[col_palette],
                             linewidths=0.2,
                             xticklabels=True, yticklabels=True)
        cg.ax_heatmap.tick_params(axis="x", rotation=45, labelsize=7)
        plt.setp(cg.ax_heatmap.get_xticklabels(), ha="right")
        cg.ax_heatmap.tick_params(axis="y", rotation=0, labelsize=7)
        cg.fig.suptitle("Hierarchical cluster of Spearman correlations", y=1.01, fontsize=12)
        cg.fig.savefig(os.path.join(FIG_DIR, "cross_metric", "cross_spearman_clustermap.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(cg.fig)
        print(f"  Saved: {FIG_DIR}/cross_metric/cross_spearman_clustermap.png")
    except Exception as e:
        print(f"  [warn] clustermap failed: {e}")

    print("  → Tables: tables/cross_metric_*.csv")
    return spearman_mat


# ─────────────────────────────────────────────────────────────────
# 4. Top-k ranking overlap (Jaccard / intersection)
# ─────────────────────────────────────────────────────────────────
def ranking_overlap_analysis(df):
    """Top-k overlap and Jaccard between all pairs at k=10,20,30."""
    print("\n[3] Top-k ranking overlap analysis ...")

    KS = [10, 20, 30]

    all_cols = []
    short_labels = {}
    for metric, info in METRIC_GROUPS.items():
        for col, lab in zip(info["cols"], info["labels"]):
            all_cols.append(col)
            short_labels[col] = f"{metric}|{lab}"

    sub = df[all_cols].copy()
    sub.columns = [short_labels[c] for c in all_cols]

    for k in KS:
        overlap_mat  = pd.DataFrame(np.nan, index=sub.columns, columns=sub.columns)
        jaccard_mat  = pd.DataFrame(np.nan, index=sub.columns, columns=sub.columns)
        records = []

        for ci in sub.columns:
            for cj in sub.columns:
                ov, jac = overlap_at_k(sub[ci], sub[cj], k)
                overlap_mat.loc[ci, cj] = ov
                jaccard_mat.loc[ci, cj] = jac
                if ci < cj:
                    records.append({"col_A": ci, "col_B": cj,
                                    f"top{k}_overlap": ov,
                                    f"top{k}_jaccard": jac})

        pd.DataFrame(records).to_csv(
            os.path.join(TBL_DIR, f"ranking_top{k}.csv"), index=False)

        # Jaccard heatmap
        fig, ax = plt.subplots(figsize=(len(sub.columns) * 0.88 + 2,
                                        len(sub.columns) * 0.84 + 2))
        sns.heatmap(jaccard_mat.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=ax, square=True,
                    annot_kws={"size": 6.5}, linewidths=0.3)
        ax.set_title(f"Top-{k} Jaccard Index", fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", rotation=0, labelsize=7.5)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR, "ranking", f"top{k}_jaccard_heatmap.png"))

        # Overlap count heatmap
        fig, ax = plt.subplots(figsize=(len(sub.columns) * 0.88 + 2,
                                        len(sub.columns) * 0.84 + 2))
        sns.heatmap(overlap_mat.astype(float), annot=True, fmt=".0f", cmap="Blues",
                    vmin=0, vmax=k, ax=ax, square=True,
                    annot_kws={"size": 6.5}, linewidths=0.3)
        ax.set_title(f"Top-{k} Overlap Count", fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", rotation=0, labelsize=7.5)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR, "ranking", f"top{k}_overlap_heatmap.png"))

    print("  → Tables: tables/ranking_top*.csv")


# ─────────────────────────────────────────────────────────────────
# 5. Binned mean plots
# ─────────────────────────────────────────────────────────────────
def binned_mean_plots(df, n_bins=8):
    """
    For every (within-metric) pair A, B:
      - rank all modules by A, split into n_bins equal-size bins
      - compute mean of B in each bin
      - plot
    """
    print(f"\n[4] Binned mean plots (n_bins={n_bins}) ...")

    for metric, info in METRIC_GROUPS.items():
        cols   = info["cols"]
        labels = info["labels"]
        sub_df = df[cols].copy()
        sub_df.columns = labels

        valid_cols = [c for c in labels if sub_df[c].notna().sum() > max(n_bins * 2, 10)]
        if len(valid_cols) < 2:
            continue

        pairs = list(combinations(valid_cols, 2))
        # Each pair gives 2 plots (A vs B, B vs A) → combine in one figure
        n_pairs = len(pairs)
        ncols = 3
        nrows = int(np.ceil(n_pairs * 2 / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.2 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()
        plot_idx  = 0

        for (cA, cB) in pairs:
            for (x_col, y_col) in [(cA, cB), (cB, cA)]:
                pair = sub_df[[x_col, y_col]].dropna()
                if len(pair) < n_bins * 2:
                    continue
                pair = pair.sort_values(x_col).reset_index(drop=True)
                bin_labels_arr = pd.qcut(pair[x_col], q=n_bins, labels=False, duplicates="drop")
                bin_means_x = pair.groupby(bin_labels_arr)[x_col].mean()
                bin_means_y = pair.groupby(bin_labels_arr)[y_col].mean()
                bin_counts  = pair.groupby(bin_labels_arr)[y_col].count()

                ax = axes_flat[plot_idx]
                ax.bar(range(len(bin_means_x)), bin_means_y.values, color="#4C72B0",
                       alpha=0.7, edgecolor="white")
                ax.set_xticks(range(len(bin_means_x)))
                ax.set_xticklabels(
                    [f"{v:.2g}" for v in bin_means_x.values], rotation=40,
                    ha="right", fontsize=7.5)
                ax.set_xlabel(f"Bin by {x_col}", fontsize=8)
                ax.set_ylabel(f"Mean({y_col})", fontsize=8)
                ax.set_title(f"{metric}: sort by {x_col} → mean {y_col}", fontsize=8.5)
                ax.tick_params(labelsize=7.5)

                # Overlay spearman
                _, sr, _ = corr_trio(sub_df[x_col], sub_df[y_col])
                ax.text(0.97, 0.97, f"ρ={sr:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"Binned mean: {metric} (within-metric)", fontsize=13, y=1.01)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR, "binned_mean", f"{metric}_binned_mean.png"))

    # Also do cross-metric binned mean for "representative" columns
    # pick one col per metric (first valid col)
    repr_cols = {}
    for metric, info in METRIC_GROUPS.items():
        for col, lab in zip(info["cols"], info["labels"]):
            if df[col].notna().sum() > 20:
                repr_cols[metric] = (col, lab)
                break

    metric_names = list(repr_cols.keys())
    cross_pairs = list(combinations(metric_names, 2))
    if cross_pairs:
        ncols = 3
        nrows = int(np.ceil(len(cross_pairs) * 2 / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.2 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()
        plot_idx  = 0

        for (mA, mB) in cross_pairs:
            colA, labA = repr_cols[mA]
            colB, labB = repr_cols[mB]
            for (x_col, x_lab, y_col, y_lab) in [
                (colA, f"{mA}|{labA}", colB, f"{mB}|{labB}"),
                (colB, f"{mB}|{labB}", colA, f"{mA}|{labA}"),
            ]:
                pair = df[[x_col, y_col]].dropna()
                if len(pair) < n_bins * 2:
                    continue
                pair = pair.sort_values(x_col).reset_index(drop=True)
                bin_labels_arr = pd.qcut(pair[x_col], q=n_bins, labels=False, duplicates="drop")
                bin_means_x = pair.groupby(bin_labels_arr)[x_col].mean()
                bin_means_y = pair.groupby(bin_labels_arr)[y_col].mean()

                ax = axes_flat[plot_idx]
                ax.bar(range(len(bin_means_x)), bin_means_y.values, color="#DD8452",
                       alpha=0.75, edgecolor="white")
                ax.set_xticks(range(len(bin_means_x)))
                ax.set_xticklabels(
                    [f"{v:.2g}" for v in bin_means_x.values], rotation=40,
                    ha="right", fontsize=7.5)
                ax.set_xlabel(f"Bin by {x_lab}", fontsize=8)
                ax.set_ylabel(f"Mean({y_lab})", fontsize=8)
                ax.set_title(f"sort by {x_lab} → mean {y_lab}", fontsize=8.5)
                ax.tick_params(labelsize=7.5)
                _, sr, _ = corr_trio(df[x_col], df[y_col])
                ax.text(0.97, 0.97, f"ρ={sr:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                plot_idx += 1

        for idx in range(plot_idx, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Binned mean: cross-metric (representative defs)", fontsize=13, y=1.01)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR, "binned_mean", "cross_metric_binned_mean.png"))

    print("  → Figures: figures/binned_mean/")


# ─────────────────────────────────────────────────────────────────
# 6. Generate Markdown report
# ─────────────────────────────────────────────────────────────────
def generate_report(df, within_corr_df):
    print("\n[5] Generating report ...")
    lines = []
    lines += [
        "# MI Plasticity Score Table — Analysis Report",
        "",
        f"**过滤条件**: 去除 `layer=-1` 的行，去除所有 `LayerNorm` 相关模块",
        f"**有效数据行数**: {len(df)}",
        "",
    ]

    # ── 各定义说明及计算成本
    lines += ["## 一、各指标各定义含义及计算成本", ""]
    for metric, info in METRIC_GROUPS.items():
        lines += [f"### {metric}", ""]
        lines += ["| 定义 | 含义 | 计算成本 |", "|------|------|----------|"]
        for lab in info["labels"]:
            desc = info["descriptions"].get(lab, "")
            note = info["cost_note"].get(lab, "")
            lines.append(f"| `{lab}` | {desc} | {note} |")
        lines += [
            "",
            f"**计算成本排序**（从低到高）: "
            + " < ".join([f"`{x}`" for x in info["cost_order"]]),
            "",
        ]

    # 全局成本分级
    lines += ["### 全局计算成本分级", ""]
    lines += ["| 成本等级 | 指标定义 |", "|----------|----------|"]
    for level, cols in GLOBAL_COST_RANK.items():
        lines.append(f"| **{level}** | {', '.join([f'`{c}`' for c in cols])} |")
    lines += [""]

    # ── Within-metric 一致性摘要
    lines += ["## 二、指标内部一致性摘要", ""]
    for metric in METRIC_GROUPS:
        sub = within_corr_df[within_corr_df["metric"] == metric]
        if sub.empty:
            continue
        lines += [f"### {metric}", ""]
        lines += ["| 定义A | 定义B | n | Pearson r | Spearman ρ | Kendall τ |",
                  "|-------|-------|---|-----------|------------|-----------|"]
        for _, row in sub.iterrows():
            def fmt(v):
                return f"{v:.3f}" if not np.isnan(v) else "—"
            lines.append(
                f"| `{row['def_A']}` | `{row['def_B']}` | {row['n_valid']} "
                f"| {fmt(row['pearson'])} | {fmt(row['spearman'])} | {fmt(row['kendall'])} |"
            )
        lines += [""]

    # ── Cross-metric
    lines += [
        "## 三、指标间两两相关性",
        "",
        "详细矩阵见 `tables/cross_metric_*.csv`，热力图见 `figures/cross_metric/`。",
        "",
    ]

    # ── Ranking overlap
    lines += [
        "## 四、Top-k 排序重合度",
        "",
        "详细结果见 `tables/ranking_top*.csv`，Jaccard 热力图见 `figures/ranking/`。",
        "",
    ]

    # ── Binned mean
    lines += [
        "## 五、分桶均值图",
        "",
        "各指标内部及跨指标的分桶均值图见 `figures/binned_mean/`。",
        "图中横轴为按参照指标排序后的分桶中位数，纵轴为目标指标在该桶内的均值，",
        "右上角标注 Spearman ρ。",
        "",
    ]

    # ── Figure index
    lines += [
        "## 六、图表索引",
        "",
        "| 目录 | 内容 |",
        "|------|------|",
        "| `figures/within_metric/` | 各指标内部散点矩阵、Spearman 热力图 |",
        "| `figures/cross_metric/`  | 跨指标 Pearson/Spearman/Kendall 热力图、层次聚类图 |",
        "| `figures/ranking/`       | Top-10/20/30 Jaccard & Overlap Count 热力图 |",
        "| `figures/binned_mean/`   | 指标内及跨指标分桶均值图 |",
        "| `tables/`                | 所有数值结果 CSV |",
        "",
    ]

    report_path = os.path.join(BASE_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → Report: {report_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    within_corr_df = within_metric_analysis(df)
    cross_metric_analysis(df)
    ranking_overlap_analysis(df)
    binned_mean_plots(df, n_bins=8)
    generate_report(df, within_corr_df)
    print("\n[Done] All analysis complete.")


if __name__ == "__main__":
    main()
