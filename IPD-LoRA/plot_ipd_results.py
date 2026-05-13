import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


QUADRANT_COLOR = {
    "high_I_high_P": "#1f77b4",
    "high_I_low_P": "#ff7f0e",
    "low_I_high_P": "#2ca02c",
    "low_I_low_P": "#d62728",
    "frozen": "#9467bd",
    "warmup": "#7f7f7f",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot IPD-LoRA experiment results.")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _safe_read_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_ip_scatter(module_scores: pd.DataFrame, output_dir: str):
    if module_scores.empty:
        return
    steps = sorted(module_scores["step"].unique().tolist())
    for step in steps:
        df = module_scores[module_scores["step"] == step].copy()
        if df.empty:
            continue
        plt.figure(figsize=(8, 6))
        for quadrant, group in df.groupby("quadrant"):
            color = QUADRANT_COLOR.get(quadrant, "#333333")
            plt.scatter(group["ema_I"], group["ema_P"], label=quadrant, color=color, alpha=0.85)
            for _, row in group.iterrows():
                label = f"{int(row['layer_index'])}-{str(row['projection_type']).replace('_proj','')}"
                plt.text(row["ema_I"], row["ema_P"], label, fontsize=7, alpha=0.8)
        plt.xlabel("ema_I")
        plt.ylabel("ema_P")
        plt.title(f"IP Scatter at step {step}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ip_scatter_step_{step}.png"), dpi=180)
        plt.close()


def plot_topk_overlap(module_scores: pd.DataFrame, output_dir: str):
    if module_scores.empty:
        return
    percentages = [0.1, 0.2, 0.3, 0.5]
    steps = sorted(module_scores["step"].unique().tolist())
    overlap = {p: [] for p in percentages}
    valid_steps = []
    for step in steps:
        df = module_scores[module_scores["step"] == step].copy()
        n = len(df)
        if n == 0:
            continue
        valid_steps.append(step)
        ranked_I = df.sort_values("ema_I", ascending=False)["module_name"].tolist()
        ranked_P = df.sort_values("ema_P", ascending=False)["module_name"].tolist()
        for p in percentages:
            k = max(1, int(round(n * p)))
            top_I = set(ranked_I[:k])
            top_P = set(ranked_P[:k])
            ov = len(top_I.intersection(top_P)) / float(k)
            overlap[p].append(ov)

    plt.figure(figsize=(8, 5))
    for p in percentages:
        if overlap[p]:
            plt.plot(valid_steps, overlap[p], marker="o", label=f"Overlap@{int(p*100)}%")
    plt.xlabel("step")
    plt.ylabel("Overlap@k")
    plt.title("I/P Top-k Overlap")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ip_topk_overlap.png"), dpi=180)
    plt.close()


def plot_correlation(module_scores: pd.DataFrame, output_dir: str):
    if module_scores.empty:
        return
    steps = sorted(module_scores["step"].unique().tolist())
    pearson_vals = []
    spearman_vals = []
    valid_steps = []
    for step in steps:
        df = module_scores[module_scores["step"] == step]
        if len(df) < 2:
            continue
        x = df["ema_I"].to_numpy(dtype=np.float64)
        y = df["ema_P"].to_numpy(dtype=np.float64)
        p = pearsonr(x, y)[0] if np.std(x) > 0 and np.std(y) > 0 else 0.0
        s = spearmanr(x, y)[0] if np.std(x) > 0 and np.std(y) > 0 else 0.0
        valid_steps.append(step)
        pearson_vals.append(p)
        spearman_vals.append(s)

    if not valid_steps:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(valid_steps, pearson_vals, marker="o", label="Pearson")
    plt.plot(valid_steps, spearman_vals, marker="s", label="Spearman")
    plt.xlabel("step")
    plt.ylabel("correlation")
    plt.title("I/P Correlation over Training")
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ip_correlation.png"), dpi=180)
    plt.close()


def plot_active_rank(rank_history: pd.DataFrame, output_dir: str):
    if rank_history.empty:
        return
    agg = rank_history.groupby("step", as_index=False)["active_rank"].sum().sort_values("step")
    plt.figure(figsize=(8, 5))
    plt.plot(agg["step"], agg["active_rank"], marker="o")
    plt.xlabel("step")
    plt.ylabel("sum(active_rank)")
    plt.title("Active Rank over Training")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "active_rank_over_time.png"), dpi=180)
    plt.close()


def plot_quadrant_distribution(quadrant_history: pd.DataFrame, output_dir: str):
    if quadrant_history.empty:
        return
    cnt = (
        quadrant_history.groupby(["step", "quadrant"], as_index=False)
        .size()
        .pivot(index="step", columns="quadrant", values="size")
        .fillna(0)
        .sort_index()
    )
    plt.figure(figsize=(9, 5))
    for col in cnt.columns:
        plt.plot(cnt.index.values, cnt[col].values, marker="o", label=str(col), color=QUADRANT_COLOR.get(col))
    plt.xlabel("step")
    plt.ylabel("module count")
    plt.title("Quadrant Distribution over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quadrant_distribution.png"), dpi=180)
    plt.close()


def plot_eval_curve(training_log: pd.DataFrame, output_dir: str):
    if training_log.empty:
        return
    df = training_log.dropna(subset=["eval_accuracy", "eval_loss"], how="all").copy()
    if df.empty:
        return
    df = df.sort_values("step")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["step"], df["eval_accuracy"], color="#1f77b4", marker="o", label="eval_accuracy")
    ax1.set_xlabel("step")
    ax1.set_ylabel("eval_accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["eval_loss"], color="#d62728", marker="s", label="eval_loss")
    ax2.set_ylabel("eval_loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Evaluation Curve")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_curve.png"), dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir

    module_scores = _safe_read_jsonl(os.path.join(output_dir, "module_scores.jsonl"))
    rank_history = _safe_read_csv(os.path.join(output_dir, "rank_history.csv"))
    quadrant_history = _safe_read_csv(os.path.join(output_dir, "quadrant_history.csv"))
    training_log = _safe_read_jsonl(os.path.join(output_dir, "training_log.jsonl"))

    plot_ip_scatter(module_scores, output_dir)
    plot_topk_overlap(module_scores, output_dir)
    plot_correlation(module_scores, output_dir)
    plot_active_rank(rank_history, output_dir)
    plot_quadrant_distribution(quadrant_history, output_dir)
    plot_eval_curve(training_log, output_dir)
    print("[done] Plots generated.")


if __name__ == "__main__":
    main()
