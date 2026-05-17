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
    parser.add_argument(
        "--future_eval_points",
        type=int,
        default=1,
        help="Use eval loss decrease from current eval point to N-th future eval point.",
    )
    parser.add_argument(
        "--future_score_points",
        type=int,
        default=1,
        help="Use module I increase from current scoring point to N-th future scoring point.",
    )
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


def _attach_future_loss_decrease(
    module_scores: pd.DataFrame,
    training_log: pd.DataFrame,
    future_eval_points: int = 1,
) -> pd.DataFrame:
    if module_scores.empty or training_log.empty:
        return module_scores.copy()
    # Only keep true evaluation records.
    # In training_log, logging rows may carry stale eval_loss for convenience, but they are not new eval points.
    eval_df = training_log.dropna(subset=["eval_loss"]).copy()
    if "train_loss" in eval_df.columns:
        eval_df = eval_df[eval_df["train_loss"].isna()].copy()
    if eval_df.empty:
        return module_scores.copy()
    eval_df = eval_df.groupby("step", as_index=False)["eval_loss"].last().sort_values("step")
    eval_steps = eval_df["step"].to_numpy(dtype=np.int64)
    eval_losses = eval_df["eval_loss"].to_numpy(dtype=np.float64)
    if len(eval_steps) < (future_eval_points + 1):
        out = module_scores.copy()
        out["loss_decrease"] = np.nan
        out["training_gain"] = np.nan
        return out

    out = module_scores.copy()
    out["loss_decrease"] = np.nan
    out["training_gain"] = np.nan

    module_steps = out["step"].to_numpy(dtype=np.int64)
    idx_now = np.searchsorted(eval_steps, module_steps, side="right") - 1
    valid_now = idx_now >= 0
    idx_future = idx_now + int(future_eval_points)
    valid_future = idx_future < len(eval_steps)
    valid = valid_now & valid_future
    if valid.any():
        now_loss = eval_losses[idx_now[valid]]
        fut_loss = eval_losses[idx_future[valid]]
        decrease = now_loss - fut_loss
        out.loc[valid, "loss_decrease"] = decrease
        rank = out.loc[valid, "active_rank"].astype(float).to_numpy()
        out.loc[valid, "training_gain"] = decrease / np.clip(rank, 1.0, None)
    return out


def _attach_future_i_increase(module_scores: pd.DataFrame, future_score_points: int = 1) -> pd.DataFrame:
    if module_scores.empty:
        return module_scores.copy()
    out = module_scores.copy()
    out = out.sort_values(["module_name", "step"]).reset_index(drop=True)
    out["future_ema_I"] = out.groupby("module_name")["ema_I"].shift(-int(future_score_points))
    out["delta_ema_I"] = out["future_ema_I"] - out["ema_I"]
    return out


def plot_quadrant_actual_gain(module_scores: pd.DataFrame, training_log: pd.DataFrame, output_dir: str, future_eval_points: int):
    if module_scores.empty:
        return
    quadrant_order = ["high_I_high_P", "high_I_low_P", "low_I_high_P", "low_I_low_P"]
    ms = module_scores[module_scores["quadrant"].isin(quadrant_order)].copy()
    if ms.empty:
        return
    ms = _attach_future_loss_decrease(ms, training_log, future_eval_points=future_eval_points)

    # Final retention ratio is computed on last scoring step.
    last_step = ms["step"].max()
    final_df = ms[ms["step"] == last_step].copy()
    retention = (
        final_df.assign(retained=final_df["active_rank"] > 0)
        .groupby("quadrant", as_index=False)["retained"]
        .mean()
        .rename(columns={"retained": "final_retention_ratio"})
    )

    summary = (
        ms.groupby("quadrant", as_index=False)
        .agg(
            avg_I=("ema_I", "mean"),
            avg_P=("ema_P", "mean"),
            avg_active_rank=("active_rank", "mean"),
            avg_training_gain=("training_gain", "mean"),
            avg_loss_decrease=("loss_decrease", "mean"),
        )
        .merge(retention, on="quadrant", how="left")
    )
    summary["final_retention_ratio"] = summary["final_retention_ratio"].fillna(0.0)
    summary["quadrant"] = pd.Categorical(summary["quadrant"], categories=quadrant_order, ordered=True)
    summary = summary.sort_values("quadrant")
    summary.to_csv(os.path.join(output_dir, "quadrant_module_stats.csv"), index=False)

    # Plot requested: actual training gain by quadrant (validation loss decrease in future steps).
    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary))
    y = summary["avg_loss_decrease"].to_numpy(dtype=float)
    colors = [QUADRANT_COLOR.get(q, "#333333") for q in summary["quadrant"].tolist()]
    plt.bar(x, y, color=colors, alpha=0.9)
    plt.xticks(x, summary["quadrant"], rotation=15)
    plt.xlabel("quadrant")
    plt.ylabel(f"validation loss decrease (+{future_eval_points} eval)")
    plt.title("Actual Training Gain by Quadrant")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quadrant_actual_training_gain.png"), dpi=180)
    plt.close()

    # Supplementary multi-metric panel for requested six statistics.
    metric_cols = [
        ("avg_I", "avg I"),
        ("avg_P", "avg P"),
        ("avg_active_rank", "avg active rank"),
        ("avg_training_gain", "avg training gain"),
        ("avg_loss_decrease", "avg loss decrease"),
        ("final_retention_ratio", "final retention ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metric_cols):
        vals = summary[col].to_numpy(dtype=float)
        ax.bar(summary["quadrant"], vals, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Quadrant Module Statistics")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quadrant_module_stats.png"), dpi=180)
    plt.close(fig)


def plot_quadrant_gain_interpretation(
    module_scores: pd.DataFrame,
    training_log: pd.DataFrame,
    output_dir: str,
    future_eval_points: int,
    future_score_points: int,
):
    """
    Extra interpretation plots to support expected conclusions:
    1) high_I_high_P has highest or most stable gain
    2) low_I_high_P can raise future I (exploration effective)
    3) high_I_low_P has low gain but should be retained (importance high)
    4) low_I_low_P can be pruned with small impact
    """
    if module_scores.empty:
        return
    quadrant_order = ["high_I_high_P", "high_I_low_P", "low_I_high_P", "low_I_low_P"]
    ms = module_scores[module_scores["quadrant"].isin(quadrant_order)].copy()
    if ms.empty:
        return

    ms = _attach_future_loss_decrease(ms, training_log, future_eval_points=future_eval_points)
    ms = _attach_future_i_increase(ms, future_score_points=future_score_points)

    # Keep rows with valid future targets for each analysis.
    ms_loss = ms.dropna(subset=["loss_decrease"]).copy()
    ms_i = ms.dropna(subset=["delta_ema_I"]).copy()

    # Plot A: gain stability (boxplot of future validation loss decrease by quadrant).
    if not ms_loss.empty:
        data = []
        labels = []
        colors = []
        for q in quadrant_order:
            qvals = ms_loss.loc[ms_loss["quadrant"] == q, "loss_decrease"].to_numpy(dtype=float)
            if len(qvals) == 0:
                continue
            data.append(qvals)
            labels.append(q)
            colors.append(QUADRANT_COLOR.get(q, "#333333"))
        if data:
            plt.figure(figsize=(9, 5))
            bp = plt.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            plt.xlabel("quadrant")
            plt.ylabel(f"validation loss decrease (+{future_eval_points} eval)")
            plt.title("Quadrant Gain Stability (Distribution)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "quadrant_gain_stability.png"), dpi=180)
            plt.close()

    # Plot B: exploration effectiveness (future I increase).
    if not ms_i.empty:
        rows = []
        for q in quadrant_order:
            qdf = ms_i[ms_i["quadrant"] == q]
            if qdf.empty:
                continue
            delta = qdf["delta_ema_I"].to_numpy(dtype=float)
            pos_ratio = float((delta > 0).mean()) if len(delta) > 0 else np.nan
            rows.append(
                {
                    "quadrant": q,
                    "avg_delta_ema_I": float(np.mean(delta)),
                    "std_delta_ema_I": float(np.std(delta)),
                    "positive_I_increase_ratio": pos_ratio,
                }
            )
        if rows:
            df_delta = pd.DataFrame(rows)
            df_delta["quadrant"] = pd.Categorical(df_delta["quadrant"], categories=quadrant_order, ordered=True)
            df_delta = df_delta.sort_values("quadrant")
            df_delta.to_csv(os.path.join(output_dir, "quadrant_exploration_stats.csv"), index=False)

            x = np.arange(len(df_delta))
            y = df_delta["avg_delta_ema_I"].to_numpy(dtype=float)
            yerr = df_delta["std_delta_ema_I"].to_numpy(dtype=float)
            colors = [QUADRANT_COLOR.get(q, "#333333") for q in df_delta["quadrant"].tolist()]
            plt.figure(figsize=(9, 5))
            bars = plt.bar(x, y, yerr=yerr, capsize=4, color=colors, alpha=0.9)
            plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
            plt.xticks(x, df_delta["quadrant"], rotation=15)
            plt.xlabel("quadrant")
            plt.ylabel(f"future Δema_I (+{future_score_points} scoring)")
            plt.title("Quadrant Exploration Effectiveness")
            for i, bar in enumerate(bars):
                txt = f"pos={df_delta.iloc[i]['positive_I_increase_ratio']:.2f}"
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "quadrant_exploration_effectiveness.png"), dpi=180)
            plt.close()

    # Plot C: importance vs gain vs retention (to explain keep/prune decisions).
    summary_path = os.path.join(output_dir, "quadrant_module_stats.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        if not summary.empty:
            summary = summary[summary["quadrant"].isin(quadrant_order)].copy()
            summary["quadrant"] = pd.Categorical(summary["quadrant"], categories=quadrant_order, ordered=True)
            summary = summary.sort_values("quadrant")
            plt.figure(figsize=(8, 6))
            for _, row in summary.iterrows():
                q = row["quadrant"]
                x = float(row["avg_I"])
                y = float(row["avg_loss_decrease"])
                size = 400.0 * float(row["final_retention_ratio"] + 0.2)
                plt.scatter(x, y, s=size, color=QUADRANT_COLOR.get(q, "#333333"), alpha=0.8)
                plt.text(x, y, q, fontsize=9)
            plt.xlabel("avg I (importance)")
            plt.ylabel(f"avg validation loss decrease (+{future_eval_points} eval)")
            plt.title("Keep vs Prune Map (Importance / Gain / Retention)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "quadrant_keep_prune_map.png"), dpi=180)
            plt.close()


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
    plot_quadrant_actual_gain(
        module_scores,
        training_log,
        output_dir,
        future_eval_points=max(1, int(args.future_eval_points)),
    )
    plot_quadrant_gain_interpretation(
        module_scores,
        training_log,
        output_dir,
        future_eval_points=max(1, int(args.future_eval_points)),
        future_score_points=max(1, int(args.future_score_points)),
    )
    print("[done] Plots generated.")


if __name__ == "__main__":
    main()
