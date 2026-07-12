import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJ_COLOR = {
    "query_proj": "#1f77b4",
    "key_proj": "#ff7f0e",
    "value_proj": "#2ca02c",
    "output_proj": "#d62728",
    "ffn_in": "#9467bd",
    "ffn_out": "#8c564b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GP-only LoRA experiment results.")
    parser.add_argument("--output_dir", type=str, required=True, help="Training output directory.")
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to write figures. Defaults to <output_dir>/plots.",
    )
    parser.add_argument(
        "--top_k_modules",
        type=int,
        default=20,
        help="Number of modules shown in summary bar plots.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_read_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.DataFrame()


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _module_label(row) -> str:
    if "layer_index" in row and "projection_type" in row:
        proj = str(row["projection_type"]).replace("_proj", "")
        return f"L{int(row['layer_index'])}-{proj}"
    return str(row.get("module_name", "module")).split(".")[-1]


def _eval_records(training_log: pd.DataFrame) -> pd.DataFrame:
    if training_log.empty or "eval_loss" not in training_log.columns:
        return pd.DataFrame()
    df = training_log.dropna(subset=["eval_loss"], how="all").copy()
    if "train_loss" in df.columns:
        # Periodic train logs may carry stale eval values; true eval rows have train_loss=None.
        df = df[df["train_loss"].isna()].copy()
    if df.empty:
        return df
    return df.groupby("step", as_index=False).last().sort_values("step")


def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_train_eval_curves(training_log: pd.DataFrame, plot_dir: str):
    if training_log.empty:
        return

    if {"step", "train_loss"}.issubset(training_log.columns):
        train_df = training_log.dropna(subset=["train_loss"]).copy().sort_values("step")
        if not train_df.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(train_df["step"], train_df["train_loss"], marker="o", linewidth=1.5)
            plt.xlabel("step")
            plt.ylabel("train_loss")
            plt.title("Training Loss")
            _savefig(os.path.join(plot_dir, "train_loss_curve.png"))

    eval_df = _eval_records(training_log)
    if eval_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    if "eval_accuracy" in eval_df.columns:
        ax1.plot(eval_df["step"], eval_df["eval_accuracy"], color="#1f77b4", marker="o", label="eval metric")
        ax1.set_ylabel("eval metric", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("step")

    if "eval_loss" in eval_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(eval_df["step"], eval_df["eval_loss"], color="#d62728", marker="s", label="eval loss")
        ax2.set_ylabel("eval loss", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Evaluation Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "eval_curve.png"), dpi=180)
    plt.close(fig)


def plot_active_rank(rank_history: pd.DataFrame, plot_dir: str):
    if rank_history.empty or not {"step", "active_rank"}.issubset(rank_history.columns):
        return
    agg = rank_history.groupby("step", as_index=False)["active_rank"].agg(["sum", "mean"]).reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(agg["step"], agg["sum"], marker="o", color="#1f77b4")
    plt.xlabel("step")
    plt.ylabel("sum(active_rank)")
    plt.title("Total Active Rank Budget over Training")
    _savefig(os.path.join(plot_dir, "active_rank_over_time.png"))

    plt.figure(figsize=(8, 5))
    plt.plot(agg["step"], agg["mean"], marker="s", color="#ff7f0e")
    plt.xlabel("step")
    plt.ylabel("mean(active_rank)")
    plt.title("Mean Active Rank per Module over Training")
    _savefig(os.path.join(plot_dir, "mean_active_rank_over_time.png"))


def _pivot_modules(df: pd.DataFrame, value_col: str) -> Optional[pd.DataFrame]:
    if df.empty or not {"step", "module_name", value_col}.issubset(df.columns):
        return None
    ordered = df.copy()
    if {"layer_index", "projection_type"}.issubset(ordered.columns):
        ordered["module_label"] = ordered.apply(_module_label, axis=1)
        index_col = "module_label"
        ordered = ordered.sort_values(["layer_index", "projection_type", "module_name"])
    else:
        index_col = "module_name"
        ordered = ordered.sort_values("module_name")
    pivot = ordered.pivot_table(index=index_col, columns="step", values=value_col, aggfunc="last")
    return pivot


def plot_heatmap(pivot: Optional[pd.DataFrame], title: str, cbar_label: str, filename: str, plot_dir: str):
    if pivot is None or pivot.empty:
        return
    fig_height = max(5.0, min(16.0, 0.22 * len(pivot.index) + 2.5))
    plt.figure(figsize=(12, fig_height))
    arr = pivot.to_numpy(dtype=np.float64)
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, label=cbar_label)
    plt.yticks(np.arange(len(pivot.index)), pivot.index, fontsize=7)
    step_labels = [str(int(c)) for c in pivot.columns]
    if len(step_labels) > 12:
        keep = np.linspace(0, len(step_labels) - 1, 12, dtype=int)
        plt.xticks(keep, [step_labels[i] for i in keep], rotation=45)
    else:
        plt.xticks(np.arange(len(step_labels)), step_labels, rotation=45)
    plt.xlabel("step")
    plt.title(title)
    _savefig(os.path.join(plot_dir, filename))


def plot_rank_and_goodput_heatmaps(module_scores: pd.DataFrame, rank_history: pd.DataFrame, plot_dir: str):
    plot_heatmap(
        _pivot_modules(module_scores, "ema_G"),
        title="Module EMA Goodput Heatmap",
        cbar_label="ema_G",
        filename="module_goodput_heatmap.png",
        plot_dir=plot_dir,
    )
    rank_source = module_scores if "active_rank" in module_scores.columns else rank_history
    plot_heatmap(
        _pivot_modules(rank_source, "active_rank"),
        title="Module Active Rank Heatmap",
        cbar_label="active_rank",
        filename="rank_heatmap.png",
        plot_dir=plot_dir,
    )


def plot_goodput_curve(module_scores: pd.DataFrame, plot_dir: str):
    if module_scores.empty or "ema_G" not in module_scores.columns:
        return
    agg_spec = {"ema_G": ["mean", "median", "std"]}
    if "current_G" in module_scores.columns:
        agg_spec["current_G"] = ["mean"]
    df = module_scores.groupby("step").agg(agg_spec)
    df.columns = ["_".join(c).strip("_") for c in df.columns.to_flat_index()]
    df = df.reset_index().sort_values("step")
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["ema_G_mean"], marker="o", label="mean ema_G")
    plt.plot(df["step"], df["ema_G_median"], marker="s", label="median ema_G")
    if "current_G_mean" in df.columns:
        plt.plot(df["step"], df["current_G_mean"], marker="^", alpha=0.65, label="mean current_G")
    if "ema_G_std" in df.columns:
        lo = df["ema_G_mean"] - df["ema_G_std"]
        hi = df["ema_G_mean"] + df["ema_G_std"]
        plt.fill_between(df["step"], lo, hi, alpha=0.18, label="ema_G ± std")
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("step")
    plt.ylabel("Module Learning Goodput")
    plt.title("Module Goodput over Training")
    plt.legend()
    _savefig(os.path.join(plot_dir, "goodput_curve.png"))


def plot_global_goodput_curve(training_log: pd.DataFrame, plot_dir: str):
    if training_log.empty or "global_goodput" not in training_log.columns:
        return
    rows = []
    for _, r in training_log.iterrows():
        gp = r.get("global_goodput")
        if isinstance(gp, dict) and gp:
            rows.append(
                {
                    "step": int(r["step"]),
                    "rank_step": float(gp.get("goodput/rank_step", np.nan)),
                    "per_second": float(gp.get("goodput/per_second", np.nan)),
                    "per_flop": float(gp.get("goodput/per_flop", np.nan)),
                    "delta_val_loss": float(gp.get("goodput/delta_val_loss", np.nan)),
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("step")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    panels = [
        ("rank_step", "Delta ValLoss / Rank-Step"),
        ("per_second", "Delta ValLoss / Second"),
        ("per_flop", "Delta ValLoss / FLOPs(proxy)"),
        ("delta_val_loss", "Window Delta ValLoss"),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(df["step"], df[col], marker="o")
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("step")
        ax.set_title(title)
    fig.suptitle("Global Goodput")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "global_goodput_curve.png"), dpi=180)
    plt.close(fig)


def plot_goodput_rank_alignment(module_scores: pd.DataFrame, plot_dir: str):
    required = {"step", "module_name", "ema_G", "active_rank"}
    if module_scores.empty or not required.issubset(module_scores.columns):
        return

    rows: List[Dict[str, float]] = []
    for step, df in module_scores.groupby("step"):
        if len(df) < 3:
            continue
        if df["ema_G"].nunique(dropna=True) <= 1 or df["active_rank"].nunique(dropna=True) <= 1:
            corr = 0.0
        else:
            corr = float(df["ema_G"].corr(df["active_rank"], method="spearman"))
        rows.append({"step": int(step), "spearman_ema_G_active_rank": corr})
    if rows:
        corr_df = pd.DataFrame(rows).sort_values("step")
        corr_df.to_csv(os.path.join(plot_dir, "goodput_rank_alignment.csv"), index=False)
        plt.figure(figsize=(8, 5))
        plt.plot(corr_df["step"], corr_df["spearman_ema_G_active_rank"], marker="o")
        plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
        plt.ylim(-1.05, 1.05)
        plt.xlabel("step")
        plt.ylabel("Spearman(ema_G, active_rank)")
        plt.title("Goodput-Rank Alignment over Training")
        _savefig(os.path.join(plot_dir, "goodput_rank_alignment.png"))

    last_step = int(module_scores["step"].max())
    last = module_scores[module_scores["step"] == last_step].copy()
    if last.empty:
        return
    colors = [
        PROJ_COLOR.get(str(v), "#333333")
        for v in last.get("projection_type", pd.Series(["module"] * len(last))).tolist()
    ]
    plt.figure(figsize=(8, 6))
    plt.scatter(last["ema_G"], last["active_rank"], c=colors, s=80, alpha=0.85, edgecolors="k", linewidths=0.4)
    for _, row in last.iterrows():
        plt.text(float(row["ema_G"]), float(row["active_rank"]), _module_label(row), fontsize=7, alpha=0.75)
    plt.axvline(float(last["ema_G"].median()), linestyle="--", color="gray", linewidth=1)
    plt.xlabel("ema_G")
    plt.ylabel("active_rank")
    plt.title(f"Final Goodput vs Rank at step {last_step}")
    _savefig(os.path.join(plot_dir, "final_goodput_rank_scatter.png"))


def plot_module_summary(module_scores: pd.DataFrame, plot_dir: str, top_k: int):
    if module_scores.empty or "ema_G" not in module_scores.columns:
        return
    df = module_scores.copy()
    df["module_label"] = df.apply(_module_label, axis=1)
    summary = (
        df.groupby(["module_name", "module_label"], as_index=False)
        .agg(
            mean_ema_G=("ema_G", "mean"),
            max_ema_G=("ema_G", "max"),
            mean_active_rank=("active_rank", "mean"),
            final_step=("step", "max"),
        )
        .sort_values("mean_ema_G", ascending=False)
    )
    final_rank = (
        df.sort_values("step")
        .groupby("module_name", as_index=False)
        .last()[["module_name", "active_rank", "ema_G"]]
        .rename(columns={"active_rank": "final_active_rank", "ema_G": "final_ema_G"})
    )
    summary = summary.merge(final_rank, on="module_name", how="left")
    summary.to_csv(os.path.join(plot_dir, "module_goodput_summary.csv"), index=False)

    top = summary.head(max(1, int(top_k))).copy().iloc[::-1]
    if top.empty:
        return
    plt.figure(figsize=(9, max(5, 0.35 * len(top) + 1.5)))
    plt.barh(top["module_label"], top["mean_ema_G"], color="#1f77b4", alpha=0.85)
    plt.xlabel("mean ema_G")
    plt.ylabel("module")
    plt.title(f"Top-{len(top)} Modules by Mean Goodput")
    _savefig(os.path.join(plot_dir, "top_modules_by_goodput.png"))


def plot_rank_budget_efficiency(training_log: pd.DataFrame, plot_dir: str):
    eval_df = _eval_records(training_log)
    if eval_df.empty or "active_total_rank" not in eval_df.columns:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    if "eval_accuracy" in eval_df.columns:
        ax1.plot(eval_df["step"], eval_df["eval_accuracy"], color="#1f77b4", marker="o", label="eval metric")
        ax1.set_ylabel("eval metric", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("step")
    ax2 = ax1.twinx()
    ax2.plot(eval_df["step"], eval_df["active_total_rank"], color="#ff7f0e", marker="s", label="active total rank")
    ax2.set_ylabel("active total rank", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    plt.title("Evaluation Progress vs Rank Budget")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "rank_budget_efficiency.png"), dpi=180)
    plt.close(fig)


def compute_efficiency_metrics(training_log: pd.DataFrame, plot_dir: str, threshold_ratio: float = 0.9):
    eval_df = _eval_records(training_log)
    if eval_df.empty or "eval_accuracy" not in eval_df.columns:
        return
    steps = eval_df["step"].to_numpy(dtype=np.float64)
    acc = eval_df["eval_accuracy"].to_numpy(dtype=np.float64)
    final_acc = float(acc[-1])
    best_acc = float(np.max(acc))

    trapz = getattr(np, "trapezoid", np.trapz)
    if len(steps) >= 2:
        span = float(steps[-1] - steps[0])
        aulc = float(trapz(acc, steps) / span) if span > 0 else final_acc
    else:
        aulc = final_acc

    target = threshold_ratio * best_acc
    hit = np.where(acc >= target)[0]
    steps_to_threshold = int(steps[hit[0]]) if len(hit) > 0 else None

    gp_means: Dict[str, float] = {}
    if "global_goodput" in training_log.columns:
        vals = {"rank_step": [], "per_second": [], "per_flop": []}
        for _, r in training_log.iterrows():
            gp = r.get("global_goodput")
            if isinstance(gp, dict) and gp:
                vals["rank_step"].append(float(gp.get("goodput/rank_step", np.nan)))
                vals["per_second"].append(float(gp.get("goodput/per_second", np.nan)))
                vals["per_flop"].append(float(gp.get("goodput/per_flop", np.nan)))
        gp_means = {
            f"mean_goodput_{k}": float(np.nanmean(v))
            for k, v in vals.items()
            if len(v) > 0
        }

    summary = {
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "aulc_accuracy": aulc,
        "steps_to_threshold": steps_to_threshold,
        "threshold_ratio": float(threshold_ratio),
        **gp_means,
    }
    with open(os.path.join(plot_dir, "efficiency_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    output_dir = args.output_dir
    plot_dir = args.plot_dir or os.path.join(output_dir, "plots")
    ensure_dir(plot_dir)

    module_scores = _safe_read_jsonl(os.path.join(output_dir, "module_scores.jsonl"))
    rank_history = _safe_read_csv(os.path.join(output_dir, "rank_history.csv"))
    training_log = _safe_read_jsonl(os.path.join(output_dir, "training_log.jsonl"))

    plot_train_eval_curves(training_log, plot_dir)
    plot_active_rank(rank_history, plot_dir)
    plot_goodput_curve(module_scores, plot_dir)
    plot_global_goodput_curve(training_log, plot_dir)
    plot_rank_and_goodput_heatmaps(module_scores, rank_history, plot_dir)
    plot_goodput_rank_alignment(module_scores, plot_dir)
    plot_module_summary(module_scores, plot_dir, top_k=max(1, int(args.top_k_modules)))
    plot_rank_budget_efficiency(training_log, plot_dir)
    compute_efficiency_metrics(training_log, plot_dir, threshold_ratio=0.9)

    print(f"[done] GP-only plots generated in {plot_dir}")


if __name__ == "__main__":
    main()
