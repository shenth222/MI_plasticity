#!/usr/bin/env python
"""Aggregate harness runs into a mean +/- std comparison table.

Scans ``outputs/<task>/<method>/seed*/eval_results.json`` and summarizes each
method across seeds (best/final accuracy, rank budget, trainable params,
wall-clock, rank-step budget). Writes both Markdown and CSV next to the task dir.

Usage:
  python scripts/aggregate_results.py --task rte
  python scripts/aggregate_results.py --task rte --outputs outputs
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

METHOD_ORDER = {"lora": 0, "gora": 1, "adalora": 2, "goodput": 3}

# GLUE-standard primary metric per task (key in the stored scores dict).
PRIMARY_METRIC_BY_TASK = {
    "cola": ("matthews_correlation", "MCC"),
    "mrpc": ("f1", "F1"),
    "qqp": ("f1", "F1"),
    "stsb": ("pearson", "pearson"),
}


# Subtasks that are degenerate under this harness (all methods collapse to a
# constant answer, scores < random). Excluded from the commonsense primary metric.
COMMONSENSE_EXCLUDED_SUBTASKS = {"wino", "winogrande"}


def task_metric(task):
    if task.lower() == "commonsense":
        return ("__excl_wino__", "excl-wino acc")
    return PRIMARY_METRIC_BY_TASK.get(task.lower(), ("accuracy", "acc"))


def _commonsense_excl_wino(d):
    """Mean accuracy over non-degenerate commonsense subtasks (equal-weight;
    subtasks are ~equal size, within <0.1pp of sample-weighted)."""
    det = d.get("final_eval_detail", {}) or {}
    accs = [
        float(v)
        for k, v in det.items()
        if k.startswith("acc_") and k[len("acc_"):] not in COMMONSENSE_EXCLUDED_SUBTASKS
    ]
    if not accs:
        return d.get("final_eval_accuracy") or d.get("best_eval_accuracy")
    return sum(accs) / len(accs)


def extract_metric(d, metric_key):
    """Read the task's primary metric from stored scores; fall back to best_eval_accuracy."""
    if metric_key == "__excl_wino__":
        return _commonsense_excl_wino(d)
    split = d.get("final_eval_primary_split")
    sr = d.get("final_eval_split_results", {}) or {}
    row = sr.get(split) if split else None
    if row is None and sr:
        row = next(iter(sr.values()))
    if row:
        scores = row.get("scores", {}) or {}
        if metric_key in scores:
            return float(scores[metric_key])
    return d.get("best_eval_accuracy")


def load_runs(task_dir):
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(task_dir, "*", "seed*", "eval_results.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        method = d.get("method") or d.get("rank_allocation") or "unknown"
        runs[method].append(d)
    return runs


def _agg(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return mean(vals), pstdev(vals), len(vals)


def summarize(runs, metric_key):
    rows = []
    for method in sorted(runs.keys(), key=lambda m: METHOD_ORDER.get(m, 99)):
        ds = runs[method]
        m_mean, m_std, n = _agg([extract_metric(d, metric_key) for d in ds])
        wall_m, _, _ = _agg([d.get("total_train_wall_seconds") for d in ds])
        rs_m, _, _ = _agg([d.get("cum_rank_steps") for d in ds])
        budget = ds[0].get("effective_rank_budget") or ds[0].get("target_rank_budget")
        params = ds[0].get("effective_trainable_params_final") or ds[0].get("trainable_params_final")
        rows.append(
            {
                "method": method,
                "n_seeds": n,
                "metric_mean": m_mean,
                "metric_std": m_std,
                "rank_budget": budget,
                "trainable_params": params,
                "cum_rank_steps_mean": rs_m,
                "wall_seconds_mean": wall_m,
            }
        )
    return rows


def fmt_pct(m, s):
    if m is None:
        return "-"
    return f"{100 * m:.2f}±{100 * s:.2f}" if s is not None else f"{100 * m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()

    task_dir = os.path.join(args.outputs, args.task)
    runs = load_runs(task_dir)
    if not runs:
        print(f"[aggregate] no eval_results.json found under {task_dir}/*/seed*/")
        return
    metric_key, metric_name = task_metric(args.task)
    rows = summarize(runs, metric_key)

    metric_col = f"{metric_name}(%)"
    header = ["method", "n_seeds", metric_col, "rank_budget", "trainable_params", "rank_steps", "wall_s"]
    print(f"\n=== {args.task.upper()} summary (mean±std over seeds, primary metric = {metric_name}) ===")
    print(" | ".join(header))
    print("-" * 92)
    md = [
        f"# {args.task.upper()} 结果汇总（mean±std over seeds，主指标 = {metric_name}）\n",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for r in rows:
        line = [
            r["method"],
            str(r["n_seeds"]),
            fmt_pct(r["metric_mean"], r["metric_std"]),
            str(r["rank_budget"]),
            str(r["trainable_params"]),
            f"{r['cum_rank_steps_mean']:.3e}" if r["cum_rank_steps_mean"] else "-",
            f"{r['wall_seconds_mean']:.0f}" if r["wall_seconds_mean"] else "-",
        ]
        print(" | ".join(line))
        md.append("| " + " | ".join(line) + " |")

    md_path = os.path.join(task_dir, "summary.md")
    csv_path = os.path.join(task_dir, "summary.csv")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[aggregate] wrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
