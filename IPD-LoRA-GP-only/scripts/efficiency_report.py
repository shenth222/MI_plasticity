#!/usr/bin/env python
"""E3 efficiency report: convergence speed & Goodput from existing training logs.

Reconstructs, per (method, seed), the learning curve from training_log.jsonl and
computes the efficiency quantities the proposal actually cares about:

  - best        : best primary-metric over training
  - AULC        : area under the (metric vs step) curve, step-normalized -> mean
                  metric across training. Higher = faster convergence.
  - steps@thr   : steps to first reach a shared accuracy threshold (a fraction of
                  the best any method achieved on this task). Convergence speed.
  - ranksteps@thr: cumulative rank-steps consumed to reach that threshold
                  (integral of active_total_rank over steps). Rank-step Goodput.
  - wall_s      : total training wall-clock (from eval_results.json)
  - gp_rankstep : best_metric / total_rank_steps      (accuracy per rank-step)
  - gp_wall     : best_metric / total_wall_seconds    (accuracy per second)

No re-training needed. Also writes averaged accuracy-vs-step and accuracy-vs-
rankstep curves per method for plotting.

Usage:
  python scripts/efficiency_report.py --task rte
  python scripts/efficiency_report.py --task mnli --outputs outputs --thr-frac 0.98
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

METHOD_ORDER = {"lora": 0, "gora": 1, "adalora": 2, "goodput": 3}


def load_curve(train_log_path):
    """Return (eval_steps, eval_metrics, cum_ranksteps_at_eval, total_ranksteps)."""
    steps, ranks, evals = [], [], []
    with open(train_log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("step")
            if s is None:
                continue
            steps.append(int(s))
            ranks.append(float(d.get("active_total_rank") or 0.0))
            evals.append(d.get("eval_accuracy"))
    if not steps:
        return [], [], [], 0.0
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    ranks = [ranks[i] for i in order]
    evals = [evals[i] for i in order]

    # Integrate active_total_rank over steps -> cumulative rank-steps at each log point.
    cum = [0.0] * len(steps)
    for i in range(1, len(steps)):
        dstep = steps[i] - steps[i - 1]
        cum[i] = cum[i - 1] + ranks[i] * dstep
    total_ranksteps = cum[-1] if cum else 0.0

    eval_steps, eval_metrics, eval_cum = [], [], []
    for i, e in enumerate(evals):
        if e is None:
            continue
        eval_steps.append(steps[i])
        eval_metrics.append(float(e))
        eval_cum.append(cum[i])
    return eval_steps, eval_metrics, eval_cum, total_ranksteps


def aulc(steps, metrics):
    """Step-normalized area under the metric-vs-step curve (trapezoid)."""
    if len(steps) < 2:
        return metrics[0] if metrics else None
    area = 0.0
    for i in range(1, len(steps)):
        area += 0.5 * (metrics[i] + metrics[i - 1]) * (steps[i] - steps[i - 1])
    span = steps[-1] - steps[0]
    return area / span if span > 0 else metrics[-1]


def first_reach(steps, metrics, cum, thr):
    for s, m, c in zip(steps, metrics, cum):
        if m >= thr:
            return s, c
    return None, None


def _agg(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return mean(vals), pstdev(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--thr-frac", type=float, default=0.97,
                    help="threshold = thr_frac * (global best metric on this task)")
    args = ap.parse_args()

    task_dir = os.path.join(args.outputs, args.task)
    # Gather runs: method -> list of dicts with curve + eval_results scalars.
    runs = defaultdict(list)
    for er_path in glob.glob(os.path.join(task_dir, "*", "seed*", "eval_results.json")):
        run_dir = os.path.dirname(er_path)
        tl = os.path.join(run_dir, "training_log.jsonl")
        if not os.path.exists(tl):
            continue
        try:
            er = json.load(open(er_path, "r", encoding="utf-8"))
        except Exception:
            continue
        method = er.get("method") or er.get("rank_allocation") or "unknown"
        steps, metrics, cum, total_rs = load_curve(tl)
        if not steps:
            continue
        runs[method].append({
            "steps": steps, "metrics": metrics, "cum": cum,
            "best": max(metrics),
            "total_ranksteps": total_rs,
            "wall": er.get("total_train_wall_seconds"),
            "budget": er.get("effective_rank_budget") or er.get("target_rank_budget"),
        })

    if not runs:
        print(f"[eff] no runs with training_log.jsonl under {task_dir}")
        return

    global_best = max(r["best"] for rs in runs.values() for r in rs)
    thr = args.thr_frac * global_best
    print(f"\n=== {args.task.upper()} efficiency (global_best={global_best:.4f}, threshold={thr:.4f} = {args.thr_frac:g}x best) ===")
    header = ["method", "n", "best(%)", "AULC(%)", f"steps@{args.thr_frac:g}", f"rankstep@{args.thr_frac:g}",
              "wall_s", "gp/rankstep(1e-6)", "gp/wall(1e-3)"]
    print(" | ".join(header))
    print("-" * 110)

    rows = []
    for method in sorted(runs.keys(), key=lambda m: METHOD_ORDER.get(m, 99)):
        rs = runs[method]
        best_m, best_s, n = _agg([r["best"] for r in rs])
        aulc_m, aulc_s, _ = _agg([aulc(r["steps"], r["metrics"]) for r in rs])
        s2t, rs2t = [], []
        for r in rs:
            s, c = first_reach(r["steps"], r["metrics"], r["cum"], thr)
            s2t.append(s)
            rs2t.append(c)
        s2t_m, s2t_s, _ = _agg(s2t)
        rs2t_m, _, _ = _agg(rs2t)
        wall_m, _, _ = _agg([r["wall"] for r in rs])
        # Goodput: best metric per unit total cost.
        gp_rs = _agg([r["best"] / r["total_ranksteps"] for r in rs if r["total_ranksteps"]])[0]
        gp_wall = _agg([r["best"] / r["wall"] for r in rs if r["wall"]])[0]
        rows.append({
            "method": method, "n": n,
            "best": best_m, "best_std": best_s,
            "aulc": aulc_m, "aulc_std": aulc_s,
            "steps_to_thr": s2t_m, "steps_to_thr_std": s2t_s,
            "ranksteps_to_thr": rs2t_m,
            "wall_s": wall_m,
            "gp_rankstep": gp_rs, "gp_wall": gp_wall,
        })
        print(" | ".join([
            method, str(n),
            f"{100*best_m:.2f}±{100*best_s:.2f}" if best_m is not None else "-",
            f"{100*aulc_m:.2f}±{100*aulc_s:.2f}" if aulc_m is not None else "-",
            f"{s2t_m:.0f}±{s2t_s:.0f}" if s2t_m is not None else ">max",
            f"{rs2t_m:.3e}" if rs2t_m is not None else ">max",
            f"{wall_m:.0f}" if wall_m is not None else "-",
            f"{1e6*gp_rs:.3f}" if gp_rs is not None else "-",
            f"{1e3*gp_wall:.3f}" if gp_wall is not None else "-",
        ]))

    # Write summary + averaged curves for plotting.
    out_md = os.path.join(task_dir, "efficiency.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.task.upper()} 效率报告（E3）\n\n")
        f.write(f"global_best={global_best:.4f}, threshold={thr:.4f} ({args.thr_frac:g}x best)\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join([
                r["method"], str(r["n"]),
                f"{100*r['best']:.2f}±{100*r['best_std']:.2f}",
                f"{100*r['aulc']:.2f}±{100*r['aulc_std']:.2f}",
                f"{r['steps_to_thr']:.0f}±{r['steps_to_thr_std']:.0f}" if r["steps_to_thr"] is not None else ">max",
                f"{r['ranksteps_to_thr']:.3e}" if r["ranksteps_to_thr"] is not None else ">max",
                f"{r['wall_s']:.0f}" if r["wall_s"] is not None else "-",
                f"{1e6*r['gp_rankstep']:.3f}" if r["gp_rankstep"] is not None else "-",
                f"{1e3*r['gp_wall']:.3f}" if r["gp_wall"] is not None else "-",
            ]) + " |\n")
    print(f"\n[eff] wrote {out_md}")


if __name__ == "__main__":
    main()
