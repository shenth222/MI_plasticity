#!/usr/bin/env python
"""E4: is the goodput signal actually informative?

The null result (online goodput allocation ties static uniform LoRA) needs a
*mechanistic* explanation. This script tests, from existing goodput runs'
module_scores.jsonl (no re-training), whether the per-module Module Learning
Goodput signal has the properties a useful online allocation signal must have:

  1. PERSISTENCE (temporal stability): does a module's realized goodput at
     window t predict its realized goodput at window t+1?
         rho_persist(t) = Spearman( current_G(t) , current_G(t+1) )   over modules
     If ~0, the signal is white noise across windows -> nothing to track ->
     online reallocation cannot help, static == online.

  2. PREDICTIVE VALIDITY of what we allocate on: the smoothed ema_G(t) is what
     the allocator uses at window t. Does it predict the *next* realized gain?
         rho_pred(t) = Spearman( ema_G(t) , current_G(t+1) )          over modules
     If ~0, we are allocating rank on a signal uncorrelated with future gain.

  3. LAG DECAY: rho_pred / rho_persist as a function of lag k (t vs t+k). How
     fast does the signal's predictive content decay?

  4. CROSS-SECTIONAL CONCENTRATION: at each window, is there real spread across
     modules to exploit? Reported as normalized entropy of the (nonneg) current_G
     distribution (1.0 = perfectly uniform, no module is preferable; lower =
     concentrated, some modules truly deserve more rank) and as coefficient of
     variation.

Aggregates mean+/-std across scoring windows and seeds, per task. Writes
e4_signal_validity.md under the task dir (or a combined report with --all).

Usage:
  python scripts/e4_signal_validity.py --task rte
  python scripts/e4_signal_validity.py --task mnli --outputs outputs
  python scripts/e4_signal_validity.py --all
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev


def _spearman(x, y):
    """Spearman rho on paired lists (ties -> average ranks). None if degenerate."""
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _norm_entropy(vals):
    """Normalized Shannon entropy of a nonneg vector (1=uniform, 0=one-hot)."""
    v = [max(0.0, x) for x in vals]
    s = sum(v)
    n = len(v)
    if n < 2 or s <= 0:
        return None
    p = [x / s for x in v]
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    return h / math.log(n)


def _cv(vals):
    """Coefficient of variation (std/|mean|) of current_G across modules."""
    if len(vals) < 2:
        return None
    m = mean(vals)
    if m == 0:
        return None
    return pstdev(vals) / abs(m)


def load_module_series(path):
    """module_scores.jsonl -> {step: {module: {current_G, ema_G}}} sorted by step."""
    by_step = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            step = d.get("step")
            name = d.get("module_name")
            if step is None or name is None:
                continue
            by_step[int(step)][name] = {
                "current_G": float(d.get("current_G") or 0.0),
                "ema_G": float(d.get("ema_G") or 0.0),
            }
    return dict(sorted(by_step.items()))


def analyze_run(path, max_lag=3):
    series = load_module_series(path)
    steps = list(series.keys())
    if len(steps) < 3:
        return None

    modules = sorted(set().union(*[set(series[s].keys()) for s in steps]))

    def vec(step, key):
        return [series[step].get(m, {}).get(key, 0.0) for m in modules]

    persist = defaultdict(list)   # lag -> [rho...]
    predict = defaultdict(list)   # lag -> [rho...]
    entropies, cvs = [], []

    for i, s in enumerate(steps):
        cg = vec(s, "current_G")
        entropies.append(_norm_entropy(cg))
        cvs.append(_cv(cg))
        for lag in range(1, max_lag + 1):
            if i + lag < len(steps):
                s2 = steps[i + lag]
                cg2 = vec(s2, "current_G")
                r_p = _spearman(cg, cg2)
                if r_p is not None:
                    persist[lag].append(r_p)
                r_pred = _spearman(vec(s, "ema_G"), cg2)
                if r_pred is not None:
                    predict[lag].append(r_pred)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return mean(xs) if xs else None

    return {
        "n_windows": len(steps),
        "n_modules": len(modules),
        "persist": {lag: avg(v) for lag, v in persist.items()},
        "predict": {lag: avg(v) for lag, v in predict.items()},
        "entropy": avg(entropies),
        "cv": avg(cvs),
    }


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return mean(vals), pstdev(vals), len(vals)


def aggregate_task(outputs, task, max_lag=3):
    runs = []
    for p in sorted(glob.glob(os.path.join(outputs, task, "goodput", "seed*", "module_scores.jsonl"))):
        r = analyze_run(p, max_lag=max_lag)
        if r is not None:
            runs.append(r)
    if not runs:
        return None
    out = {
        "task": task, "n_runs": len(runs),
        "n_windows": _agg([r["n_windows"] for r in runs])[0],
        "n_modules": int(_agg([r["n_modules"] for r in runs])[0]),
        "entropy": _agg([r["entropy"] for r in runs]),
        "cv": _agg([r["cv"] for r in runs]),
        "persist": {}, "predict": {},
    }
    for lag in range(1, max_lag + 1):
        out["persist"][lag] = _agg([r["persist"].get(lag) for r in runs])
        out["predict"][lag] = _agg([r["predict"].get(lag) for r in runs])
    return out


def fmt(agg):
    if agg is None or agg[0] is None:
        return "-"
    m, s, _ = agg
    return f"{m:+.3f}±{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--all", action="store_true", help="scan every task under outputs/")
    ap.add_argument("--max-lag", type=int, default=3)
    args = ap.parse_args()

    if args.all:
        tasks = sorted(
            d for d in os.listdir(args.outputs)
            if os.path.isdir(os.path.join(args.outputs, d, "goodput"))
        )
    elif args.task:
        tasks = [args.task]
    else:
        ap.error("provide --task or --all")

    results = [aggregate_task(args.outputs, t, args.max_lag) for t in tasks]
    results = [r for r in results if r is not None]
    if not results:
        print(f"[e4] no goodput module_scores.jsonl found under {args.outputs}")
        return

    lags = list(range(1, args.max_lag + 1))
    header = (["task", "runs", "modules", "entropy(1=unif)", "cv"]
              + [f"persist@lag{l}" for l in lags]
              + [f"predict@lag{l}" for l in lags])

    def row(r):
        return ([r["task"], str(r["n_runs"]), str(r["n_modules"]),
                 fmt(r["entropy"]), fmt(r["cv"])]
                + [fmt(r["persist"][l]) for l in lags]
                + [fmt(r["predict"][l]) for l in lags])

    print("\n=== E4 signal validity (Module Learning Goodput) ===")
    print(" | ".join(header))
    print("-" * 140)
    for r in results:
        print(" | ".join(row(r)))

    # Written report (per-task file, or combined at outputs root when --all).
    if args.all or len(results) > 1:
        out_md = os.path.join(args.outputs, "e4_signal_validity.md")
    else:
        out_md = os.path.join(args.outputs, results[0]["task"], "e4_signal_validity.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# E4 信号有效性：Module Learning Goodput 是否可用作在线分配信号\n\n")
        f.write("- **persist@lag_k** = Spearman(current_G(t), current_G(t+k))：模块 goodput 的时间持续性（≈0 → 白噪声，无可追踪结构）。\n")
        f.write("- **predict@lag_k** = Spearman(ema_G(t), current_G(t+k))：分配所用的平滑信号对未来真实收益的预测力。\n")
        f.write("- **entropy** = current_G 分布的归一化熵（1=各模块均等，无可利用差异；越低越集中）。\n")
        f.write("- **cv** = current_G 的变异系数（截面离散度）。\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in results:
            f.write("| " + " | ".join(row(r)) + " |\n")
        f.write("\n## 判读\n")
        f.write("- persist / predict 接近 0：在线信号无时间可预测性，动态重分配等价于随机重排 → 解释\"online == static\"的 null result。\n")
        f.write("- entropy 接近 1：各模块 goodput 差异极小，分配没有可利用的杠杆。\n")
        f.write("- 若 predict 在 lag=1 尚可但迅速衰减：信号有短时效性，支持\"仅早期在线、随后冻结\"的设计。\n")
    print(f"\n[e4] wrote {out_md}")


if __name__ == "__main__":
    main()
