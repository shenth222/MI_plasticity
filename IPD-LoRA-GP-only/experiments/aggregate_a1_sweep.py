"""Aggregate A.1 planted-rank sweep into the criterion-vs-gain table (C-iii figure).

For each planted_rank, averages over seeds:
  - uniform / gora / goodput / oracle final MSE
  - gain = uniform_mse / dynamic_mse (>1 => dynamic helps; log10 reported)
  - goodput signal predictability (persistence, planted recall)
Shows the gain grows as the scenario demands more concentration (planted_rank
exceeds the uniform per-module rank = budget/M).
"""
import glob
import json
import os
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob.glob(os.path.join(HERE, "results", "sweep", "pr*_s*.json")))

by_pr = defaultdict(list)
for f in FILES:
    d = json.load(open(f))
    s = d["summary"]
    pr = s["config"]["planted_rank"]
    by_pr[pr].append(s)

rows = []
for pr in sorted(by_pr):
    runs = by_pr[pr]
    budget = runs[0]["config"]["budget"]
    M = runs[0]["config"]["n_layers"]
    uni_per_mod = budget // M

    def avg(key, method):
        return mean(r["final_mse"][method] for r in runs)

    uni = avg("final_mse", "uniform")
    gp = avg("final_mse", "goodput")
    go = avg("final_mse", "gora")
    orc = avg("final_mse", "oracle")
    persist = mean(r["goodput_predictability"]["persistence"] for r in runs
                   if r.get("goodput_predictability", {}).get("persistence") is not None) if runs else None
    recall = mean(r["goodput_predictability"]["planted_topk_recall"] for r in runs
                  if r.get("goodput_predictability", {}).get("planted_topk_recall") is not None) if runs else None
    import math
    gain_gp = uni / gp if gp > 0 else float("inf")
    rows.append({
        "planted_rank": pr, "uni_per_mod": uni_per_mod, "n_seeds": len(runs),
        "uniform_mse": uni, "gora_mse": go, "goodput_mse": gp, "oracle_mse": orc,
        "gain_log10": math.log10(gain_gp) if gain_gp not in (0, float("inf")) else None,
        "persistence": persist, "planted_recall": recall,
    })

hdr = ["planted_rank", "uni_per_mod", "n_seeds", "uniform_mse", "gora_mse", "goodput_mse",
       "oracle_mse", "gain_log10", "persistence", "planted_recall"]
print("\n=== A.1 planted-rank sweep (mean over seeds) ===")
print(" | ".join(hdr))
print("-" * 110)
md = ["# A.1 Positive control: criterion vs gain\n",
      "> gain_log10 = log10(uniform_mse / goodput_mse). Dynamic allocation should",
      "> help (gain>0) once planted_rank > uniform per-module rank (budget/M).\n",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3e}" if (abs(v) < 1e-2 or abs(v) > 1e3) else f"{v:.3f}"
        return str(v)
    line = [fmt(r[h]) for h in hdr]
    print(" | ".join(line))
    md.append("| " + " | ".join(line) + " |")

out = os.path.join(HERE, "results", "a1_sweep_summary.md")
open(out, "w").write("\n".join(md) + "\n")
print(f"\n[A.1] wrote {out}")
