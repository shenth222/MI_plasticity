# src/analysis/aggregate.py
import os, json, argparse
import numpy as np
from typing import Dict, List, Tuple

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between x and y.
    Spearman = Pearson correlation of ranks.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) == 0:
        return 0.0
    
    # Rank x and y (using scipy-style ranking: average rank for ties)
    def rankdata(arr):
        sorter = np.argsort(arr)
        ranks = np.empty_like(sorter, dtype=float)
        ranks[sorter] = np.arange(1, len(arr) + 1)

        # Handle ties by assigning average rank per unique value
        uniq, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            sum_ranks = np.zeros_like(uniq, dtype=float)
            for i, r in zip(inv, ranks):
                sum_ranks[i] += r
            avg_ranks = sum_ranks / counts
            ranks = avg_ranks[inv]
        return ranks
    
    rx = rankdata(x)
    ry = rankdata(y)
    
    # Pearson correlation of ranks
    return float(np.corrcoef(rx, ry)[0, 1])


def top_k_overlap(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    Compute top-k overlap between x and y.
    Returns the fraction of shared indices in top-k of x and top-k of y.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if k > len(x):
        k = len(x)
    topk_x = set(np.argsort(x)[-k:])
    topk_y = set(np.argsort(y)[-k:])
    return len(topk_x & topk_y) / k


def percentile_threshold(arr: np.ndarray, percentile: float) -> float:
    """Return the value at given percentile (0-100)."""
    return float(np.percentile(arr, percentile))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g. outputs/MNLI/seed1")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    
    # Load all jsonl files
    importance_pre_path = os.path.join(args.exp_dir, "importance_pre.jsonl")
    importance_post_path = os.path.join(args.exp_dir, "importance_post.jsonl")
    gradfisher_pre_path = os.path.join(args.exp_dir, "gradfisher_pre.jsonl")
    update_path = os.path.join(args.exp_dir, "update.jsonl")
    
    if not os.path.exists(importance_pre_path):
        raise FileNotFoundError(f"Missing {importance_pre_path}")
    if not os.path.exists(update_path):
        raise FileNotFoundError(f"Missing {update_path}")
    
    importance_pre = load_jsonl(importance_pre_path)
    importance_post = load_jsonl(importance_post_path) if os.path.exists(importance_post_path) else []
    gradfisher_pre = load_jsonl(gradfisher_pre_path) if os.path.exists(gradfisher_pre_path) else []
    update = load_jsonl(update_path)
    
    # Build a dict keyed by (layer, head)
    data = {}
    for rec in importance_pre:
        key = (rec["layer"], rec["head"])
        data[key] = {"layer": rec["layer"], "head": rec["head"], "I_pre": rec["I"]}
    
    for rec in importance_post:
        key = (rec["layer"], rec["head"])
        if key in data:
            data[key]["I_post"] = rec["I"]
    
    for rec in gradfisher_pre:
        key = (rec["layer"], rec["head"])
        if key in data:
            data[key]["G"] = rec["G"]
            data[key]["F"] = rec["F"]
            data[key]["Ppred"] = rec["Ppred"]
    
    for rec in update:
        key = (rec["layer"], rec["head"])
        if key in data:
            data[key]["U"] = rec["U"]
            data[key]["Urel"] = rec["Urel"]
            data[key]["Uq"] = rec["Uq"]
            data[key]["Uk"] = rec["Uk"]
            data[key]["Uv"] = rec["Uv"]
            data[key]["Uo"] = rec["Uo"]
    
    # Convert to list and fill missing fields with NaN
    heads = []
    for key in sorted(data.keys()):
        rec = data[key]
        rec.setdefault("I_post", float("nan"))
        rec.setdefault("G", float("nan"))
        rec.setdefault("F", float("nan"))
        rec.setdefault("Ppred", float("nan"))
        rec.setdefault("U", float("nan"))
        rec.setdefault("Urel", float("nan"))
        rec.setdefault("Uq", float("nan"))
        rec.setdefault("Uk", float("nan"))
        rec.setdefault("Uv", float("nan"))
        rec.setdefault("Uo", float("nan"))
        heads.append(rec)
    
    # Save heads.csv
    csv_path = os.path.join(args.exp_dir, "heads.csv")
    with open(csv_path, "w") as f:
        f.write("layer,head,I_pre,I_post,U,Urel,G,F,Ppred,Uq,Uk,Uv,Uo\n")
        for rec in heads:
            f.write(f"{rec['layer']},{rec['head']},{rec['I_pre']},{rec['I_post']},{rec['U']},{rec['Urel']},{rec['G']},{rec['F']},{rec['Ppred']},{rec['Uq']},{rec['Uk']},{rec['Uv']},{rec['Uo']}\n")
    
    print(f"Saved {csv_path}")
    
    # Compute stats
    I_pre = np.array([rec["I_pre"] for rec in heads])
    I_post = np.array([rec["I_post"] for rec in heads])
    U = np.array([rec["U"] for rec in heads])
    Urel = np.array([rec["Urel"] for rec in heads])
    G = np.array([rec["G"] for rec in heads])
    F = np.array([rec["F"] for rec in heads])
    Ppred = np.array([rec["Ppred"] for rec in heads])
    
    # Spearman correlation I_pre vs U
    rho_Ipre_U = rank_correlation(I_pre, U)
    rho_Ipre_Urel = rank_correlation(I_pre, Urel)

    # Spearman correlation I_post vs (U, Urel, G, F, Ppred) if available
    valid_Ipost_mask = ~np.isnan(I_post)
    if np.any(valid_Ipost_mask):
        Ipost_v = I_post[valid_Ipost_mask]
        rho_Ipost_U = rank_correlation(Ipost_v, U[valid_Ipost_mask])
        rho_Ipost_Urel = rank_correlation(Ipost_v, Urel[valid_Ipost_mask])

        valid_G = valid_Ipost_mask & ~np.isnan(G)
        valid_F = valid_Ipost_mask & ~np.isnan(F)
        valid_Ppred = valid_Ipost_mask & ~np.isnan(Ppred)

        rho_Ipost_G = rank_correlation(I_post[valid_G], G[valid_G]) if np.any(valid_G) else float("nan")
        rho_Ipost_F = rank_correlation(I_post[valid_F], F[valid_F]) if np.any(valid_F) else float("nan")
        rho_Ipost_Ppred = rank_correlation(I_post[valid_Ppred], Ppred[valid_Ppred]) if np.any(valid_Ppred) else float("nan")
        n_heads_Ipost = int(valid_Ipost_mask.sum())
    else:
        rho_Ipost_U = float("nan")
        rho_Ipost_Urel = float("nan")
        rho_Ipost_G = float("nan")
        rho_Ipost_F = float("nan")
        rho_Ipost_Ppred = float("nan")
        n_heads_Ipost = 0
    
    # Top-k overlap I_pre vs U
    overlap_Ipre_U = top_k_overlap(I_pre, U, k=args.topk)
    overlap_Ipre_Urel = top_k_overlap(I_pre, Urel, k=args.topk)

    # Top-k overlap I_pre vs I_post (if available)
    if np.any(valid_Ipost_mask):
        overlap_Ipre_Ipost = top_k_overlap(I_pre[valid_Ipost_mask], I_post[valid_Ipost_mask], k=args.topk)
    else:
        overlap_Ipre_Ipost = float("nan")
    
    stats = {
        "spearman_rho_Ipre_U": rho_Ipre_U,
        "spearman_rho_Ipre_Urel": rho_Ipre_Urel,
        "topk_overlap_Ipre_U": overlap_Ipre_U,
        "topk_overlap_Ipre_Urel": overlap_Ipre_Urel,
        "topk": args.topk,
        "n_heads": len(heads),
        "n_heads_Ipost": n_heads_Ipost,
        "topk_overlap_Ipre_Ipost": overlap_Ipre_Ipost,
        "spearman_rho_Ipost_U": rho_Ipost_U,
        "spearman_rho_Ipost_Urel": rho_Ipost_Urel,
        "spearman_rho_Ipost_G": rho_Ipost_G,
        "spearman_rho_Ipost_F": rho_Ipost_F,
        "spearman_rho_Ipost_Ppred": rho_Ipost_Ppred,
    }
    
    # Compute case sets
    # A) important-but-static: I_pre top 10% AND Urel bottom 30%
    # B) plastic-but-unimportant: Urel top 10% AND I_pre bottom 30%
    
    p90_I = percentile_threshold(I_pre, 90)
    p10_I = percentile_threshold(I_pre, 10)
    p30_I = percentile_threshold(I_pre, 30)
    
    p90_Urel = percentile_threshold(Urel, 90)
    p10_Urel = percentile_threshold(Urel, 10)
    p30_Urel = percentile_threshold(Urel, 30)
    
    important_but_static = []
    plastic_but_unimportant = []
    
    for i, rec in enumerate(heads):
        I = rec["I_pre"]
        U_rel = rec["Urel"]
        
        # A) I_pre >= p90 AND Urel <= p30
        if I >= p90_I and U_rel <= p30_Urel:
            important_but_static.append({
                "layer": rec["layer"],
                "head": rec["head"],
                "I_pre": I,
                "Urel": U_rel,
                "U": rec["U"],
                "G": rec["G"],
                "Ppred": rec["Ppred"],
            })
        
        # B) Urel >= p90 AND I_pre <= p30
        if U_rel >= p90_Urel and I <= p30_I:
            plastic_but_unimportant.append({
                "layer": rec["layer"],
                "head": rec["head"],
                "I_pre": I,
                "Urel": U_rel,
                "U": rec["U"],
                "G": rec["G"],
                "Ppred": rec["Ppred"],
            })
    
    stats["n_important_but_static"] = len(important_but_static)
    stats["n_plastic_but_unimportant"] = len(plastic_but_unimportant)
    
    # Save stats.json
    stats_path = os.path.join(args.exp_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {stats_path}")
    
    # Save cases.json
    cases_path = os.path.join(args.exp_dir, "cases.json")
    cases = {
        "important_but_static": important_but_static,
        "plastic_but_unimportant": plastic_but_unimportant,
        "thresholds": {
            "I_pre_p90": p90_I,
            "I_pre_p30": p30_I,
            "Urel_p90": p90_Urel,
            "Urel_p30": p30_Urel,
        }
    }
    with open(cases_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved {cases_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total heads: {len(heads)}")
    print(f"Spearman ρ(I_pre, U): {rho_Ipre_U:.3f}")
    print(f"Spearman ρ(I_pre, Urel): {rho_Ipre_Urel:.3f}")
    print(f"Top-{args.topk} overlap (I_pre, U): {overlap_Ipre_U:.3f}")
    print(f"Top-{args.topk} overlap (I_pre, Urel): {overlap_Ipre_Urel:.3f}")
    if not np.isnan(overlap_Ipre_Ipost):
        print(f"Top-{args.topk} overlap (I_pre, I_post): {overlap_Ipre_Ipost:.3f}")
    if not np.isnan(rho_Ipost_U):
        print(f"Spearman ρ(I_post, U): {rho_Ipost_U:.3f}")
        print(f"Spearman ρ(I_post, Urel): {rho_Ipost_Urel:.3f}")
        print(f"Spearman ρ(I_post, G): {rho_Ipost_G:.3f}")
        print(f"Spearman ρ(I_post, F): {rho_Ipost_F:.3f}")
        print(f"Spearman ρ(I_post, Ppred): {rho_Ipost_Ppred:.3f}")
    print(f"Important-but-static cases: {len(important_but_static)}")
    print(f"Plastic-but-unimportant cases: {len(plastic_but_unimportant)}")


if __name__ == "__main__":
    main()
