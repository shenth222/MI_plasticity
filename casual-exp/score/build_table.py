#!/usr/bin/env python3
"""
将 score/ 下四个指标（actual_update, pre_importance, training_gain, update_response）
的输出结果整理成一张表格。

行粒度（从粗到细）：
  - layer      : 整个 Transformer 层（对该层所有子模块求和）
  - attention  : 注意力块（对该层所有 attention 子模块求和）
  - ffn        : 前馈网络块（对该层所有 ffn 子模块求和）
  - submodule  : 最细粒度的叶模块（直接来自 module_scores）

列：每个指标的每种定义（含多子指标的会展开为多列）

输出：score_table.csv + score_table.xlsx
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
SCORE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_SUBPATH = "outputs/FFT/MNLI/seed42/lr1e-5"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def try_load(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  [WARN] 文件不存在: {path}")
        return {}
    return load_json(path)


def extract_flat_scores(data: dict) -> dict[str, float]:
    """从 data["module_scores"] 中提取 {模块名: 标量} 字典。"""
    raw = data.get("module_scores", {})
    result = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
    return result


def extract_subkey_scores(data: dict, subkey: str) -> dict[str, float]:
    """从 data["module_scores"][module][subkey] 中提取标量。"""
    raw = data.get("module_scores", {})
    result = {}
    for k, v in raw.items():
        if isinstance(v, dict) and subkey in v:
            val = v[subkey]
            if isinstance(val, (int, float)):
                result[k] = float(val)
    return result


# ──────────────────────────────────────────────
# 加载所有指标列
# ──────────────────────────────────────────────
def build_metric_columns() -> dict[str, dict[str, float]]:
    """
    返回 OrderedDict: 列名 -> {模块名: 标量值}
    列名格式: "指标 | 定义 | 子指标（可选）"
    """
    cols: dict[str, dict[str, float]] = {}

    # ── actual_update ────────────────────────
    au_dir = f"{SCORE_DIR}/actual_update/{EXP_SUBPATH}/actual_update"
    print("Loading actual_update ...")
    cols["U_m | def1_abs"] = extract_flat_scores(
        try_load(f"{au_dir}/def1_absolute.json"))
    cols["U_m | def2_rel"] = extract_flat_scores(
        try_load(f"{au_dir}/def2_relative.json"))
    cols["U_m | def3_path"] = extract_flat_scores(
        try_load(f"{au_dir}/def3_path_length.json"))

    # ── pre_importance ───────────────────────
    pi_dir = f"{SCORE_DIR}/pre_importance/{EXP_SUBPATH}/pre_importance"
    print("Loading pre_importance ...")
    cols["I_pre | fisher"] = extract_flat_scores(
        try_load(f"{pi_dir}/fisher.json"))

    saliency = try_load(f"{pi_dir}/saliency.json")
    cols["I_pre | saliency_grad_norm"] = (
        saliency.get("grad_norm", {}).get("module_scores", {}))
    cols["I_pre | saliency_taylor"] = (
        saliency.get("taylor", {}).get("module_scores", {}))
    # 确保值是 float（saliency 的 module_scores 已经是标量）
    for k in ("I_pre | saliency_grad_norm", "I_pre | saliency_taylor"):
        cols[k] = {m: float(v) for m, v in cols[k].items()
                   if isinstance(v, (int, float))}

    cols["I_pre | perturbation"] = extract_flat_scores(
        try_load(f"{pi_dir}/perturbation.json"))

    sv_data = try_load(f"{pi_dir}/singular_value.json")
    for sub in ["nuclear_norm", "top32_sum", "max_sv", "min_sv"]:
        cols[f"I_pre | sv_{sub}"] = extract_subkey_scores(sv_data, sub)

    se_data = try_load(f"{pi_dir}/spectral_entropy.json")
    for sub in ["spectral_entropy", "raw_entropy"]:
        cols[f"I_pre | se_{sub}"] = extract_subkey_scores(se_data, sub)

    # ── training_gain ────────────────────────
    tg_dir = f"{SCORE_DIR}/training_gain/{EXP_SUBPATH}/training_gain"
    print("Loading training_gain ...")
    cols["G_m | def1_rollback_loss"] = extract_flat_scores(
        try_load(f"{tg_dir}/def1_rollback_loss.json"))
    cols["G_m | def2_rollback_acc"] = extract_flat_scores(
        try_load(f"{tg_dir}/def2_rollback_acc.json"))
    cols["G_m | def3_path_integral"] = extract_flat_scores(
        try_load(f"{tg_dir}/def3_path_integral.json"))

    # ── update_response ──────────────────────
    ur_dir = f"{SCORE_DIR}/update_response/{EXP_SUBPATH}/update_response"
    print("Loading update_response ...")
    cols["R_hat | def1_probe_delta"] = extract_flat_scores(
        try_load(f"{ur_dir}/def1_probe_delta.json"))
    cols["R_hat | def2_grad_curvature"] = extract_flat_scores(
        try_load(f"{ur_dir}/def2_grad_curvature.json"))
    cols["R_hat | def3_early_grad_norm"] = extract_flat_scores(
        try_load(f"{ur_dir}/def3_early_grad_norm.json"))
    cols["R_hat | def4_ppred"] = extract_flat_scores(
        try_load(f"{ur_dir}/def4_ppred.json"))

    print(f"共加载 {len(cols)} 列指标。\n")
    return cols


# ──────────────────────────────────────────────
# 模块名解析
# ──────────────────────────────────────────────
def parse_module(name: str) -> tuple[str, int | None, str, str]:
    """
    返回 (level, layer_idx, block, submodule_path)
    level: embedding / attention / ffn / other
    """
    if re.match(r"deberta\.embeddings\.", name):
        sub = name[len("deberta.embeddings."):]
        return "embedding", None, "embedding", sub

    m = re.match(r"deberta\.encoder\.layer\.(\d+)\.(.*)", name)
    if m:
        layer = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("attention."):
            return "attention", layer, "attention", rest
        else:
            return "ffn", layer, "ffn", rest

    # classifier / pooler 等
    return "other", None, "other", name


# ──────────────────────────────────────────────
# 主逻辑：构造 DataFrame
# ──────────────────────────────────────────────
def build_dataframe(metric_cols: dict[str, dict[str, float]]) -> pd.DataFrame:
    # 收集所有叶模块名
    all_modules: set[str] = set()
    for scores in metric_cols.values():
        all_modules.update(scores.keys())

    # 按解析结果归类（用于聚合）
    layer_members: dict[str, list[str]] = defaultdict(list)
    attn_members:  dict[str, list[str]] = defaultdict(list)
    ffn_members:   dict[str, list[str]] = defaultdict(list)

    for m in all_modules:
        level, layer, block, _ = parse_module(m)
        if layer is not None:
            lkey = f"layer_{layer:02d}"
            layer_members[lkey].append(m)
            if block == "attention":
                attn_members[f"layer_{layer:02d}.attention"].append(m)
            else:
                ffn_members[f"layer_{layer:02d}.ffn"].append(m)

    def sum_group(members: list[str], col_scores: dict[str, float]) -> float | None:
        vals = [col_scores[m] for m in members if m in col_scores]
        return sum(vals) if vals else None

    rows: list[dict] = []

    # ── 1. 每个 Transformer 层的聚合行 ──
    for lkey in sorted(layer_members):
        members = layer_members[lkey]
        layer_idx = int(lkey.split("_")[1])
        row = {
            "module": lkey,
            "level": "layer",
            "layer": layer_idx,
            "block": "layer",
            "submodule": "",
        }
        for col, scores in metric_cols.items():
            row[col] = sum_group(members, scores)
        rows.append(row)

    # ── 2. 每个 Attention 块的聚合行 ──
    for akey in sorted(attn_members):
        members = attn_members[akey]
        layer_idx = int(akey.split(".")[0].split("_")[1])
        row = {
            "module": akey,
            "level": "attention",
            "layer": layer_idx,
            "block": "attention",
            "submodule": "",
        }
        for col, scores in metric_cols.items():
            row[col] = sum_group(members, scores)
        rows.append(row)

    # ── 3. 每个 FFN 块的聚合行 ──
    for fkey in sorted(ffn_members):
        members = ffn_members[fkey]
        layer_idx = int(fkey.split(".")[0].split("_")[1])
        row = {
            "module": fkey,
            "level": "ffn",
            "layer": layer_idx,
            "block": "ffn",
            "submodule": "",
        }
        for col, scores in metric_cols.items():
            row[col] = sum_group(members, scores)
        rows.append(row)

    # ── 4. 最细粒度子模块行 ──
    for m in sorted(all_modules):
        _, layer, block, sub = parse_module(m)
        row = {
            "module": m,
            "level": "submodule",           # 统一标记为叶节点
            "layer": layer if layer is not None else -1,
            "block": block,
            "submodule": sub,
        }
        for col, scores in metric_cols.items():
            row[col] = scores.get(m)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── 排序策略 ──
    # 每行赋予一个 5 元组排序键：
    #   (layer_sort, group_order, subgroup_order, submodule)
    # 对于每一层 N:
    #   layer_00 聚合行:        (N, 0, 0, "")
    #   layer_00.attention 聚合: (N, 1, 0, "")
    #   attention 子模块:        (N, 1, 1, submodule)
    #   layer_00.ffn 聚合:       (N, 2, 0, "")
    #   ffn 子模块:              (N, 2, 1, submodule)
    # embedding / other 子模块: (-1, block_order, 1, submodule)

    def row_sort_key(row):
        layer = row["layer"]      # int or -1
        level = row["level"]      # layer/attention/ffn/submodule
        block = row["block"]      # layer/attention/ffn/embedding/other
        sub   = row["submodule"] if pd.notna(row["submodule"]) else ""

        if layer == -1:
            # embedding / other 模块，最前面
            block_o = {"embedding": 0, "other": 1}.get(block, 2)
            return (-1, block_o, 1, sub)

        if level == "layer":
            return (layer, 0, 0, "")
        if level == "attention":
            return (layer, 1, 0, "")
        if level == "ffn":
            return (layer, 2, 0, "")
        # submodule
        if block == "attention":
            return (layer, 1, 1, sub)
        if block == "ffn":
            return (layer, 2, 1, sub)
        return (layer, 9, 1, sub)

    df["_sort_key"] = df.apply(row_sort_key, axis=1)
    df = (df
          .sort_values("_sort_key")
          .drop(columns=["_sort_key"])
          .reset_index(drop=True))

    return df


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
def main():
    metric_cols = build_metric_columns()
    df = build_dataframe(metric_cols)

    # 保存 CSV
    out_csv = os.path.join(SCORE_DIR, "score_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"✓ CSV 已保存: {out_csv}")

    # 保存 Excel（带冻结首行首列）
    out_xlsx = os.path.join(SCORE_DIR, "score_table.xlsx")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="scores")
            ws = writer.sheets["scores"]
            ws.freeze_panes = "F2"   # 冻结前5列 + 首行
        print(f"✓ Excel 已保存: {out_xlsx}")
    except Exception as e:
        print(f"  [WARN] Excel 保存失败: {e}")

    print(f"\n表格大小: {df.shape[0]} 行 × {df.shape[1]} 列")
    print("\n列名预览:")
    for c in df.columns:
        print(f"  {c}")
    print("\n前 10 行 (仅元数据列):")
    print(df[["module", "level", "layer", "block", "submodule"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
