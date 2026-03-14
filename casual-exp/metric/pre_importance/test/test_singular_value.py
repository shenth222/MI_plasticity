"""
metric/pre_importance/test/test_singular_value.py

定义 4 奇异值重要性 — 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.pre_importance.test.test_singular_value

覆盖点：
  1. 返回结构完整（module_scores / top_k）
  2. 每个模块分数包含 nuclear_norm / top{k}_sum / max_sv / min_sv 等字段
  3. nuclear_norm ≥ top{k}_sum（核范数 ≥ 截断和）
  4. max_sv ≥ min_sv ≥ 0（奇异值均非负）
  5. 不需要 dataloader（needs_data=False）
  6. save() / load() 往返一致
  7. top_k 参数生效（top8_sum ≤ top32_sum）
"""

import json
import tempfile

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.pre_importance.singular_value import SingularValueImportance
from metric.pre_importance.test.conftest import TinyClassifier


def test_sv_structure():
    """返回字典包含 module_scores 和 top_k"""
    model  = TinyClassifier()
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=8)

    assert "module_scores" in result, "缺少 module_scores"
    assert "top_k"         in result, "缺少 top_k"
    assert len(result["module_scores"]) > 0, "module_scores 为空"
    print(f"✓ 返回结构正确，共 {len(result['module_scores'])} 个模块")


def test_sv_per_module_fields():
    """每个模块分数字典包含所有必要字段"""
    model  = TinyClassifier()
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=8)

    for name, scores in result["module_scores"].items():
        for field in ("nuclear_norm", "top8_sum", "max_sv", "min_sv",
                      "num_singular_values", "matrix_shape"):
            assert field in scores, f"模块 {name} 缺少字段 {field}"
    print("✓ 所有模块分数字段完整")


def test_sv_nuclear_ge_topk():
    """核范数 ≥ 前 k 截断和（截断只能更小或相等）"""
    model  = TinyClassifier()
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=8)

    for name, scores in result["module_scores"].items():
        nn   = scores["nuclear_norm"]
        topk = scores["top8_sum"]
        assert nn >= topk - 1e-6, \
            f"{name}: nuclear_norm={nn:.4f} < top8_sum={topk:.4f}"
    print("✓ 所有模块 nuclear_norm ≥ top8_sum")


def test_sv_non_negative():
    """所有奇异值均非负"""
    model  = TinyClassifier()
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=8)

    for name, scores in result["module_scores"].items():
        assert scores["max_sv"] >= 0, f"{name} max_sv 为负"
        assert scores["min_sv"] >= 0, f"{name} min_sv 为负"
        assert scores["max_sv"] >= scores["min_sv"] - 1e-6, \
            f"{name} max_sv < min_sv"
    print("✓ 所有奇异值非负且 max_sv ≥ min_sv")


def test_sv_no_dataloader_needed():
    """SingularValueImportance.needs_data 应为 False，不需要 dataloader"""
    metric = SingularValueImportance()
    assert metric.needs_data is False, "needs_data 应为 False"

    model  = TinyClassifier()
    # 不传入 dataloader（传 None）应正常运行
    result = metric.compute(model, dataloader=None, device=None)
    assert "module_scores" in result
    print("✓ needs_data=False，无 dataloader 也能运行")


def test_sv_topk_param():
    """不同 top_k 参数结果应不同（top8_sum ≤ top32_sum）"""
    model   = TinyClassifier()
    metric  = SingularValueImportance()
    res8    = metric.compute(model, top_k=8)
    res32   = metric.compute(model, top_k=32)

    common = set(res8["module_scores"]) & set(res32["module_scores"])
    for name in list(common)[:3]:
        s8  = res8["module_scores"][name]["top8_sum"]
        s32 = res32["module_scores"][name]["top32_sum"]
        assert s32 >= s8 - 1e-6, \
            f"{name}: top32_sum={s32:.4f} < top8_sum={s8:.4f}"
    print("✓ top_k 参数生效：top32_sum ≥ top8_sum")


def test_sv_save_load():
    """save() / load() 往返一致"""
    model  = TinyClassifier()
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists()

        with open(saved_path) as f:
            loaded = json.load(f)

        for name in list(result["module_scores"].keys())[:3]:
            orig = result["module_scores"][name]["nuclear_norm"]
            load = loaded["module_scores"][name]["nuclear_norm"]
            assert abs(orig - load) < 1e-9, f"{name} nuclear_norm 不一致"

    print("✓ save/load 往返一致")


def test_sv_print_topk(top_k: int = 5):
    """打印核范数 Top-k 模块，供人工核验"""
    model  = TinyClassifier()
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=8)

    scores = {
        name: v["nuclear_norm"]
        for name, v in result["module_scores"].items()
    }
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n奇异值重要性（核范数）Top-{top_k} 模块：")
    print(f"{'模块名':<55} {'nuclear_norm':>14} {'top8_sum':>10}")
    print("-" * 82)
    for name, _ in sorted_s[:top_k]:
        v = result["module_scores"][name]
        print(f"{name:<55} {v['nuclear_norm']:>14.4f} {v['top8_sum']:>10.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 奇异值重要性（定义 4）")
    print("=" * 60)
    test_sv_structure()
    test_sv_per_module_fields()
    test_sv_nuclear_ge_topk()
    test_sv_non_negative()
    test_sv_no_dataloader_needed()
    test_sv_topk_param()
    test_sv_save_load()
    test_sv_print_topk()
    print("\n所有测试通过 ✓")
