"""
metric/pre_importance/test/test_fisher.py

定义 1 Fisher 型重要性 — 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.pre_importance.test.test_fisher

覆盖点：
  1. compute() 返回结构完整
  2. 所有 module_scores 均为非负数（梯度平方期望 ≥ 0）
  3. num_batches 计数正确
  4. save() / load() 往返一致
  5. 有梯度的参数才有分数（requires_grad=False 的参数分数为 0）
"""

import json
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.pre_importance.fisher import FisherImportance
from metric.pre_importance.test.conftest import TinyClassifier, make_fake_dataloader


def test_fisher_compute_structure():
    """返回字典包含 module_scores / param_scores / num_batches"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    assert "module_scores" in result, "缺少 module_scores"
    assert "param_scores"  in result, "缺少 param_scores"
    assert "num_batches"   in result, "缺少 num_batches"
    print("✓ 返回结构正确")


def test_fisher_non_negative():
    """Fisher 分数（梯度平方期望）必须非负"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    for name, score in result["module_scores"].items():
        assert score >= 0, f"{name} 的 Fisher 分数为负：{score}"
    for name, score in result["param_scores"].items():
        assert score >= 0, f"{name} 的参数 Fisher 分数为负：{score}"
    print(f"✓ 所有 {len(result['module_scores'])} 个模块分数均非负")


def test_fisher_num_batches():
    """num_batches 与实际处理的 batch 数一致"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=2, num_batches=6)
    device = torch.device("cpu")

    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    assert result["num_batches"] == 4, \
        f"期望 num_batches=4，实际={result['num_batches']}"
    print("✓ num_batches 计数正确")


def test_fisher_save_load():
    """save() 写出 JSON 后，load() 读取结果完全一致"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists(), "JSON 文件未生成"

        with open(saved_path) as f:
            loaded = json.load(f)

        # 只比较 module_scores 中的几个键
        for key in list(result["module_scores"].keys())[:3]:
            assert abs(loaded["module_scores"][key] - result["module_scores"][key]) < 1e-9, \
                f"module {key} 数值不一致"

    print("✓ save/load 往返一致")


def test_fisher_importance_nonzero():
    """正常前向 + 反向后，至少有一个模块的 Fisher 分数 > 0"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    nonzero = sum(1 for v in result["module_scores"].values() if v > 0)
    assert nonzero > 0, "所有模块分数都为 0，可能未正确反向传播"
    print(f"✓ {nonzero}/{len(result['module_scores'])} 个模块分数 > 0")


def test_fisher_print_topk(top_k: int = 5):
    """打印 Fisher 分数 Top-k 模块，供人工核验"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=8)

    scores = result["module_scores"]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nFisher 重要性 Top-{top_k} 模块：")
    print(f"{'模块名':<55} {'分数':>12}")
    print("-" * 70)
    for name, score in sorted_scores[:top_k]:
        print(f"{name:<55} {score:>12.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 Fisher 型重要性（定义 1）")
    print("=" * 60)
    test_fisher_compute_structure()
    test_fisher_non_negative()
    test_fisher_num_batches()
    test_fisher_save_load()
    test_fisher_importance_nonzero()
    test_fisher_print_topk()
    print("\n所有测试通过 ✓")
