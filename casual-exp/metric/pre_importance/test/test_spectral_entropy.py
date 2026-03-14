"""
metric/pre_importance/test/test_spectral_entropy.py

定义 5 谱熵重要性 — 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.pre_importance.test.test_spectral_entropy

覆盖点：
  1. 返回结构完整（module_scores / eps）
  2. 每个模块包含 spectral_entropy / raw_entropy / rank 字段
  3. spectral_entropy ∈ [0, 1]（归一化保证）
  4. raw_entropy ≥ 0（香农熵非负）
  5. 单位矩阵的谱熵应接近 1（各向同性）
  6. 秩为 1 的矩阵谱熵为 0（能量完全集中）
  7. save() / load() 往返一致
"""

import json
import math
import tempfile

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import torch.nn as nn

from metric.pre_importance.spectral_entropy import SpectralEntropyImportance
from metric.pre_importance.test.conftest import TinyClassifier


def test_se_structure():
    """返回字典包含 module_scores 和 eps"""
    model  = TinyClassifier()
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    assert "module_scores" in result
    assert "eps"           in result
    assert len(result["module_scores"]) > 0
    print(f"✓ 返回结构正确，共 {len(result['module_scores'])} 个模块")


def test_se_per_module_fields():
    """每个模块分数包含 spectral_entropy / raw_entropy / rank / matrix_shape"""
    model  = TinyClassifier()
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    for name, scores in result["module_scores"].items():
        for field in ("spectral_entropy", "raw_entropy", "rank", "matrix_shape"):
            assert field in scores, f"模块 {name} 缺少字段 {field}"
    print("✓ 所有模块分数字段完整")


def test_se_range():
    """归一化谱熵 ∈ [0, 1]，raw_entropy ≥ 0"""
    model  = TinyClassifier()
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    for name, scores in result["module_scores"].items():
        se = scores["spectral_entropy"]
        re = scores["raw_entropy"]
        assert -1e-6 <= se <= 1 + 1e-6, \
            f"{name} spectral_entropy={se:.4f} 超出 [0,1]"
        assert re >= -1e-6, \
            f"{name} raw_entropy={re:.4f} 为负"
    print("✓ 所有谱熵值在 [0, 1] 范围内")


def test_se_identity_near_one():
    """
    单位矩阵（各向同性）的归一化谱熵应接近 1。
    构造一个权重为单位矩阵的单层模型。
    """
    class IdentityModel(nn.Module):
        def __init__(self, n=16):
            super().__init__()
            self.linear = nn.Linear(n, n, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(n))

    model  = IdentityModel(n=16)
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    se = result["module_scores"]["linear"]["spectral_entropy"]
    assert se > 0.99, \
        f"单位矩阵谱熵应接近 1，实际={se:.4f}"
    print(f"✓ 单位矩阵谱熵={se:.4f}（接近 1，正确）")


def test_se_rank1_near_zero():
    """
    秩为 1 的矩阵（能量完全集中）谱熵应接近 0。
    """
    class Rank1Model(nn.Module):
        def __init__(self, m=8, n=16):
            super().__init__()
            self.linear = nn.Linear(n, m, bias=False)
            with torch.no_grad():
                # 所有行相同 → 秩为 1
                row = torch.randn(1, n)
                self.linear.weight.copy_(row.expand(m, n))

    model  = Rank1Model()
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    se = result["module_scores"]["linear"]["spectral_entropy"]
    assert se < 0.01, \
        f"秩为 1 的矩阵谱熵应接近 0，实际={se:.4f}"
    print(f"✓ 秩为 1 矩阵谱熵={se:.4f}（接近 0，正确）")


def test_se_no_dataloader_needed():
    """needs_data 应为 False，可以不传 dataloader"""
    metric = SpectralEntropyImportance()
    assert metric.needs_data is False

    model  = TinyClassifier()
    result = metric.compute(model, dataloader=None, device=None)
    assert "module_scores" in result
    print("✓ needs_data=False，无 dataloader 也能运行")


def test_se_save_load():
    """save() / load() 往返一致"""
    model  = TinyClassifier()
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists()

        with open(saved_path) as f:
            loaded = json.load(f)

        for name in list(result["module_scores"].keys())[:3]:
            orig = result["module_scores"][name]["spectral_entropy"]
            load = loaded["module_scores"][name]["spectral_entropy"]
            assert abs(orig - load) < 1e-9, f"{name} 谱熵数值不一致"

    print("✓ save/load 往返一致")


def test_se_print_all():
    """打印所有模块的谱熵，供人工核验"""
    model  = TinyClassifier()
    metric = SpectralEntropyImportance()
    result = metric.compute(model)

    scores = {
        name: v["spectral_entropy"]
        for name, v in result["module_scores"].items()
    }
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n谱熵重要性（所有模块，降序）：")
    print(f"{'模块名':<55} {'谱熵':>8} {'rank':>6} {'shape'}")
    print("-" * 90)
    for name, se in sorted_s:
        v = result["module_scores"][name]
        shape = str(v["matrix_shape"])
        print(f"{name:<55} {se:>8.4f} {v['rank']:>6} {shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 谱熵重要性（定义 5）")
    print("=" * 60)
    test_se_structure()
    test_se_per_module_fields()
    test_se_range()
    test_se_identity_near_one()
    test_se_rank1_near_zero()
    test_se_no_dataloader_needed()
    test_se_save_load()
    test_se_print_all()
    print("\n所有测试通过 ✓")
