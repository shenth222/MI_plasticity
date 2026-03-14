"""
metric/pre_importance/test/test_saliency.py

定义 2 梯度敏感度 / Saliency — 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.pre_importance.test.test_saliency

覆盖点：
  1. 返回字典同时包含 grad_norm 和 taylor 两种子变体
  2. 两种变体各自包含 module_scores 和 param_scores
  3. 所有分数非负（绝对值之和）
  4. grad_norm 与 taylor 分数不完全相同（两种变体确实不同）
  5. save() / load() 往返一致
  6. 打印 Top-k 对比两种变体
"""

import json
import tempfile

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.pre_importance.saliency import SaliencyImportance
from metric.pre_importance.test.conftest import TinyClassifier, make_fake_dataloader


def test_saliency_structure():
    """返回字典同时包含 grad_norm / taylor / num_batches"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = SaliencyImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    assert "grad_norm"   in result, "缺少 grad_norm"
    assert "taylor"      in result, "缺少 taylor"
    assert "num_batches" in result, "缺少 num_batches"

    for variant in ("grad_norm", "taylor"):
        assert "module_scores" in result[variant], f"{variant} 缺少 module_scores"
        assert "param_scores"  in result[variant], f"{variant} 缺少 param_scores"

    print("✓ 返回结构正确（grad_norm + taylor 均存在）")


def test_saliency_non_negative():
    """两种变体的所有分数均 ≥ 0（基于绝对值）"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = SaliencyImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    for variant in ("grad_norm", "taylor"):
        for name, score in result[variant]["module_scores"].items():
            assert score >= 0, f"{variant}/{name} 分数为负：{score}"
    print("✓ 所有分数非负")


def test_saliency_variants_differ():
    """
    grad_norm 和 taylor 分数不应完全相同
    （除非所有参数初始化为 ±1，极低概率）
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = SaliencyImportance()
    result = metric.compute(model, dl, device, num_batches=8)

    gn = result["grad_norm"]["module_scores"]
    ta = result["taylor"]["module_scores"]

    # 至少一个模块的两种变体值不同
    differs = any(
        abs(gn[k] - ta[k]) > 1e-9
        for k in gn if k in ta
    )
    assert differs, "grad_norm 和 taylor 分数完全相同，可能实现有误"
    print("✓ grad_norm 与 taylor 两种变体分数存在差异")


def test_saliency_save_load():
    """save() 写出 JSON 后，load() 读取结果完全一致"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = SaliencyImportance()
    result = metric.compute(model, dl, device, num_batches=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists(), "JSON 文件未生成"

        with open(saved_path) as f:
            loaded = json.load(f)

        for variant in ("grad_norm", "taylor"):
            for key in list(result[variant]["module_scores"].keys())[:3]:
                orig = result[variant]["module_scores"][key]
                load = loaded[variant]["module_scores"][key]
                assert abs(orig - load) < 1e-9, \
                    f"{variant}/{key} 数值不一致：{orig} vs {load}"

    print("✓ save/load 往返一致")


def test_saliency_print_topk(top_k: int = 5):
    """打印两种变体的 Top-k 模块，供人工核验"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = SaliencyImportance()
    result = metric.compute(model, dl, device, num_batches=8)

    for variant in ("grad_norm", "taylor"):
        scores = result[variant]["module_scores"]
        sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nSaliency [{variant}] Top-{top_k} 模块：")
        print(f"{'模块名':<55} {'分数':>12}")
        print("-" * 70)
        for name, score in sorted_s[:top_k]:
            print(f"{name:<55} {score:>12.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 梯度敏感度 / Saliency（定义 2）")
    print("=" * 60)
    test_saliency_structure()
    test_saliency_non_negative()
    test_saliency_variants_differ()
    test_saliency_save_load()
    test_saliency_print_topk()
    print("\n所有测试通过 ✓")
