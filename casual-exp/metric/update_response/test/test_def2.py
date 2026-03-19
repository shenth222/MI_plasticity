"""
metric/update_response/test/test_def2.py

定义 2 梯度-曲率归一化（GradCurvatureMetric）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_def2

覆盖点：
  1. compute() 返回结构完整
  2. module_scores 所有值非负（分子/分母均非负）
  3. num_batches 计数正确
  4. grad_norm_mean 与 fisher_module 字段均有非零值（梯度确实存在）
  5. module_scores ≤ 1 的比例（定义 2 < 1 当 Fisher 存在时，Cauchy-Schwarz）
  6. save/load 往返一致
  7. 打印 Top-k 分数模块（供人工核验）
"""

import json
import tempfile

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.update_response.def2_grad_curvature import GradCurvatureMetric
from metric.update_response.test.conftest import TinyClassifier, make_fake_dataloader


def test_def2_structure():
    """返回字典包含所有必需字段。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    for key in ("module_scores", "grad_norm_mean", "fisher_module", "num_batches", "epsilon"):
        assert key in result, f"缺少字段: {key}"
    print("✓ 返回结构正确")


def test_def2_non_negative():
    """module_scores / grad_norm_mean / fisher_module 均非负。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    for name, v in result["module_scores"].items():
        assert v >= 0, f"{name}: module_scores={v} < 0"
    for name, v in result["grad_norm_mean"].items():
        assert v >= 0, f"{name}: grad_norm_mean={v} < 0"
    for name, v in result["fisher_module"].items():
        assert v >= 0, f"{name}: fisher_module={v} < 0"
    print(f"✓ 所有 {len(result['module_scores'])} 个模块分数均非负")


def test_def2_num_batches():
    """num_batches 与实际使用的 batch 数量一致。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=8)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    assert result["num_batches"] == 4, \
        f"期望 num_batches=4，实际={result['num_batches']}"
    print("✓ num_batches 计数正确")


def test_def2_nonzero_grad():
    """梯度存在时，至少一个模块的 grad_norm_mean > 0。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    nonzero = sum(1 for v in result["grad_norm_mean"].values() if v > 0)
    assert nonzero > 0, "所有模块 grad_norm_mean = 0，可能未正确反向传播"
    print(f"✓ {nonzero}/{len(result['grad_norm_mean'])} 个模块 grad_norm_mean > 0")


def test_def2_cauchy_schwarz():
    """
    由 Cauchy-Schwarz：E[‖g‖] ≤ √E[‖g‖²] → module_scores ≤ 1。
    （当 epsilon > 0 时略小于 1）
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=8, epsilon=1e-8)

    for name, score in result["module_scores"].items():
        assert score <= 1.0 + 1e-6, \
            f"{name}: module_scores={score:.4f} > 1（违反 Cauchy-Schwarz）"

    print(f"✓ 所有 module_scores ≤ 1（Cauchy-Schwarz 验证通过）")


def test_def2_save_load():
    """save/load 往返一致。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists()
        with open(saved_path) as f:
            loaded = json.load(f)
        for key in list(result["module_scores"].keys())[:3]:
            assert abs(loaded["module_scores"][key] - result["module_scores"][key]) < 1e-9

    print("✓ save/load 往返一致")


def test_def2_print_topk(top_k: int = 5):
    """打印 Top-k 模块（供人工核验）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device, num_batches=8)

    scores = result["module_scores"]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n定义 2：梯度-曲率归一化 Top-{top_k} 模块（num_batches=8）：")
    print(f"{'模块名':<55} {'R̂_m':>10} {'E[‖g‖]':>12} {'Fisher_m':>12}")
    print("-" * 92)
    gn  = result["grad_norm_mean"]
    fim = result["fisher_module"]
    for name, score in sorted_scores[:top_k]:
        print(f"{name:<55} {score:>10.6f} {gn.get(name, 0):>12.6f} "
              f"{fim.get(name, 0):>12.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试定义 2：梯度-曲率归一化（GradCurvatureMetric）")
    print("=" * 60)
    test_def2_structure()
    test_def2_non_negative()
    test_def2_num_batches()
    test_def2_nonzero_grad()
    test_def2_cauchy_schwarz()
    test_def2_save_load()
    test_def2_print_topk()
    print("\n所有测试通过 ✓")
