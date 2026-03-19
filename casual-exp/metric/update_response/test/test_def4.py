"""
metric/update_response/test/test_def4.py

定义 4 梯度信噪比（PpredMetric）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_def4

覆盖点：
  1. compute() 返回结构完整
  2. module_scores ∈ [0, 1]（Cauchy-Schwarz 必须保证）
  3. param_scores ∈ [0, 1]
  4. num_batches 计数正确
  5. 至少一个模块 Ppred > 0（梯度不全为零）
  6. G_module ≤ sqrt(F_module)（分子的平方根 ≤ 分母的平方根）
  7. save/load 往返一致
  8. 打印 Top-k 模块（供人工核验）
"""

import json
import math
import tempfile

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.update_response.def4_ppred import PpredMetric
from metric.update_response.test.conftest import TinyClassifier, make_fake_dataloader


def test_def4_structure():
    """返回字典包含所有必需字段。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    for key in ("module_scores", "param_scores", "G_module", "F_module",
                "num_batches", "epsilon"):
        assert key in result, f"缺少字段: {key}"
    print("✓ 返回结构正确")


def test_def4_range_module():
    """module_scores ∈ [0, 1]（Cauchy-Schwarz）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=8)

    for name, v in result["module_scores"].items():
        assert 0 <= v <= 1.0 + 1e-6, \
            f"{name}: Ppred={v:.6f} 不在 [0, 1] 范围内"
    print(f"✓ 所有 {len(result['module_scores'])} 个模块 Ppred ∈ [0, 1]")


def test_def4_range_param():
    """param_scores ∈ [0, 1]（元素级 Ppred 均值）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=8)

    for name, v in result["param_scores"].items():
        assert 0 <= v <= 1.0 + 1e-6, \
            f"{name}: param Ppred={v:.6f} 不在 [0, 1] 范围内"
    print(f"✓ 所有参数级 Ppred ∈ [0, 1]")


def test_def4_num_batches():
    """num_batches 计数正确。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=8)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    assert result["num_batches"] == 4, \
        f"期望 num_batches=4，实际={result['num_batches']}"
    print("✓ num_batches 计数正确")


def test_def4_nonzero():
    """至少一个模块 Ppred > 0（梯度不全为零）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=8)

    nonzero = sum(1 for v in result["module_scores"].values() if v > 0)
    assert nonzero > 0, "所有模块 Ppred = 0，可能未正确反向传播"
    print(f"✓ {nonzero}/{len(result['module_scores'])} 个模块 Ppred > 0")


def test_def4_g_vs_f():
    """
    G_module ≤ sqrt(F_module + ε)（E[|g|] ≤ sqrt(E[g²])，Jensen 不等式）。
    即分子 ≤ 分母，保证 Ppred ≤ 1。
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=8, epsilon=1e-8)

    eps = result["epsilon"]
    for name in result["G_module"]:
        g = result["G_module"][name]
        f = result["F_module"][name]
        assert g <= (f + eps) ** 0.5 + 1e-6, \
            f"{name}: G_module={g:.6f} > sqrt(F_module+ε)={(f+eps)**0.5:.6f}"

    print(f"✓ 所有模块 G_m ≤ √(F_m + ε)（Jensen 不等式验证通过）")


def test_def4_save_load():
    """save/load 往返一致。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists()
        with open(saved_path) as f:
            loaded = json.load(f)
        for key in list(result["module_scores"].keys())[:3]:
            assert abs(loaded["module_scores"][key] - result["module_scores"][key]) < 1e-9

    print("✓ save/load 往返一致")


def test_def4_print_topk(top_k: int = 5):
    """打印 Top-k 模块（供人工核验）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=8)

    scores = result["module_scores"]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n定义 4：梯度 SNR（Ppred）Top-{top_k} 模块（num_batches=8）：")
    print(f"{'模块名':<55} {'Ppred':>8} {'G_m':>10} {'F_m':>10}")
    print("-" * 86)
    gm = result["G_module"]
    fm = result["F_module"]
    for name, score in sorted_scores[:top_k]:
        print(f"{name:<55} {score:>8.4f} {gm.get(name, 0):>10.6f} "
              f"{fm.get(name, 0):>10.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试定义 4：梯度信噪比 Ppred（PpredMetric）")
    print("=" * 60)
    test_def4_structure()
    test_def4_range_module()
    test_def4_range_param()
    test_def4_num_batches()
    test_def4_nonzero()
    test_def4_g_vs_f()
    test_def4_save_load()
    test_def4_print_topk()
    print("\n所有测试通过 ✓")
