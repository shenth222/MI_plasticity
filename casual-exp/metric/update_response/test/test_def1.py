"""
metric/update_response/test/test_def1.py

定义 1 短程试跑更新量（ProbeDeltaMetric）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_def1

覆盖点：
  1. compute() 返回结构完整（module_scores / param_scores / probe_steps / probe_lr）
  2. 所有模块分数非负（L2 范数必须 ≥ 0）
  3. θ^(0) 在 compute() 后被完全恢复（参数数值不变）
  4. 位移非零（探针步确实引发了参数变化）
  5. probe_steps 计数字段正确
  6. save/load 往返一致
  7. 打印 Top-k 位移模块（供人工核验）
"""

import json
import copy
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.update_response.def1_probe_delta import ProbeDeltaMetric
from metric.update_response.test.conftest import TinyClassifier, make_fake_dataloader


def test_def1_structure():
    """返回字典包含所有必需字段。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device, probe_steps=3, probe_lr=1e-3)

    for key in ("module_scores", "param_scores", "probe_steps", "probe_lr"):
        assert key in result, f"缺少字段: {key}"
    assert result["probe_steps"] == 3
    print("✓ 返回结构正确")


def test_def1_non_negative():
    """位移范数（L2）必须非负。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device, probe_steps=3, probe_lr=1e-3)

    for name, score in result["module_scores"].items():
        assert score >= 0, f"{name} 的 L2 位移为负: {score}"
    print(f"✓ 所有 {len(result['module_scores'])} 个模块分数均非负")


def test_def1_restore_theta0():
    """compute() 后模型参数应完全恢复至 θ^(0)。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    # 保存初始参数快照
    theta0 = {n: p.data.clone() for n, p in model.named_parameters()}

    metric = ProbeDeltaMetric()
    metric.compute(model, dl, device, probe_steps=5, probe_lr=1e-3)

    # 比对恢复后的参数
    for n, p in model.named_parameters():
        delta = (p.data - theta0[n]).abs().max().item()
        assert delta < 1e-7, f"参数 {n} 未正确恢复，最大偏差={delta:.2e}"

    print("✓ θ^(0) 完全恢复")


def test_def1_nonzero_delta():
    """探针步应导致非零参数位移（至少一个模块分数 > 0）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device, probe_steps=5, probe_lr=1e-3)

    nonzero = sum(1 for v in result["module_scores"].values() if v > 0)
    assert nonzero > 0, "所有模块位移为 0，可能未正确进行梯度下降"
    print(f"✓ {nonzero}/{len(result['module_scores'])} 个模块有非零位移")


def test_def1_save_load():
    """save() 写 JSON 后 load() 读取值与原始一致。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device, probe_steps=3, probe_lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists(), "JSON 文件未生成"
        with open(saved_path) as f:
            loaded = json.load(f)
        for key in list(result["module_scores"].keys())[:3]:
            assert abs(loaded["module_scores"][key] - result["module_scores"][key]) < 1e-9

    print("✓ save/load 往返一致")


def test_def1_print_topk(top_k: int = 5):
    """打印位移 Top-k 模块（供人工核验）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device, probe_steps=10, probe_lr=1e-3)

    scores = result["module_scores"]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n定义 1：参数位移 Top-{top_k} 模块（probe_steps=10, lr=1e-3）：")
    print(f"{'模块名':<55} {'‖Δθ‖₂':>12}")
    print("-" * 70)
    for name, score in sorted_scores[:top_k]:
        print(f"{name:<55} {score:>12.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试定义 1：短程试跑更新量（ProbeDeltaMetric）")
    print("=" * 60)
    test_def1_structure()
    test_def1_non_negative()
    test_def1_restore_theta0()
    test_def1_nonzero_delta()
    test_def1_save_load()
    test_def1_print_topk()
    print("\n所有测试通过 ✓")
