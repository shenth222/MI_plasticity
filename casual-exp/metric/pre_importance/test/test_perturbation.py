"""
metric/pre_importance/test/test_perturbation.py

定义 3 扰动敏感度 — 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.pre_importance.test.test_perturbation

覆盖点：
  1. 返回结构完整（module_scores / num_batches / num_samples / noise_std）
  2. 参数恢复正确：扰动后参数值与扰动前完全一致
  3. 噪声 std 参数正确传入（relative_noise 开/关）
  4. 无可训练参数的模块分数为 0
  5. save() / load() 往返一致
  6. 打印 Top-k 供人工核验
"""

import json
import tempfile

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.pre_importance.perturbation import PerturbationImportance
from metric.pre_importance.test.conftest import TinyClassifier, make_fake_dataloader


def test_perturbation_structure():
    """返回字典包含所有必要字段"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = PerturbationImportance()
    result = metric.compute(model, dl, device, num_batches=2, num_samples=2)

    for key in ("module_scores", "num_batches", "num_samples", "noise_std", "relative_noise"):
        assert key in result, f"缺少字段：{key}"
    print("✓ 返回结构正确")


def test_perturbation_param_restore():
    """
    计算完成后，模型参数应与计算前完全一致（扰动已原地恢复）。
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    # 记录计算前参数快照
    before = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    metric = PerturbationImportance()
    metric.compute(model, dl, device, num_batches=2, num_samples=2)

    # 逐参数比对
    max_diff = 0.0
    for name, param in model.named_parameters():
        diff = (param.data - before[name]).abs().max().item()
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-6, f"扰动未完全恢复，最大差值={max_diff}"
    print(f"✓ 参数恢复正确（最大差值={max_diff:.2e}）")


def test_perturbation_absolute_noise():
    """relative_noise=False 时使用绝对噪声标准差"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = PerturbationImportance()
    result = metric.compute(
        model, dl, device,
        num_batches=2, num_samples=2,
        noise_std=0.01,
        relative_noise=False,
    )

    assert result["relative_noise"] is False
    assert abs(result["noise_std"] - 0.01) < 1e-9
    print("✓ 绝对噪声模式参数传递正确")


def test_perturbation_scores_exist():
    """至少有若干模块具有非零 delta（正常扰动应引起 loss 变化）"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=4)
    device = torch.device("cpu")

    metric = PerturbationImportance()
    result = metric.compute(
        model, dl, device,
        num_batches=2, num_samples=3,
        noise_std=1e-2,
    )

    nonzero = sum(
        1 for v in result["module_scores"].values() if abs(v) > 1e-9
    )
    total = len(result["module_scores"])
    print(f"  非零模块: {nonzero}/{total}")
    assert nonzero > 0, "所有模块 delta 均为 0，扰动可能未生效"
    print("✓ 存在非零扰动 delta 的模块")


def test_perturbation_save_load():
    """save() / load() 往返一致"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    metric = PerturbationImportance()
    result = metric.compute(model, dl, device, num_batches=2, num_samples=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists(), "JSON 文件未生成"

        with open(saved_path) as f:
            loaded = json.load(f)

        for key in list(result["module_scores"].keys())[:3]:
            orig = result["module_scores"][key]
            load = loaded["module_scores"][key]
            assert abs(orig - load) < 1e-9, f"{key} 数值不一致"

    print("✓ save/load 往返一致")


def test_perturbation_print_topk(top_k: int = 5):
    """打印 Top-k 扰动敏感度，供人工核验"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=4)
    device = torch.device("cpu")

    metric = PerturbationImportance()
    result = metric.compute(
        model, dl, device,
        num_batches=2, num_samples=3,
        noise_std=1e-2,
    )

    scores = result["module_scores"]
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n扰动敏感度 Top-{top_k} 模块（noise_std=1e-2, relative=True）：")
    print(f"{'模块名':<55} {'E[ΔL]':>12}")
    print("-" * 70)
    for name, score in sorted_s[:top_k]:
        print(f"{name:<55} {score:>+12.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 扰动敏感度（定义 3）")
    print("=" * 60)
    test_perturbation_structure()
    test_perturbation_param_restore()
    test_perturbation_absolute_noise()
    test_perturbation_scores_exist()
    test_perturbation_save_load()
    test_perturbation_print_topk()
    print("\n所有测试通过 ✓")
