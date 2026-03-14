"""
metric/pre_importance/test/test_runner.py

PreImportanceRunner 组合运行器 — 集成测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.pre_importance.test.test_runner

覆盖点：
  1. from_str() 正确解析逗号分隔的指标名
  2. 未知指标名抛出 ValueError
  3. run() 同时计算多个指标并各自保存 JSON
  4. 仅 SVD 指标（无数据）时传入 dataloader=None 也能运行
  5. metric_kwargs 正确传递到各指标的 compute()
  6. 结果字典包含所有选定指标的输出
"""

import json
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import pytest

from metric.pre_importance.runner import PreImportanceRunner
from metric.pre_importance.test.conftest import TinyClassifier, make_fake_dataloader


def test_runner_from_str():
    """from_str() 正确解析逗号分隔字符串"""
    runner = PreImportanceRunner.from_str("fisher,saliency,spectral_entropy")
    assert set(runner.metrics.keys()) == {"fisher", "saliency", "spectral_entropy"}
    print("✓ from_str() 解析正确")


def test_runner_unknown_metric():
    """未知指标名应抛出 ValueError"""
    try:
        PreImportanceRunner(metrics=["fisher", "unknown_metric"])
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "unknown_metric" in str(e)
        print(f"✓ 未知指标正确报错：{e}")


def test_runner_multi_metric():
    """同时运行多个指标，每个指标均生成独立 JSON 文件"""
    model   = TinyClassifier()
    dl      = make_fake_dataloader(batch_size=2, num_batches=4)
    device  = torch.device("cpu")

    runner = PreImportanceRunner(
        metrics=["fisher", "singular_value", "spectral_entropy"],
        metric_kwargs={"fisher": {"num_batches": 2}},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = runner.run(model, dl, device, save_dir=tmpdir)

        # 返回字典包含所有指标
        assert set(results.keys()) == {"fisher", "singular_value", "spectral_entropy"}

        # 各指标均生成独立 JSON 文件
        for name in ("fisher", "singular_value", "spectral_entropy"):
            path = Path(tmpdir) / f"{name}.json"
            assert path.exists(), f"{name}.json 未生成"
            with open(path) as f:
                data = json.load(f)
            assert "module_scores" in data, f"{name}.json 缺少 module_scores"

    print("✓ 多指标运行正确，各自生成独立 JSON")


def test_runner_svd_only_no_dataloader():
    """仅 SVD 类指标（不需要数据）时，可以传入 dataloader=None"""
    model  = TinyClassifier()
    device = torch.device("cpu")

    runner = PreImportanceRunner(
        metrics=["singular_value", "spectral_entropy"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = runner.run(model, dataloader=None, device=device, save_dir=tmpdir)

        assert "singular_value"   in results
        assert "spectral_entropy" in results
        assert (Path(tmpdir) / "singular_value.json").exists()
        assert (Path(tmpdir) / "spectral_entropy.json").exists()

    print("✓ 纯 SVD 指标无需 dataloader 也能运行")


def test_runner_metric_kwargs():
    """metric_kwargs 参数正确传递（通过 num_batches 验证）"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=10)
    device = torch.device("cpu")

    runner = PreImportanceRunner(
        metrics=["fisher"],
        metric_kwargs={"fisher": {"num_batches": 3}},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = runner.run(model, dl, device, save_dir=tmpdir)
        assert results["fisher"]["num_batches"] == 3, \
            f"期望 num_batches=3，实际={results['fisher']['num_batches']}"

    print("✓ metric_kwargs 正确传递（num_batches=3 生效）")


def test_runner_all_five_metrics():
    """同时运行全部五种指标"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    runner = PreImportanceRunner(
        metrics=list(PreImportanceRunner(metrics=["fisher"]).available_metrics),
        metric_kwargs={
            "fisher":       {"num_batches": 2},
            "saliency":     {"num_batches": 2},
            "perturbation": {"num_batches": 1, "num_samples": 1},
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = runner.run(model, dl, device, save_dir=tmpdir)

        expected = {"fisher", "saliency", "perturbation",
                    "singular_value", "spectral_entropy"}
        assert set(results.keys()) == expected

        for name in expected:
            path = Path(tmpdir) / f"{name}.json"
            assert path.exists(), f"{name}.json 未生成"

    print("✓ 五种指标全部运行成功，各生成独立 JSON")


def test_runner_available_metrics():
    """available_metrics 属性返回所有注册的指标名"""
    runner = PreImportanceRunner(metrics=["fisher"])
    avail  = runner.available_metrics
    expected = {"fisher", "saliency", "perturbation",
                "singular_value", "spectral_entropy"}
    assert set(avail) == expected
    print(f"✓ available_metrics = {avail}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 PreImportanceRunner（组合运行器）")
    print("=" * 60)
    test_runner_from_str()
    test_runner_unknown_metric()
    test_runner_multi_metric()
    test_runner_svd_only_no_dataloader()
    test_runner_metric_kwargs()
    test_runner_all_five_metrics()
    test_runner_available_metrics()
    print("\n所有测试通过 ✓")
