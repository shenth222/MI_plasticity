"""
metric/update_response/test/test_runner.py

UpdateResponseRunner — 组合运行器测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_runner

覆盖点：
  1. from_str() 正确解析指标名
  2. 未知指标名抛出 ValueError
  3. run_pre() 运行 def1/def2/def4 并保存对应 JSON 文件
  4. make_training_callbacks() 返回 def3 的 callback（def3 选中时）
  5. 无训练中指标时 make_training_callbacks() 返回空列表
  6. 混合组合（def1+def2+def3+def4）端到端流程
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
from torch.utils.data import DataLoader

from metric.update_response.runner import UpdateResponseRunner
from metric.update_response.def3_early_grad_norm import EarlyGradNormCallback
from metric.update_response.test.conftest import TinyClassifier, make_fake_dataloader


# ---------------------------------------------------------------------------
# 辅助：Trainer 状态模拟（与 test_def3 相同）
# ---------------------------------------------------------------------------

@dataclass
class FakeTrainerState:
    is_world_process_zero: bool = True

@dataclass
class FakeTrainerArgs:
    local_rank: int = -1

@dataclass
class FakeTrainerControl:
    pass

def _simulate_training(cb, model, dl, total_steps, device):
    """手动模拟 backward + on_step_end。"""
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    state   = FakeTrainerState()
    args    = FakeTrainerArgs()
    control = FakeTrainerControl()
    model.train()
    data_iter = iter(dl)
    for _ in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        model(**inputs).loss.backward()
        optimizer.step()
        cb.on_step_end(args=args, state=state, control=control)
        if cb._done:
            break
    if not cb._done:
        cb.on_train_end(args=args, state=state, control=control)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_runner_from_str():
    """from_str 解析逗号分隔字符串，正确划分 pre / in-training 指标。"""
    runner = UpdateResponseRunner.from_str("def1,def2,def3,def4")
    assert set(runner.selected_pre_metrics) == {"def1", "def2", "def4"}
    assert set(runner.selected_in_metrics)  == {"def3"}
    print("✓ from_str 解析正确，pre={def1,def2,def4}，in={def3}")


def test_runner_unknown_metric():
    """未知指标名应抛出 ValueError。"""
    try:
        UpdateResponseRunner.from_str("def1,def_unknown")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "def_unknown" in str(e)
    print("✓ 未知指标名正确抛出 ValueError")


def test_runner_run_pre_saves_files():
    """run_pre 运行后，selected pre 指标均生成对应 JSON 文件。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=8)
    device = torch.device("cpu")

    runner = UpdateResponseRunner.from_str(
        "def1,def2,def4",
        metric_kwargs={
            "def1": {"probe_steps": 3, "probe_lr": 1e-3},
            "def2": {"num_batches": 4},
            "def4": {"num_batches": 4},
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        runner.run_pre(model, dl, device, save_dir=tmpdir)

        for name in ("def1_probe_delta", "def2_grad_curvature", "def4_ppred"):
            p = Path(tmpdir) / f"{name}.json"
            assert p.exists(), f"JSON 文件 {name}.json 未生成"
            with open(p) as f:
                data = json.load(f)
            assert "module_scores" in data

    print("✓ def1/def2/def4 均生成 JSON 文件")


def test_runner_no_in_metrics():
    """只选 pre 指标时，make_training_callbacks() 返回空列表。"""
    runner = UpdateResponseRunner.from_str("def1,def2")
    assert runner.make_training_callbacks(TinyClassifier(), save_dir="/tmp") == []
    print("✓ 无 in-training 指标时 make_training_callbacks 返回空列表")


def test_runner_in_training_callback():
    """选 def3 时，make_training_callbacks() 返回 EarlyGradNormCallback。"""
    model  = TinyClassifier()
    runner = UpdateResponseRunner.from_str(
        "def3", metric_kwargs={"def3": {"T_early": 5}}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cbs = runner.make_training_callbacks(model, save_dir=tmpdir)
        assert len(cbs) == 1
        assert isinstance(cbs[0], EarlyGradNormCallback)
        assert cbs[0].T_early == 5
    print("✓ def3 返回正确的 EarlyGradNormCallback（T_early=5）")


def test_runner_full_pipeline():
    """
    混合组合 def1+def2+def3+def4 的端到端流程：
      1. run_pre → 保存 def1/def2/def4 文件
      2. make_training_callbacks → 返回 def3 回调
      3. 模拟训练 → def3 文件保存
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=20)
    device = torch.device("cpu")

    runner = UpdateResponseRunner.from_str(
        "def1,def2,def3,def4",
        metric_kwargs={
            "def1": {"probe_steps": 3, "probe_lr": 1e-3},
            "def2": {"num_batches": 4},
            "def3": {"T_early": 5},
            "def4": {"num_batches": 4},
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 嵌入点 1：训练前
        runner.run_pre(model, dl, device, save_dir=tmpdir)

        # 嵌入点 2：构造 callback
        cbs = runner.make_training_callbacks(model, save_dir=tmpdir)
        assert len(cbs) == 1

        # 模拟训练
        _simulate_training(cbs[0], model, dl, total_steps=8, device=device)

        # 验证所有 4 个文件都生成
        expected_files = [
            "def1_probe_delta.json",
            "def2_grad_curvature.json",
            "def3_early_grad_norm.json",
            "def4_ppred.json",
        ]
        for fname in expected_files:
            p = Path(tmpdir) / fname
            assert p.exists(), f"JSON 文件 {fname} 未生成"
            with open(p) as f:
                data = json.load(f)
            assert "module_scores" in data, f"{fname} 缺少 module_scores"

    print("✓ 端到端流程：def1+def2+def3+def4 全部正确生成 JSON")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试 UpdateResponseRunner（组合运行器）")
    print("=" * 60)
    test_runner_from_str()
    test_runner_unknown_metric()
    test_runner_run_pre_saves_files()
    test_runner_no_in_metrics()
    test_runner_in_training_callback()
    test_runner_full_pipeline()
    print("\n所有测试通过 ✓")
