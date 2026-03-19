"""
metric/update_response/test/test_def3.py

定义 3 累积早期梯度范数（EarlyGradNormMetric / EarlyGradNormCallback）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_def3

覆盖点：
  1. EarlyGradNormCallback 可正常构造，钩子注册数量 = 可训练参数数量
  2. 手动模拟 Trainer 步：backward + on_step_end，累积 T_early 步后结果正确
  3. module_scores 非负（L2 范数之和必须 ≥ 0）
  4. steps_collected 与 T_early 一致（或等于总步数）
  5. 钩子在 T_early 后被卸载（_done=True，_hooks 为空）
  6. on_train_end 提前结束时也能正确保存
  7. save 文件存在且可解析
  8. 打印 Top-k 模块（供人工核验）
"""

import json
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
from torch.utils.data import DataLoader

from metric.update_response.def3_early_grad_norm import (
    EarlyGradNormMetric,
    EarlyGradNormCallback,
)
from metric.update_response.test.conftest import TinyClassifier, make_fake_dataloader


# ---------------------------------------------------------------------------
# 最简 Trainer 状态模拟（用于传入 callback 方法）
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


def _run_fake_training(
    model: torch.nn.Module,
    dl: DataLoader,
    T_early: int,
    save_dir: str,
    total_steps: int,
    device: torch.device,
) -> EarlyGradNormCallback:
    """
    手动模拟 HuggingFace Trainer 的核心训练循环：
      optimizer.zero_grad() → forward → backward → on_step_end
    用于单元测试，不需要真实 Trainer。
    """
    cb = EarlyGradNormCallback(model=model, T_early=T_early, save_dir=save_dir)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    state   = FakeTrainerState()
    args    = FakeTrainerArgs()
    control = FakeTrainerControl()

    model.train()
    data_iter = iter(dl)

    for step in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        loss = model(**inputs).loss
        loss.backward()                          # 触发梯度钩子
        optimizer.step()

        cb.on_step_end(args=args, state=state, control=control)

        if cb._done:
            break

    return cb


# ---------------------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------------------

def test_def3_hook_count():
    """注册钩子数量 = 可训练参数数量。"""
    model = TinyClassifier()
    n_params = sum(1 for p in model.named_parameters() if p[1].requires_grad)

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EarlyGradNormCallback(model=model, T_early=10, save_dir=tmpdir)
        assert len(cb._hooks) == n_params, \
            f"期望 {n_params} 个钩子，实际 {len(cb._hooks)}"

    print(f"✓ 注册了 {n_params} 个梯度钩子")


def test_def3_accumulation():
    """经过 T_early 步后，至少一个模块分数 > 0。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=16)
    device = torch.device("cpu")
    T_early = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = _run_fake_training(model, dl, T_early, tmpdir, total_steps=T_early + 2, device=device)

        nonzero = sum(1 for v in cb._module_acc.values() if v > 0)
        assert nonzero > 0, "所有模块累积梯度范数为 0"

    print(f"✓ {nonzero}/{len(cb._module_acc)} 个模块有非零累积梯度范数")


def test_def3_steps_collected():
    """steps_collected 与 T_early 一致（不超过，也不少于）。"""
    model   = TinyClassifier()
    dl      = make_fake_dataloader(batch_size=2, num_batches=20)
    device  = torch.device("cpu")
    T_early = 6

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = _run_fake_training(model, dl, T_early, tmpdir, total_steps=T_early + 3, device=device)

        assert cb._steps_collected == T_early, \
            f"期望 steps_collected={T_early}，实际={cb._steps_collected}"
        assert cb._done, "T_early 步后 _done 应为 True"
        assert len(cb._hooks) == 0, "T_early 步后钩子应已卸载"

    print(f"✓ steps_collected={T_early}，钩子已卸载，_done=True")


def test_def3_non_negative():
    """所有模块分数非负。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=12)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = _run_fake_training(model, dl, T_early=5, save_dir=tmpdir,
                                total_steps=6, device=device)
        for m_name, v in cb._module_acc.items():
            assert v >= 0, f"{m_name}: module_acc={v} < 0"

    print(f"✓ 所有 {len(cb._module_acc)} 个模块分数均非负")


def test_def3_early_end():
    """训练步数 < T_early 时，on_train_end 能正确保存。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=12)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = _run_fake_training(model, dl, T_early=100, save_dir=tmpdir,
                                total_steps=4, device=device)

        assert not cb._done, "训练步数 < T_early，此时 _done 应为 False"

        # 模拟 on_train_end
        state   = FakeTrainerState()
        args    = FakeTrainerArgs()
        control = FakeTrainerControl()
        cb.on_train_end(args=args, state=state, control=control)

        assert cb._done, "on_train_end 后 _done 应为 True"
        out_file = Path(tmpdir) / "def3_early_grad_norm.json"
        assert out_file.exists(), "JSON 文件未生成"
        with open(out_file) as f:
            data = json.load(f)
        assert data["steps_collected"] == 4

    print(f"✓ 训练提前结束（4步 < T_early=100），on_train_end 正确保存")


def test_def3_save_file():
    """T_early 步后 JSON 文件存在且字段完整。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=2, num_batches=16)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        _run_fake_training(model, dl, T_early=5, save_dir=tmpdir,
                           total_steps=8, device=device)

        out_file = Path(tmpdir) / "def3_early_grad_norm.json"
        assert out_file.exists(), "JSON 文件未生成"
        with open(out_file) as f:
            data = json.load(f)
        for key in ("module_scores", "param_scores", "T_early", "steps_collected"):
            assert key in data, f"JSON 缺少字段: {key}"

    print("✓ JSON 文件存在，字段完整")


def test_def3_print_topk(top_k: int = 5):
    """打印 Top-k 模块（供人工核验）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=20)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = _run_fake_training(model, dl, T_early=10, save_dir=tmpdir,
                                total_steps=12, device=device)
        scores = dict(cb._module_acc)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n定义 3：早期累积梯度范数 Top-{top_k}（T_early=10）：")
    print(f"{'模块名':<55} {'Σ‖g_m‖':>12}")
    print("-" * 70)
    for name, score in sorted_scores[:top_k]:
        print(f"{name:<55} {score:>12.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  测试定义 3：累积早期梯度范数（EarlyGradNormCallback）")
    print("=" * 60)
    test_def3_hook_count()
    test_def3_accumulation()
    test_def3_steps_collected()
    test_def3_non_negative()
    test_def3_early_end()
    test_def3_save_file()
    test_def3_print_topk()
    print("\n所有测试通过 ✓")
