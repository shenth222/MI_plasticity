"""
metric/actual_update/test/conftest.py

复用 pre_importance 的测试 fixtures（TinyClassifier / make_fake_dataloader），
额外提供轻量模拟训练工具函数，用于 def1 / def2 / def3 / runner 的单元测试。

运行方式（从 casual-exp 根目录）：
    python -m metric.actual_update.test.test_def1
    python -m metric.actual_update.test.test_def2
    python -m metric.actual_update.test.test_def3
    python -m metric.actual_update.test.test_runner
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# ── 复用 pre_importance 的 fixtures ──────────────────────────────────────────
from metric.pre_importance.test.conftest import (
    TinyClassifier,
    TinyHFClassifier,
    TinyConfig,
    make_fake_dataloader,
)

# ── 轻量 Trainer 状态 mock（避免构造 TrainingArguments）────────────────────────

class FakeArgs:
    """最小化的 TrainingArguments mock，仅需保留 callback 实际访问的字段。"""
    output_dir = "/tmp/actual_update_test"


class FakeState:
    """最小化的 TrainerState mock。"""
    is_world_process_zero: bool = True
    global_step: int = 0
    epoch: float = 0.0


class FakeControl:
    """最小化的 TrainerControl mock。"""
    should_training_stop: bool = False


def train_n_steps(
    model,
    dl,
    callbacks=None,
    steps: int = 5,
    lr: float = 1e-2,
):
    """
    轻量模拟训练循环：手动执行 AdamW 更新，并在每步调用 on_step_end callback。

    Args:
        model:     待训练模型
        dl:        DataLoader
        callbacks: TrainerCallback 列表（可为 None）
        steps:     训练步数
        lr:        学习率

    Returns:
        (args, state, control) — 可供测试用例进一步调用 on_train_end
    """
    import torch

    callbacks = callbacks or []
    args    = FakeArgs()
    state   = FakeState()
    control = FakeControl()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    data_iter = iter(dl)

    for _ in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        optimizer.zero_grad()
        model(**batch).loss.backward()
        optimizer.step()

        for cb in callbacks:
            cb.on_step_end(args, state, control, model=model)

        state.global_step += 1

    return args, state, control


def fire_train_end(callbacks, model, args=None, state=None, control=None):
    """触发所有 callback 的 on_train_end，用于验证保存行为。"""
    args    = args    or FakeArgs()
    state   = state   or FakeState()
    control = control or FakeControl()
    for cb in callbacks:
        cb.on_train_end(args, state, control, model=model)
    return args, state, control


__all__ = [
    # pre_importance fixtures（透传）
    "TinyClassifier",
    "TinyHFClassifier",
    "TinyConfig",
    "make_fake_dataloader",
    # 本包工具
    "FakeArgs",
    "FakeState",
    "FakeControl",
    "train_n_steps",
    "fire_train_end",
]
