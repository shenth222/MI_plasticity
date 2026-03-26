"""
metric/training_gain/test/conftest.py

测试公共 fixtures，适用于 training_gain 所有单元测试。

与 actual_update/test/conftest.py 的关键区别：
  ① train_n_steps 额外调用 on_step_begin（PathIntegralCallback 依赖此事件清空
    梯度缓冲区；若跳过则梯度跨步累积，导致 g·Δθ 计算错误）。
  ② 新增 make_fake_eval_fn：返回轻量 EvalResult，供 def1/def2 测试使用，
    不依赖 GLUE 数据集或真实分词器。

运行方式（从 casual-exp 根目录）：
    python -m metric.training_gain.test.test_def12
    python -m metric.training_gain.test.test_def3
    python -m metric.training_gain.test.test_runner
"""

import sys
import os
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# ── 复用 pre_importance 的模型 fixtures ────────────────────────────────────
from metric.pre_importance.test.conftest import (
    TinyClassifier,
    TinyHFClassifier,
    TinyConfig,
    make_fake_dataloader,
)

from metric.training_gain.base import EvalResult


# ============================================================================
# Trainer mock（与 actual_update 相同）
# ============================================================================

class FakeArgs:
    """最小化的 TrainingArguments mock。"""
    output_dir = "/tmp/training_gain_test"


class FakeState:
    """最小化的 TrainerState mock。"""
    is_world_process_zero: bool = True
    global_step: int = 0
    epoch: float = 0.0


class FakeControl:
    """最小化的 TrainerControl mock。"""
    should_training_stop: bool = False


# ============================================================================
# 假评估函数工厂（def1/def2 需要 eval_fn）
# ============================================================================

def make_fake_eval_fn(
    num_batches: int = 4,
    batch_size:  int = 4,
    num_labels:  int = 3,
    seed:        int = 42,
) -> Callable[[nn.Module, torch.device], EvalResult]:
    """
    构建轻量假评估函数，不依赖 GLUE 数据集或分词器。

    内部使用随机固定数据（seeded），确保测试可重复。
    eval_fn(model, device) → EvalResult(avg_loss, primary_metric_name, primary_metric_value)

    Args:
        num_batches: 评估 batch 数（越少测试越快）
        batch_size:  每 batch 样本数
        num_labels:  分类类别数（与 TinyClassifier 默认值一致）
        seed:        随机种子（固定数据集，保证可重复性）
    """
    # 预生成固定数据（不随 model 更新而变化）
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = batch_size * num_batches
    input_ids      = torch.randint(0, 200, (n, 16), generator=rng)
    attention_mask = torch.ones(n, 16, dtype=torch.long)
    labels         = torch.randint(0, num_labels, (n,), generator=rng)
    dataset        = TensorDataset(input_ids, attention_mask, labels)

    def eval_fn(model: nn.Module, device: torch.device) -> EvalResult:
        model.eval()
        total_loss    = 0.0
        total_batches = 0
        all_preds:  list = []
        all_labels: list = []

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for ids, masks, lbls in loader:
                ids, masks, lbls = ids.to(device), masks.to(device), lbls.to(device)
                out = model(input_ids=ids, attention_mask=masks, labels=lbls)
                if out.loss is not None:
                    total_loss    += out.loss.item()
                    total_batches += 1
                all_preds.append(out.logits.cpu().float().numpy())
                all_labels.append(lbls.cpu().numpy())

        avg_loss   = total_loss / max(total_batches, 1)
        preds_arr  = np.concatenate(all_preds,  axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        pred_cls   = preds_arr.argmax(axis=-1)
        accuracy   = float((pred_cls == labels_arr).mean())

        return EvalResult(
            avg_loss=avg_loss,
            primary_metric_name="accuracy",
            primary_metric_value=accuracy,
            all_metrics={"accuracy": accuracy},
        )

    return eval_fn


# ============================================================================
# 轻量训练循环（关键：比 actual_update 多调用 on_step_begin）
# ============================================================================

def train_n_steps(
    model,
    dl,
    callbacks=None,
    steps:    int   = 5,
    lr:       float = 1e-2,
    use_sgd:  bool  = False,
):
    """
    轻量模拟训练循环，调用顺序与 HF Trainer 对齐：

      on_step_begin → zero_grad → forward → backward → step → on_step_end

    注意：
      · on_step_begin 在 zero_grad 之前调用（PathIntegralCallback 借此清空梯度缓冲区）。
      · gradient hook 在 backward 触发（自动填充梯度缓冲区）。
      · 与 actual_update conftest 的区别：此处调用 on_step_begin。

    Args:
        model:    待训练模型
        dl:       DataLoader
        callbacks: TrainerCallback 列表（可为 None）
        steps:    训练步数
        lr:       学习率
        use_sgd:  True 则使用 SGD（def3 符号验证测试用，Δθ = -α·g 精确成立）

    Returns:
        (args, state, control)
    """
    callbacks = callbacks or []
    args      = FakeArgs()
    state     = FakeState()
    control   = FakeControl()

    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    data_iter = iter(dl)

    for _ in range(steps):
        # 1. on_step_begin（PathIntegralCallback 清空梯度缓冲区）
        for cb in callbacks:
            if hasattr(cb, "on_step_begin"):
                cb.on_step_begin(args, state, control)

        # 2. 获取数据
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        # 3. 前向 + 反向（触发梯度 hook → 填充 grad_buffer）
        optimizer.zero_grad()
        model(**batch).loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. on_step_end
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
    # training_gain 特有
    "FakeArgs",
    "FakeState",
    "FakeControl",
    "make_fake_eval_fn",
    "train_n_steps",
    "fire_train_end",
]
