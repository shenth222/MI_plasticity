"""
metric/update_response/test/conftest.py

复用 pre_importance 的测试 fixtures（TinyClassifier + make_fake_dataloader），
额外提供 make_fake_trainer_args 用于 def3 的 Trainer 回调测试。

运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_def1
    python -m metric.update_response.test.test_def3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 直接复用 pre_importance 的 conftest 内容，避免重复定义
from metric.pre_importance.test.conftest import (
    TinyClassifier,
    TinyHFClassifier,
    TinyConfig,
    make_fake_dataloader,
)

__all__ = [
    "TinyClassifier",
    "TinyHFClassifier",
    "TinyConfig",
    "make_fake_dataloader",
]
