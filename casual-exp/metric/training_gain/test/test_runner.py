"""
metric/training_gain/test/test_runner.py

TrainingGainRunner —— 组合运行器测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.training_gain.test.test_runner

覆盖点：
  1.  REGISTRY 包含 def1 / def2 / def3
  2.  from_str 正确解析逗号分隔字符串（含空格容忍）
  3.  未知指标名称抛出 ValueError
  4.  def1+def2 合并为 1 个 RollbackCallback（共用前向）
  5.  def1+def2+def3 → 2 个 callback（1 个 Rollback + 1 个 PathIntegral）
  6.  def1 独立运行：生成 def1_rollback_loss.json，不生成 def2/def3
  7.  def2 独立运行：生成 def2_rollback_acc.json，不生成 def1/def3
  8.  def3 独立运行：生成 def3_path_integral.json，steps_collected 正确
  9.  三种指标同时运行：生成 3 个独立 JSON
  10. metric_kwargs 中 def3 的 log_every 正确透传
  11. head_granularity 全局开关：JSON 含 head_scores（TinyHFClassifier）
  12. available_metrics 属性返回有序列表
"""

import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.training_gain.runner import TrainingGainRunner, REGISTRY
from metric.training_gain.test.conftest import (
    TinyClassifier,
    TinyHFClassifier,
    TinyConfig,
    make_fake_dataloader,
    make_fake_eval_fn,
    train_n_steps,
    fire_train_end,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _run_end_to_end(
    metrics_str:  str,
    tmpdir:       str,
    steps:        int   = 5,
    metric_kwargs       = None,
    head_granularity:   bool = False,
    use_hf_model:       bool = False,
):
    """
    完整模拟：构造 runner → make_callbacks → 训练 → on_train_end → 返回生成文件列表。

    Args:
        use_hf_model: True 时使用 TinyHFClassifier（支持头级别测试）
    """
    if use_hf_model:
        cfg   = TinyConfig(hidden_size=8, num_attention_heads=2)
        model = TinyHFClassifier(cfg)
    else:
        model = TinyClassifier()

    dl      = make_fake_dataloader(batch_size=4, num_batches=8)
    eval_fn = make_fake_eval_fn(num_batches=4, batch_size=4)

    runner    = TrainingGainRunner.from_str(
        metrics_str,
        head_granularity=head_granularity,
    )
    callbacks = runner.make_callbacks(
        model,
        save_dir=tmpdir,
        eval_fn=eval_fn,
        metric_kwargs=metric_kwargs,
    )

    args, state, control = train_n_steps(model, dl, callbacks=callbacks, steps=steps)
    fire_train_end(callbacks, model, args, state, control)

    return [f for f in os.listdir(tmpdir) if f.endswith(".json")]


# ---------------------------------------------------------------------------
# 注册表与基本属性
# ---------------------------------------------------------------------------

def test_runner_registry():
    """REGISTRY 包含 def1 / def2 / def3。"""
    assert set(REGISTRY.keys()) == {"def1", "def2", "def3"}
    print(f"✓ REGISTRY 包含 {sorted(REGISTRY.keys())}")


def test_runner_from_str_single():
    """from_str 正确解析单个指标。"""
    r = TrainingGainRunner.from_str("def3")
    assert r.selected_metrics == ["def3"]
    print("✓ from_str 单指标解析正确")


def test_runner_from_str_all():
    """from_str 正确解析多指标（含空格容忍）。"""
    r = TrainingGainRunner.from_str("def1 , def2,def3 ")
    assert set(r.selected_metrics) == {"def1", "def2", "def3"}
    print("✓ from_str 多指标解析正确（容忍空格）")


def test_runner_unknown_metric():
    """未知指标名称应抛出 ValueError 并提示可用指标。"""
    try:
        TrainingGainRunner(metrics=["def99"])
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "def99" in str(e)
        assert "Available" in str(e)
    print("✓ 未知指标正确抛出 ValueError")


def test_runner_available_metrics():
    """available_metrics 属性返回所有注册指标的有序列表。"""
    r = TrainingGainRunner.from_str("def1")
    assert r.available_metrics == ["def1", "def2", "def3"]
    print(f"✓ available_metrics = {r.available_metrics}")


# ---------------------------------------------------------------------------
# callback 数量：def1+def2 合并
# ---------------------------------------------------------------------------

def test_runner_def12_merged_callback():
    """
    def1+def2 应合并为 1 个 RollbackCallback（共用前向传播），
    而非分别创建 2 个 callback。
    """
    model   = TinyClassifier()
    eval_fn = make_fake_eval_fn()

    with tempfile.TemporaryDirectory() as tmpdir:
        runner    = TrainingGainRunner.from_str("def1,def2")
        callbacks = runner.make_callbacks(model, save_dir=tmpdir, eval_fn=eval_fn)
        assert len(callbacks) == 1, (
            f"def1+def2 应合并为1个callback，实际={len(callbacks)}"
        )
    print("✓ def1+def2 合并为 1 个 RollbackCallback")


def test_runner_all_three_callback_count():
    """def1+def2+def3 → 2 个 callback（1 Rollback + 1 PathIntegral）。"""
    model   = TinyClassifier()
    eval_fn = make_fake_eval_fn()

    with tempfile.TemporaryDirectory() as tmpdir:
        runner    = TrainingGainRunner.from_str("def1,def2,def3")
        callbacks = runner.make_callbacks(model, save_dir=tmpdir, eval_fn=eval_fn)
        assert len(callbacks) == 2, (
            f"def1+def2+def3 应有2个callback，实际={len(callbacks)}"
        )
    print("✓ def1+def2+def3 → 2 个 callback（合并策略正确）")


# ---------------------------------------------------------------------------
# 端到端：各指标独立生成正确 JSON
# ---------------------------------------------------------------------------

def test_runner_def1_only():
    """仅 def1 → def1_rollback_loss.json，不生成 def2/def3。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end("def1", tmpdir, steps=5)
        assert "def1_rollback_loss.json" in files, f"缺少 def1，实际: {files}"
        assert "def2_rollback_acc.json"  not in files
        assert "def3_path_integral.json" not in files

        with open(os.path.join(tmpdir, "def1_rollback_loss.json")) as f:
            d = json.load(f)
        assert "baseline_loss"  in d
        assert "module_scores"  in d
        assert len(d["module_scores"]) > 0

    print("✓ def1 独立运行：仅生成 def1_rollback_loss.json，结构正确")


def test_runner_def2_only():
    """仅 def2 → def2_rollback_acc.json，不生成 def1/def3。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end("def2", tmpdir, steps=5)
        assert "def2_rollback_acc.json"  in files, f"缺少 def2，实际: {files}"
        assert "def1_rollback_loss.json" not in files
        assert "def3_path_integral.json" not in files

        with open(os.path.join(tmpdir, "def2_rollback_acc.json")) as f:
            d = json.load(f)
        assert "baseline_acc"   in d
        assert "module_scores"  in d
        assert "primary_metric" in d

    print("✓ def2 独立运行：仅生成 def2_rollback_acc.json，结构正确")


def test_runner_def3_only():
    """仅 def3 → def3_path_integral.json，steps_collected 正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def3", tmpdir, steps=6,
            metric_kwargs={"def3": {"log_every": 1}},
        )
        assert "def3_path_integral.json" in files, f"缺少 def3，实际: {files}"
        assert "def1_rollback_loss.json" not in files
        assert "def2_rollback_acc.json"  not in files

        with open(os.path.join(tmpdir, "def3_path_integral.json")) as f:
            d = json.load(f)
        assert d["steps_collected"] == 6, (
            f"期望 steps_collected=6，实际={d['steps_collected']}"
        )

    print("✓ def3 独立运行：def3_path_integral.json，steps_collected=6 正确")


def test_runner_all_three():
    """三种指标同时运行：生成 3 个独立 JSON，互不干扰。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def1,def2,def3", tmpdir, steps=5,
            metric_kwargs={"def3": {"log_every": 1}},
        )
        assert len(files) == 3, f"期望3个JSON，实际: {files}"
        for expected in (
            "def1_rollback_loss.json",
            "def2_rollback_acc.json",
            "def3_path_integral.json",
        ):
            assert expected in files, f"缺少 {expected}"

        # def1 的 module_scores 与 def2 的 module_scores 键集合相同
        with open(os.path.join(tmpdir, "def1_rollback_loss.json")) as f:
            d1 = json.load(f)
        with open(os.path.join(tmpdir, "def2_rollback_acc.json")) as f:
            d2 = json.load(f)
        assert set(d1["module_scores"].keys()) == set(d2["module_scores"].keys()), (
            "def1 和 def2 的模块键集合不一致"
        )

    print("✓ 三种指标同时运行：3 个 JSON 独立生成，模块键集合一致")


# ---------------------------------------------------------------------------
# metric_kwargs 透传
# ---------------------------------------------------------------------------

def test_runner_metric_kwargs_log_every():
    """metric_kwargs 中 def3 的 log_every 正确透传到 PathIntegralCallback。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_end_to_end(
            "def3", tmpdir, steps=9,
            metric_kwargs={"def3": {"log_every": 3}},
        )
        with open(os.path.join(tmpdir, "def3_path_integral.json")) as f:
            d = json.load(f)
        assert d["log_every"] == 3
        assert d["steps_collected"] == 3, (
            f"log_every=3，9步，期望 steps_collected=3，实际={d['steps_collected']}"
        )
    print("✓ metric_kwargs log_every=3 透传正确，steps_collected=3")


# ---------------------------------------------------------------------------
# head_granularity
# ---------------------------------------------------------------------------

def test_runner_head_granularity_flag():
    """from_str 正确接收并存储 head_granularity 标志。"""
    r_no_head   = TrainingGainRunner.from_str("def1,def3", head_granularity=False)
    r_with_head = TrainingGainRunner.from_str("def1,def3", head_granularity=True)
    assert r_no_head.head_granularity   is False
    assert r_with_head.head_granularity is True
    print("✓ from_str 正确传递 head_granularity 标志")


def test_runner_head_granularity_all():
    """
    head_granularity=True 时，def1/def2/def3 生成的 JSON 均含 head_scores。
    使用 TinyHFClassifier（有 model.config）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def1,def2,def3", tmpdir, steps=5,
            metric_kwargs={"def3": {"log_every": 1}},
            head_granularity=True,
            use_hf_model=True,
        )
        for fname in (
            "def1_rollback_loss.json",
            "def2_rollback_acc.json",
            "def3_path_integral.json",
        ):
            assert fname in files, f"{fname} 未生成"
            with open(os.path.join(tmpdir, fname)) as f:
                d = json.load(f)
            assert "head_scores" in d, f"{fname} 缺少 head_scores"
            assert len(d["head_scores"]) > 0

    print("✓ head_granularity=True：全部3个JSON均含 head_scores")


def test_runner_head_granularity_no_config():
    """
    head_granularity=True 但模型无 config（TinyClassifier）时，
    不崩溃，JSON 不含 head_scores（安全降级）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def1,def2,def3", tmpdir, steps=5,
            metric_kwargs={"def3": {"log_every": 1}},
            head_granularity=True,
            use_hf_model=False,  # TinyClassifier，无 config
        )
        for fname in (
            "def1_rollback_loss.json",
            "def2_rollback_acc.json",
            "def3_path_integral.json",
        ):
            with open(os.path.join(tmpdir, fname)) as f:
                d = json.load(f)
            assert "head_scores" not in d, (
                f"{fname}: 无 config 时不应含 head_scores"
            )

    print("✓ 无 config 时 head_granularity 安全降级（不崩溃，不含 head_scores）")


# ---------------------------------------------------------------------------
# 错误处理：def1/def2 缺少 eval_fn
# ---------------------------------------------------------------------------

def test_runner_missing_eval_fn():
    """def1/def2 未提供 eval_fn 时应抛出 ValueError。"""
    model = TinyClassifier()
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = TrainingGainRunner.from_str("def1")
        try:
            runner.make_callbacks(model, save_dir=tmpdir, eval_fn=None)
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "eval_fn" in str(e)
    print("✓ 缺少 eval_fn 时正确抛出 ValueError")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试 TrainingGainRunner")
    print("=" * 65)
    test_runner_registry()
    test_runner_from_str_single()
    test_runner_from_str_all()
    test_runner_unknown_metric()
    test_runner_available_metrics()
    test_runner_def12_merged_callback()
    test_runner_all_three_callback_count()
    test_runner_def1_only()
    test_runner_def2_only()
    test_runner_def3_only()
    test_runner_all_three()
    test_runner_metric_kwargs_log_every()
    test_runner_missing_eval_fn()
    print()
    print("── 头级别（head_granularity）测试 ──")
    test_runner_head_granularity_flag()
    test_runner_head_granularity_all()
    test_runner_head_granularity_no_config()
    print("\n所有测试通过 ✓")
