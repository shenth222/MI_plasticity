"""
metric/actual_update/test/test_runner.py

ActualUpdateRunner — 组合运行器测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.actual_update.test.test_runner

覆盖点：
  1. from_str 正确解析逗号分隔字符串
  2. 未知指标名称抛出 ValueError
  3. make_callbacks 返回正确数量的 callback
  4. 单一指标（def1/def2/def3）端到端：训练后生成对应 JSON
  5. 全部三种指标组合端到端：生成三个独立 JSON，互不干扰
  6. metric_kwargs 正确透传（def3 的 log_every）
"""

import json
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.actual_update.runner import ActualUpdateRunner, REGISTRY
from metric.actual_update.test.conftest import (
    TinyClassifier, make_fake_dataloader, train_n_steps, fire_train_end,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _run_end_to_end(metrics_str, tmpdir, steps=5, metric_kwargs=None):
    """
    完整模拟：构造 runner → make_callbacks → 训练 → on_train_end → 返回生成文件列表。
    """
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)

    runner    = ActualUpdateRunner.from_str(metrics_str, metric_kwargs=metric_kwargs)
    callbacks = runner.make_callbacks(model, save_dir=tmpdir)

    args, state, control = train_n_steps(model, dl, callbacks=callbacks, steps=steps)
    fire_train_end(callbacks, model, args, state, control)

    return [f for f in os.listdir(tmpdir) if f.endswith(".json")]


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

def test_runner_registry():
    """REGISTRY 包含所有预期指标。"""
    assert set(REGISTRY.keys()) == {"def1", "def2", "def3"}
    print(f"✓ REGISTRY 包含 {sorted(REGISTRY.keys())}")


def test_runner_from_str_single():
    """from_str 正确解析单个指标。"""
    r = ActualUpdateRunner.from_str("def1")
    assert r.selected_metrics == ["def1"]
    print("✓ from_str 单指标解析正确")


def test_runner_from_str_all():
    """from_str 正确解析逗号分隔的多指标（含空格）。"""
    r = ActualUpdateRunner.from_str("def1, def2 , def3")
    assert set(r.selected_metrics) == {"def1", "def2", "def3"}
    print("✓ from_str 多指标解析正确（容忍空格）")


def test_runner_unknown_metric():
    """未知指标名称应抛出 ValueError 并提示可用指标。"""
    try:
        ActualUpdateRunner(metrics=["def99"])
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "def99" in str(e)
        assert "Available" in str(e)
    print("✓ 未知指标正确抛出 ValueError")


def test_runner_make_callbacks_count():
    """make_callbacks 返回的 callback 数量等于指标数量。"""
    model = TinyClassifier()
    with tempfile.TemporaryDirectory() as tmpdir:
        r = ActualUpdateRunner.from_str("def1,def2,def3")
        cbs = r.make_callbacks(model, save_dir=tmpdir)
        assert len(cbs) == 3, f"期望3个callback，实际{len(cbs)}"
    print("✓ make_callbacks 返回3个 callback（def1+def2+def3）")


def test_runner_def1_only():
    """仅 def1 端到端：生成 def1_absolute.json，不生成其他文件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end("def1", tmpdir)
        assert "def1_absolute.json" in files, f"缺少 def1_absolute.json，实际: {files}"
        assert "def2_relative.json" not in files, "不应生成 def2 文件"
        assert "def3_path_length.json" not in files, "不应生成 def3 文件"

        with open(os.path.join(tmpdir, "def1_absolute.json")) as f:
            d = json.load(f)
        assert "module_scores" in d and "weight_only_scores" in d

    print("✓ def1 独立运行：仅生成 def1_absolute.json，结构正确")


def test_runner_def2_only():
    """仅 def2 端到端：生成 def2_relative.json。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end("def2", tmpdir)
        assert "def2_relative.json" in files

        with open(os.path.join(tmpdir, "def2_relative.json")) as f:
            d = json.load(f)
        assert "module_scores" in d and "abs_module_scores" in d and "epsilon" in d

    print("✓ def2 独立运行：仅生成 def2_relative.json，结构正确")


def test_runner_def3_only():
    """仅 def3 端到端：生成 def3_path_length.json，steps_collected 正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def3", tmpdir, steps=6,
            metric_kwargs={"def3": {"log_every": 1}},
        )
        assert "def3_path_length.json" in files

        with open(os.path.join(tmpdir, "def3_path_length.json")) as f:
            d = json.load(f)
        assert d["steps_collected"] == 6, (
            f"期望 steps_collected=6，实际={d['steps_collected']}"
        )

    print("✓ def3 独立运行：生成 def3_path_length.json，steps_collected=6 正确")


def test_runner_all_three():
    """三种指标同时运行：生成3个独立 JSON，互不干扰。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def1,def2,def3", tmpdir, steps=5,
            metric_kwargs={"def3": {"log_every": 1}},
        )
        assert len(files) == 3, f"期望3个JSON，实际: {files}"
        for expected in ("def1_absolute.json", "def2_relative.json", "def3_path_length.json"):
            assert expected in files, f"缺少 {expected}"

        # 交叉验证：def1 的 module_scores == def2 的 abs_module_scores
        with open(os.path.join(tmpdir, "def1_absolute.json")) as f:
            d1 = json.load(f)
        with open(os.path.join(tmpdir, "def2_relative.json")) as f:
            d2 = json.load(f)
        for m_name in d1["module_scores"]:
            assert abs(d1["module_scores"][m_name] - d2["abs_module_scores"][m_name]) < 1e-6, (
                f"{m_name}: def1={d1['module_scores'][m_name]:.8f}, "
                f"def2分子={d2['abs_module_scores'][m_name]:.8f}"
            )

    print("✓ 三种指标同时运行：3个JSON独立生成，def1×def2交叉验证通过")


def test_runner_metric_kwargs_log_every():
    """metric_kwargs 中 def3 的 log_every 正确透传。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = _run_end_to_end(
            "def3", tmpdir, steps=9,
            metric_kwargs={"def3": {"log_every": 3}},
        )
        with open(os.path.join(tmpdir, "def3_path_length.json")) as f:
            d = json.load(f)
        assert d["log_every"] == 3
        # step=3,6,9 → 3次
        assert d["steps_collected"] == 3, (
            f"log_every=3, 9步，期望 steps_collected=3，实际={d['steps_collected']}"
        )

    print("✓ metric_kwargs log_every=3 透传正确，steps_collected=3")


def test_runner_available_metrics():
    """available_metrics 属性返回所有注册指标的有序列表。"""
    r = ActualUpdateRunner.from_str("def1")
    assert r.available_metrics == ["def1", "def2", "def3"]
    print(f"✓ available_metrics = {r.available_metrics}")


# ---------------------------------------------------------------------------
# 头级别（head_granularity）测试（使用 TinyHFClassifier）
# ---------------------------------------------------------------------------

def test_runner_head_granularity_from_str():
    """from_str 正确接收并存储 head_granularity 标志。"""
    r_no_head  = ActualUpdateRunner.from_str("def1,def2", head_granularity=False)
    r_with_head = ActualUpdateRunner.from_str("def1,def2", head_granularity=True)
    assert r_no_head.head_granularity  is False
    assert r_with_head.head_granularity is True
    print("✓ from_str 正确传递 head_granularity 标志")


def test_runner_head_granularity_all():
    """
    head_granularity=True 时，三种指标均生成含 head_scores 的 JSON。
    使用 TinyHFClassifier（有 config）。
    """
    from metric.actual_update.test.conftest import TinyHFClassifier, TinyConfig

    cfg   = TinyConfig(hidden_size=8, num_attention_heads=2)
    model = TinyHFClassifier(cfg)
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ActualUpdateRunner.from_str(
            "def1,def2,def3",
            metric_kwargs={"def3": {"log_every": 1}},
            head_granularity=True,
        )
        callbacks = runner.make_callbacks(model, save_dir=tmpdir)
        args, state, control = train_n_steps(model, dl, callbacks=callbacks, steps=5)
        fire_train_end(callbacks, model, args, state, control)

        for fname in ("def1_absolute.json", "def2_relative.json", "def3_path_length.json"):
            path = os.path.join(tmpdir, fname)
            assert os.path.exists(path), f"{fname} 未生成"
            with open(path) as f:
                d = json.load(f)
            assert "head_scores" in d, f"{fname} 缺少 head_scores"
            assert len(d["head_scores"]) > 0, f"{fname}.head_scores 为空"
            # 验证头数量
            for m_name, per_head in d["head_scores"].items():
                assert len(per_head) == cfg.num_attention_heads, (
                    f"{fname}.{m_name}: head 数量={len(per_head)}，"
                    f"期望={cfg.num_attention_heads}"
                )

    print(
        f"✓ head_granularity=True：全部3个JSON均含 head_scores，"
        f"每模块 {cfg.num_attention_heads} 头"
    )


def test_runner_head_granularity_per_metric_override():
    """
    metric_kwargs 中单独设置某个指标的 head_granularity，可覆盖全局设置。
    例：全局 False，但 def1 通过 metric_kwargs 设置 True。
    """
    from metric.actual_update.test.conftest import TinyHFClassifier

    model = TinyHFClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ActualUpdateRunner.from_str(
            "def1,def2",
            metric_kwargs={
                "def1": {"head_granularity": True},   # 单独开启
                # def2 不设置，将继承全局 False
            },
            head_granularity=False,   # 全局 False
        )
        callbacks = runner.make_callbacks(model, save_dir=tmpdir)
        args, state, control = train_n_steps(model, dl, callbacks=callbacks, steps=5)
        fire_train_end(callbacks, model, args, state, control)

        # def1 应有 head_scores（metric_kwargs 覆盖全局）
        with open(os.path.join(tmpdir, "def1_absolute.json")) as f:
            d1 = json.load(f)
        assert "head_scores" in d1, "def1 metric_kwargs 覆盖 head_granularity=True 未生效"

        # def2 不应有 head_scores（继承全局 False）
        with open(os.path.join(tmpdir, "def2_relative.json")) as f:
            d2 = json.load(f)
        assert "head_scores" not in d2, "def2 应继承全局 head_granularity=False"

    print("✓ metric_kwargs 可单独覆盖 head_granularity（def1=True, def2=False）")


def test_runner_head_granularity_no_config():
    """
    head_granularity=True 但模型无 config（TinyClassifier）时，
    不崩溃，JSON 不含 head_scores（降级处理）。
    """
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ActualUpdateRunner.from_str("def1,def2", head_granularity=True)
        callbacks = runner.make_callbacks(model, save_dir=tmpdir)
        args, state, control = train_n_steps(model, dl, callbacks=callbacks, steps=5)
        fire_train_end(callbacks, model, args, state, control)

        for fname in ("def1_absolute.json", "def2_relative.json"):
            with open(os.path.join(tmpdir, fname)) as f:
                d = json.load(f)
            assert "head_scores" not in d, (
                f"{fname}: 无 config 时不应含 head_scores"
            )

    print("✓ 无 config 时 head_granularity 安全降级（不崩溃，不含 head_scores）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试 ActualUpdateRunner")
    print("=" * 65)
    test_runner_registry()
    test_runner_from_str_single()
    test_runner_from_str_all()
    test_runner_unknown_metric()
    test_runner_make_callbacks_count()
    test_runner_def1_only()
    test_runner_def2_only()
    test_runner_def3_only()
    test_runner_all_three()
    test_runner_metric_kwargs_log_every()
    test_runner_available_metrics()
    print()
    print("── 头级别（head_granularity）测试 ──")
    test_runner_head_granularity_from_str()
    test_runner_head_granularity_all()
    test_runner_head_granularity_per_metric_override()
    test_runner_head_granularity_no_config()
    print("\n所有测试通过 ✓")
