"""
metric/actual_update/test/test_def3.py

定义三：累计路径长度（PathLengthMetric）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.actual_update.test.test_def3

覆盖点：
  1. 路径长度 ≥ 直线距离（三角不等式验证）
  2. 路径长度单调非减（每步只能增加，不能减小）
  3. 零步时路径长度为零
  4. steps_collected 字段与实际训练步数一致（log_every=1）
  5. log_every > 1 时 steps_collected = total_steps // log_every
  6. save / load 往返一致
  7. callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成
  8. 打印 Top-k 模块（供人工核验）
"""

import json
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.actual_update.def3_path_length import PathLengthMetric, PathLengthCallback
from metric.actual_update.def1_absolute import AbsoluteUpdateMetric
from metric.actual_update.base import snapshot_params
from metric.actual_update.test.conftest import (
    TinyClassifier, make_fake_dataloader,
    FakeArgs, FakeState, FakeControl,
    train_n_steps, fire_train_end,
)


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

def test_def3_path_geq_direct():
    """
    路径长度 ≥ 直线距离（三角不等式）。
    U_m^path = Σ_t ||θ^(t) - θ^(t-1)|| ≥ ||θ^(T) - θ^(0)|| = U_m^(A)
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    device = torch.device("cpu")

    # 记录 θ^(0)（用于定义一）
    theta0 = snapshot_params(model)

    # 创建定义三 callback 并训练
    metric3 = PathLengthMetric()
    cb = metric3.make_callback(model, save_dir="/tmp", log_every=1)
    train_n_steps(model, dl, callbacks=[cb], steps=10, lr=1e-2)

    # 用训练后模型计算定义一（直线距离）
    result1 = AbsoluteUpdateMetric().compute(theta0, model, device)

    for m_name in result1["module_scores"]:
        direct = result1["module_scores"][m_name]      # ||θ^(T) - θ^(0)||_2
        path   = cb._module_acc.get(m_name, 0.0)       # U_m^path
        assert path >= direct - 1e-9, (
            f"{m_name}: 路径长度({path:.8f}) < 直线距离({direct:.8f})，"
            f"违反三角不等式"
        )
    print("✓ 所有模块路径长度 ≥ 直线距离（三角不等式验证通过）")


def test_def3_monotone_non_decreasing():
    """路径长度每步单调非减（步进贡献 ≥ 0）。"""
    model = TinyClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=16)

    metric = PathLengthMetric()
    cb = metric.make_callback(model, save_dir="/tmp")

    args    = FakeArgs()
    state   = FakeState()
    control = FakeControl()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    data_iter = iter(dl)

    prev_acc = {m: 0.0 for m in cb._module_acc}

    for step in range(10):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        optimizer.zero_grad()
        model(**batch).loss.backward()
        optimizer.step()
        cb.on_step_end(args, state, control, model=model)

        for m_name, val in cb._module_acc.items():
            assert val >= prev_acc[m_name] - 1e-9, (
                f"步 {step+1}: {m_name} 路径长度从 {prev_acc[m_name]:.8f} "
                f"减小至 {val:.8f}"
            )
        prev_acc = dict(cb._module_acc)

    print("✓ 路径长度单调非减（10步验证通过）")


def test_def3_zero_before_training():
    """未进行任何训练步时，路径长度应全为零。"""
    model  = TinyClassifier()
    metric = PathLengthMetric()
    cb = metric.make_callback(model, save_dir="/tmp")

    # 不调用 on_step_end，路径长度应全为零
    for m_name, val in cb._module_acc.items():
        assert abs(val) < 1e-12, f"{m_name} 初始路径长度不为零: {val}"
    assert cb._steps_collected == 0
    print("✓ 训练前路径长度全为零，steps_collected=0")


def test_def3_steps_collected_exact():
    """log_every=1 时，steps_collected 应等于实际训练步数。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    metric = PathLengthMetric()
    cb = metric.make_callback(model, save_dir="/tmp", log_every=1)

    train_n_steps(model, dl, callbacks=[cb], steps=7)

    assert cb._steps_collected == 7, (
        f"期望 steps_collected=7，实际={cb._steps_collected}"
    )
    print(f"✓ log_every=1 时 steps_collected={cb._steps_collected}（等于训练步数7）")


def test_def3_log_every():
    """
    log_every=N 时，steps_collected 应为 total_steps // N（精确整除时）。
    具体：9 步，log_every=3，step=3,6,9 时计算 → collected=3。
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=20)
    metric = PathLengthMetric()
    cb = metric.make_callback(model, save_dir="/tmp", log_every=3)

    train_n_steps(model, dl, callbacks=[cb], steps=9)

    assert cb._steps_collected == 3, (
        f"log_every=3, 9步，期望 steps_collected=3，实际={cb._steps_collected}"
    )
    print(f"✓ log_every=3，9步 → steps_collected={cb._steps_collected}（正确）")


def test_def3_log_every_path_approx():
    """
    log_every > 1 时路径长度为近似值（不精确），但仍应满足非负和非减。
    同时验证其始终 ≤ log_every=1 的精确路径长度（因为跳过了中间步）。
    实际上近似值 ≤ 精确值并不保证，这里仅验证近似结果非负。
    """
    model   = TinyClassifier()
    dl      = make_fake_dataloader(batch_size=4, num_batches=16)

    metric  = PathLengthMetric()
    cb_approx = metric.make_callback(model, save_dir="/tmp", log_every=5)
    train_n_steps(model, dl, callbacks=[cb_approx], steps=10, lr=1e-2)

    for m_name, val in cb_approx._module_acc.items():
        assert val >= 0, f"{m_name} 近似路径长度为负: {val}"

    print(
        f"✓ log_every=5 近似路径长度非负，steps_collected={cb_approx._steps_collected}"
    )


def test_def3_save_load():
    """on_train_end 保存 JSON 后，加载值与累积器一致。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    metric = PathLengthMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, log_every=1)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        assert os.path.exists(json_path), "JSON 文件未生成"

        with open(json_path) as f:
            loaded = json.load(f)

        assert loaded["steps_collected"] == 5
        assert loaded["log_every"] == 1

        # 验证几个模块的值与 callback 内部累积器一致
        for m_name in list(cb._module_acc.keys())[:3]:
            assert abs(loaded["module_scores"][m_name] - cb._module_acc[m_name]) < 1e-9

    print("✓ save/load 往返一致（JSON 值与累积器完全匹配）")


def test_def3_callback_end_to_end():
    """callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成及字段校验。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    metric = PathLengthMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, log_every=1)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=8)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            saved = json.load(f)

        for key in ("module_scores", "param_scores", "steps_collected", "log_every"):
            assert key in saved, f"JSON 缺少字段: {key}"

        assert saved["steps_collected"] == 8
        nonzero = sum(1 for v in saved["module_scores"].values() if v > 0)
        assert nonzero > 0

    print(f"✓ callback 端到端：JSON 已生成，{nonzero} 个模块路径长度 > 0")


def test_def3_print_topk(top_k: int = 5):
    """打印路径长度 Top-k 模块，同时展示与直线距离的对比（供人工核验）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    metric = PathLengthMetric()
    cb = metric.make_callback(model, save_dir="/tmp")
    train_n_steps(model, dl, callbacks=[cb], steps=20, lr=1e-2)

    result1 = AbsoluteUpdateMetric().compute(theta0, model, device)

    sorted_modules = sorted(
        cb._module_acc.keys(),
        key=lambda k: cb._module_acc[k],
        reverse=True,
    )

    print(f"\n定义三：路径长度 Top-{top_k} 模块（steps=20, lr=1e-2）：")
    print(f"{'模块名':<55} {'路径长度':>14} {'直线距离':>12} {'比值':>8}")
    print("-" * 93)
    for name in sorted_modules[:top_k]:
        path   = cb._module_acc[name]
        direct = result1["module_scores"].get(name, 0.0)
        ratio  = path / (direct + 1e-12)
        print(f"{name:<55} {path:>14.8f} {direct:>12.8f} {ratio:>8.3f}×")


# ---------------------------------------------------------------------------
# 头级别测试（使用 TinyHFClassifier）
# ---------------------------------------------------------------------------

def test_def3_head_granularity_structure():
    """
    head_granularity=True 时，on_train_end 保存的 JSON 含 head_scores 字段，
    结构 {module_name: {"head_0": float, "head_1": float, ...}}，
    头数量等于 config.num_attention_heads。
    """
    from metric.actual_update.test.conftest import TinyHFClassifier, TinyConfig

    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathLengthMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, head_granularity=True)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        with open(json_path) as f:
            saved = json.load(f)

    assert "head_scores" in saved, "head_granularity=True 时 JSON 缺少 head_scores"
    assert len(saved["head_scores"]) > 0

    for m_name, per_head in saved["head_scores"].items():
        assert len(per_head) == cfg.num_attention_heads, (
            f"{m_name}: 头数量={len(per_head)}，期望={cfg.num_attention_heads}"
        )
        for h in range(cfg.num_attention_heads):
            key = f"head_{h}"
            assert key in per_head
            assert per_head[key] >= 0, f"{m_name}.{key} 头路径长度为负: {per_head[key]}"

    print(
        f"✓ head_granularity（def3）: {len(saved['head_scores'])} 个注意力模块，"
        f"每模块 {cfg.num_attention_heads} 头，结构正确"
    )


def test_def3_head_granularity_absent_by_default():
    """head_granularity=False（默认）时，JSON 不含 head_scores。"""
    from metric.actual_update.test.conftest import TinyHFClassifier

    model  = TinyHFClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathLengthMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, head_granularity=False)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        with open(json_path) as f:
            saved = json.load(f)

    assert "head_scores" not in saved, "head_granularity=False 时不应含 head_scores"
    print("✓ head_granularity=False 时 JSON 不含 head_scores（正确）")


def test_def3_head_granularity_monotone():
    """头级别路径长度随训练步数单调非减（逐步验证累积器不减）。"""
    from metric.actual_update.test.conftest import TinyHFClassifier, TinyConfig

    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    metric = PathLengthMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", head_granularity=True)

    args    = FakeArgs()
    state   = FakeState()
    control = FakeControl()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    data_iter = iter(dl)

    # 初始头级别累积器快照
    prev_head = {
        m: dict(per_h)
        for m, per_h in cb._head_acc.items()
    }

    for step in range(8):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        optimizer.zero_grad()
        model(**batch).loss.backward()
        optimizer.step()
        cb.on_step_end(args, state, control, model=model)

        for m_name, per_h in cb._head_acc.items():
            for h, val in per_h.items():
                prev_val = prev_head[m_name][h]
                assert val >= prev_val - 1e-9, (
                    f"步 {step+1}: {m_name}.head_{h} 路径从 {prev_val:.8f} 减小至 {val:.8f}"
                )
        prev_head = {m: dict(ph) for m, ph in cb._head_acc.items()}

    print(f"✓ 头级别路径长度单调非减（8步验证通过）")


def test_def3_head_granularity_print(top_k: int = 3):
    """打印头级别路径长度（供人工核验）。"""
    from metric.actual_update.test.conftest import TinyHFClassifier, TinyConfig

    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    metric = PathLengthMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", head_granularity=True)
    train_n_steps(model, dl, callbacks=[cb], steps=10, lr=1e-2)

    print(f"\n定义三头级别路径长度 Top-{top_k} 注意力模块（steps=10）：")
    head_items = sorted(
        cb._head_acc.items(),
        key=lambda kv: cb._module_acc.get(kv[0], 0.0),
        reverse=True,
    )[:top_k]
    for m_name, per_h in head_items:
        mod_path = cb._module_acc.get(m_name, 0.0)
        head_str = "  ".join(f"h{h}={v:.6f}" for h, v in per_h.items())
        print(f"  {m_name:<60} module={mod_path:.6f}  {head_str}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试定义三：累计路径长度（PathLengthMetric）")
    print("=" * 65)
    test_def3_path_geq_direct()
    test_def3_monotone_non_decreasing()
    test_def3_zero_before_training()
    test_def3_steps_collected_exact()
    test_def3_log_every()
    test_def3_log_every_path_approx()
    test_def3_save_load()
    test_def3_callback_end_to_end()
    test_def3_print_topk()
    print()
    print("── 头级别（head_granularity）测试 ──")
    test_def3_head_granularity_structure()
    test_def3_head_granularity_absent_by_default()
    test_def3_head_granularity_monotone()
    test_def3_head_granularity_print()
    print("\n所有测试通过 ✓")
