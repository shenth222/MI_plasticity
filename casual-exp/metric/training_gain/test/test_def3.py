"""
metric/training_gain/test/test_def3.py

定义三：路径积分（PathIntegralGainMetric）—— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.training_gain.test.test_def3

覆盖点：
  1.  训练前累积器全为零
  2.  steps_collected 与实际训练步数一致（log_every=1）
  3.  log_every > 1 时 steps_collected = total_steps // log_every
  4.  SGD 下 G_m^PI ≤ 0（Δθ = -α·g → g·Δθ = -α||g||² ≤ 0，符号性质严格成立）
  5.  param_scores 到 module_scores 的聚合一致性
  6.  save/load 往返一致（JSON 值与累积器匹配）
  7.  callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成
  8.  head_granularity=True 时 head_scores 结构正确
  9.  head_granularity=False（默认）时不含 head_scores
  10. callback + head_granularity=True 端到端
  11. 打印 Top-k 模块（|G_m^PI| 降序，供人工核验）
"""

import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.training_gain.def3_path_integral import PathIntegralGainMetric, PathIntegralCallback
from metric.training_gain.test.conftest import (
    TinyClassifier,
    TinyHFClassifier,
    TinyConfig,
    make_fake_dataloader,
    FakeArgs,
    FakeState,
    FakeControl,
    train_n_steps,
    fire_train_end,
)


# ---------------------------------------------------------------------------
# 单元测试：训练前状态
# ---------------------------------------------------------------------------

def test_def3_zero_before_training():
    """未进行任何训练步时，所有累积器应为零，steps_collected=0。"""
    model  = TinyClassifier()
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp")

    for m_name, val in cb._module_acc.items():
        assert abs(val) < 1e-12, f"{m_name}: 初始路径积分不为零: {val}"
    for p_name, val in cb._param_acc.items():
        assert abs(val) < 1e-12, f"{p_name}: 初始参数路径积分不为零: {val}"
    assert cb._steps_collected == 0
    print("✓ 训练前：所有累积器为零，steps_collected=0")


# ---------------------------------------------------------------------------
# 单元测试：steps_collected 计数
# ---------------------------------------------------------------------------

def test_def3_steps_collected_exact():
    """log_every=1 时，steps_collected 应等于实际训练步数。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", log_every=1)

    train_n_steps(model, dl, callbacks=[cb], steps=7)

    assert cb._steps_collected == 7, (
        f"期望 steps_collected=7，实际={cb._steps_collected}"
    )
    print(f"✓ log_every=1：7 步训练后 steps_collected={cb._steps_collected}")


def test_def3_log_every():
    """log_every=3，9 步训练 → steps_collected=3（step=3,6,9 时计算）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=20)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", log_every=3)

    train_n_steps(model, dl, callbacks=[cb], steps=9)

    assert cb._steps_collected == 3, (
        f"log_every=3, 9步，期望 steps_collected=3，实际={cb._steps_collected}"
    )
    print(f"✓ log_every=3，9步 → steps_collected={cb._steps_collected}（step=3,6,9）")


# ---------------------------------------------------------------------------
# 单元测试：SGD 下路径积分符号严格 ≤ 0
# ---------------------------------------------------------------------------

def test_def3_sign_with_sgd():
    """
    使用 SGD（无动量、无权重衰减）时，路径积分符号严格满足 G_m^PI ≤ 0。

    证明：
      SGD：Δθ_t = −α · g_t
      → g_t · Δθ_t = g_t · (−α · g_t) = −α · ||g_t||² ≤ 0（等号仅在 g=0 时成立）
      → G_m^PI = Σ_t g_t · Δθ_t ≤ 0

    此测试专门使用 use_sgd=True 以保证符号严格成立。
    （AdamW 中 Δθ ≠ −α·g，不保证每步符号，但累积和通常 ≤ 0）
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", log_every=1)

    # 使用 SGD（精确符号验证）
    train_n_steps(model, dl, callbacks=[cb], steps=10, lr=1e-2, use_sgd=True)

    for m_name, val in cb._module_acc.items():
        assert val <= 1e-9, (
            f"{m_name}: SGD 下 G_m^PI = {val:.8f} > 0，违反 g·Δθ ≤ 0"
        )
    nonzero = sum(1 for v in cb._module_acc.values() if v < -1e-12)
    assert nonzero > 0, "所有模块路径积分为零，梯度可能未正确传播"
    print(f"✓ SGD 下路径积分全部 ≤ 0（{nonzero}/{len(cb._module_acc)} 个模块严格 < 0）")


# ---------------------------------------------------------------------------
# 单元测试：param_scores 到 module_scores 聚合一致性
# ---------------------------------------------------------------------------

def test_def3_param_module_consistency():
    """
    module_acc[m] 应等于该模块所有参数的 param_acc 之和（非 L2，直接加法）。

    公式：G_m^PI = Σ_{p ∈ θ_m} G_p^PI
    """
    from metric.training_gain.base import group_params_by_module

    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", log_every=1)

    train_n_steps(model, dl, callbacks=[cb], steps=5)

    groups = group_params_by_module(model)
    for m_name, param_names in groups.items():
        recomputed = sum(cb._param_acc.get(pn, 0.0) for pn in param_names)
        stored     = cb._module_acc.get(m_name, 0.0)
        assert abs(recomputed - stored) < 1e-9, (
            f"{m_name}: 重算={recomputed:.8f}, 存储={stored:.8f}"
        )
    print("✓ module_scores = Σ param_scores 聚合一致性验证通过")


# ---------------------------------------------------------------------------
# 单元测试：save/load 往返一致
# ---------------------------------------------------------------------------

def test_def3_save_load():
    """on_train_end 保存 JSON 后，加载值与累积器完全一致。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathIntegralGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, log_every=1)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, "def3_path_integral.json")
        assert os.path.exists(json_path), "def3_path_integral.json 未生成"

        with open(json_path) as f:
            loaded = json.load(f)

        assert loaded["steps_collected"] == 5
        assert loaded["log_every"]       == 1

        # 验证模块值与累积器一致
        for m_name in list(cb._module_acc.keys())[:5]:
            assert abs(loaded["module_scores"][m_name] - cb._module_acc[m_name]) < 1e-9, (
                f"{m_name}: JSON={loaded['module_scores'][m_name]:.8f}, "
                f"累积器={cb._module_acc[m_name]:.8f}"
            )

    print("✓ save/load 往返一致（JSON 值与累积器完全匹配）")


# ---------------------------------------------------------------------------
# 单元测试：callback 端到端
# ---------------------------------------------------------------------------

def test_def3_callback_end_to_end():
    """callback 端到端：make_callback → 训练 → on_train_end → JSON 字段校验。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathIntegralGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, log_every=1)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=8)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, "def3_path_integral.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            saved = json.load(f)

        for key in ("module_scores", "param_scores", "steps_collected", "log_every"):
            assert key in saved, f"JSON 缺少字段: {key}"

        assert saved["steps_collected"] == 8
        # 至少有一些非零（有符号）分数（梯度下降中通常有负值）
        nonzero = sum(1 for v in saved["module_scores"].values() if abs(v) > 1e-12)
        assert nonzero > 0, "所有模块路径积分为零"

    print(f"✓ callback 端到端：JSON 已生成，{nonzero} 个模块路径积分非零")


# ---------------------------------------------------------------------------
# 单元测试：head_granularity
# ---------------------------------------------------------------------------

def test_def3_head_granularity_structure():
    """
    head_granularity=True 时，on_train_end 保存的 JSON 含 head_scores，
    结构 {module_name: {"head_0": float, ...}}，头数量等于 num_attention_heads。
    """
    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathIntegralGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, head_granularity=True)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, "def3_path_integral.json")
        with open(json_path) as f:
            saved = json.load(f)

    assert "head_scores" in saved, "head_granularity=True 时 JSON 缺少 head_scores"
    assert len(saved["head_scores"]) > 0

    for m_name, per_head in saved["head_scores"].items():
        assert len(per_head) == cfg.num_attention_heads, (
            f"{m_name}: 头数量={len(per_head)}，期望={cfg.num_attention_heads}"
        )
        for h in range(cfg.num_attention_heads):
            assert f"head_{h}" in per_head, f"{m_name} 缺少 head_{h}"

    print(
        f"✓ head_granularity：{len(saved['head_scores'])} 个注意力模块，"
        f"每模块 {cfg.num_attention_heads} 头，结构正确"
    )


def test_def3_head_granularity_absent_by_default():
    """head_granularity=False（默认）时，JSON 不含 head_scores。"""
    model  = TinyHFClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = PathIntegralGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, head_granularity=False)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, "def3_path_integral.json")
        with open(json_path) as f:
            saved = json.load(f)

    assert "head_scores" not in saved, "head_granularity=False 时不应含 head_scores"
    print("✓ head_granularity=False 时 JSON 不含 head_scores（正确）")


def test_def3_head_sgn_with_sgd():
    """
    head_granularity=True + SGD：各头路径积分应 ≤ 0（与模块级别相同的符号性质）。
    """
    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", head_granularity=True)

    train_n_steps(model, dl, callbacks=[cb], steps=10, lr=1e-2, use_sgd=True)

    for m_name, per_h in cb._head_acc.items():
        for h_idx, val in per_h.items():
            assert val <= 1e-9, (
                f"{m_name}.head_{h_idx}: SGD 下头路径积分={val:.8f} > 0"
            )
    print("✓ SGD 下头级别路径积分全部 ≤ 0")


# ---------------------------------------------------------------------------
# 打印（人工核验）
# ---------------------------------------------------------------------------

def test_def3_print_topk(top_k: int = 5):
    """按 |G_m^PI| 降序打印 Top-k 模块（供人工核验）。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", log_every=1)

    train_n_steps(model, dl, callbacks=[cb], steps=20, lr=1e-2, use_sgd=True)

    sorted_modules = sorted(
        cb._module_acc.keys(),
        key=lambda k: abs(cb._module_acc[k]),
        reverse=True,
    )

    total_pi = sum(cb._module_acc.values())
    print(f"\n定义三：路径积分 Top-{top_k} 模块（SGD，steps=20，lr=1e-2）：")
    print(f"  总 loss 变化近似（Σ G_m^PI = {total_pi:.6f}）")
    print(f"{'模块名':<55} {'G_m^PI':>14} {'|G_m^PI|':>12}")
    print("-" * 84)
    for name in sorted_modules[:top_k]:
        val = cb._module_acc[name]
        print(f"{name:<55} {val:>+14.8f} {abs(val):>12.8f}")


def test_def3_print_head_topk(top_k: int = 3):
    """打印头级别路径积分（供人工核验）。"""
    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=16)
    metric = PathIntegralGainMetric()
    cb     = metric.make_callback(model, save_dir="/tmp", head_granularity=True)

    train_n_steps(model, dl, callbacks=[cb], steps=10, lr=1e-2, use_sgd=True)

    head_items = sorted(
        cb._head_acc.items(),
        key=lambda kv: abs(cb._module_acc.get(kv[0], 0.0)),
        reverse=True,
    )[:top_k]

    print(f"\n定义三头级别路径积分 Top-{top_k} 注意力模块（SGD，steps=10）：")
    for m_name, per_h in head_items:
        mod_pi = cb._module_acc.get(m_name, 0.0)
        head_str = "  ".join(f"h{h}={v:+.6f}" for h, v in per_h.items())
        print(f"  {m_name:<60} module={mod_pi:+.6f}  {head_str}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试定义三：路径积分（PathIntegralGainMetric）")
    print("=" * 65)
    test_def3_zero_before_training()
    test_def3_steps_collected_exact()
    test_def3_log_every()
    test_def3_sign_with_sgd()
    test_def3_param_module_consistency()
    test_def3_save_load()
    test_def3_callback_end_to_end()
    test_def3_print_topk()
    print()
    print("── 头级别（head_granularity）测试 ──")
    test_def3_head_granularity_structure()
    test_def3_head_granularity_absent_by_default()
    test_def3_head_sgn_with_sgd()
    test_def3_print_head_topk()
    print("\n所有测试通过 ✓")
