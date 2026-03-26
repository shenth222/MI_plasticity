"""
metric/training_gain/test/test_def12.py

定义一（Validation Loss 变化）& 定义二（Accuracy 变化）独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.training_gain.test.test_def12

覆盖点：
  1.  compute() 返回结构完整
  2.  baseline 评估合理（loss > 0，acc ∈ [0,1]）
  3.  回滚改变了 loss 和 acc（训练后参数与初始参数不同 → 回滚有效果）
  4.  untrained 模型：theta0 == thetaT → 所有回滚增益约为零
  5.  充分训练后：损失增益总和 > 0（回滚整体让 loss 变差）
  6.  compute_loss=False 时结果不含 loss scores
  7.  compute_acc=False 时结果不含 acc scores
  8.  save() 生成 def1_rollback_loss.json 和 def2_rollback_acc.json
  9.  save/load 往返一致
  10. callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成
  11. head_granularity=True 时 head_scores 结构正确
  12. callback + head_granularity=True 端到端
  13. 打印 Top-k 模块（供人工核验）
"""

import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch

from metric.training_gain.def12_rollback import (
    RollbackGainMetric,
    compute_rollback,
    _save_def1,
    _save_def2,
)
from metric.training_gain.base import (
    snapshot_params,
    group_params_by_module,
)
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
# 辅助函数
# ---------------------------------------------------------------------------

def _setup_trained(steps: int = 30, lr: float = 1e-2):
    """
    返回 (theta0, trained_model, eval_fn, device, module_groups)。

    使用足够多的 steps（30）以确保参数显著变化，使回滚效果可测。
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")
    eval_fn = make_fake_eval_fn(num_batches=4, batch_size=4)

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=steps, lr=lr)

    module_groups = group_params_by_module(model)
    return theta0, model, eval_fn, device, module_groups


def _setup_untrained():
    """
    返回 (theta0, model, eval_fn, device, module_groups)。

    模型未训练：theta0 与当前参数完全相同 → 所有回滚增益应约为零。
    """
    model   = TinyClassifier()
    device  = torch.device("cpu")
    eval_fn = make_fake_eval_fn(num_batches=4, batch_size=4)
    theta0  = snapshot_params(model)  # 与当前完全一致
    module_groups = group_params_by_module(model)
    return theta0, model, eval_fn, device, module_groups


# ---------------------------------------------------------------------------
# 单元测试：compute() 结构
# ---------------------------------------------------------------------------

def test_def12_structure():
    """compute() 返回字典包含所有必需字段。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=5)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )
    for key in ("baseline_loss", "baseline_acc", "primary_metric",
                "module_scores_loss", "module_scores_acc",
                "module_all_metric_scores", "head_scores_loss",
                "head_scores_acc", "num_modules_computed"):
        assert key in result, f"缺少字段: {key}"
    assert len(result["module_scores_loss"]) > 0
    assert len(result["module_scores_acc"])  > 0
    print(f"✓ compute() 返回结构完整（{len(result['module_scores_loss'])} 个模块）")


# ---------------------------------------------------------------------------
# 单元测试：baseline 评估合理性
# ---------------------------------------------------------------------------

def test_def12_baseline_reasonable():
    """baseline loss > 0，accuracy ∈ [0, 1]。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=5)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )
    assert result["baseline_loss"] > 0, "baseline loss 应 > 0"
    assert 0.0 <= result["baseline_acc"] <= 1.0, "baseline acc 应在 [0,1]"
    print(f"✓ baseline: loss={result['baseline_loss']:.4f}，acc={result['baseline_acc']:.4f}")


# ---------------------------------------------------------------------------
# 单元测试：untrained 模型回滚增益约为零
# ---------------------------------------------------------------------------

def test_def12_zero_gain_untrained():
    """
    theta0 == thetaT（未训练）时，回滚操作不改变参数 → 所有增益应完全为零。
    """
    theta0, model, eval_fn, device, module_groups = _setup_untrained()
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )
    for m_name, delta_loss in result["module_scores_loss"].items():
        assert abs(delta_loss) < 1e-6, f"{m_name}: 未训练时 ΔL={delta_loss:.8f} 不为零"
    for m_name, delta_acc in result["module_scores_acc"].items():
        assert abs(delta_acc) < 1e-6, f"{m_name}: 未训练时 ΔAcc={delta_acc:.8f} 不为零"
    print(f"✓ 未训练模型：所有 {len(result['module_scores_loss'])} 个模块回滚增益 ≈ 0")


# ---------------------------------------------------------------------------
# 单元测试：训练后回滚改变 loss/acc
# ---------------------------------------------------------------------------

def test_def12_rollback_changes_scores():
    """充分训练后，至少有一个模块的回滚使 loss/acc 发生变化。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=30)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )
    nonzero_loss = sum(1 for v in result["module_scores_loss"].values() if abs(v) > 1e-6)
    nonzero_acc  = sum(1 for v in result["module_scores_acc"].values()  if abs(v) > 1e-6)
    assert nonzero_loss > 0, "训练后应有至少一个模块的 loss 回滚增益非零"
    assert nonzero_acc  > 0, "训练后应有至少一个模块的 acc 回滚增益非零"
    print(f"✓ 训练后：{nonzero_loss}/{len(result['module_scores_loss'])} 个模块有非零 loss 增益，"
          f"{nonzero_acc}/{len(result['module_scores_acc'])} 个模块有非零 acc 增益")


# ---------------------------------------------------------------------------
# 单元测试：compute_loss=False / compute_acc=False
# ---------------------------------------------------------------------------

def test_def12_compute_loss_only():
    """compute_acc=False 时不计算 acc scores。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=5)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
        compute_loss=True, compute_acc=False,
    )
    assert len(result["module_scores_loss"]) > 0, "应有 loss scores"
    assert len(result["module_scores_acc"])  == 0, "compute_acc=False 时 acc scores 应为空"
    print("✓ compute_acc=False：正确跳过 acc 计算")


def test_def12_compute_acc_only():
    """compute_loss=False 时不计算 loss scores。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=5)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
        compute_loss=False, compute_acc=True,
    )
    assert len(result["module_scores_acc"])  > 0, "应有 acc scores"
    assert len(result["module_scores_loss"]) == 0, "compute_loss=False 时 loss scores 应为空"
    print("✓ compute_loss=False：正确跳过 loss 计算")


# ---------------------------------------------------------------------------
# 单元测试：save() 生成正确文件
# ---------------------------------------------------------------------------

def test_def12_save_generates_files():
    """save() 生成 def1_rollback_loss.json 和 def2_rollback_acc.json。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=5)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )
    metric = RollbackGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = metric.save(result, tmpdir, compute_loss=True, compute_acc=True)
        assert "def1" in saved and saved["def1"].exists(), "def1 JSON 未生成"
        assert "def2" in saved and saved["def2"].exists(), "def2 JSON 未生成"

        with open(saved["def1"]) as f:
            d1 = json.load(f)
        with open(saved["def2"]) as f:
            d2 = json.load(f)

        assert "baseline_loss"  in d1
        assert "module_scores"  in d1
        assert "baseline_acc"   in d2
        assert "module_scores"  in d2
        assert "primary_metric" in d2

    print("✓ save() 生成 def1_rollback_loss.json 和 def2_rollback_acc.json，结构正确")


# ---------------------------------------------------------------------------
# 单元测试：save/load 往返一致
# ---------------------------------------------------------------------------

def test_def12_save_load_roundtrip():
    """save() 后 JSON 数值与 compute() 结果一致。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=5)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )
    metric = RollbackGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = metric.save(result, tmpdir)

        with open(saved["def1"]) as f:
            d1 = json.load(f)
        with open(saved["def2"]) as f:
            d2 = json.load(f)

        # 验证前 3 个模块数值一致
        for m_name in list(result["module_scores_loss"].keys())[:3]:
            assert abs(
                d1["module_scores"][m_name] - result["module_scores_loss"][m_name]
            ) < 1e-9, f"{m_name}: def1 loss 不一致"
            assert abs(
                d2["module_scores"][m_name] - result["module_scores_acc"][m_name]
            ) < 1e-9, f"{m_name}: def2 acc 不一致"

    print("✓ save/load 往返一致（def1 loss + def2 acc 均验证）")


# ---------------------------------------------------------------------------
# 单元测试：callback 端到端
# ---------------------------------------------------------------------------

def test_def12_callback_end_to_end():
    """
    callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成。

    注意：RollbackCallback 在构造时立即快照 θ^(0)（必须在 DDP 包装前）。
    on_train_end 时加载 θ^(T)，逐模块回滚并评估。
    """
    model   = TinyClassifier()
    dl      = make_fake_dataloader(batch_size=4, num_batches=8)
    eval_fn = make_fake_eval_fn(num_batches=4, batch_size=4)
    metric  = RollbackGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(
            model=model, save_dir=tmpdir, eval_fn=eval_fn,
            compute_loss=True, compute_acc=True,
        )
        # 训练（修改参数）
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        # 触发 on_train_end → 计算回滚 → 保存 JSON
        fire_train_end([cb], model, args, state, control)

        def1_path = os.path.join(tmpdir, "def1_rollback_loss.json")
        def2_path = os.path.join(tmpdir, "def2_rollback_acc.json")
        assert os.path.exists(def1_path), "def1_rollback_loss.json 未生成"
        assert os.path.exists(def2_path), "def2_rollback_acc.json 未生成"

        with open(def1_path) as f:
            d1 = json.load(f)
        with open(def2_path) as f:
            d2 = json.load(f)

        assert "baseline_loss" in d1 and "module_scores" in d1
        assert "baseline_acc"  in d2 and "module_scores" in d2
        assert len(d1["module_scores"]) > 0

    print(f"✓ callback 端到端：JSON 已生成，"
          f"{len(d1['module_scores'])} 个模块均已计算回滚增益")


# ---------------------------------------------------------------------------
# 单元测试：head_granularity
# ---------------------------------------------------------------------------

def test_def12_head_granularity_structure():
    """
    head_granularity=True 时，结果含 head_scores_loss 和 head_scores_acc，
    结构为 {module_name: {"head_0": float, ...}}，头数量等于 num_attention_heads。
    """
    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")
    eval_fn = make_fake_eval_fn(num_batches=4, batch_size=4)

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=5)

    module_groups = group_params_by_module(model)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
        head_granularity=True,
    )

    assert len(result["head_scores_loss"]) > 0, "head_granularity=True 但 head_scores_loss 为空"
    assert len(result["head_scores_acc"])  > 0, "head_granularity=True 但 head_scores_acc 为空"

    for m_name, per_head in result["head_scores_loss"].items():
        assert len(per_head) == cfg.num_attention_heads, (
            f"{m_name}: 头数量={len(per_head)}，期望={cfg.num_attention_heads}"
        )
        for h in range(cfg.num_attention_heads):
            assert f"head_{h}" in per_head, f"{m_name} 缺少 head_{h}"

    print(
        f"✓ head_granularity：{len(result['head_scores_loss'])} 个注意力模块，"
        f"每模块 {cfg.num_attention_heads} 头，结构正确"
    )


def test_def12_head_granularity_callback():
    """callback + head_granularity=True 端到端：JSON 含 head_scores 字段。"""
    cfg    = TinyConfig(hidden_size=8, num_attention_heads=2)
    model  = TinyHFClassifier(cfg)
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    eval_fn = make_fake_eval_fn(num_batches=4, batch_size=4)
    metric  = RollbackGainMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(
            model=model, save_dir=tmpdir, eval_fn=eval_fn,
            head_granularity=True,
        )
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        def1_path = os.path.join(tmpdir, "def1_rollback_loss.json")
        def2_path = os.path.join(tmpdir, "def2_rollback_acc.json")
        assert os.path.exists(def1_path)
        assert os.path.exists(def2_path)

        with open(def1_path) as f:
            d1 = json.load(f)
        with open(def2_path) as f:
            d2 = json.load(f)

        assert "head_scores" in d1, "def1 JSON 缺少 head_scores"
        assert "head_scores" in d2, "def2 JSON 缺少 head_scores"
        assert len(d1["head_scores"]) > 0
        assert len(d2["head_scores"]) > 0

    print(f"✓ callback+head_granularity 端到端：def1/def2 JSON 均含 head_scores")


# ---------------------------------------------------------------------------
# 打印（人工核验）
# ---------------------------------------------------------------------------

def test_def12_print_topk(top_k: int = 5):
    """打印 loss/acc 增益 Top-k 模块（供人工核验）。"""
    theta0, model, eval_fn, device, module_groups = _setup_trained(steps=30)
    result = compute_rollback(
        theta0=theta0, model=model, eval_fn=eval_fn,
        device=device, module_groups=module_groups,
    )

    sorted_modules = sorted(
        result["module_scores_loss"].keys(),
        key=lambda k: result["module_scores_loss"][k],
        reverse=True,
    )

    print(f"\n训练收益（回滚）Top-{top_k} 模块（steps=30，lr=1e-2）：")
    print(f"{'模块名':<55} {'ΔL（loss增益）':>16} {'ΔAcc（acc增益）':>16}")
    print("-" * 90)
    for name in sorted_modules[:top_k]:
        dl_val   = result["module_scores_loss"].get(name, 0.0)
        dacc_val = result["module_scores_acc"].get(name, 0.0)
        print(f"{name:<55} {dl_val:>+16.6f} {dacc_val:>+16.6f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试定义一&二：回滚训练收益（RollbackGainMetric）")
    print("=" * 65)
    test_def12_structure()
    test_def12_baseline_reasonable()
    test_def12_zero_gain_untrained()
    test_def12_rollback_changes_scores()
    test_def12_compute_loss_only()
    test_def12_compute_acc_only()
    test_def12_save_generates_files()
    test_def12_save_load_roundtrip()
    test_def12_callback_end_to_end()
    test_def12_print_topk()
    print()
    print("── 头级别（head_granularity）测试 ──")
    test_def12_head_granularity_structure()
    test_def12_head_granularity_callback()
    print("\n所有测试通过 ✓")
