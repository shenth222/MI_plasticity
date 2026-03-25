"""
metric/actual_update/test/test_def1.py

定义一：绝对更新量（AbsoluteUpdateMetric）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.actual_update.test.test_def1

覆盖点：
  1. compute() 返回结构完整（含 module_scores / weight_only_scores / param_scores）
  2. 变体A（全参数L2）≥ 变体B（仅weight Frobenius）—— 子集关系约束
  3. 训练前更新量为零
  4. 训练后有非零更新量
  5. 所有分数非负
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

from metric.actual_update.def1_absolute import AbsoluteUpdateMetric
from metric.actual_update.base import snapshot_params
from metric.actual_update.test.conftest import (
    TinyClassifier, make_fake_dataloader, train_n_steps, fire_train_end,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _setup(steps=5, lr=1e-2):
    """返回 (theta0, model_after_training, result)。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=steps, lr=lr)

    result = AbsoluteUpdateMetric().compute(theta0, model, device)
    return theta0, model, result


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

def test_def1_structure():
    """compute() 返回字典包含所有必需字段。"""
    _, _, result = _setup()
    for key in ("module_scores", "weight_only_scores", "param_scores"):
        assert key in result, f"缺少字段: {key}"
    assert len(result["module_scores"]) > 0, "module_scores 为空"
    print("✓ 返回结构正确（module_scores / weight_only_scores / param_scores）")


def test_def1_non_negative():
    """所有分数必须非负（L2/Frobenius 范数 ≥ 0）。"""
    _, _, result = _setup()
    for name, score in result["module_scores"].items():
        assert score >= 0, f"变体A: {name} 的更新量为负: {score}"
    for name, score in result["weight_only_scores"].items():
        assert score >= 0, f"变体B: {name} 的更新量为负: {score}"
    print(f"✓ 所有 {len(result['module_scores'])} 个模块分数非负（变体A和B）")


def test_def1_variantA_geq_variantB():
    """
    变体A（含bias）≥ 变体B（仅weight）的模块分数。
    证明：变体A = sqrt(Σ_{all p} ||Δp||²)，变体B = sqrt(Σ_{non-bias p} ||Δp||²)，
    由于 bias 项 ||Δbias||² ≥ 0，故变体A ≥ 变体B。
    """
    _, _, result = _setup()
    for m_name in result["module_scores"]:
        a = result["module_scores"][m_name]
        b = result["weight_only_scores"][m_name]
        assert a >= b - 1e-9, f"{m_name}: 变体A({a:.8f}) < 变体B({b:.8f})"
    print("✓ 变体A ≥ 变体B（对所有模块成立，子集约束验证通过）")


def test_def1_zero_before_training():
    """训练前 theta0 == 当前参数，更新量应全为零。"""
    model  = TinyClassifier()
    device = torch.device("cpu")
    theta0 = snapshot_params(model)   # 此时模型未训练

    result = AbsoluteUpdateMetric().compute(theta0, model, device)

    for name, score in result["module_scores"].items():
        assert abs(score) < 1e-9, f"{name} 训练前变体A更新量不为零: {score}"
    for name, score in result["weight_only_scores"].items():
        assert abs(score) < 1e-9, f"{name} 训练前变体B更新量不为零: {score}"
    print("✓ 训练前绝对更新量均为 0（变体A和B）")


def test_def1_nonzero_after_training():
    """训练后至少有一个模块的更新量非零。"""
    _, _, result = _setup(steps=5, lr=1e-2)
    nonzero_A = sum(1 for v in result["module_scores"].values() if v > 0)
    nonzero_B = sum(1 for v in result["weight_only_scores"].values() if v > 0)
    assert nonzero_A > 0, "所有模块变体A更新量为0，参数未发生变化"
    assert nonzero_B > 0, "所有模块变体B更新量为0，参数未发生变化"
    print(f"✓ {nonzero_A}/{len(result['module_scores'])} 个模块有非零变体A更新量")
    print(f"✓ {nonzero_B}/{len(result['weight_only_scores'])} 个模块有非零变体B更新量")


def test_def1_param_scores_consistent():
    """
    module_scores（变体A）应等于按模块聚合的 param_scores 的 L2 组合。
    即：module_score = sqrt(Σ_{p∈m} param_score[p]²)
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    device = torch.device("cpu")
    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=5)

    from metric.actual_update.base import group_params_by_module
    metric  = AbsoluteUpdateMetric()
    result  = metric.compute(theta0, model, device)
    groups  = group_params_by_module(model)

    for m_name, param_names in groups.items():
        recomputed = sum(result["param_scores"].get(pn, 0.0) ** 2 for pn in param_names) ** 0.5
        stored     = result["module_scores"][m_name]
        assert abs(recomputed - stored) < 1e-9, (
            f"{m_name}: 重算={recomputed:.8f}, 存储={stored:.8f}"
        )
    print("✓ module_scores 与 param_scores 聚合一致")


def test_def1_save_load():
    """save() 写 JSON 后 load() 读取值与原始一致。"""
    _, _, result = _setup()
    metric = AbsoluteUpdateMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists(), "JSON 文件未生成"
        with open(saved_path) as f:
            loaded = json.load(f)

        for key in list(result["module_scores"].keys())[:3]:
            assert abs(loaded["module_scores"][key] - result["module_scores"][key]) < 1e-9
            assert abs(
                loaded["weight_only_scores"][key] - result["weight_only_scores"][key]
            ) < 1e-9

    print("✓ save/load 往返一致（变体A和B均验证）")


def test_def1_callback_end_to_end():
    """callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    metric = AbsoluteUpdateMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        assert os.path.exists(json_path), f"JSON 文件未生成: {json_path}"

        with open(json_path) as f:
            saved = json.load(f)

        assert "module_scores"      in saved
        assert "weight_only_scores" in saved
        assert "param_scores"       in saved

        # 验证训练后有非零更新量
        nonzero = sum(1 for v in saved["module_scores"].values() if v > 0)
        assert nonzero > 0

    print(f"✓ callback 端到端：JSON 已生成，{nonzero} 个模块有非零更新量")


def test_def1_print_topk(top_k: int = 5):
    """打印变体A/B Top-k 模块（供人工核验）。"""
    _, _, result = _setup(steps=10, lr=1e-2)

    sorted_modules = sorted(
        result["module_scores"].keys(),
        key=lambda k: result["module_scores"][k],
        reverse=True,
    )

    print(f"\n定义一：绝对更新量 Top-{top_k} 模块（steps=10, lr=1e-2）：")
    print(f"{'模块名':<55} {'变体A ||Δθ||₂':>16} {'变体B ||ΔW||_F':>16}")
    print("-" * 90)
    for name in sorted_modules[:top_k]:
        a = result["module_scores"][name]
        b = result["weight_only_scores"][name]
        print(f"{name:<55} {a:>16.8f} {b:>16.8f}")


# ---------------------------------------------------------------------------
# 头级别测试（使用 TinyHFClassifier）
# ---------------------------------------------------------------------------

def test_def1_head_granularity_structure():
    """
    head_granularity=True 时，compute() 结果包含 head_scores 字段，
    结构为 {module_name: {"head_0": float, "head_1": float, ...}}，
    头数量等于 config.num_attention_heads。
    """
    from metric.actual_update.test.conftest import TinyHFClassifier, TinyConfig

    cfg   = TinyConfig(hidden_size=8, num_attention_heads=2)
    model = TinyHFClassifier(cfg)
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=5, lr=1e-2)

    result = AbsoluteUpdateMetric().compute(
        theta0, model, device, head_granularity=True
    )

    assert "head_scores" in result, "head_granularity=True 时结果缺少 head_scores"

    head_scores = result["head_scores"]
    assert len(head_scores) > 0, "head_scores 为空"

    # 验证每个模块的头数量正确
    for m_name, per_head in head_scores.items():
        assert len(per_head) == cfg.num_attention_heads, (
            f"{m_name}: head 数量={len(per_head)}，期望={cfg.num_attention_heads}"
        )
        for h in range(cfg.num_attention_heads):
            key = f"head_{h}"
            assert key in per_head, f"{m_name} 缺少 {key}"
            assert per_head[key] >= 0, f"{m_name}.{key} 为负值: {per_head[key]}"

    print(
        f"✓ head_granularity: {len(head_scores)} 个注意力模块，"
        f"每模块 {cfg.num_attention_heads} 头，所有头分数非负"
    )


def test_def1_head_granularity_absent_by_default():
    """head_granularity=False（默认）时，结果不包含 head_scores 字段。"""
    from metric.actual_update.test.conftest import TinyHFClassifier

    model = TinyHFClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=3, lr=1e-2)

    result = AbsoluteUpdateMetric().compute(theta0, model, device, head_granularity=False)
    assert "head_scores" not in result, "head_granularity=False 时不应包含 head_scores"
    print("✓ head_granularity=False 时结果不含 head_scores（正确）")


def test_def1_head_granularity_non_negative():
    """训练后各头的绝对更新量非负。"""
    from metric.actual_update.test.conftest import TinyHFClassifier

    model = TinyHFClassifier()
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")
    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=5, lr=1e-2)

    result = AbsoluteUpdateMetric().compute(theta0, model, device, head_granularity=True)

    for m_name, per_head in result["head_scores"].items():
        for hk, val in per_head.items():
            assert val >= 0, f"{m_name}.{hk} 头级别绝对更新量为负: {val}"

    print(f"✓ 头级别绝对更新量全部非负")


def test_def1_head_granularity_callback():
    """callback + head_granularity=True 端到端：JSON 含 head_scores 字段。"""
    from metric.actual_update.test.conftest import TinyHFClassifier

    model  = TinyHFClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    metric = AbsoluteUpdateMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir, head_granularity=True)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            saved = json.load(f)

        assert "head_scores" in saved, "callback JSON 缺少 head_scores"
        assert len(saved["head_scores"]) > 0

    print(f"✓ callback 端到端（head_granularity=True）：JSON 含 head_scores")


def test_def1_head_granularity_print(top_k: int = 3):
    """打印头级别绝对更新量（供人工核验）。"""
    from metric.actual_update.test.conftest import TinyHFClassifier, TinyConfig

    cfg   = TinyConfig(hidden_size=8, num_attention_heads=2)
    model = TinyHFClassifier(cfg)
    dl    = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=10, lr=1e-2)
    result = AbsoluteUpdateMetric().compute(theta0, model, device, head_granularity=True)

    print(f"\n定义一头级别 Top-{top_k} 注意力模块（steps=10, lr=1e-2）：")
    head_items = sorted(
        result["head_scores"].items(),
        key=lambda kv: result["module_scores"].get(kv[0], 0.0),
        reverse=True,
    )[:top_k]
    for m_name, per_head in head_items:
        mod_score = result["module_scores"].get(m_name, 0.0)
        head_str = "  ".join(f"h{h}={v:.6f}" for h, v in enumerate(per_head.values()))
        print(f"  {m_name:<60} module={mod_score:.6f}  {head_str}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试定义一：绝对更新量（AbsoluteUpdateMetric）")
    print("=" * 65)
    test_def1_structure()
    test_def1_non_negative()
    test_def1_variantA_geq_variantB()
    test_def1_zero_before_training()
    test_def1_nonzero_after_training()
    test_def1_param_scores_consistent()
    test_def1_save_load()
    test_def1_callback_end_to_end()
    test_def1_print_topk()
    print()
    print("── 头级别（head_granularity）测试 ──")
    test_def1_head_granularity_structure()
    test_def1_head_granularity_absent_by_default()
    test_def1_head_granularity_non_negative()
    test_def1_head_granularity_callback()
    test_def1_head_granularity_print()
    print("\n所有测试通过 ✓")
