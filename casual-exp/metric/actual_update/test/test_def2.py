"""
metric/actual_update/test/test_def2.py

定义二：相对更新量（RelativeUpdateMetric）— 独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.actual_update.test.test_def2

覆盖点：
  1. compute() 返回结构完整
  2. 所有分数非负
  3. 训练前相对更新量为零
  4. 分子（abs_module_scores）与定义一的 module_scores 完全一致（交叉验证）
  5. epsilon 字段正确写入
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

from metric.actual_update.def2_relative import RelativeUpdateMetric
from metric.actual_update.def1_absolute import AbsoluteUpdateMetric
from metric.actual_update.base import snapshot_params
from metric.actual_update.test.conftest import (
    TinyClassifier, make_fake_dataloader, train_n_steps, fire_train_end,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _setup(steps=5, lr=1e-2, epsilon=1e-8):
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=steps, lr=lr)

    result = RelativeUpdateMetric().compute(theta0, model, device, epsilon=epsilon)
    return theta0, model, result, device


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

def test_def2_structure():
    """compute() 返回字典包含所有必需字段。"""
    _, _, result, _ = _setup()
    for key in ("module_scores", "abs_module_scores", "init_norm_scores",
                "param_scores", "epsilon"):
        assert key in result, f"缺少字段: {key}"
    print("✓ 返回结构正确（含5个必需字段）")


def test_def2_non_negative():
    """相对更新量（分子/分母均非负，所以比值非负）。"""
    _, _, result, _ = _setup()
    for name, score in result["module_scores"].items():
        assert score >= 0, f"{name} 的相对更新量为负: {score}"
    for name, score in result["abs_module_scores"].items():
        assert score >= 0, f"{name} 的分子（||Δθ||₂）为负: {score}"
    for name, score in result["init_norm_scores"].items():
        assert score >= 0, f"{name} 的分母（||θ^(0)||₂）为负: {score}"
    print(f"✓ 所有 {len(result['module_scores'])} 个模块分数非负")


def test_def2_zero_before_training():
    """训练前 Δθ = 0，分子为 0，相对更新量应为 0。"""
    model  = TinyClassifier()
    device = torch.device("cpu")
    theta0 = snapshot_params(model)

    result = RelativeUpdateMetric().compute(theta0, model, device)

    for name, score in result["module_scores"].items():
        assert abs(score) < 1e-9, f"{name} 训练前相对更新量不为零: {score}"
    for name, score in result["abs_module_scores"].items():
        assert abs(score) < 1e-9, f"{name} 训练前分子不为零: {score}"
    print("✓ 训练前相对更新量均为 0")


def test_def2_numerator_equals_def1():
    """
    def2 的分子（abs_module_scores）应等于 def1 的 module_scores。
    两者均为 ||Δθ_m||_2（全参数 L2 组合），交叉验证实现一致性。
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader(batch_size=4, num_batches=8)
    device = torch.device("cpu")

    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=5)

    r1 = AbsoluteUpdateMetric().compute(theta0, model, device)
    r2 = RelativeUpdateMetric().compute(theta0, model, device)

    for m_name in r1["module_scores"]:
        def1_val = r1["module_scores"][m_name]
        def2_num = r2["abs_module_scores"][m_name]
        assert abs(def1_val - def2_num) < 1e-9, (
            f"{m_name}: def1={def1_val:.10f}, def2分子={def2_num:.10f}"
        )
    print("✓ def2 分子（abs_module_scores）与 def1 module_scores 完全一致")


def test_def2_epsilon_stored():
    """epsilon 应正确写入结果字典。"""
    eps    = 1e-6
    _, _, result, _ = _setup(epsilon=eps)
    assert abs(result["epsilon"] - eps) < 1e-15, (
        f"存储的 epsilon={result['epsilon']}，期望={eps}"
    )
    print(f"✓ epsilon={eps} 正确写入")


def test_def2_relative_ordering():
    """
    相对更新量的排序可以与绝对更新量不同（归一化效果验证）。
    具体：找到绝对排名前 3 和相对排名前 3，检查它们可能不完全相同
    （对于小模型可能相同，此测试仅记录，不强制断言顺序不同）。
    """
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    device = torch.device("cpu")
    theta0 = snapshot_params(model)
    train_n_steps(model, dl, steps=10)

    r1 = AbsoluteUpdateMetric().compute(theta0, model, device)
    r2 = RelativeUpdateMetric().compute(theta0, model, device)

    top3_abs = sorted(r1["module_scores"], key=r1["module_scores"].get, reverse=True)[:3]
    top3_rel = sorted(r2["module_scores"], key=r2["module_scores"].get, reverse=True)[:3]
    print(f"  绝对Top3: {top3_abs}")
    print(f"  相对Top3: {top3_rel}")
    print("✓ 相对/绝对排序已打印（可能相同或不同，取决于模型结构）")


def test_def2_save_load():
    """save() 写 JSON 后 load() 读取值与原始一致。"""
    _, _, result, _ = _setup()
    metric = RelativeUpdateMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = metric.save(result, tmpdir)
        assert saved_path.exists(), "JSON 文件未生成"
        with open(saved_path) as f:
            loaded = json.load(f)

        for key in list(result["module_scores"].keys())[:3]:
            assert abs(loaded["module_scores"][key] - result["module_scores"][key]) < 1e-9
            assert abs(
                loaded["abs_module_scores"][key] - result["abs_module_scores"][key]
            ) < 1e-9

    print("✓ save/load 往返一致")


def test_def2_callback_end_to_end():
    """callback 端到端：make_callback → 训练 → on_train_end → JSON 文件生成。"""
    model  = TinyClassifier()
    dl     = make_fake_dataloader()
    metric = RelativeUpdateMetric()

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = metric.make_callback(model, save_dir=tmpdir)
        args, state, control = train_n_steps(model, dl, callbacks=[cb], steps=5)
        fire_train_end([cb], model, args, state, control)

        json_path = os.path.join(tmpdir, f"{metric.name}.json")
        assert os.path.exists(json_path), f"JSON 未生成: {json_path}"

        with open(json_path) as f:
            saved = json.load(f)

        for key in ("module_scores", "abs_module_scores", "init_norm_scores",
                    "param_scores", "epsilon"):
            assert key in saved, f"JSON 缺少字段: {key}"

        nonzero = sum(1 for v in saved["module_scores"].values() if v > 0)
        assert nonzero > 0

    print(f"✓ callback 端到端：JSON 已生成，{nonzero} 个模块有非零相对更新量")


def test_def2_print_topk(top_k: int = 5):
    """打印相对更新量 Top-k 模块（供人工核验）。"""
    _, _, result, _ = _setup(steps=10, lr=1e-2)

    sorted_modules = sorted(
        result["module_scores"].keys(),
        key=lambda k: result["module_scores"][k],
        reverse=True,
    )

    print(f"\n定义二：相对更新量 Top-{top_k} 模块（steps=10, lr=1e-2）：")
    print(f"{'模块名':<55} {'U_rel':>10} {'||Δθ||₂':>12} {'||θ^(0)||₂':>12}")
    print("-" * 93)
    for name in sorted_modules[:top_k]:
        u = result["module_scores"][name]
        d = result["abs_module_scores"][name]
        i = result["init_norm_scores"][name]
        print(f"{name:<55} {u:>10.6f} {d:>12.8f} {i:>12.8f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  测试定义二：相对更新量（RelativeUpdateMetric）")
    print("=" * 65)
    test_def2_structure()
    test_def2_non_negative()
    test_def2_zero_before_training()
    test_def2_numerator_equals_def1()
    test_def2_epsilon_stored()
    test_def2_relative_ordering()
    test_def2_save_load()
    test_def2_callback_end_to_end()
    test_def2_print_topk()
    print("\n所有测试通过 ✓")
