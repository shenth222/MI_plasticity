"""
metric/pre_importance/test/test_attn_head.py

注意力头级别粒度 — 完整独立测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    /data1/shenth/miniconda3/envs/MI/bin/python \
        -m metric.pre_importance.test.test_attn_head

覆盖点：
  Part A  attn_head 工具函数单元测试
    A1. get_attn_head_config — 从 TinyHFClassifier 正确提取 config
    A2. classify_attn_module — QKV / OUT / None 分类正确
    A3. get_attn_modules     — 扫描 TinyHFClassifier 返回正确模块及类型
    A4. get_head_weight_view — 权重视图形状和共享内存验证
    A5. get_head_bias_view   — QKV 偏置视图 / OUT 返回 None
    A6. 原地修改通过 view 正确反映到原参数（扰动恢复逻辑验证）

  Part B  各指标的 head_granularity 集成测试
    B1. Fisher  — head_scores 存在，每头非负，形状正确
    B2. Saliency — grad_norm + taylor 均含 head_scores
    B3. Perturbation — head_scores 存在，参数恢复正确
    B4. SingularValue — head_scores 包含 nuclear_norm / top{k}_sum
    B5. SpectralEntropy — head_scores 谱熵 ∈ [0, 1]

  Part C  向后兼容测试
    C1. 所有指标默认（head_granularity=False）不产生 head_scores
    C2. 模型无 config 时，head_granularity=True 安全降级

  Part D  Runner 集成测试
    D1. PreImportanceRunner(head_granularity=True).run() 生成含 head_scores 的 JSON
    D2. from_str(..., head_granularity=True) 正确传递标志

  Part E  结果打印
    E1. 打印 TinyHFClassifier 各注意力模块的头级别谱熵，供人工核验
"""

import json
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import torch.nn as nn

from metric.pre_importance.attn_head import (
    AttnHeadConfig,
    get_attn_head_config,
    classify_attn_module,
    get_attn_modules,
    get_head_weight_view,
    get_head_bias_view,
    agg_head_scores_from_acc,
    compute_head_svd_scores,
)
from metric.pre_importance.fisher import FisherImportance
from metric.pre_importance.saliency import SaliencyImportance
from metric.pre_importance.perturbation import PerturbationImportance
from metric.pre_importance.singular_value import SingularValueImportance
from metric.pre_importance.spectral_entropy import SpectralEntropyImportance
from metric.pre_importance.runner import PreImportanceRunner
from metric.pre_importance.test.conftest import (
    TinyClassifier,
    TinyHFClassifier,
    TinyConfig,
    make_fake_dataloader,
)


# ============================================================================
# 帮助函数
# ============================================================================

def _make_hf_model_and_dl():
    model = TinyHFClassifier(TinyConfig(hidden_size=8, num_attention_heads=2))
    dl    = make_fake_dataloader(batch_size=2, num_batches=4,
                                 vocab_size=200, num_labels=3)
    device = torch.device("cpu")
    return model, dl, device


def _expected_attn_modules():
    """返回 TinyHFClassifier 中预期被识别为注意力模块的名称集合"""
    return {
        "deberta.encoder.layer.0.attention.self.query_proj",
        "deberta.encoder.layer.0.attention.self.key_proj",
        "deberta.encoder.layer.0.attention.self.value_proj",
        "deberta.encoder.layer.0.attention.output.dense",
        "deberta.encoder.layer.1.attention.self.query_proj",
        "deberta.encoder.layer.1.attention.self.key_proj",
        "deberta.encoder.layer.1.attention.self.value_proj",
        "deberta.encoder.layer.1.attention.output.dense",
    }


# ============================================================================
# Part A：attn_head 工具函数单元测试
# ============================================================================

def test_A1_get_attn_head_config():
    """从 TinyHFClassifier 正确提取注意力头配置"""
    model = TinyHFClassifier(TinyConfig(hidden_size=8, num_attention_heads=2))
    cfg   = get_attn_head_config(model)

    assert cfg is not None,          "应返回 AttnHeadConfig，不应为 None"
    assert cfg.num_heads   == 2,     f"num_heads 应为 2，实际={cfg.num_heads}"
    assert cfg.hidden_size == 8,     f"hidden_size 应为 8，实际={cfg.hidden_size}"
    assert cfg.head_dim    == 4,     f"head_dim 应为 4，实际={cfg.head_dim}"
    assert cfg.attn_dim    == 8,     f"attn_dim 应为 8，实际={cfg.attn_dim}"
    print("✓ A1 get_attn_head_config 正确")


def test_A2_classify_attn_module():
    """classify_attn_module 对各类路径的分类应正确"""
    assert classify_attn_module("deberta.encoder.layer.0.attention.self.query_proj") == "qkv"
    assert classify_attn_module("deberta.encoder.layer.0.attention.self.key_proj")   == "qkv"
    assert classify_attn_module("deberta.encoder.layer.0.attention.self.value_proj") == "qkv"
    assert classify_attn_module("deberta.encoder.layer.0.attention.output.dense")    == "out"
    assert classify_attn_module("deberta.encoder.layer.0.intermediate.dense")        is None
    assert classify_attn_module("deberta.encoder.layer.0.output_ffn")                is None
    assert classify_attn_module("classifier")                                         is None
    # GPT-2 style
    assert classify_attn_module("transformer.h.0.attn.c_attn")                       == "qkv"
    assert classify_attn_module("transformer.h.0.attn.out_proj")                     == "out"
    print("✓ A2 classify_attn_module 分类正确")


def test_A3_get_attn_modules():
    """get_attn_modules 扫描 TinyHFClassifier，返回正确的模块集合及类型"""
    model = TinyHFClassifier(TinyConfig(hidden_size=8, num_attention_heads=2))
    cfg   = get_attn_head_config(model)
    mods  = get_attn_modules(model, cfg)

    expected = _expected_attn_modules()
    assert set(mods.keys()) == expected, (
        f"期望模块集合：\n{sorted(expected)}\n"
        f"实际模块集合：\n{sorted(mods.keys())}"
    )
    # 验证类型分配
    for name, mtype in mods.items():
        if "output.dense" in name:
            assert mtype == "out",  f"{name} 应为 'out'，实际={mtype}"
        else:
            assert mtype == "qkv",  f"{name} 应为 'qkv'，实际={mtype}"

    print(f"✓ A3 get_attn_modules 正确识别 {len(mods)} 个注意力模块")


def test_A4_head_weight_view_shape():
    """get_head_weight_view 返回的视图形状正确"""
    cfg = TinyConfig(hidden_size=8, num_attention_heads=2)   # head_dim=4
    W_qkv = torch.randn(8, 8)   # [num_heads*head_dim, hidden]
    W_out  = torch.randn(8, 8)  # [hidden, num_heads*head_dim]

    for h in range(2):
        v_q = get_head_weight_view(W_qkv, "qkv", h, head_dim=4)
        assert v_q.shape == (4, 8), f"QKV head_{h} view 形状应为 (4,8)，实际={v_q.shape}"

        v_o = get_head_weight_view(W_out, "out", h, head_dim=4)
        assert v_o.shape == (8, 4), f"OUT head_{h} view 形状应为 (8,4)，实际={v_o.shape}"

    print("✓ A4 head weight view 形状正确")


def test_A5_head_bias_view():
    """get_head_bias_view：QKV 返回视图，OUT 返回 None"""
    b_qkv = torch.randn(8)   # [num_heads*head_dim]
    b_out  = torch.randn(8)  # [hidden]

    for h in range(2):
        v = get_head_bias_view(b_qkv, "qkv", h, head_dim=4)
        assert v is not None and v.shape == (4,), \
            f"QKV bias 视图形状应为 (4,)，实际={v.shape if v is not None else None}"

        v_o = get_head_bias_view(b_out, "out", h, head_dim=4)
        assert v_o is None, "OUT bias 视图应为 None"

    print("✓ A5 head bias view 正确")


def test_A6_inplace_view_shared_memory():
    """通过 view 的原地修改应反映到原张量（用于验证扰动恢复逻辑）"""
    W = torch.ones(8, 8)
    noise = torch.ones(4, 8) * 0.5

    view = get_head_weight_view(W, "qkv", h=0, head_dim=4)
    view.add_(noise)    # 原地加噪
    assert W[0:4, :].mean().item() == 1.5, "原地加噪后原张量应更新"

    view.sub_(noise)    # 原地恢复
    assert (W - 1.0).abs().max().item() < 1e-6, "原地恢复后原张量应回到原值"

    print("✓ A6 view 原地操作共享内存验证通过")


# ============================================================================
# Part B：各指标的 head_granularity 集成测试
# ============================================================================

def test_B1_fisher_head_granularity():
    """Fisher head_granularity=True 时 head_scores 结构和数值正确"""
    model, dl, device = _make_hf_model_and_dl()
    metric = FisherImportance()
    result = metric.compute(model, dl, device, num_batches=4, head_granularity=True)

    assert "head_scores" in result, "head_granularity=True 应产生 head_scores"

    cfg  = get_attn_head_config(model)
    mods = _expected_attn_modules()

    for mod in mods:
        assert mod in result["head_scores"], f"head_scores 缺少模块 {mod}"
        h_scores = result["head_scores"][mod]
        assert len(h_scores) == cfg.num_heads, \
            f"{mod} 应有 {cfg.num_heads} 个头，实际={len(h_scores)}"
        for h in range(cfg.num_heads):
            v = h_scores[f"head_{h}"]
            assert v >= 0, f"{mod} head_{h} Fisher 分数为负: {v}"

    # 模块级分数必须仍然存在（向后兼容）
    assert "module_scores" in result
    print(f"✓ B1 Fisher head_granularity: {len(mods)} 模块 × {cfg.num_heads} 头，均非负")


def test_B2_saliency_head_granularity():
    """Saliency grad_norm + taylor 均含 head_scores"""
    model, dl, device = _make_hf_model_and_dl()
    metric = SaliencyImportance()
    result = metric.compute(model, dl, device, num_batches=4, head_granularity=True)

    for variant in ("grad_norm", "taylor"):
        assert "head_scores" in result[variant], \
            f"saliency.{variant} 缺少 head_scores"
        h_scores = result[variant]["head_scores"]
        for mod in _expected_attn_modules():
            assert mod in h_scores, f"{variant} head_scores 缺少 {mod}"
        # 所有值非负（绝对值）
        for mod, hs in h_scores.items():
            for hk, v in hs.items():
                assert v >= 0, f"{variant}/{mod}/{hk} 分数为负: {v}"

    print("✓ B2 Saliency head_granularity: grad_norm + taylor 均正确")


def test_B3_perturbation_head_granularity():
    """Perturbation head_scores 存在，且参数在计算后完全恢复"""
    model, dl, device = _make_hf_model_and_dl()

    # 记录计算前参数快照
    before = {n: p.data.clone() for n, p in model.named_parameters()}

    metric = PerturbationImportance()
    result = metric.compute(
        model, dl, device,
        num_batches=1, num_samples=1,
        noise_std=1e-3,
        head_granularity=True,
    )

    assert "head_scores" in result, "Perturbation head_granularity=True 应产生 head_scores"

    # 参数恢复验证
    max_diff = max(
        (p.data - before[n]).abs().max().item()
        for n, p in model.named_parameters()
    )
    assert max_diff < 1e-5, f"参数未完全恢复，最大差值={max_diff}"

    # head_scores 结构验证
    for mod in _expected_attn_modules():
        assert mod in result["head_scores"], f"head_scores 缺少 {mod}"

    print(f"✓ B3 Perturbation head_granularity: 参数恢复正确（max_diff={max_diff:.2e}）")


def test_B4_singular_value_head_granularity():
    """SingularValue head_scores 包含各头的 nuclear_norm / top_k_sum"""
    model  = TinyHFClassifier(TinyConfig(hidden_size=8, num_attention_heads=2))
    metric = SingularValueImportance()
    result = metric.compute(model, top_k=4, head_granularity=True)

    assert "head_scores" in result

    cfg = get_attn_head_config(model)
    for mod in _expected_attn_modules():
        assert mod in result["head_scores"], f"head_scores 缺少 {mod}"
        hs = result["head_scores"][mod]
        assert len(hs) == cfg.num_heads
        for h in range(cfg.num_heads):
            hd = hs[f"head_{h}"]
            assert "nuclear_norm" in hd and "top4_sum" in hd, \
                f"{mod} head_{h} 缺少必要字段"
            assert hd["nuclear_norm"] >= hd["top4_sum"] - 1e-6, \
                f"{mod} head_{h} nuclear_norm < top4_sum"
            # 头子矩阵形状应为 [head_dim, hidden_size] 或 [hidden_size, head_dim]
            shape = hd["matrix_shape"]
            assert cfg.head_dim in shape, f"{mod} head_{h} 子矩阵不含 head_dim={cfg.head_dim}"

    print(f"✓ B4 SingularValue head_granularity: nuclear_norm ≥ top4_sum，形状正确")


def test_B5_spectral_entropy_head_granularity():
    """SpectralEntropy head_scores 谱熵 ∈ [0, 1]"""
    model  = TinyHFClassifier(TinyConfig(hidden_size=8, num_attention_heads=2))
    metric = SpectralEntropyImportance()
    result = metric.compute(model, head_granularity=True)

    assert "head_scores" in result

    cfg = get_attn_head_config(model)
    for mod in _expected_attn_modules():
        hs = result["head_scores"][mod]
        for h in range(cfg.num_heads):
            se = hs[f"head_{h}"]["spectral_entropy"]
            assert -1e-6 <= se <= 1 + 1e-6, \
                f"{mod} head_{h} 谱熵={se:.4f} 超出 [0,1]"

    print("✓ B5 SpectralEntropy head_granularity: 所有头谱熵 ∈ [0, 1]")


# ============================================================================
# Part C：向后兼容测试
# ============================================================================

def test_C1_no_head_scores_by_default():
    """head_granularity=False（默认）时，所有指标均不产生 head_scores"""
    model, dl, device = _make_hf_model_and_dl()

    for MetricCls, needs_data in [
        (FisherImportance,       True),
        (SaliencyImportance,     True),
        (PerturbationImportance, True),
        (SingularValueImportance, False),
        (SpectralEntropyImportance, False),
    ]:
        metric = MetricCls()
        kw = {"num_batches": 2, "num_samples": 1} if needs_data else {}
        if needs_data:
            r = metric.compute(model, dl, device, **kw)
        else:
            r = metric.compute(model)

        # 检查顶层
        assert "head_scores" not in r, \
            f"{MetricCls.name} 默认不应含 head_scores，但发现了"
        # Saliency 嵌套检查
        if MetricCls == SaliencyImportance:
            assert "head_scores" not in r["grad_norm"]
            assert "head_scores" not in r["taylor"]

    print("✓ C1 所有指标默认不含 head_scores（向后兼容）")


def test_C2_no_config_graceful_degradation():
    """模型无 config 时，head_granularity=True 安全降级，不抛出异常"""
    model  = TinyClassifier()   # 无 config
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    for MetricCls, needs_data in [
        (FisherImportance,       True),
        (SaliencyImportance,     True),
        (SingularValueImportance, False),
        (SpectralEntropyImportance, False),
    ]:
        metric = MetricCls()
        kw = {"num_batches": 2} if needs_data else {}
        try:
            if needs_data:
                r = metric.compute(model, dl, device, head_granularity=True, **kw)
            else:
                r = metric.compute(model, head_granularity=True)
            assert "head_scores" not in r, \
                f"{MetricCls.name} 无 config 时不应含 head_scores"
        except Exception as e:
            raise AssertionError(f"{MetricCls.name} 无 config 时抛出异常: {e}")

    print("✓ C2 无 config 时 head_granularity=True 安全降级")


# ============================================================================
# Part D：Runner 集成测试
# ============================================================================

def test_D1_runner_head_granularity():
    """PreImportanceRunner(head_granularity=True).run() 生成含 head_scores 的 JSON"""
    model, dl, device = _make_hf_model_and_dl()

    runner = PreImportanceRunner(
        metrics=["fisher", "spectral_entropy"],
        metric_kwargs={"fisher": {"num_batches": 2}},
        head_granularity=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = runner.run(model, dl, device, save_dir=tmpdir)

        for name in ("fisher", "spectral_entropy"):
            assert "head_scores" in results[name], \
                f"{name} 结果缺少 head_scores"
            path = Path(tmpdir) / f"{name}.json"
            assert path.exists()
            with open(path) as f:
                saved = json.load(f)
            assert "head_scores" in saved, f"{name}.json 缺少 head_scores"

    print("✓ D1 Runner head_granularity=True 正确传递并保存 head_scores")


def test_D2_runner_from_str_head_granularity():
    """from_str(..., head_granularity=True) 正确传递标志"""
    runner = PreImportanceRunner.from_str(
        "singular_value,spectral_entropy",
        head_granularity=True,
    )
    assert runner.head_granularity is True

    model = TinyHFClassifier()
    with tempfile.TemporaryDirectory() as tmpdir:
        results = runner.run(model, dataloader=None, device=torch.device("cpu"),
                             save_dir=tmpdir)
        for name in ("singular_value", "spectral_entropy"):
            assert "head_scores" in results[name]

    print("✓ D2 Runner.from_str head_granularity 传递正确")


# ============================================================================
# Part E：结果打印
# ============================================================================

def test_E1_print_head_spectral_entropy():
    """打印 TinyHFClassifier 各注意力模块的头级别谱熵，供人工核验"""
    model  = TinyHFClassifier(TinyConfig(hidden_size=8, num_attention_heads=2))
    metric = SpectralEntropyImportance()
    result = metric.compute(model, head_granularity=True)

    print(f"\n谱熵头级别分布（TinyHFClassifier，hidden=8，2 头，head_dim=4）：")
    print(f"{'模块名（末三级）':<45} {'head_0':>8} {'head_1':>8} {'模块谱熵':>10}")
    print("-" * 75)

    for mod, hs in result["head_scores"].items():
        short = ".".join(mod.split(".")[-3:])
        mod_se = result["module_scores"].get(mod, {}).get("spectral_entropy", float("nan"))
        h0 = hs.get("head_0", {}).get("spectral_entropy", float("nan"))
        h1 = hs.get("head_1", {}).get("spectral_entropy", float("nan"))
        print(f"{short:<45} {h0:>8.4f} {h1:>8.4f} {mod_se:>10.4f}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  注意力头级别粒度测试（test_attn_head.py）")
    print("=" * 65)

    print("\n── Part A：工具函数单元测试 ──")
    test_A1_get_attn_head_config()
    test_A2_classify_attn_module()
    test_A3_get_attn_modules()
    test_A4_head_weight_view_shape()
    test_A5_head_bias_view()
    test_A6_inplace_view_shared_memory()

    print("\n── Part B：各指标头级别集成测试 ──")
    test_B1_fisher_head_granularity()
    test_B2_saliency_head_granularity()
    test_B3_perturbation_head_granularity()
    test_B4_singular_value_head_granularity()
    test_B5_spectral_entropy_head_granularity()

    print("\n── Part C：向后兼容测试 ──")
    test_C1_no_head_scores_by_default()
    test_C2_no_config_graceful_degradation()

    print("\n── Part D：Runner 集成测试 ──")
    test_D1_runner_head_granularity()
    test_D2_runner_from_str_head_granularity()

    print("\n── Part E：结果打印 ──")
    test_E1_print_head_spectral_entropy()

    print("\n\n所有测试通过 ✓")
