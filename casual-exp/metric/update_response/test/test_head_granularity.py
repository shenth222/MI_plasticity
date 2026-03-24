"""
metric/update_response/test/test_head_granularity.py

update_response 各定义的注意力头级别粒度测试
─────────────────────────────────────────────────────────────────────────────
运行方式（从 casual-exp 根目录）：
    python -m metric.update_response.test.test_head_granularity

测试结构：
  Part A  工具函数（attn_head 转发导入验证）
    A1. update_response.attn_head 转发导入正常
    A2. TinyHFClassifier 的注意力模块识别正确

  Part B  各定义 head_granularity=True 集成测试
    B1. def1 — head_scores 存在；头分数非负；结构正确
    B2. def2 — head_scores ≤ 1（Cauchy-Schwarz）；结构正确
    B3. def3 — head_scores 存在；头分数非负；steps_collected 正确
    B4. def4 — head_scores ∈ [0, 1]（Jensen 不等式）；结构正确

  Part C  向后兼容
    C1. head_granularity=False（默认）时所有定义均不产生 head_scores
    C2. 模型无 config 时 head_granularity=True 安全降级

  Part D  Runner 集成
    D1. UpdateResponseRunner(head_granularity=True).run_pre() 生成含 head_scores 的 JSON
    D2. make_training_callbacks(head_granularity=True) 正确传递标志
    D3. from_str(..., head_granularity=True) 正确

  Part E  结果打印（供人工核验）
    E1. 打印 def4 Ppred 各头分数
    E2. 打印 def1 各头位移分数
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
from torch.utils.data import DataLoader

# 被测模块
from metric.update_response.def1_probe_delta    import ProbeDeltaMetric
from metric.update_response.def2_grad_curvature  import GradCurvatureMetric
from metric.update_response.def3_early_grad_norm import EarlyGradNormCallback, EarlyGradNormMetric
from metric.update_response.def4_ppred           import PpredMetric
from metric.update_response.runner               import UpdateResponseRunner
from metric.update_response.attn_head import (
    get_attn_head_config, get_attn_modules,
)

# 共享 fixtures（含 TinyHFClassifier）
from metric.update_response.test.conftest import (
    TinyHFClassifier, TinyConfig, make_fake_dataloader,
    TinyClassifier,
)


# ============================================================================
# 共用辅助
# ============================================================================

def _make_hf_model_and_dl(hidden=8, num_heads=2, num_labels=3):
    """返回带 config 的 TinyHFClassifier + DataLoader + device"""
    model  = TinyHFClassifier(TinyConfig(hidden_size=hidden,
                                         num_attention_heads=num_heads,
                                         num_labels=num_labels))
    dl     = make_fake_dataloader(batch_size=2, num_batches=8,
                                  vocab_size=200, num_labels=num_labels)
    device = torch.device("cpu")
    return model, dl, device


def _expected_attn_mods(model):
    """返回期望被识别的注意力模块名称集合"""
    cfg = get_attn_head_config(model)
    return set(get_attn_modules(model, cfg).keys())


# Trainer 状态模拟
@dataclass
class FakeTrainerState:
    is_world_process_zero: bool = True

@dataclass
class FakeTrainerArgs:
    local_rank: int = -1

@dataclass
class FakeTrainerControl:
    pass


def _simulate_training(cb, model, dl, total_steps, device):
    """手动模拟 backward + on_step_end 循环。"""
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    state, args, control = FakeTrainerState(), FakeTrainerArgs(), FakeTrainerControl()
    model.train()
    data_iter = iter(dl)
    for _ in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        model(**inputs).loss.backward()
        optimizer.step()
        cb.on_step_end(args=args, state=state, control=control)
        if cb._done:
            break
    if not cb._done:
        cb.on_train_end(args=args, state=state, control=control)


def _check_head_scores_structure(head_scores, expected_mods, num_heads, metric_name):
    """通用结构验证：head_scores 包含所有预期模块，每模块有 num_heads 个头键。"""
    assert head_scores is not None, f"{metric_name}: head_scores 为 None"
    for mod in expected_mods:
        assert mod in head_scores, f"{metric_name}: head_scores 缺少模块 {mod}"
        for h in range(num_heads):
            assert f"head_{h}" in head_scores[mod], \
                f"{metric_name}: {mod} 缺少 head_{h}"


# ============================================================================
# Part A：工具函数
# ============================================================================

def test_A1_attn_head_import():
    """update_response.attn_head 转发导入应正常工作。"""
    from metric.update_response.attn_head import (
        AttnHeadConfig, get_attn_head_config, get_attn_modules,
        get_head_weight_view, get_head_bias_view, agg_head_scores_from_acc,
        classify_attn_module,
    )
    cfg = TinyConfig(hidden_size=8, num_attention_heads=2)
    model = TinyHFClassifier(cfg)
    attn_cfg = get_attn_head_config(model)
    assert attn_cfg is not None
    assert attn_cfg.num_heads == 2 and attn_cfg.head_dim == 4
    print("✓ A1 update_response.attn_head 转发导入正常")


def test_A2_attn_module_scan():
    """TinyHFClassifier 的注意力模块应被正确识别（8 个：2层×4投影）。"""
    model = TinyHFClassifier()
    cfg   = get_attn_head_config(model)
    mods  = get_attn_modules(model, cfg)
    assert len(mods) == 8, f"期望 8 个注意力模块，实际={len(mods)}"
    qkv_count = sum(1 for t in mods.values() if t == "qkv")
    out_count  = sum(1 for t in mods.values() if t == "out")
    assert qkv_count == 6, f"期望 6 个 qkv 模块（2层×3），实际={qkv_count}"
    assert out_count  == 2, f"期望 2 个 out 模块（2层×1），实际={out_count}"
    print(f"✓ A2 识别到 8 个注意力模块（6 qkv + 2 out）")


# ============================================================================
# Part B：各定义头级别集成测试
# ============================================================================

def test_B1_def1_head_granularity():
    """def1 head_granularity=True：head_scores 存在，值非负，θ⁰ 仍被恢复。"""
    model, dl, device = _make_hf_model_and_dl()
    theta0 = {n: p.data.clone() for n, p in model.named_parameters()}
    num_heads = model.config.num_attention_heads

    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device,
                             probe_steps=5, probe_lr=1e-3,
                             head_granularity=True)

    assert "head_scores" in result, "def1: 缺少 head_scores"
    expected_mods = _expected_attn_mods(model)
    _check_head_scores_structure(result["head_scores"], expected_mods,
                                  num_heads, "def1")

    # 所有头分数非负
    for m, hs in result["head_scores"].items():
        for hk, v in hs.items():
            assert v >= 0, f"def1 {m}/{hk}: 分数为负 {v}"

    # θ⁰ 完整恢复
    for n, p in model.named_parameters():
        delta = (p.data - theta0[n]).abs().max().item()
        assert delta < 1e-7, f"def1: 参数 {n} 未恢复，偏差={delta:.2e}"

    print(f"✓ B1 def1 head_granularity: {len(expected_mods)} 模块 × {num_heads} 头，"
          f"非负，θ⁰ 已恢复")


def test_B2_def2_head_granularity():
    """def2 head_granularity=True：head_scores ≤ 1（Cauchy-Schwarz），结构正确。"""
    model, dl, device = _make_hf_model_and_dl()
    num_heads = model.config.num_attention_heads

    metric = GradCurvatureMetric()
    result = metric.compute(model, dl, device,
                             num_batches=4, head_granularity=True)

    assert "head_scores" in result, "def2: 缺少 head_scores"
    expected_mods = _expected_attn_mods(model)
    _check_head_scores_structure(result["head_scores"], expected_mods,
                                  num_heads, "def2")

    # 所有头分数 ≤ 1（Cauchy-Schwarz 保证）
    for m, hs in result["head_scores"].items():
        for hk, v in hs.items():
            assert 0 <= v <= 1.0 + 1e-6, \
                f"def2 {m}/{hk}: score={v:.4f} 超出 [0,1]"

    # module_scores 应仍存在（向后兼容）
    assert "module_scores" in result

    print(f"✓ B2 def2 head_granularity: {len(expected_mods)} 模块 × {num_heads} 头，"
          f"所有值 ≤ 1")


def test_B3_def3_head_granularity():
    """def3 head_granularity=True：head_scores 存在，非负，steps_collected 正确。"""
    model, dl, device = _make_hf_model_and_dl()
    num_heads  = model.config.num_attention_heads
    T_early    = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EarlyGradNormCallback(model=model, T_early=T_early,
                                   save_dir=tmpdir, head_granularity=True)
        _simulate_training(cb, model, dl, total_steps=T_early + 2, device=device)

        # 头级别累积器应有值
        assert cb._attn_cfg is not None, "def3: attn_cfg 为 None"
        for m, heads in cb._head_acc.items():
            for h, v in heads.items():
                assert v >= 0, f"def3 {m}/head_{h}: head_acc={v} < 0"

        assert cb._steps_collected == T_early

        # JSON 中应有 head_scores
        out_file = Path(tmpdir) / "def3_early_grad_norm.json"
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "head_scores" in data, "def3: JSON 中缺少 head_scores"

        expected_mods = _expected_attn_mods(model)
        _check_head_scores_structure(data["head_scores"], expected_mods,
                                      num_heads, "def3")
        for m, hs in data["head_scores"].items():
            for hk, v in hs.items():
                assert v >= 0, f"def3 {m}/{hk}: 分数为负 {v}"

    print(f"✓ B3 def3 head_granularity: {len(expected_mods)} 模块 × {num_heads} 头，"
          f"非负，JSON 含 head_scores")


def test_B4_def4_head_granularity():
    """def4 head_granularity=True：head_scores ∈ [0, 1]（Jensen 不等式），结构正确。"""
    model, dl, device = _make_hf_model_and_dl()
    num_heads = model.config.num_attention_heads

    metric = PpredMetric()
    result = metric.compute(model, dl, device,
                             num_batches=4, head_granularity=True)

    assert "head_scores" in result, "def4: 缺少 head_scores"
    expected_mods = _expected_attn_mods(model)
    _check_head_scores_structure(result["head_scores"], expected_mods,
                                  num_heads, "def4")

    for m, hs in result["head_scores"].items():
        for hk, v in hs.items():
            assert 0 <= v <= 1.0 + 1e-6, \
                f"def4 {m}/{hk}: Ppred={v:.4f} 超出 [0, 1]"

    print(f"✓ B4 def4 head_granularity: {len(expected_mods)} 模块 × {num_heads} 头，"
          f"所有 Ppred ∈ [0, 1]")


# ============================================================================
# Part C：向后兼容测试
# ============================================================================

def test_C1_no_head_scores_by_default():
    """head_granularity=False（默认）时，所有定义均不产生 head_scores。"""
    model, dl, device = _make_hf_model_and_dl()

    for name, MetricCls, kw in [
        ("def1", ProbeDeltaMetric,   {"probe_steps": 3, "probe_lr": 1e-3}),
        ("def2", GradCurvatureMetric, {"num_batches": 4}),
        ("def4", PpredMetric,         {"num_batches": 4}),
    ]:
        metric = MetricCls()
        result = metric.compute(model, dl, device, **kw)
        assert "head_scores" not in result, \
            f"{name} 默认不应含 head_scores，但检测到了"

    # def3 callback
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EarlyGradNormCallback(model=model, T_early=3,
                                   save_dir=tmpdir, head_granularity=False)
        _simulate_training(cb, model, dl, total_steps=5, device=device)
        out_file = Path(tmpdir) / "def3_early_grad_norm.json"
        with open(out_file) as f:
            data = json.load(f)
        assert "head_scores" not in data, "def3 默认不应含 head_scores"

    print("✓ C1 所有定义默认不含 head_scores（向后兼容）")


def test_C2_no_config_graceful_degradation():
    """模型无 config 时，head_granularity=True 安全降级，不抛出异常。"""
    model  = TinyClassifier()  # 无 config
    dl     = make_fake_dataloader(batch_size=2, num_batches=4)
    device = torch.device("cpu")

    for name, MetricCls, kw in [
        ("def1", ProbeDeltaMetric,   {"probe_steps": 3, "probe_lr": 1e-3}),
        ("def2", GradCurvatureMetric, {"num_batches": 4}),
        ("def4", PpredMetric,         {"num_batches": 4}),
    ]:
        metric = MetricCls()
        try:
            result = metric.compute(model, dl, device,
                                     head_granularity=True, **kw)
            assert "head_scores" not in result, \
                f"{name} 无 config 时不应含 head_scores"
        except Exception as e:
            raise AssertionError(f"{name} 无 config 时抛出异常: {e}")

    # def3 callback
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EarlyGradNormCallback(model=model, T_early=3,
                                   save_dir=tmpdir, head_granularity=True)
        assert cb._attn_cfg is None, "无 config 时 _attn_cfg 应为 None"
        _simulate_training(cb, model, dl, total_steps=5, device=device)
        out_file = Path(tmpdir) / "def3_early_grad_norm.json"
        with open(out_file) as f:
            data = json.load(f)
        assert "head_scores" not in data, "def3 无 config 时不应含 head_scores"

    print("✓ C2 所有定义无 config 时 head_granularity=True 安全降级")


# ============================================================================
# Part D：Runner 集成测试
# ============================================================================

def test_D1_runner_run_pre_head_granularity():
    """UpdateResponseRunner(head_granularity=True).run_pre() 生成含 head_scores 的 JSON。"""
    model, dl, device = _make_hf_model_and_dl()

    runner = UpdateResponseRunner.from_str(
        "def1,def2,def4",
        metric_kwargs={
            "def1": {"probe_steps": 3, "probe_lr": 1e-3},
            "def2": {"num_batches": 4},
            "def4": {"num_batches": 4},
        },
        head_granularity=True,
    )
    assert runner.head_granularity is True

    with tempfile.TemporaryDirectory() as tmpdir:
        runner.run_pre(model, dl, device, save_dir=tmpdir)

        for fname in ("def1_probe_delta.json",
                      "def2_grad_curvature.json",
                      "def4_ppred.json"):
            p = Path(tmpdir) / fname
            assert p.exists(), f"{fname} 未生成"
            with open(p) as f:
                data = json.load(f)
            assert "head_scores" in data, f"{fname} 缺少 head_scores"

    print("✓ D1 Runner head_granularity=True: def1/def2/def4 JSON 均含 head_scores")


def test_D2_runner_training_callback_head_granularity():
    """make_training_callbacks(head_granularity=True) 正确传递给 def3。"""
    model, dl, device = _make_hf_model_and_dl()
    runner = UpdateResponseRunner.from_str(
        "def3",
        metric_kwargs={"def3": {"T_early": 5}},
        head_granularity=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cbs = runner.make_training_callbacks(model, save_dir=tmpdir)
        assert len(cbs) == 1
        cb = cbs[0]
        assert isinstance(cb, EarlyGradNormCallback)
        assert cb._head_granularity is True
        assert cb._attn_cfg is not None, "def3 callback 的 _attn_cfg 不应为 None"

    print("✓ D2 Runner def3 callback head_granularity 正确传递")


def test_D3_runner_from_str_head_granularity():
    """from_str(..., head_granularity=True) 正确设置 runner.head_granularity。"""
    runner = UpdateResponseRunner.from_str(
        "def1,def2,def3,def4", head_granularity=True
    )
    assert runner.head_granularity is True
    print("✓ D3 from_str head_granularity=True 正确")


def test_D4_runner_full_pipeline_head_granularity():
    """def1+def2+def3+def4 全流程，head_granularity=True，所有 JSON 含 head_scores。"""
    model, dl, device = _make_hf_model_and_dl()

    runner = UpdateResponseRunner.from_str(
        "def1,def2,def3,def4",
        metric_kwargs={
            "def1": {"probe_steps": 3, "probe_lr": 1e-3},
            "def2": {"num_batches": 4},
            "def3": {"T_early": 5},
            "def4": {"num_batches": 4},
        },
        head_granularity=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        runner.run_pre(model, dl, device, save_dir=tmpdir)
        cbs = runner.make_training_callbacks(model, save_dir=tmpdir)
        _simulate_training(cbs[0], model, dl, total_steps=7, device=device)

        for fname in ("def1_probe_delta.json", "def2_grad_curvature.json",
                      "def3_early_grad_norm.json", "def4_ppred.json"):
            p = Path(tmpdir) / fname
            assert p.exists(), f"{fname} 未生成"
            with open(p) as f:
                data = json.load(f)
            assert "module_scores" in data, f"{fname} 缺少 module_scores"
            assert "head_scores"   in data, f"{fname} 缺少 head_scores"

    print("✓ D4 完整流程：def1+def2+def3+def4 全部含 head_scores")


# ============================================================================
# Part E：结果打印（供人工核验）
# ============================================================================

def test_E1_print_def4_head_ppred():
    """打印 def4 各注意力头 Ppred，供人工核验。"""
    model, dl, device = _make_hf_model_and_dl()
    metric = PpredMetric()
    result = metric.compute(model, dl, device, num_batches=8, head_granularity=True)

    num_heads = model.config.num_attention_heads
    print(f"\ndef4 头级别 Ppred（TinyHFClassifier，hidden=8，{num_heads} 头）：")
    header = f"{'模块名（末三级）':<45}" + "".join(
        f"  {'head_'+str(h):>8}" for h in range(num_heads)
    ) + f"  {'模块':>8}"
    print(header)
    print("-" * (47 + 10 * num_heads + 10))

    for mod, hs in result["head_scores"].items():
        short = ".".join(mod.split(".")[-3:])
        mod_score = result["module_scores"].get(mod, float("nan"))
        row = f"{short:<45}" + "".join(
            f"  {hs.get(f'head_{h}', float('nan')):>8.4f}" for h in range(num_heads)
        ) + f"  {mod_score:>8.4f}"
        print(row)


def test_E2_print_def1_head_delta():
    """打印 def1 各注意力头参数位移，供人工核验。"""
    model, dl, device = _make_hf_model_and_dl()
    metric = ProbeDeltaMetric()
    result = metric.compute(model, dl, device,
                             probe_steps=10, probe_lr=1e-3, head_granularity=True)

    num_heads = model.config.num_attention_heads
    print(f"\ndef1 头级别参数位移（TinyHFClassifier，probe_steps=10）：")
    header = f"{'模块名（末三级）':<45}" + "".join(
        f"  {'head_'+str(h):>8}" for h in range(num_heads)
    ) + f"  {'模块':>10}"
    print(header)
    print("-" * (47 + 10 * num_heads + 12))

    for mod, hs in result["head_scores"].items():
        short = ".".join(mod.split(".")[-3:])
        mod_score = result["module_scores"].get(mod, float("nan"))
        row = f"{short:<45}" + "".join(
            f"  {hs.get(f'head_{h}', float('nan')):>8.4f}" for h in range(num_heads)
        ) + f"  {mod_score:>10.6f}"
        print(row)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  update_response 注意力头级别粒度测试")
    print("=" * 65)

    print("\n── Part A：工具函数 ──")
    test_A1_attn_head_import()
    test_A2_attn_module_scan()

    print("\n── Part B：各定义头级别集成测试 ──")
    test_B1_def1_head_granularity()
    test_B2_def2_head_granularity()
    test_B3_def3_head_granularity()
    test_B4_def4_head_granularity()

    print("\n── Part C：向后兼容测试 ──")
    test_C1_no_head_scores_by_default()
    test_C2_no_config_graceful_degradation()

    print("\n── Part D：Runner 集成测试 ──")
    test_D1_runner_run_pre_head_granularity()
    test_D2_runner_training_callback_head_granularity()
    test_D3_runner_from_str_head_granularity()
    test_D4_runner_full_pipeline_head_granularity()

    print("\n── Part E：结果打印 ──")
    test_E1_print_def4_head_ppred()
    test_E2_print_def1_head_delta()

    print("\n\n所有测试通过 ✓")
