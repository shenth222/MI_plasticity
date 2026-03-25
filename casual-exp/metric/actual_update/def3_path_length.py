"""
metric/actual_update/def3_path_length.py

定义三：累计路径长度
─────────────────────────────────────────────────────────────────────────────
公式：

    U_m^{path} = Σ_{t=1}^{T} ||θ_m^{(t)} - θ_m^{(t-1)}||_2

其中：
    ||θ_m^{(t)} - θ_m^{(t-1)}||_2 = sqrt( Σ_{p ∈ θ_m} ||Δp_t||_F² )

头级别（head_granularity=True）：

    U_h^{path} = Σ_{t=1}^{T} ||θ_h^{(t)} - θ_h^{(t-1)}||_2

    ||θ_h^{(t)} - θ_h^{(t-1)}||_2 = sqrt( Σ_{p∈{w,b}} ||view_h(Δp_t)||_F² )

实现细节：
    · head_granularity=False：每步只计算参数级标量范数（prev_params 必须保留）
    · head_granularity=True：每步额外切片 Δp 张量按头聚合，切片后即可释放 delta 张量
      峰值内存 ≈ 1 份参数快照（CPU） + 1 份步进 delta 张量（短暂，处理后释放）
      delta 张量大小 ≈ 参数量级别（约 700MB fp32 / 350MB bf16 for DeBERTa）
      但仅在 on_step_end 期间保留，随后立即释放，长期额外内存为 0

─────────────────────────────────────────────────────────────────────────────
保存格式：
    def3_path_length.json
    {
      "module_scores":   {module_name: float, ...},  # 累计路径长度
      "param_scores":    {param_name:  float, ...},  # 参数粒度路径长度
      "steps_collected": int,
      "log_every":       int,
      "head_scores":     {module_name: {"head_0": float, ...}}  # 仅 head_granularity=True
    }
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base import PathMetric, group_params_by_module, snapshot_params, resolve_param_dict
from .attn_head import (
    get_attn_head_config,
    get_attn_modules,
    compute_head_delta_l2,
    AttnHeadConfig,
)


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class PathLengthCallback(TrainerCallback):
    """
    逐步累积各模块参数更新轨迹的路径长度（可选地按注意力头细分）。

    核心循环（每个计算步 t）：
      1. 从 GPU 拉取当前参数至 CPU：curr_p
      2. 计算步进变化量：delta_p = curr_p - prev_p
      3. 各模块路径长度 += sqrt( Σ_{p∈m} ||delta_p||_F² )（模块 L2 组合）
      4. 参数路径长度   += ||delta_p||_F
      5. [仅 head_granularity] 各头路径长度 += sqrt( Σ_{p∈h} ||view_h(delta_p)||_F² )
      6. 更新 prev_params = curr_p

    内存开销：
      · ~1 份参数快照（prev_params，CPU，始终保留）
      · head_granularity=True 时：步进 delta 张量额外在 on_step_end 期间保留，
        完成头级别累积后即刻释放，不增加长期内存占用
    """

    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        log_every: int = 1,
        head_granularity: bool = False,
        name: str = "def3_path_length",
    ):
        self._save_dir        = save_dir
        self._log_every       = log_every
        self._name            = name
        self._head_granularity = head_granularity
        self._module_groups   = group_params_by_module(model)

        self._step:            int = 0
        self._steps_collected: int = 0

        # θ^(0) 作为初始 prev_params
        self._prev_params: Dict[str, torch.Tensor] = snapshot_params(model)

        # 模块级累积器
        self._module_acc: Dict[str, float] = {m: 0.0 for m in self._module_groups}
        # 参数级累积器
        self._param_acc:  Dict[str, float] = {n: 0.0 for n in self._prev_params}

        # 头级别累积器与配置（仅 head_granularity=True）
        self._attn_cfg:  Optional[AttnHeadConfig] = None
        self._attn_mods: Dict[str, str]           = {}
        self._head_acc:  Dict[str, Dict[int, float]] = {}

        if head_granularity:
            self._attn_cfg = get_attn_head_config(model)
            if self._attn_cfg is None:
                print(f"[{name}] head_granularity=True 但模型无 config，跳过头级别计算")
            else:
                self._attn_mods = get_attn_modules(model, self._attn_cfg)
                self._head_acc  = {
                    m: {h: 0.0 for h in range(self._attn_cfg.num_heads)}
                    for m in self._attn_mods
                }

        log_str = f"log_every={log_every}" if log_every > 1 else "逐步精确计算"
        head_str = f"，头级别 {len(self._attn_mods)} 模块" if self._attn_cfg else ""
        print(
            f"[{name}] 路径长度收集器已初始化（"
            f"{len(self._prev_params)} 个可训练参数，{log_str}{head_str}）"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ) -> TrainerControl:
        self._step += 1

        if model is None or (self._log_every > 1 and self._step % self._log_every != 0):
            return control

        param_dict = resolve_param_dict(model)

        # ── 计算步进 delta 张量（head_granularity 时保留张量，否则只保留标量）──
        step_delta_scalars: Dict[str, float] = {}
        step_delta_tensors: Dict[str, torch.Tensor] = {}  # 仅 head_granularity=True

        for name, prev in self._prev_params.items():
            p = param_dict.get(name)
            if p is None:
                continue
            delta = p.data.cpu() - prev
            step_delta_scalars[name] = delta.norm(p=2).item()
            if self._head_granularity and self._attn_cfg is not None:
                step_delta_tensors[name] = delta

        # ── 参数级路径长度累积 ────────────────────────────────────────────────
        for name, d in step_delta_scalars.items():
            self._param_acc[name] = self._param_acc.get(name, 0.0) + d

        # ── 模块级路径长度累积 ────────────────────────────────────────────────
        for m_name, param_names in self._module_groups.items():
            sq_sum = sum(step_delta_scalars.get(pn, 0.0) ** 2 for pn in param_names)
            self._module_acc[m_name] += sq_sum ** 0.5

        # ── 头级别路径长度累积 ────────────────────────────────────────────────
        if self._attn_cfg is not None and step_delta_tensors:
            step_head_deltas = compute_head_delta_l2(
                step_delta_tensors, self._attn_cfg, self._attn_mods
            )
            for m_name, per_head in step_head_deltas.items():
                for hk, val in per_head.items():
                    h = int(hk.split("_")[1])
                    self._head_acc[m_name][h] += val
            # 立即释放 delta 张量，不增加长期内存
            step_delta_tensors.clear()

        # ── 更新 prev_params = 当前参数 ──────────────────────────────────────
        for name in self._prev_params:
            p = param_dict.get(name)
            if p is not None:
                self._prev_params[name] = p.data.clone().cpu()

        self._steps_collected += 1
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero:
            return control

        result: Dict[str, Any] = {
            "module_scores":   dict(self._module_acc),
            "param_scores":    dict(self._param_acc),
            "steps_collected": self._steps_collected,
            "log_every":       self._log_every,
        }

        if self._attn_cfg is not None and self._head_acc:
            result["head_scores"] = {
                m_name: {f"head_{h}": v for h, v in heads.items()}
                for m_name, heads in self._head_acc.items()
            }

        save_dir = Path(self._save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{self._name}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[{self._name}] 路径长度计算完成，"
            f"共累积 {self._steps_collected} 步 → {path}"
        )

        return control


# ---------------------------------------------------------------------------
# PathMetric 包装类
# ---------------------------------------------------------------------------

class PathLengthMetric(PathMetric):
    """
    累计路径长度（定义三）—— PathMetric 包装类。

    输出字段：
      · module_scores  U_m^path（主指标）
      · param_scores   参数粒度路径长度
      · head_scores    注意力头级别路径长度（仅 head_granularity=True）
    """

    name = "def3_path_length"

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        log_every: int = 1,
        head_granularity: bool = False,
        **kwargs,
    ) -> PathLengthCallback:
        """
        Args:
            model:            未经 DDP 包装的原始模型
            save_dir:         结果保存目录
            log_every:        每隔多少 step 计算一次（1=精确；>1=近似）
            head_granularity: 是否额外计算注意力头级别路径长度
        """
        return PathLengthCallback(
            model=model, save_dir=save_dir,
            log_every=log_every, head_granularity=head_granularity, name=self.name,
        )
