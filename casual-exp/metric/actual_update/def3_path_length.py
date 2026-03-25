"""
metric/actual_update/def3_path_length.py

定义三：累计路径长度
─────────────────────────────────────────────────────────────────────────────
公式：

    U_m^{path} = Σ_{t=1}^{T} ||θ_m^{(t)} - θ_m^{(t-1)}||_2

其中：
    ||θ_m^{(t)} - θ_m^{(t-1)}||_2 = sqrt( Σ_{p ∈ θ_m} ||p^{(t)} - p^{(t-1)}||_F² )
    t 的粒度为每个 optimizer step（精确）或每 log_every 步（近似，省计算）

与定义一的关系（三角不等式）：
    U_m^{path} ≥ U_m^{(A)}（路径长度 ≥ 直线距离）
    当且仅当优化轨迹严格单向时等号成立；
    路径长度越大于直线距离，说明参数在参数空间中越"曲折"（振荡或震荡）。

实现细节：
    · on_step_end 时，将当前参数拷贝到 CPU，与 prev_params 做差，
      计算各模块的步进 L2 变化量，累积到 module_acc，然后更新 prev_params。
    · prev_params 始终存于 CPU（float，与原始数据类型一致），
      避免占用 GPU 显存（DeBERTa-v3-base fp32 ≈ 700MB CPU RAM）。
    · log_every > 1 时为近似路径长度：仅在 t % log_every == 0 的步骤计算，
      适合参数量大或步数极多的场景。

嵌入训练（最小侵入）：
    metric = PathLengthMetric()
    cb = metric.make_callback(model, save_dir, log_every=1)
    trainer = Trainer(..., callbacks=[..., cb])

─────────────────────────────────────────────────────────────────────────────
保存格式：
    def3_path_length.json
    {
      "module_scores":   {module_name: float, ...},  # 累计路径长度（主指标）
      "param_scores":    {param_name:  float, ...},  # 参数粒度路径长度
      "steps_collected": int,                         # 实际计算的步数
      "log_every":       int,                         # 记录粒度
    }
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base import PathMetric, group_params_by_module, snapshot_params, resolve_param_dict


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class PathLengthCallback(TrainerCallback):
    """
    逐步累积各模块参数更新轨迹的路径长度。

    核心循环（每个计算步 t）：
      1. 从 GPU 拉取当前参数至 CPU：curr_p
      2. 计算步进变化量：delta_p = curr_p - prev_p
      3. 各模块路径长度 += sqrt( Σ_{p∈m} ||delta_p||_F² )
      4. 参数路径长度   += ||delta_p||_F（各参数独立）
      5. 更新 prev_params = curr_p

    内存开销：~1 份参数快照（CPU），DeBERTa-v3-base fp32 ≈ 700MB CPU RAM。
    若训练使用 bf16/fp16，快照也以对应精度存储（约 350MB）。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        log_every: int = 1,
        name: str = "def3_path_length",
    ):
        """
        Args:
            model:     未经 DDP 包装的原始模型（在 Trainer 创建前传入）
            save_dir:  结果保存目录
            log_every: 计算步进变化量的频率（1 = 每步精确计算；N > 1 = 近似，每 N 步计算一次）
            name:      JSON 文件名（无需修改）
        """
        self._save_dir      = save_dir
        self._log_every     = log_every
        self._name          = name
        self._module_groups = group_params_by_module(model)

        # 全局步计数（实际 optimizer step 数，含跳过的步）
        self._step: int = 0
        # 实际累积的步数（扣除 log_every 跳过的步）
        self._steps_collected: int = 0

        # 初始化 prev_params = θ^(0)（存于 CPU）
        self._prev_params: Dict[str, torch.Tensor] = snapshot_params(model)

        # 累积器（模块级）：module_acc[m] = U_m^path
        self._module_acc: Dict[str, float] = {
            m: 0.0 for m in self._module_groups
        }
        # 累积器（参数级）：param_acc[p] = Σ_t ||delta_p_t||_F
        self._param_acc: Dict[str, float] = {
            n: 0.0 for n in self._prev_params
        }

        log_str = f"log_every={log_every}" if log_every > 1 else "逐步精确计算"
        print(
            f"[{name}] 路径长度收集器已初始化（"
            f"{len(self._prev_params)} 个可训练参数，{log_str}）"
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

        # log_every 跳过逻辑：仅在 step % log_every == 0 时计算
        if model is None or (self._log_every > 1 and self._step % self._log_every != 0):
            return control

        param_dict = resolve_param_dict(model)  # 自动处理 DDP 前缀

        # ── 计算步进变化量，累积到路径长度 ────────────────────────────────────
        step_delta: Dict[str, float] = {}  # {param_name: ||delta_p||_F}

        for name, prev in self._prev_params.items():
            p = param_dict.get(name)
            if p is None:
                continue
            delta = p.data.cpu() - prev          # delta_p = θ^(t) - θ^(t-1)
            step_delta[name] = delta.norm(p=2).item()  # ||delta_p||_F

        # 参数级路径长度累积
        for name, d in step_delta.items():
            self._param_acc[name] = self._param_acc.get(name, 0.0) + d

        # 模块级路径长度累积（L2 组合：sqrt(Σ_p ||delta_p||² ) per module per step）
        for m_name, param_names in self._module_groups.items():
            sq_sum = sum(step_delta.get(pn, 0.0) ** 2 for pn in param_names)
            self._module_acc[m_name] += sq_sum ** 0.5

        # ── 更新 prev_params = 当前参数（CPU 快照）────────────────────────────
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
      · module_scores    U_m^path = Σ_t ||θ_m^(t) - θ_m^(t-1)||_2（主指标）
      · param_scores     参数粒度路径长度 = Σ_t ||p^(t) - p^(t-1)||_F
      · steps_collected  实际累积的步数
      · log_every        记录粒度（1 = 精确；>1 = 近似）
    """

    name = "def3_path_length"

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        log_every: int = 1,
        **kwargs,
    ) -> PathLengthCallback:
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:     未经 DDP 包装的原始模型
            save_dir:  结果保存目录
            log_every: 计算频率（1 = 每步精确；N > 1 = 每 N 步近似，
                       推荐大模型使用 log_every=10 或 100 以减少 CPU 压力）
        """
        return PathLengthCallback(
            model=model, save_dir=save_dir, log_every=log_every, name=self.name
        )
