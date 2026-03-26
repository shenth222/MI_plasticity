"""
metric/training_gain/def3_path_integral.py

定义三：路径积分（Path Integral）
─────────────────────────────────────────────────────────────────────────────
公式：
    G_m^(PI) = Σ_{t=1}^{T} ∇_{θ_m} L(θ^(t)) · Δθ_{m,t}

其中：
    ∇_{θ_m} L(θ^(t)) ∈ R^{|θ_m|}  — 第 t 步模块 m 的梯度向量
    Δθ_{m,t} = θ_m^(t) − θ_m^{(t-1)} ∈ R^{|θ_m|}  — 第 t 步参数更新量
    · = 向量内积（element-wise 乘积后求和）

物理含义：
  · 一阶 Taylor 近似：Σ_t g_t · Δθ_t ≈ L(θ^(T)) − L(θ^(0))（对全体参数求和等于总 loss 变化）
  · 对模块 m 单独计算：G_m^(PI) 衡量模块 m 的参数更新在整个训练路径上
    对总 loss 变化的累积贡献（一阶近似）
  · 通常 G_m^(PI) ≤ 0（梯度下降使 loss 持续降低）；|G_m^(PI)| 越大贡献越多

与 saliency.py 中 taylor 定义的区别：
  saliency.py（taylor）:
    I_m^taylor = Σ_{i ∈ θ_m} |θ_i · g_i|   @θ^(0) 单点估计（训练前）
    含义：若将参数置零，loss 约增加多少
    特点：取绝对值，无方向，单步估计，计算简单
  路径积分（PI）：
    G_m^PI = Σ_t Σ_{i ∈ θ_m} g_{i,t} · Δθ_{i,t}  沿整个训练路径累积
    含义：训练过程中模块 m 参数更新实际使 loss 下降了多少
    特点：有符号，沿路径累积，反映真实训练动态，需要训练中持续记录

头级别（head_granularity=True）：
    G_h^(PI) = Σ_t Σ_{i ∈ view_h(θ_m)} g_{i,t} · Δθ_{i,t}
    对注意力模块额外按头切片计算各头的路径积分贡献。

梯度捕获机制：
    通过 param.register_hook 在 backward 期间捕获梯度（此时梯度尚未被 zero_grad 清除）。
    支持梯度累积（gradient_accumulation_steps > 1）：
      · on_step_begin 清空 _grad_buffer
      · 每个 substep 的 backward 触发 hook，梯度在 _grad_buffer 中累积
      · on_step_end 计算 Δθ（curr − prev），与 _grad_buffer 中的累积梯度做内积

注意：
    · 梯度 hook 捕获的是梯度裁剪（clip_grad_norm）前的原始梯度
    · 分布式训练时，hook 捕获的是该进程的本地梯度（all-reduce 前）；
      def3 仅在主进程（accelerator.is_main_process）上创建 callback 以避免重复

─────────────────────────────────────────────────────────────────────────────
保存格式：

def3_path_integral.json
{
  "module_scores":   {module_name: G_m_PI, ...},   # 主指标（有符号）
  "param_scores":    {param_name:  G_p_PI, ...},   # 参数粒度（有符号）
  "head_scores":     {module_name: {"head_0": G_h_PI, ...}, ...},  # head_granularity=True
  "steps_collected": int,
  "log_every":       int
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .base import (
    PathIntegralMetric,
    group_params_by_module,
    resolve_param_dict,
    snapshot_params,
)
from .attn_head import (
    AttnHeadConfig,
    get_attn_head_config,
    get_attn_modules,
    compute_head_pi_dot,
)


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class PathIntegralCallback(TrainerCallback):
    """
    逐步累积路径积分 G_m^(PI) = Σ_t ∇_{θ_m}L(θ^(t)) · Δθ_{m,t}。

    核心循环（每个 optimizer step t）：
      1. on_step_begin：清空 _grad_buffer（为本步梯度累积做准备）
      2. backward（可能多个 substep）：
           → gradient hook 触发，将本步所有 substep 梯度累积到 _grad_buffer
      3. optimizer.step() → zero_grad()
      4. on_step_end：
           a. curr_params = 从模型读取当前参数（GPU → CPU）
           b. Δθ = curr_params − prev_params
           c. param_acc[name] += (grad_buffer[name] · Δθ).sum()
           d. module_acc[m]   += Σ_{p ∈ m} param_acc_step[p]   ← 步进贡献累积
           e. head_acc[m][h]  += 步进头级别内积                  ← 仅 head_granularity
           f. prev_params = curr_params

    内存开销：
      · _prev_params：~1 份参数（CPU，始终保留）
      · _grad_buffer：~1 份参数（CPU fp32，on_step_begin 清空，on_step_end 使用后仍保留到下个 on_step_begin）
      · head_granularity=True 时：步进 delta_tensors 和 grad_tensors
        在 on_step_end 期间短暂保留，处理后立即释放
    """

    def __init__(
        self,
        model:            torch.nn.Module,
        save_dir:         str,
        log_every:        int = 1,
        head_granularity: bool = False,
        name:             str = "def3_path_integral",
    ):
        """
        Args:
            model:            未经 DDP 包装的原始模型（Trainer 创建前传入）
            save_dir:         结果保存目录
            log_every:        每隔多少 optimizer step 计算一次（1=精确；>1=近似，省计算）
            head_granularity: 是否额外计算注意力头级别路径积分
            name:             用于日志和文件名
        """
        self._save_dir         = save_dir
        self._log_every        = log_every
        self._name             = name
        self._head_granularity = head_granularity
        self._module_groups    = group_params_by_module(model)

        self._step:            int = 0
        self._steps_collected: int = 0

        # θ^(0) 作为初始 prev_params
        self._prev_params: Dict[str, torch.Tensor] = snapshot_params(model)

        # 梯度缓冲区（在每步 on_step_begin 清空，hook 填充）
        self._grad_buffer: Dict[str, torch.Tensor] = {}

        # 参数级路径积分累积器（scalar，整个训练保留）
        self._param_acc: Dict[str, float] = {n: 0.0 for n in self._prev_params}

        # 模块级路径积分累积器（scalar，整个训练保留）
        self._module_acc: Dict[str, float] = {m: 0.0 for m in self._module_groups}

        # 头级别路径积分累积器（scalar，整个训练保留）
        self._attn_cfg:  Optional[AttnHeadConfig]         = None
        self._attn_mods: Optional[Dict[str, str]]         = None
        self._head_acc:  Dict[str, Dict[int, float]]      = {}

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

        # ── 注册梯度 hook（在 backward 期间捕获梯度到 _grad_buffer）──────────
        self._hooks = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                hook = self._make_grad_hook(param_name)
                handle = param.register_hook(hook)
                self._hooks.append(handle)

        log_str  = f"log_every={log_every}" if log_every > 1 else "逐步精确计算"
        head_str = (
            f"，头级别 {len(self._attn_mods)} 模块 × {self._attn_cfg.num_heads} 头"
            if self._attn_cfg and self._attn_mods else ""
        )
        print(
            f"[{name}] 路径积分收集器已初始化（"
            f"{len(self._prev_params)} 个参数，{log_str}{head_str}，"
            f"已注册 {len(self._hooks)} 个梯度 hook）"
        )

    def _make_grad_hook(self, param_name: str):
        """为参数创建梯度 hook。在每次 backward 触发时，将梯度累积到 _grad_buffer。"""

        def hook(grad: torch.Tensor) -> None:
            if grad is None:
                return
            # 转为 float32 CPU（与 prev_params 格式一致）
            g = grad.detach().float().cpu()
            if param_name in self._grad_buffer:
                self._grad_buffer[param_name] += g
            else:
                self._grad_buffer[param_name] = g.clone()

        return hook

    def on_step_begin(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """每个 optimizer step 开始时清空梯度缓冲区（含梯度累积的所有 substep）。"""
        self._grad_buffer.clear()
        return control

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        model:   torch.nn.Module = None,
        **kwargs,
    ) -> TrainerControl:
        self._step += 1

        if model is None:
            return control
        if self._log_every > 1 and self._step % self._log_every != 0:
            return control

        param_dict = resolve_param_dict(model)

        # ── 计算步进 Δθ ──────────────────────────────────────────────────────
        step_param_pi:      Dict[str, float]         = {}  # 参数级步进贡献
        step_delta_tensors: Dict[str, torch.Tensor]  = {}  # head_granularity 时保留
        step_grad_tensors:  Dict[str, torch.Tensor]  = {}  # head_granularity 时保留

        for pname, prev in self._prev_params.items():
            p = param_dict.get(pname)
            g = self._grad_buffer.get(pname)
            if p is None:
                continue

            delta = p.data.float().cpu() - prev.float()

            if g is not None:
                pi_step = (g * delta).sum().item()
                step_param_pi[pname] = pi_step
                self._param_acc[pname] = self._param_acc.get(pname, 0.0) + pi_step

                if self._head_granularity and self._attn_cfg is not None:
                    step_delta_tensors[pname] = delta
                    step_grad_tensors[pname]  = g

        # ── 模块级累积 ───────────────────────────────────────────────────────
        for m_name, param_names in self._module_groups.items():
            m_pi = sum(step_param_pi.get(pn, 0.0) for pn in param_names)
            self._module_acc[m_name] = self._module_acc.get(m_name, 0.0) + m_pi

        # ── 头级别累积 ───────────────────────────────────────────────────────
        if (
            self._attn_cfg is not None
            and self._attn_mods is not None
            and step_delta_tensors
        ):
            step_head = compute_head_pi_dot(
                step_delta_tensors, step_grad_tensors,
                self._attn_cfg, self._attn_mods,
            )
            for m_name, per_head in step_head.items():
                for hk, val in per_head.items():
                    h = int(hk.split("_")[1])
                    self._head_acc[m_name][h] = (
                        self._head_acc[m_name].get(h, 0.0) + val
                    )
            # 立即释放，不增加长期内存
            step_delta_tensors.clear()
            step_grad_tensors.clear()

        # ── 更新 prev_params ─────────────────────────────────────────────────
        for pname in self._prev_params:
            p = param_dict.get(pname)
            if p is not None:
                self._prev_params[pname] = p.data.float().cpu()

        self._steps_collected += 1
        return control

    def on_train_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not state.is_world_process_zero:
            return control

        # 移除梯度 hook（避免残留）
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

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
            f"[{self._name}] 路径积分计算完成，"
            f"共累积 {self._steps_collected} 步 → {path}"
        )
        return control


# ---------------------------------------------------------------------------
# PathIntegralMetric 包装类
# ---------------------------------------------------------------------------

class PathIntegralGainMetric(PathIntegralMetric):
    """
    路径积分训练收益（定义三）—— PathIntegralMetric 包装类。

    输出字段：
      · module_scores  G_m^(PI) 主指标（有符号，通常 ≤ 0）
      · param_scores   参数粒度路径积分
      · head_scores    注意力头级别路径积分（仅 head_granularity=True）
      · steps_collected 累积的 optimizer step 数
    """

    name = "def3_path_integral"

    def make_callback(
        self,
        model:            torch.nn.Module,
        save_dir:         str,
        log_every:        int = 1,
        head_granularity: bool = False,
        **kwargs,
    ) -> PathIntegralCallback:
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:            未经 DDP 包装的原始模型
            save_dir:         结果保存目录
            log_every:        每隔多少 optimizer step 计算一次路径积分贡献
                              （1=精确；>1=近似，省计算开销；建议大任务 MNLI 设为 10）
            head_granularity: 是否额外计算注意力头级别路径积分
        """
        return PathIntegralCallback(
            model=model,
            save_dir=save_dir,
            log_every=log_every,
            head_granularity=head_granularity,
            name=self.name,
        )
