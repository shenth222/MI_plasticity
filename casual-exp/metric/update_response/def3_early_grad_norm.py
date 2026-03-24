"""
metric/update_response/def3_early_grad_norm.py

定义 3：累积早期梯度范数
─────────────────────────────────────────────────────────────────────────────
公式（模块级精确计算）：

    \hat{R}_m = \sum_{t=1}^{T_{early}} \|\nabla_{\theta_m}\mathcal{L}^{(t)}\|_2

头级别公式（head_granularity=True）：

    \hat{R}_h = \sum_{t=1}^{T_{early}} \|\nabla_{\theta_h}\mathcal{L}^{(t)}\|_2

    其中 ‖∇_{θ_h}L^(t)‖₂ = sqrt(Σ_{i∈head_h} g_i^(t)²)。

实现：
    · 非注意力参数（或 head_granularity=False）：
        hook 累积每步的梯度范数平方（标量），on_step_end 聚合。
    · 注意力参数（head_granularity=True）：
        hook 额外累积每步的元素级梯度平方张量，
        on_step_end 按头切片，计算各头梯度范数并累积到 head_acc。
        注意：element-wise 张量仅在 on_step_end 之前保存，之后清空，
        单步内存开销约等于注意力参数的大小（约 14 MB/DeBERTa-v3-base）。

保存格式：
    def3_early_grad_norm.json
    {
      "module_scores":    {module_name: float, ...},
      "param_scores":     {param_name:  float, ...},
      "T_early":          int,
      "steps_collected":  int,
      "head_scores":      {module_name: {"head_0": float, ...}, ...}  # 仅 head_granularity=True
    }
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base import InTrainingMetric, group_params_by_module
from .attn_head import (
    get_attn_head_config,
    get_attn_modules,
    get_head_weight_view,
    get_head_bias_view,
)


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class EarlyGradNormCallback(TrainerCallback):
    """
    在真实训练的前 T_early 步累积各模块的梯度 L2 范数（并可选地输出头级别分数）。

    头级别机制：
        注意力参数除了标量范数钩子外，额外存储 per-step 元素级梯度平方张量
        （保存在 _step_attn_sq_tensor），on_step_end 时按头切片计算头梯度范数，
        随后清空，峰值内存约等于注意力参数的参数量 × 4 字节（float32）。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        T_early: int,
        save_dir: str,
        head_granularity: bool = False,
    ):
        self.T_early          = T_early
        self.save_dir         = save_dir
        self._head_granularity = head_granularity

        self._module_groups = group_params_by_module(model)

        # 标量缓冲（每步内各 micro-batch 梯度范数平方之和）
        self._step_param_sq:    Dict[str, float] = {}
        # 元素级缓冲（仅注意力参数，head_granularity=True 时使用）
        self._step_attn_sq_tensor: Dict[str, torch.Tensor] = {}

        # 累积器
        self._module_acc: Dict[str, float] = {m: 0.0 for m in self._module_groups}
        self._param_acc:  Dict[str, float] = {}

        # 头级别
        self._attn_cfg  = None
        self._attn_mods: Dict[str, str] = {}
        self._head_acc:  Dict[str, Dict[int, float]] = {}
        self._attn_param_set: set = set()

        if head_granularity:
            self._attn_cfg = get_attn_head_config(model)
            if self._attn_cfg is None:
                print("  [def3_early_grad_norm] head_granularity=True 但模型无 config，"
                      "跳过头级别计算")
            else:
                self._attn_mods = get_attn_modules(model, self._attn_cfg)
                self._head_acc = {
                    m: {h: 0.0 for h in range(self._attn_cfg.num_heads)}
                    for m in self._attn_mods
                }
                # 预计算注意力参数名称集合，用于钩子分发
                for m_name in self._attn_mods:
                    for suffix in ("weight", "bias"):
                        self._attn_param_set.add(f"{m_name}.{suffix}")

        # 控制标志
        self._hooks:           List  = []
        self._step:            int   = 0
        self._done:            bool  = False
        self._steps_collected: int   = 0

        # 注册梯度钩子
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._param_acc[name] = 0.0
                is_attn = head_granularity and (name in self._attn_param_set)
                handle = param.register_hook(self._make_hook(name, is_attn=is_attn))
                self._hooks.append(handle)

        print(f"[def3_early_grad_norm] 已注册 {len(self._hooks)} 个梯度钩子，"
              f"将收集前 {T_early} 步"
              + (f"（+头级别 {len(self._attn_mods)} 模块）" if self._attn_cfg else ""))

    def _make_hook(self, param_name: str, is_attn: bool = False):
        """
        工厂函数：生成梯度钩子。

        非注意力参数：累积标量梯度范数平方到 _step_param_sq。
        注意力参数（head_granularity=True）：
            同时累积元素级梯度平方张量到 _step_attn_sq_tensor，
            以及标量到 _step_param_sq（与模块聚合保持一致）。
        """
        def hook(grad: torch.Tensor) -> None:
            if self._done or self._step >= self.T_early:
                return
            norm_sq = grad.detach().pow(2).sum().item()
            self._step_param_sq[param_name] = (
                self._step_param_sq.get(param_name, 0.0) + norm_sq
            )
            if is_attn:
                g_sq = grad.detach().pow(2)
                if param_name in self._step_attn_sq_tensor:
                    self._step_attn_sq_tensor[param_name].add_(g_sq)
                else:
                    self._step_attn_sq_tensor[param_name] = g_sq.clone()
        return hook

    def _commit_step(self):
        """将当前步的梯度数据提交到累积器，并清空临时缓冲。"""
        # 模块级和参数级（精确公式：sqrt(Σ ||g_param||²) per module）
        for m_name, param_names in self._module_groups.items():
            sq_sum = sum(self._step_param_sq.get(pn, 0.0) for pn in param_names)
            self._module_acc[m_name] += sq_sum ** 0.5

        for pn, sq in self._step_param_sq.items():
            self._param_acc[pn] = self._param_acc.get(pn, 0.0) + sq ** 0.5

        self._step_param_sq.clear()
        self._steps_collected += 1

        # 头级别（仅 head_granularity=True 且模型有 config）
        if self._attn_cfg is not None and self._step_attn_sq_tensor:
            for m_name, m_type in self._attn_mods.items():
                for h in range(self._attn_cfg.num_heads):
                    sq_sum = 0.0
                    for suffix in ("weight", "bias"):
                        pn = f"{m_name}.{suffix}"
                        sq_tensor = self._step_attn_sq_tensor.get(pn)
                        if sq_tensor is None:
                            continue
                        if suffix == "weight":
                            view = get_head_weight_view(
                                sq_tensor, m_type, h, self._attn_cfg.head_dim
                            )
                        else:
                            view = get_head_bias_view(
                                sq_tensor, m_type, h, self._attn_cfg.head_dim
                            )
                        if view is not None:
                            sq_sum += view.sum().item()
                    self._head_acc[m_name][h] += sq_sum ** 0.5

            self._step_attn_sq_tensor.clear()

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not self._done and self._step < self.T_early:
            self._commit_step()

        self._step += 1
        if self._step >= self.T_early and not self._done:
            self._done = True
            self._remove_hooks()
            if state.is_world_process_zero:
                self._save()
            print(f"[def3_early_grad_norm] 已完成 {self._steps_collected} 步梯度收集，"
                  f"钩子已卸载")

        return control

    def on_train_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """训练提前结束时的兜底保存。"""
        if not self._done:
            self._done = True
            if self._step_param_sq or self._step_attn_sq_tensor:
                self._commit_step()
            self._remove_hooks()
            if state.is_world_process_zero:
                self._save()
            print(f"[def3_early_grad_norm] 训练结束，共收集 "
                  f"{self._steps_collected} 步（< T_early={self.T_early}）")
        return control

    def _remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _save(self):
        """将累积结果序列化为 JSON。"""
        result: Dict[str, Any] = {
            "module_scores":   dict(self._module_acc),
            "param_scores":    dict(self._param_acc),
            "T_early":         self.T_early,
            "steps_collected": self._steps_collected,
        }
        if self._attn_cfg is not None and self._head_acc:
            result["head_scores"] = {
                m_name: {f"head_{h}": v for h, v in heads.items()}
                for m_name, heads in self._head_acc.items()
            }
        save_path = Path(self.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out_file = save_path / "def3_early_grad_norm.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[def3_early_grad_norm] Saved → {out_file}")


# ---------------------------------------------------------------------------
# InTrainingMetric 包装
# ---------------------------------------------------------------------------

class EarlyGradNormMetric(InTrainingMetric):
    """
    累积早期梯度范数（定义 3）——InTrainingMetric 包装类。
    """

    name = "def3_early_grad_norm"
    needs_data = False

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        T_early: int = 100,
        head_granularity: bool = False,
        **kwargs,
    ) -> EarlyGradNormCallback:
        """
        Args:
            model:            未经 DDP 包装的原始模型（在 Trainer 创建前传入）
            save_dir:         结果保存目录
            T_early:          累积步数（建议 50–200）
            head_granularity: 若为 True，额外输出注意力头级别分数
        """
        return EarlyGradNormCallback(
            model=model, T_early=T_early,
            save_dir=save_dir, head_granularity=head_granularity,
        )
