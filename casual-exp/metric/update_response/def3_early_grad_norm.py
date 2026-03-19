"""
metric/update_response/def3_early_grad_norm.py

定义 3：累积早期梯度范数
─────────────────────────────────────────────────────────────────────────────
公式（模块级精确计算）：

    \hat{R}_m = \sum_{t=1}^{T_{early}} \|\nabla_{\theta_m}\mathcal{L}^{(t)}\|_2

其中 ‖∇_{θ_m}L^(t)‖₂ = sqrt(Σ_{param∈m} ‖g_param^(t)‖₂²)，
即将模块内所有参数梯度向量拼接后取 L2 范数。

含义：
    统计真实训练的前 T_early 步内，各模块梯度范数的累积和。
    值越大，表示该模块在训练早期持续受到强烈梯度驱动，预测其
    参数会积累更大的净变化（更新响应更强）。

    与定义 1 的区别：
        定义 1 测量实际位移（经过 Adam 二阶矩缩放，受 optimizer 状态影响）；
        定义 3 测量原始梯度信号的累积强度（不受 optimizer 曲率压制影响）。
    与定义 2 的区别：
        定义 2 是对初始点的静态预测（训练前，多 batch 平均）；
        定义 3 是训练动态过程中的真实累积（随训练步演化）。

实现：
    · 通过 param.register_hook() 在 backward() 期间捕获各参数梯度平方和，
      存入每步临时缓冲 _step_param_sq，不修改 Trainer 任何代码。
    · on_step_end 时将临时缓冲聚合为模块级范数并累加到 _module_acc，
      同时清空缓冲，为下一步做准备。
    · T_early 步后自动卸载所有钩子并保存结果；
      若训练提前结束，on_train_end 也会触发保存。

为什么不用 Trainer warmup？
    warmup_ratio / warmup_steps 仅控制 LR 调度（渐升曲线），
    无法定义"收集前 N 步梯度"的范围，且 warmup ≠ T_early。

嵌入示例（最小侵入）：
─────────────────────────────────────────────────────────────────────────────
    runner = UpdateResponseRunner.from_str("def3", metric_kwargs={
        "def3": {"T_early": 100}
    })
    cb = runner.make_training_callback(model, save_dir)   # model 为未包装模型
    trainer = Trainer(..., callbacks=[cb])                # 唯一嵌入点
─────────────────────────────────────────────────────────────────────────────

保存格式：
    def3_early_grad_norm.json
    {
      "module_scores":    {module_name: float, ...},  # Σ‖g_m^(t)‖₂（模块级）
      "param_scores":     {param_name:  float, ...},  # Σ‖g_param^(t)‖₂（参数级）
      "T_early":          int,                         # 目标收集步数
      "steps_collected":  int,                         # 实际收集步数（≤ T_early）
    }
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base import InTrainingMetric, group_params_by_module


# ---------------------------------------------------------------------------
# TrainerCallback 实现
# ---------------------------------------------------------------------------

class EarlyGradNormCallback(TrainerCallback):
    """
    在真实训练的前 T_early 步累积各模块的梯度 L2 范数。

    算法流程（每步）：
      1. backward()     → 梯度钩子触发，将 ‖g_param‖² 写入 _step_param_sq
      2. on_step_end    → 将 _step_param_sq 聚合为模块级范数，累加到 _module_acc，
                         清空 _step_param_sq，更新步计数
      3. 达到 T_early  → 卸载钩子，保存 JSON

    DDP 说明：
        梯度钩子在 all-reduce 完成后触发（各进程梯度已同步），
        因此不同进程累积值相同。仅主进程（is_world_process_zero）写文件。

    梯度累积（gradient_accumulation_steps > 1）说明：
        钩子在每次 micro-batch backward 时触发，每次 on_step_end 对应一个
        optimizer.step()（已完成所有 micro-batch）。_step_param_sq 在多次
        micro-batch 中持续累积，on_step_end 时一次性提交，语义正确。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        T_early: int,
        save_dir: str,
    ):
        self.T_early   = T_early
        self.save_dir  = save_dir

        self._module_groups = group_params_by_module(model)

        # 每步临时缓冲：{param_name: Σ(‖g^(micro_batch)‖²) within current step}
        self._step_param_sq: Dict[str, float] = {}

        # 累积器
        self._module_acc: Dict[str, float] = {m: 0.0 for m in self._module_groups}
        self._param_acc:  Dict[str, float] = {}

        # 控制标志
        self._hooks:          List  = []
        self._step:           int   = 0   # 已完成的 optimizer step 数
        self._done:           bool  = False
        self._steps_collected: int  = 0

        # 在原始（未 DDP 包装）模型上注册钩子
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._param_acc[name] = 0.0
                handle = param.register_hook(self._make_hook(name))
                self._hooks.append(handle)

        print(f"[def3_early_grad_norm] 已注册 {len(self._hooks)} 个梯度钩子，"
              f"将收集前 {T_early} 步")

    def _make_hook(self, param_name: str):
        """
        工厂函数：为单个参数生成梯度钩子（闭包捕获 param_name）。

        存储梯度范数的平方（而非范数本身），延迟到 on_step_end 时
        再取平方根，以支持多 micro-batch 的正确平方和累积。
        """
        def hook(grad: torch.Tensor) -> None:
            if not self._done and self._step < self.T_early:
                norm_sq = grad.detach().pow(2).sum().item()
                self._step_param_sq[param_name] = (
                    self._step_param_sq.get(param_name, 0.0) + norm_sq
                )
        return hook

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not self._done and self._step < self.T_early:
            # 模块级：‖g_m^(step)‖₂ = sqrt(Σ_{param∈m} ‖g_param‖²)
            for m_name, param_names in self._module_groups.items():
                sq_sum = sum(
                    self._step_param_sq.get(pn, 0.0) for pn in param_names
                )
                self._module_acc[m_name] += sq_sum ** 0.5

            # 参数级：‖g_param^(step)‖₂ = sqrt(Σ_micro ‖g_micro‖²)
            for pn, sq in self._step_param_sq.items():
                self._param_acc[pn] = self._param_acc.get(pn, 0.0) + sq ** 0.5

            self._step_param_sq.clear()
            self._steps_collected += 1

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
        """训练结束时兜底保存（当 T_early > 总训练步数 时触发）。"""
        if not self._done:
            self._done = True
            # 提交最后可能未提交的缓冲（正常情况下为空）
            if self._step_param_sq:
                for m_name, param_names in self._module_groups.items():
                    sq_sum = sum(
                        self._step_param_sq.get(pn, 0.0) for pn in param_names
                    )
                    self._module_acc[m_name] += sq_sum ** 0.5
                for pn, sq in self._step_param_sq.items():
                    self._param_acc[pn] = self._param_acc.get(pn, 0.0) + sq ** 0.5
                self._step_param_sq.clear()
                self._steps_collected += 1

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
        result = {
            "module_scores":   dict(self._module_acc),
            "param_scores":    dict(self._param_acc),
            "T_early":         self.T_early,
            "steps_collected": self._steps_collected,
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

    调用 make_callback() 获取 EarlyGradNormCallback，
    注册到 Trainer 的 callbacks 列表即完成嵌入，无需其他修改。
    """

    name = "def3_early_grad_norm"
    needs_data = False  # 数据由 Trainer 管理，本指标不直接消费 DataLoader

    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        T_early: int = 100,
        **kwargs,
    ) -> EarlyGradNormCallback:
        """
        Args:
            model:    未经 DDP 包装的原始模型（必须在 Trainer 创建前传入）
            save_dir: 结果保存目录
            T_early:  累积步数（建议 50–200；超过总训练步则全程收集）
        """
        return EarlyGradNormCallback(model=model, T_early=T_early, save_dir=save_dir)
