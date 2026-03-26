"""
metric/training_gain/base.py

训练收益（Training Gain, G_m）指标的抽象基类与共用工具函数。

─────────────────────────────────────────────────────────────────────────────
训练收益衡量的是：微调过程中，模块 m（或注意力头 h）的参数更新
对模型性能提升的实际贡献。

三种定义：

  RollbackMetric（def1 / def2）
    回滚模块 m 的参数到 θ^(0)，在验证集上测量：
      def1 — val loss 变化：
          G_m^(loss) = L_val(θ^(T)[m ← θ_m^(0)]) − L_val(θ^(T))
          > 0 表示该模块的训练降低了 val loss（有正贡献）
      def2 — val accuracy 变化：
          G_m^(acc)  = Acc_val(θ^(T)[m ← θ_m^(0)]) − Acc_val(θ^(T))
          < 0 表示该模块的训练提升了准确率（有正贡献）
    def1 与 def2 合并计算（共用一次前向），分别保存：
      def1_rollback_loss.json
      def2_rollback_acc.json

  PathIntegralMetric（def3）
    路径积分（一阶 Taylor 近似）：
      G_m^(PI) = Σ_{t=1}^{T} ∇_{θ_m} L(θ^(t)) · Δθ_{m,t}
    其中 Δθ_{m,t} = θ_m^(t) − θ_m^(t-1)
    通常 G_m^(PI) ≤ 0（标准梯度下降 + 模块持续降低 loss），
    绝对值越大表示该模块在训练路径上对 loss 下降的累积贡献越大。
    保存：def3_path_integral.json

─────────────────────────────────────────────────────────────────────────────
计算时机：
  RollbackMetric  — 通过 make_callback() 嵌入训练，
                    or 独立使用：compute(theta0, model, eval_fn, device)
  PathIntegralMetric — 通过 make_callback() 在每步 on_step_end 累积，
                       on_train_end 保存结果
─────────────────────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# 数据类：评估结果（eval_fn 返回值）
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    验证集评估结果。

    Attributes:
        avg_loss:            各 split 上的平均交叉熵 loss
        primary_metric_name: 主指标名称（如 "accuracy", "matthews_correlation"）
        primary_metric_value: 主指标值
        all_metrics:         全部指标 {metric_name: value}
    """
    avg_loss:             float
    primary_metric_name:  str
    primary_metric_value: float
    all_metrics:          Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class TrainingGainBase(ABC):
    """
    训练收益指标基类：提供统一的 save / load 接口。
    子类必须覆盖 `name`。
    """

    name: str  # 子类覆盖，用于文件命名和日志

    def save(self, scores: Dict[str, Any], save_dir: str) -> Path:
        """将结果序列化为 {name}.json，目录不存在时自动创建。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{self.name}.json"
        with open(path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"[{self.name}] Saved → {path}")
        return path

    @classmethod
    def load(cls, save_dir: str) -> Dict[str, Any]:
        """从 JSON 文件加载之前保存的结果。"""
        path = Path(save_dir) / f"{cls.name}.json"
        with open(path) as f:
            return json.load(f)


class RollbackMetric(TrainingGainBase):
    """
    回滚参数后评估的训练收益指标基类（定义一和定义二）。

    子类必须实现 compute() 和 make_callback()。
    """

    @abstractmethod
    def compute(
        self,
        theta0: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        eval_fn: Callable[["torch.nn.Module", "torch.device"], EvalResult],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        独立测试接口。

        Args:
            theta0:   微调前参数快照 {param_name: tensor_cpu}
            model:    微调后的模型（θ^(T) 状态）
            eval_fn:  评估函数 (model, device) → EvalResult
            device:   计算设备
            **kwargs: 子类特定超参（head_granularity, module_names 等）

        Returns:
            包含 module_scores_loss / module_scores_acc 等字段的字典
        """

    @abstractmethod
    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        eval_fn: Callable,
        **kwargs,
    ):
        """
        创建 TrainerCallback（必须在 Trainer 创建前调用）。

        Args:
            model:    未经 DDP 包装的原始模型
            save_dir: 结果保存目录
            eval_fn:  评估函数 (model, device) → EvalResult
            **kwargs: 子类特定超参
        """


class PathIntegralMetric(TrainingGainBase):
    """
    路径积分训练收益指标基类（定义三）。

    子类必须实现 make_callback()。
    """

    @abstractmethod
    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        **kwargs,
    ):
        """
        创建 TrainerCallback，在训练步中逐步累积梯度·参数更新量。

        Args:
            model:    未经 DDP 包装的原始模型（在 Trainer 创建前传入）
            save_dir: 结果保存目录
            **kwargs: 子类特定超参（head_granularity, log_every 等）
        """


# ---------------------------------------------------------------------------
# 共用工具函数
# ---------------------------------------------------------------------------

def group_params_by_module(model: torch.nn.Module) -> Dict[str, List[str]]:
    """
    将模型的命名参数按"直接父叶模块"分组。

    遍历 model.named_modules()，对每个含有本地参数（recurse=False）的模块，
    收集其参数的完整名称（含模块前缀）。

    Returns:
        {module_name: [full_param_name, ...]}

    示例（DeBERTa 某层）：
        "deberta.encoder.layer.0.attention.self.query_proj":
            ["deberta.encoder.layer.0.attention.self.query_proj.weight",
             "deberta.encoder.layer.0.attention.self.query_proj.bias"]
    """
    groups: Dict[str, List[str]] = {}
    for module_name, module in model.named_modules():
        local_params = list(module.named_parameters(recurse=False))
        if not local_params:
            continue
        full_names = [
            f"{module_name}.{p_name}" if module_name else p_name
            for p_name, _ in local_params
        ]
        groups[module_name] = full_names
    return groups


def snapshot_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    快照当前模型所有可训练参数，存于 CPU，保留原始数据类型。

    Returns:
        {param_name: tensor_cpu}（仅含 requires_grad=True 的参数）
    """
    return {
        n: p.data.clone().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def resolve_param_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    返回模型参数字典，自动剥离 DDP 包装引入的 'module.' 前缀。

    Returns:
        {param_name_without_ddp_prefix: param_tensor}
    """
    result: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        key = n[len("module."):] if n.startswith("module.") else n
        result[key] = p
    return result


def resolve_named_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    返回模型所有模块字典，自动剥离 DDP 包装引入的 'module.' 前缀。

    Returns:
        {module_name_without_ddp_prefix: module}
    """
    result: Dict[str, torch.nn.Module] = {}
    for n, m in model.named_modules():
        key = n[len("module."):] if n.startswith("module.") else n
        result[key] = m
    return result
