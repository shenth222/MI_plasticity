"""
metric/actual_update/base.py

实际更新量（Actual Update Magnitude, U_m）指标的抽象基类与共用工具函数。

─────────────────────────────────────────────────────────────────────────────
指标按计算时机分两类：

  SnapshotMetric  — 基于 θ^(0) 和 θ^(T) 快照计算（定义一、二）
                    通过 make_callback() 在 Trainer 创建前立即记录 θ^(0) 快照
                    （存于 CPU），训练结束时（on_train_end）自动计算并保存。
                    同时提供独立 compute(theta0, model, device) 接口用于单元测试。

  PathMetric      — 训练过程中逐步累积路径长度（定义三）
                    通过 make_callback() 返回 TrainerCallback，
                    在每个 optimizer step 结束后（on_step_end）计算步进变化量
                    并累积到路径长度总量，on_train_end 时保存结果。
─────────────────────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class ActualUpdateBase(ABC):
    """实际更新量指标基类：提供统一的 save / load 接口。"""

    name: str  # 子类必须覆盖，用于 JSON 文件命名和日志

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


class SnapshotMetric(ActualUpdateBase):
    """
    基于参数快照的更新量指标基类（定义一、二）。

    计算逻辑：
      在 make_callback() 时立即对未包装的原始模型调用 snapshot_params()，
      将 θ^(0) 存于 CPU；训练结束（on_train_end）时与 θ^(T) 比较，
      调用 compute() 计算更新量并保存。

    子类必须实现 compute() 和 make_callback()。
    """

    @abstractmethod
    def compute(
        self,
        theta0: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        根据初始参数快照 θ^(0) 和当前模型 θ^(T) 计算更新量。

        Args:
            theta0: 初始参数快照字典 {param_name: tensor_cpu}
            model:  当前模型（处于 θ^(T) 状态，可能已被 DDP 包装）
            device: 计算设备（保留接口一致性，当前实现均在 CPU 上差值）

        Returns:
            字典，至少包含 "module_scores" ({module_name: float})，
            可附带其他辅助字段。
        """

    @abstractmethod
    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        **kwargs,
    ):
        """
        创建 TrainerCallback。

        必须在 Trainer 创建前调用，此时 model 尚未被 DDP 包装，
        参数名与 named_parameters() 完全一致，可安全记录 θ^(0) 快照。

        Args:
            model:    未经 DDP 包装的原始模型
            save_dir: 结果保存目录
        """


class PathMetric(ActualUpdateBase):
    """
    路径长度类更新量指标基类（定义三）。

    计算逻辑：
      在 make_callback().__init__ 时立即记录 θ^(0) 作为 prev_params（存于 CPU）；
      每个 on_step_end 计算 ||θ^(t) - θ^(t-1)||_2 并累积；
      on_train_end 时保存结果。

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
        创建 TrainerCallback，在训练步中逐步累积路径长度。

        Args:
            model:    未经 DDP 包装的原始模型（在 Trainer 创建前传入，
                      此时参数名准确，立即记录 θ^(0) 快照）
            save_dir: 结果保存目录
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

    用于在训练开始前记录 θ^(0)。

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

    在 TrainerCallback 的 on_train_end 中调用时，model 可能已被 DDP 包装，
    此函数将参数名统一为未包装模型的格式，与 θ^(0) 快照的键保持一致。

    Returns:
        {param_name_without_ddp_prefix: param_tensor}
    """
    result: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        key = n[len("module."):] if n.startswith("module.") else n
        result[key] = p
    return result
