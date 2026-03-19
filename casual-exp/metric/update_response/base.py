"""
metric/update_response/base.py

更新响应预测（Update Response Prediction）指标的抽象基类与工具函数。

指标按计算时机分两类：

  PreTrainingMetric  — 训练前独立计算（定义 1、2、4）
                       在 Trainer 创建前调用 compute()，无需侵入训练循环。

  InTrainingMetric   — 训练过程中计算（定义 3）
                       通过 make_callback() 返回 TrainerCallback，
                       注册为 Trainer callback 后自动在训练步中收集数据并保存。
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class UpdateResponseBase(ABC):
    """
    更新响应预测指标基类。

    子类必须覆盖：
      - name         : str  —— 指标唯一标识符，用于 JSON 文件命名和日志
      - is_pre_training : bool —— True = 训练前；False = 训练中
    """

    name: str
    is_pre_training: bool
    needs_data: bool = True

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


class PreTrainingMetric(UpdateResponseBase):
    """
    训练前计算的指标基类（定义 1、2、4）。

    在 Trainer 创建前、θ₀ 保存后调用 compute()，
    计算完成后模型状态不变（定义 1 会临时改变参数但会恢复）。
    """

    is_pre_training = True

    @abstractmethod
    def compute(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        计算更新响应指标。

        Returns:
            字典，至少包含键 "module_scores"（{module_name: float}），
            可附带其他分析字段。
        """


class InTrainingMetric(UpdateResponseBase):
    """
    训练过程中计算的指标基类（定义 3）。

    通过 make_callback() 返回 TrainerCallback，
    注册到 Trainer 后随训练循环自动收集梯度数据并保存结果。
    """

    is_pre_training = False

    @abstractmethod
    def make_callback(
        self,
        model: torch.nn.Module,
        save_dir: str,
        **kwargs,
    ):
        """
        构造并返回 TrainerCallback。

        Args:
            model:    未经 DDP 包装的原始模型（必须在 Trainer 创建前传入，
                      此时参数名与 named_parameters() 一致）
            save_dir: 结果保存目录
        """


# ---------------------------------------------------------------------------
# 共用工具函数
# （与 metric.pre_importance.base.group_params_by_module 逻辑相同，
#  此处内联以保持本包自包含，无跨包依赖）
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
