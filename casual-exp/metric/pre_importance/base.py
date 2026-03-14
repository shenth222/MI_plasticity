"""
metric/pre_importance/base.py

所有训练前重要性（pre-training importance）指标的抽象基类，
以及将模型参数按叶模块（leaf module）分组的工具函数。
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


class ImportanceBase(ABC):
    """
    训练前重要性评估基类。

    子类需要：
      1. 设置类属性 `name`（用于文件命名和日志）
      2. 实现 `compute()` 方法
      3. 若不需要梯度 / 数据，将 `needs_data = False`
    """

    name: str        # 子类必须覆盖：指标唯一标识符
    needs_data: bool = True  # 是否需要 DataLoader（SVD 类指标不需要）

    @abstractmethod
    def compute(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        计算各模块重要性分数。

        Args:
            model:      待评估模型（应处于微调前状态 θ₀）
            dataloader: 数据加载器；needs_data=False 的子类可接受 None
            device:     计算设备
            **kwargs:   各子类特定超参数

        Returns:
            字典，至少包含键 "module_scores"（{module_name: score}），
            可附带其他元信息。
        """

    def save(self, scores: Dict[str, Any], save_dir: str) -> Path:
        """
        将结果序列化为 JSON 文件，文件名为 ``{name}.json``。

        Args:
            scores:   compute() 的返回值
            save_dir: 保存目录（不存在时自动创建）

        Returns:
            保存路径
        """
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


# ---------------------------------------------------------------------------
# 工具函数
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
