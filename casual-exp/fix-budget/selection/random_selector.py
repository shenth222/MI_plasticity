"""
fix-budget/selection/random_selector.py

策略①：随机头选择。

从所有注意力头中以均匀分布随机采样 budget_count 个。
可指定随机种子保证复现性；不指定时每次重新选择结果不同（适合测试变化性）。
"""

import random
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .base import HeadSelector
from .head_utils import get_all_conceptual_heads_from_model


class RandomSelector(HeadSelector):
    """
    随机选择策略。

    分数为 [0, 1) 均匀随机数，排名靠前的头被选中。
    每次调用 compute_head_scores() 会重新随机打分（除非固定种子）。

    Args:
        seed : 随机种子（None 表示每次随机）
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    @property
    def selector_name(self) -> str:
        return "random"

    def compute_head_scores(
        self,
        model: torch.nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        为每个模块的每个头生成随机分数。

        Returns:
            {"random": {module_name: {"head_0": float, ...}}}
        """
        from metric.pre_importance.attn_head import get_attn_head_config, get_attn_modules

        attn_cfg = get_attn_head_config(model)
        if attn_cfg is None:
            raise RuntimeError("[RandomSelector] 无法获取模型注意力头配置。")

        attn_mods = get_attn_modules(model, attn_cfg)
        rng = random.Random(self._seed)

        head_scores: Dict[str, Dict[str, float]] = {}
        for m_name in attn_mods:
            head_scores[m_name] = {
                f"head_{h}": rng.random()
                for h in range(attn_cfg.num_heads)
            }

        return {"random": head_scores}
