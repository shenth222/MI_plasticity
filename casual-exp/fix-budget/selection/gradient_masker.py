"""
fix-budget/selection/gradient_masker.py

GradientMasker：通过梯度钩子实现注意力头粒度的参数遮蔽，
使非选中头对应的参数切片在训练中实际不更新。

设计原则：
  1. 非注意力参数（FFN、归一化层等）通过 requires_grad=False 完全冻结；
     此步骤应在 Trainer/优化器创建前完成（调用 freeze_non_attn_params()）。
  2. 注意力 Q/K/V/O 参数保持 requires_grad=True，
     通过梯度钩子对非选中头的参数切片置零，
     使对应参数的实际更新量接近零（AdamW 零梯度 → 一阶矩衰减 → 有效更新极小）。
  3. 支持动态更新：调用 update_selection() 可随时更换选中头集合并重新注册钩子，
     适用于每隔 n 步重新选择的场景。

注意：
  - freeze_non_attn_params() 仅调用一次（优化器创建前）；
  - GradientMasker 初始化时同时完成钩子注册；
  - 动态更新只更新钩子，不修改 requires_grad（优化器无感知）。
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .head_utils import ConceptualHead, get_layer_key


class GradientMasker:
    """
    注意力头梯度遮蔽器。

    使用方式（典型流程）：
        # 1. 训练前：冻结非注意力参数
        GradientMasker.freeze_non_attn_params(model)

        # 2. 创建 Trainer（优化器此时确定参数组）

        # 3. 初始化 masker，注册梯度 hooks
        masker = GradientMasker(model, initial_selected_set)

        # 4. 训练中途重新选择时
        masker.update_selection(new_selected_set)

        # 5. 训练结束
        masker.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        selected_set: Set[ConceptualHead],
    ):
        """
        Args:
            model        : 已通过 freeze_non_attn_params() 处理过的模型
            selected_set : 被选中的概念头集合 {(layer_key, head_idx), ...}
        """
        from metric.pre_importance.attn_head import get_attn_head_config, get_attn_modules

        self._model = model
        self._hooks: List = []

        # 提取注意力头配置
        self._attn_cfg = get_attn_head_config(model)
        if self._attn_cfg is None:
            raise RuntimeError(
                "[GradientMasker] 无法从模型 config 中提取注意力头配置，"
                "请确保模型具有 num_attention_heads / hidden_size 字段。"
            )

        self._attn_mods: Dict[str, str] = get_attn_modules(model, self._attn_cfg)
        self._register_hooks(selected_set)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        n_selected  = len(selected_set)
        n_total_h   = len({(get_layer_key(m), h)
                            for m in self._attn_mods
                            for h in range(self._attn_cfg.num_heads)})
        print(
            f"[GradientMasker] 初始化完成。"
            f" 可训练参数: {n_trainable:,}/{n_total:,}"
            f" ({n_trainable/n_total*100:.1f}%)"
            f" | 选中头: {n_selected}/{n_total_h}"
        )

    @staticmethod
    def freeze_non_attn_params(model: nn.Module) -> int:
        """
        冻结所有非注意力参数（requires_grad → False），解冻注意力参数。

        应在优化器/Trainer 创建之前调用，确保优化器的参数组仅包含注意力参数。

        Returns:
            解冻的注意力参数个数
        """
        from metric.pre_importance.attn_head import get_attn_head_config, get_attn_modules

        attn_cfg = get_attn_head_config(model)
        if attn_cfg is None:
            raise RuntimeError(
                "[GradientMasker] freeze_non_attn_params 失败："
                "模型无注意力头配置。"
            )

        attn_mods = get_attn_modules(model, attn_cfg)
        attn_param_names: Set[str] = set()
        for m_name in attn_mods:
            for suffix in ("weight", "bias"):
                attn_param_names.add(f"{m_name}.{suffix}")

        frozen_count   = 0
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if name in attn_param_names:
                param.requires_grad_(True)
                unfrozen_count += 1
            else:
                param.requires_grad_(False)
                frozen_count += 1

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        print(
            f"[GradientMasker.freeze_non_attn_params] "
            f"冻结 {frozen_count} 个参数层，解冻 {unfrozen_count} 个注意力参数层。"
            f" 可训练: {n_trainable:,}/{n_total:,} ({n_trainable/n_total*100:.1f}%)"
        )
        return unfrozen_count

    def _register_hooks(self, selected_set: Set[ConceptualHead]) -> None:
        """为每个注意力参数注册梯度遮蔽 hook。"""
        attn_cfg   = self._attn_cfg
        named_params = dict(self._model.named_parameters())

        for m_name, m_type in self._attn_mods.items():
            layer_key = get_layer_key(m_name)
            # 该层被选中的头下标集合
            selected_heads_in_layer: Set[int] = {
                head_idx
                for (lk, head_idx) in selected_set
                if lk == layer_key
            }

            for suffix in ("weight", "bias"):
                pname = f"{m_name}.{suffix}"
                param = named_params.get(pname)
                if param is None or not param.requires_grad:
                    continue

                mask = self._build_mask(
                    shape=param.shape,
                    module_type=m_type,
                    suffix=suffix,
                    num_heads=attn_cfg.num_heads,
                    head_dim=attn_cfg.head_dim,
                    selected_heads=selected_heads_in_layer,
                )
                handle = param.register_hook(self._make_mask_hook(mask))
                self._hooks.append(handle)

    @staticmethod
    def _make_mask_hook(mask: torch.Tensor):
        """工厂函数：闭包捕获 mask，返回梯度钩子。"""
        def hook(grad: torch.Tensor) -> torch.Tensor:
            return grad * mask.to(grad.device, grad.dtype)
        return hook

    @staticmethod
    def _build_mask(
        shape: torch.Size,
        module_type: str,
        suffix: str,
        num_heads: int,
        head_dim: int,
        selected_heads: Set[int],
    ) -> torch.Tensor:
        """
        构建梯度遮蔽张量（1.0 = 保留梯度，0.0 = 置零梯度）。

        切片规则（与 attn_head.py 保持一致）：
          QKV weight : weight[h*d : (h+1)*d, :]   ← 按行（输出维度）分头
          QKV bias   : bias  [h*d : (h+1)*d]
          Out weight : weight[:, h*d : (h+1)*d]   ← 按列（输入维度）分头
          Out bias   : 属于整个模块，不按头分割；
                       只要该层有选中头就保留完整偏置梯度
        """
        mask = torch.zeros(shape)

        if suffix == "weight":
            for h in selected_heads:
                if module_type == "qkv":
                    mask[h * head_dim: (h + 1) * head_dim, :] = 1.0
                else:  # "out"
                    mask[:, h * head_dim: (h + 1) * head_dim] = 1.0
        else:  # bias
            if module_type == "qkv":
                for h in selected_heads:
                    mask[h * head_dim: (h + 1) * head_dim] = 1.0
            else:
                # out 投影的偏置不按头分割；只要该层有选中头就保留完整梯度
                if selected_heads:
                    mask.fill_(1.0)

        return mask

    def update_selection(self, selected_set: Set[ConceptualHead]) -> None:
        """
        更新选中头集合，移除旧钩子并重新注册。

        调用时机：每隔 n 步重新选择后（reselect_every_n_steps）。
        此方法不修改 requires_grad，优化器参数组保持不变，
        仅通过钩子控制哪些头的梯度被保留。
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._register_hooks(selected_set)
        print(f"[GradientMasker] 钩子已更新，当前选中 {len(selected_set)} 个概念头。")

    def remove_hooks(self) -> None:
        """移除所有梯度 hooks（训练结束时调用）。"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @property
    def num_hooks(self) -> int:
        return len(self._hooks)
