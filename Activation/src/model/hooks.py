import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np
from .metrics import OnlineStats
from ..utils.logging import get_logger


logger = get_logger(__name__)


class HookManager:
    """
    精简版 Hook Manager：直接在 o_proj 处捕获 per-head 输出。
    
    核心思路：
    1. Head Output: 在 o_proj 的输入处捕获，此时是 reshape 前的 [bs, seq_len, hidden_size]
       可以直接 view 为 [bs, seq_len, num_heads, head_dim]
    2. Head Residual Contribution: 对每个 head 的输出，应用 o_proj 权重的对应切片
    
    这样避免了重复计算 attention。
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 token_agg: str = "last"):
        """
        初始化 hook manager。
        
        Args:
            model: 模型
            num_layers: Transformer 层数
            num_heads: 每层的 attention head 数量
            head_dim: 每个 head 的维度
            token_agg: Token 聚合策略 ("last" 或 "all")
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.token_agg = token_agg
        
        # 初始化在线统计
        self.head_output_norm_stats = OnlineStats((num_layers, num_heads))
        self.head_resid_contrib_norm_stats = OnlineStats((num_layers, num_heads))
        
        # 存储当前 batch 的 attention mask
        self.current_attention_mask = None
        
        # 存储当前 batch 的中间结果（保存每个样本的值，而不是批平均）
        self._batch_head_output_norms = {}  # {layer_idx: [bs, num_heads]}
        self._batch_head_resid_norms = {}  # {layer_idx: [bs, num_heads]}
        
        # Hook handles
        self.hook_handles = []
        
        # Attach hooks
        self._attach_hooks()
        
        logger.info(f"HookManager 初始化完成（精简版）:")
        logger.info(f"  num_layers: {num_layers}")
        logger.info(f"  num_heads: {num_heads}")
        logger.info(f"  head_dim: {head_dim}")
        logger.info(f"  token_agg: {token_agg}")
    
    def _attach_hooks(self) -> None:
        """在 o_proj 层添加 pre-hook 来捕获 per-head 输出。"""
        logger.info("Attaching hooks to o_proj layers...")
        
        # 获取 attention layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise ValueError("Cannot find transformer layers in model")
        
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                o_proj = layer.self_attn.o_proj
                
                # 在 o_proj 上添加 pre-hook
                handle = o_proj.register_forward_pre_hook(
                    self._make_o_proj_hook(layer_idx, layer.self_attn)
                )
                self.hook_handles.append(handle)
            else:
                logger.warning(f"Layer {layer_idx} has no self_attn.o_proj")
        
        logger.info(f"Attached {len(self.hook_handles)} hooks")
    
    def _make_o_proj_hook(self, layer_idx: int, attn_module: nn.Module):
        """
        创建 o_proj 的 pre-hook 来捕获输入（即 per-head 输出）。
        
        Args:
            layer_idx: 层索引
            attn_module: Attention 模块
            
        Returns:
            Pre-hook 函数
        """
        def pre_hook(module, args):
            """
            o_proj 的 pre-hook。
            args[0] 是 o_proj 的输入：[bs, seq_len, hidden_size]
            这个输入就是 attention 的输出（reshape 后，o_proj 之前）。
            """
            try:
                # o_proj 的输入
                attn_output_before_proj = args[0]  # [bs, seq_len, hidden_size]
                
                bs, seq_len, hidden_size = attn_output_before_proj.shape
                
                # View 为 per-head: [bs, seq_len, num_heads, head_dim]
                head_outputs = attn_output_before_proj.view(bs, seq_len, self.num_heads, self.head_dim)
                
                # 计算指标
                self._compute_and_update_metrics(
                    layer_idx,
                    head_outputs,
                    attn_module
                )
                
            except Exception as e:
                logger.error(f"Error in o_proj pre-hook for layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        return pre_hook
    
    def _compute_and_update_metrics(self,
                                   layer_idx: int,
                                   head_outputs: torch.Tensor,
                                   attn_module: nn.Module) -> None:
        """
        计算 head 指标并更新统计。
        
        Args:
            layer_idx: 层索引
            head_outputs: Per-head outputs [bs, seq_len, num_heads, head_dim]
            attn_module: Attention 模块
        """
        bs, seq_len, num_heads, head_dim = head_outputs.shape
        
        # 获取 token 聚合位置
        if self.token_agg == "last":
            token_positions = self._get_last_token_positions(bs, seq_len, device=head_outputs.device)
        else:  # "all"
            token_positions = None
        
        # 1. 计算 Head Output Norm
        head_output_norms = self._compute_head_output_norm(
            head_outputs, token_positions
        )  # [bs, num_heads]
        
        # 2. 计算 Head Residual Contribution Norm
        head_resid_contrib_norms = self._compute_head_resid_contrib_norm(
            head_outputs, attn_module, token_positions
        )  # [bs, num_heads]
        
        # 存储到 batch 缓存（保存每个样本的值，不求平均）
        self._batch_head_output_norms[layer_idx] = head_output_norms.cpu().numpy()  # [bs, num_heads]
        self._batch_head_resid_norms[layer_idx] = head_resid_contrib_norms.cpu().numpy()  # [bs, num_heads]
    
    def _get_last_token_positions(self, bs: int, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        获取每个样本的最后一个有效 token 位置（非 padding）。
        
        Args:
            bs: Batch size
            seq_len: Sequence length
            device: 目标设备（如果为 None，则在 CPU 上）
            
        Returns:
            形状 [bs] 的 tensor
        """
        if self.current_attention_mask is not None:
            # 使用 attention_mask 找到每个样本最后一个非 padding token
            # attention_mask: [bs, seq_len]，1 表示有效 token，0 表示 padding
            mask = self.current_attention_mask.to(device)
            # 对每个样本，找到最后一个 1 的位置
            # 方法：cumsum 后找到最大值的位置
            last_positions = mask.sum(dim=1) - 1  # [bs]
            # 确保至少为 0（处理全 0 mask 的边界情况）
            last_positions = torch.clamp(last_positions, min=0)
            return last_positions.long()
        else:
            # 如果没有 mask，回退到使用最后一个位置
            logger.warning("attention_mask 未设置，使用 seq_len-1 作为最后 token 位置")
            return torch.full((bs,), seq_len - 1, dtype=torch.long, device=device)
    
    def _compute_head_output_norm(self,
                                  head_outputs: torch.Tensor,
                                  token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        计算每个 head 输出的 L2 范数。
        
        Args:
            head_outputs: [bs, seq_len, num_heads, head_dim]
            token_positions: [bs] 或 None
            
        Returns:
            [bs, num_heads]
        """
        bs, seq_len, num_heads, head_dim = head_outputs.shape
        
        if token_positions is not None:
            # "last" 聚合
            indices = token_positions.view(bs, 1, 1, 1).expand(bs, 1, num_heads, head_dim)
            selected = torch.gather(head_outputs, 1, indices).squeeze(1)  # [bs, num_heads, head_dim]
            norms = torch.norm(selected, p=2, dim=2)  # [bs, num_heads]
        else:
            # "all" 聚合 - 只对有效 token 求平均
            norms_per_token = torch.norm(head_outputs, p=2, dim=3)  # [bs, seq_len, num_heads]
            
            if self.current_attention_mask is not None:
                # 使用 mask 过滤 padding token
                mask = self.current_attention_mask.to(head_outputs.device)  # [bs, seq_len]
                mask = mask.unsqueeze(2)  # [bs, seq_len, 1]
                
                # 计算加权平均：sum(norms * mask) / sum(mask)
                masked_norms = norms_per_token * mask  # [bs, seq_len, num_heads]
                sum_norms = masked_norms.sum(dim=1)  # [bs, num_heads]
                count = mask.sum(dim=1)  # [bs, 1]
                
                # 避免除以 0
                count = torch.clamp(count, min=1)
                norms = sum_norms / count  # [bs, num_heads]
            else:
                # 如果没有 mask，回退到所有 token 的平均
                logger.warning("attention_mask 未设置，使用所有 token 进行聚合")
                norms = norms_per_token.mean(dim=1)  # [bs, num_heads]
        
        return norms
    
    def _compute_head_resid_contrib_norm(self,
                                         head_outputs: torch.Tensor,
                                         attn_module: nn.Module,
                                         token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        计算每个 head 经过 o_proj 后贡献的 L2 范数。
        
        方法：对每个 head，用 o_proj 权重的对应列切片计算贡献。
        
        Args:
            head_outputs: [bs, seq_len, num_heads, head_dim]
            attn_module: Attention 模块
            token_positions: [bs] 或 None
            
        Returns:
            [bs, num_heads]
        """
        bs, seq_len, num_heads, head_dim = head_outputs.shape
        
        if not hasattr(attn_module, 'o_proj'):
            logger.warning("No o_proj found")
            return torch.zeros(bs, num_heads, device=head_outputs.device)
        
        # o_proj 权重: [hidden_size, hidden_size]
        o_proj_weight = attn_module.o_proj.weight
        
        if token_positions is not None:
            # "last" 聚合
            indices = token_positions.view(bs, 1, 1, 1).expand(bs, 1, num_heads, head_dim)
            selected = torch.gather(head_outputs, 1, indices).squeeze(1)  # [bs, num_heads, head_dim]
            
            # 批量计算所有 head 的贡献
            norms = []
            for h in range(num_heads):
                head_out = selected[:, h, :]  # [bs, head_dim]
                # o_proj 对应列切片: [:, h*head_dim:(h+1)*head_dim]
                o_proj_slice = o_proj_weight[:, h*head_dim:(h+1)*head_dim]  # [hidden_size, head_dim]
                # 贡献: head_out @ o_proj_slice^T
                contrib = torch.matmul(head_out, o_proj_slice.T)  # [bs, hidden_size]
                norm = torch.norm(contrib, p=2, dim=1)  # [bs]
                norms.append(norm)
            
            norms = torch.stack(norms, dim=1)  # [bs, num_heads]
        else:
            # "all" 聚合 - 只对有效 token 求平均
            head_outputs_flat = head_outputs.view(bs * seq_len, num_heads, head_dim)
            
            norms_per_token = []
            for h in range(num_heads):
                head_out = head_outputs_flat[:, h, :]  # [bs*seq_len, head_dim]
                o_proj_slice = o_proj_weight[:, h*head_dim:(h+1)*head_dim]
                contrib = torch.matmul(head_out, o_proj_slice.T)
                norm = torch.norm(contrib, p=2, dim=1)
                norms_per_token.append(norm)
            
            norms_per_token = torch.stack(norms_per_token, dim=1).view(bs, seq_len, num_heads)
            
            if self.current_attention_mask is not None:
                # 使用 mask 过滤 padding token
                mask = self.current_attention_mask.to(head_outputs.device)  # [bs, seq_len]
                mask = mask.unsqueeze(2)  # [bs, seq_len, 1]
                
                # 计算加权平均
                masked_norms = norms_per_token * mask  # [bs, seq_len, num_heads]
                sum_norms = masked_norms.sum(dim=1)  # [bs, num_heads]
                count = mask.sum(dim=1)  # [bs, 1]
                
                # 避免除以 0
                count = torch.clamp(count, min=1)
                norms = sum_norms / count  # [bs, num_heads]
            else:
                # 如果没有 mask，回退到所有 token 的平均
                logger.warning("attention_mask 未设置，使用所有 token 进行聚合")
                norms = norms_per_token.mean(dim=1)  # [bs, num_heads]
        
        return norms
    
    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        """设置当前 forward pass 的 attention mask。"""
        self.current_attention_mask = attention_mask
    
    def finalize_batch(self) -> None:
        """
        完成当前 batch 的指标计算并更新在线统计。
        在每次 forward pass 后调用。
        """
        if not self._batch_head_output_norms:
            logger.warning("finalize_batch 被调用但没有数据")
            return
        
        # 检查缺失的层
        missing_layers = []
        for layer_idx in range(self.num_layers):
            if layer_idx not in self._batch_head_output_norms:
                missing_layers.append(layer_idx)
        
        if missing_layers:
            logger.warning(f"以下层没有收集到数据: {missing_layers}")
        
        # 获取 batch size（从第一个有数据的层推断）
        first_layer = next(iter(self._batch_head_output_norms.keys()))
        batch_size = self._batch_head_output_norms[first_layer].shape[0]
        
        # 对每个样本更新统计（使用样本级加权）
        for sample_idx in range(batch_size):
            # 为当前样本聚合所有层的指标
            sample_head_output_norms = np.zeros((self.num_layers, self.num_heads))
            sample_head_resid_norms = np.zeros((self.num_layers, self.num_heads))
            
            for layer_idx in range(self.num_layers):
                if layer_idx in self._batch_head_output_norms:
                    sample_head_output_norms[layer_idx, :] = self._batch_head_output_norms[layer_idx][sample_idx, :]
                    sample_head_resid_norms[layer_idx, :] = self._batch_head_resid_norms[layer_idx][sample_idx, :]
                # 注意：缺失层保持为 0，但这不会影响统计（因为我们不会对它们更新）
            
            # 只更新有数据的层
            if len(missing_layers) < self.num_layers:  # 至少有一层有数据
                self.head_output_norm_stats.update(sample_head_output_norms)
                self.head_resid_contrib_norm_stats.update(sample_head_resid_norms)
        
        # 清空 batch 缓存
        self._batch_head_output_norms = {}
        self._batch_head_resid_norms = {}
    
    def get_results(self) -> Dict[str, np.ndarray]:
        """
        获取最终统计结果。
        
        Returns:
            包含统计信息的字典
        """
        return {
            "head_output_norm_mean": self.head_output_norm_stats.get_mean(),
            "head_output_norm_std": self.head_output_norm_stats.get_std(),
            "head_resid_contrib_norm_mean": self.head_resid_contrib_norm_stats.get_mean(),
            "head_resid_contrib_norm_std": self.head_resid_contrib_norm_stats.get_std(),
            "count": self.head_output_norm_stats.get_count()
        }
    
    def remove_hooks(self) -> None:
        """移除所有 hooks。"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.info("Removed all hooks")
