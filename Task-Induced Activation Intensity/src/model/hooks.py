"""
Forward hooks 模块

用于捕获模型内部的中间激活，如 attention 输出、head 输出等
"""
import torch
from typing import Dict, List, Optional, Callable


class AttentionOutputHook:
    """
    用于捕获每层 attention 模块输出的 hook
    """
    
    def __init__(self):
        self.outputs = []
        self.handles = []
    
    def hook_fn(self, module, input, output):
        """
        Hook 函数
        
        Args:
            module: 被 hook 的模块
            input: 模块的输入
            output: 模块的输出
        """
        # 对于 Llama 的 attention 模块，output 通常是一个 tuple
        # output[0] 是 attention 输出，output[1] 是 attention weights（如果有）
        if isinstance(output, tuple):
            attn_output = output[0]  # (batch, seq_len, hidden_size)
            attn_weights = output[1] if len(output) > 1 else None
        else:
            attn_output = output
            attn_weights = None
        
        # 保存（需要 clone 以避免被后续操作覆盖）
        self.outputs.append({
            "attn_output": attn_output.detach() if attn_output is not None else None,
            "attn_weights": attn_weights.detach() if attn_weights is not None else None
        })
    
    def register(self, model, layer_name_pattern: str = "self_attn"):
        """
        注册 hooks 到模型的所有匹配的层
        
        Args:
            model: 模型
            layer_name_pattern: 层名称模式（用于匹配）
        """
        for name, module in model.named_modules():
            if layer_name_pattern in name:
                handle = module.register_forward_hook(self.hook_fn)
                self.handles.append(handle)
    
    def clear(self):
        """清除缓存的输出"""
        self.outputs = []
    
    def remove(self):
        """移除所有 hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


class HeadOutputHook:
    """
    用于捕获每个 attention head 的输出
    
    这需要在 attention 模块内部的特定位置进行 hook，
    因为 Llama 的实现可能不会直接暴露 head 输出
    """
    
    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.outputs = []
        self.handles = []
    
    def hook_fn(self, module, input, output):
        """
        Hook 函数
        
        对于 Llama attention 模块，我们需要从 output 中重构 head outputs
        """
        if isinstance(output, tuple):
            attn_output = output[0]  # (batch, seq_len, hidden_size)
        else:
            attn_output = output
        
        if attn_output is None:
            self.outputs.append(None)
            return
        
        batch_size, seq_len, hidden_size = attn_output.shape
        
        # 重塑为 (batch, seq_len, num_heads, head_dim)
        head_outputs = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 保存
        self.outputs.append(head_outputs.detach())
    
    def register(self, model, layer_name_pattern: str = "self_attn"):
        """注册 hooks"""
        for name, module in model.named_modules():
            if layer_name_pattern in name:
                handle = module.register_forward_hook(self.hook_fn)
                self.handles.append(handle)
    
    def clear(self):
        """清除缓存的输出"""
        self.outputs = []
    
    def remove(self):
        """移除所有 hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def extract_head_outputs_from_attention_output(
    attn_output: torch.Tensor,
    num_heads: int
) -> torch.Tensor:
    """
    从 attention 输出中提取 head outputs
    
    Args:
        attn_output: attention 输出，shape (batch, seq_len, hidden_size)
        num_heads: head 数量
        
    Returns:
        head_outputs: shape (batch, seq_len, num_heads, head_dim)
    """
    batch_size, seq_len, hidden_size = attn_output.shape
    head_dim = hidden_size // num_heads
    
    # 重塑
    head_outputs = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    
    return head_outputs

