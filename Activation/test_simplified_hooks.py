#!/usr/bin/env python3
"""
简单测试脚本：验证简化后的 HookManager 是否工作正常。
"""

import torch
import torch.nn as nn


# 模拟一个简单的 LLaMA-like attention 模块
class MockLlamaAttention(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, hidden_states):
        bs, seq_len, _ = hidden_states.shape
        
        # Q, K, V projections
        query = self.q_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        query = query.transpose(1, 2)  # [bs, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bs, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class MockLlamaLayer(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super().__init__()
        self.self_attn = MockLlamaAttention(hidden_size, num_heads)
    
    def forward(self, hidden_states):
        return self.self_attn(hidden_states)


class MockLlamaModel(nn.Module):
    def __init__(self, num_layers=2, hidden_size=128, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            MockLlamaLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MockModel(nn.Module):
    def __init__(self, num_layers=2, hidden_size=128, num_heads=4):
        super().__init__()
        self.model = MockLlamaModel(num_layers, hidden_size, num_heads)
    
    def forward(self, input_ids, attention_mask=None):
        # 简化：直接使用 embedding (假设 input_ids 已经是 embeddings)
        return self.model(input_ids)


def test_hook_manager():
    """测试 HookManager 的基本功能。"""
    print("=" * 60)
    print("测试简化版 HookManager")
    print("=" * 60)
    
    # 创建模拟模型
    num_layers = 2
    num_heads = 4
    head_dim = 32
    hidden_size = num_heads * head_dim
    
    model = MockModel(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads)
    model.eval()
    
    print(f"\n模型配置:")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  hidden_size: {hidden_size}")
    
    # 导入 HookManager
    try:
        from src.model.hooks import HookManager
        print("\n✓ 成功导入 HookManager")
    except Exception as e:
        print(f"\n✗ 导入 HookManager 失败: {e}")
        return False
    
    # 初始化 HookManager
    try:
        hook_manager = HookManager(
            model=model,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            token_agg="last"
        )
        print("✓ HookManager 初始化成功")
    except Exception as e:
        print(f"✗ HookManager 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建测试数据
    bs = 2
    seq_len = 8
    test_input = torch.randn(bs, seq_len, hidden_size)
    
    print(f"\n测试输入:")
    print(f"  shape: {test_input.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass 成功")
        print(f"  输出 shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Finalize batch
    try:
        hook_manager.finalize_batch()
        print("✓ Batch finalization 成功")
    except Exception as e:
        print(f"✗ Batch finalization 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 获取结果
    try:
        results = hook_manager.get_results()
        print("\n✓ 获取结果成功")
        print(f"\n结果:")
        print(f"  head_output_norm_mean shape: {results['head_output_norm_mean'].shape}")
        print(f"  head_output_norm_mean:\n{results['head_output_norm_mean']}")
        print(f"  head_resid_contrib_norm_mean shape: {results['head_resid_contrib_norm_mean'].shape}")
        print(f"  head_resid_contrib_norm_mean:\n{results['head_resid_contrib_norm_mean']}")
        print(f"  count: {results['count']}")
    except Exception as e:
        print(f"✗ 获取结果失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证结果的合理性
    try:
        import numpy as np
        
        head_out_mean = results['head_output_norm_mean']
        head_resid_mean = results['head_resid_contrib_norm_mean']
        
        # 检查形状
        assert head_out_mean.shape == (num_layers, num_heads), f"Head output norm shape 错误: {head_out_mean.shape}"
        assert head_resid_mean.shape == (num_layers, num_heads), f"Head resid norm shape 错误: {head_resid_mean.shape}"
        
        # 检查是否有 NaN 或 Inf
        assert not np.isnan(head_out_mean).any(), "Head output norm 包含 NaN"
        assert not np.isinf(head_out_mean).any(), "Head output norm 包含 Inf"
        assert not np.isnan(head_resid_mean).any(), "Head resid norm 包含 NaN"
        assert not np.isinf(head_resid_mean).any(), "Head resid norm 包含 Inf"
        
        # 检查是否为正数
        assert (head_out_mean >= 0).all(), "Head output norm 应该全为非负"
        assert (head_resid_mean >= 0).all(), "Head resid norm 应该全为非负"
        
        print("\n✓ 结果验证通过")
        
    except AssertionError as e:
        print(f"\n✗ 结果验证失败: {e}")
        return False
    
    # 移除 hooks
    try:
        hook_manager.remove_hooks()
        print("✓ Hooks 移除成功")
    except Exception as e:
        print(f"✗ Hooks 移除失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_hook_manager()
    exit(0 if success else 1)

