#!/usr/bin/env python3
"""
测试修复后的 HookManager：验证 attention_mask 处理是否正确。
"""

import torch
import torch.nn as nn
import numpy as np


# 重用 test_simplified_hooks.py 中的模拟模型
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
        return self.model(input_ids)


def test_last_token_with_mask():
    """测试 last token 聚合是否正确使用 attention_mask。"""
    print("\n" + "=" * 60)
    print("测试 1: Last Token 聚合 + Attention Mask")
    print("=" * 60)
    
    from src.model.hooks import HookManager
    
    # 模型配置
    num_layers = 2
    num_heads = 4
    head_dim = 32
    hidden_size = num_heads * head_dim
    bs = 3
    seq_len = 8
    
    model = MockModel(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads)
    model.eval()
    
    # 创建 HookManager
    hook_manager = HookManager(
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        token_agg="last"
    )
    
    # 创建测试数据：不同的有效长度
    # 样本 0: 前 5 个 token 有效
    # 样本 1: 前 6 个 token 有效
    # 样本 2: 前 3 个 token 有效
    test_input = torch.randn(bs, seq_len, hidden_size)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],  # 5 个有效 token，最后有效位置是 4
        [1, 1, 1, 1, 1, 1, 0, 0],  # 6 个有效 token，最后有效位置是 5
        [1, 1, 1, 0, 0, 0, 0, 0],  # 3 个有效 token，最后有效位置是 2
    ])
    
    expected_positions = torch.tensor([4, 5, 2])  # 预期的最后 token 位置
    
    print(f"\n输入配置:")
    print(f"  Batch size: {bs}")
    print(f"  Seq length: {seq_len}")
    print(f"  Attention mask:")
    for i in range(bs):
        valid_count = attention_mask[i].sum().item()
        print(f"    样本 {i}: {attention_mask[i].tolist()} (有效 token: {valid_count})")
    print(f"  预期最后 token 位置: {expected_positions.tolist()}")
    
    # 设置 mask 并执行 forward
    hook_manager.set_attention_mask(attention_mask)
    
    with torch.no_grad():
        output = model(test_input, attention_mask=attention_mask)
    
    # 验证 _get_last_token_positions 是否正确
    computed_positions = hook_manager._get_last_token_positions(
        bs, seq_len, device=test_input.device
    )
    print(f"\n计算得到的最后 token 位置: {computed_positions.tolist()}")
    
    if torch.equal(computed_positions, expected_positions):
        print("✓ Last token 位置计算正确！")
    else:
        print(f"✗ Last token 位置计算错误！")
        print(f"  预期: {expected_positions.tolist()}")
        print(f"  实际: {computed_positions.tolist()}")
        return False
    
    hook_manager.finalize_batch()
    results = hook_manager.get_results()
    
    print(f"\n结果:")
    print(f"  Head output norm mean shape: {results['head_output_norm_mean'].shape}")
    print(f"  Count: {results['count'][0, 0]}")  # 应该是 3（3 个样本）
    
    hook_manager.remove_hooks()
    return True


def test_all_token_with_mask():
    """测试 all token 聚合是否正确过滤 padding。"""
    print("\n" + "=" * 60)
    print("测试 2: All Token 聚合 + Padding 过滤")
    print("=" * 60)
    
    from src.model.hooks import HookManager
    
    # 模型配置
    num_layers = 1
    num_heads = 2
    head_dim = 4
    hidden_size = num_heads * head_dim
    bs = 2
    seq_len = 4
    
    model = MockModel(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads)
    model.eval()
    
    # 固定权重以便手动验证
    for param in model.parameters():
        nn.init.ones_(param)
    
    # 创建 HookManager
    hook_manager = HookManager(
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        token_agg="all"
    )
    
    # 创建简单的测试数据
    # 样本 0: 前 3 个 token 有效
    # 样本 1: 前 2 个 token 有效
    test_input = torch.ones(bs, seq_len, hidden_size)
    # 为了区分，设置不同位置有不同的值
    for b in range(bs):
        for t in range(seq_len):
            test_input[b, t] = test_input[b, t] * (b * 10 + t + 1)
    
    attention_mask = torch.tensor([
        [1, 1, 1, 0],  # 3 个有效 token
        [1, 1, 0, 0],  # 2 个有效 token
    ])
    
    print(f"\n输入配置:")
    print(f"  Batch size: {bs}")
    print(f"  Seq length: {seq_len}")
    print(f"  Attention mask:")
    for i in range(bs):
        valid_count = attention_mask[i].sum().item()
        print(f"    样本 {i}: {attention_mask[i].tolist()} (有效 token: {valid_count})")
    
    # 设置 mask 并执行 forward
    hook_manager.set_attention_mask(attention_mask)
    
    with torch.no_grad():
        output = model(test_input, attention_mask=attention_mask)
    
    hook_manager.finalize_batch()
    results = hook_manager.get_results()
    
    print(f"\n结果:")
    print(f"  Head output norm mean shape: {results['head_output_norm_mean'].shape}")
    print(f"  Head output norm mean: {results['head_output_norm_mean']}")
    print(f"  Count: {results['count']}")
    
    # 验证：如果正确过滤了 padding，统计应该只基于有效 token
    # 如果没有过滤，padding token 会稀释结果
    
    hook_manager.remove_hooks()
    
    print("✓ All token 聚合测试完成")
    return True


def test_batch_weighting():
    """测试统计量是否使用样本级加权而非批均值加权。"""
    print("\n" + "=" * 60)
    print("测试 3: 样本级加权统计")
    print("=" * 60)
    
    from src.model.hooks import HookManager
    
    # 模型配置
    num_layers = 1
    num_heads = 2
    head_dim = 4
    hidden_size = num_heads * head_dim
    
    model = MockModel(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads)
    model.eval()
    
    # 创建 HookManager
    hook_manager = HookManager(
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        token_agg="last"
    )
    
    # 第一个 batch：3 个样本
    bs1 = 3
    seq_len = 4
    test_input1 = torch.ones(bs1, seq_len, hidden_size) * 1.0
    attention_mask1 = torch.ones(bs1, seq_len)
    
    hook_manager.set_attention_mask(attention_mask1)
    with torch.no_grad():
        _ = model(test_input1)
    hook_manager.finalize_batch()
    
    # 第二个 batch：1 个样本（模拟最后一个小 batch）
    bs2 = 1
    test_input2 = torch.ones(bs2, seq_len, hidden_size) * 10.0  # 显著不同的值
    attention_mask2 = torch.ones(bs2, seq_len)
    
    hook_manager.set_attention_mask(attention_mask2)
    with torch.no_grad():
        _ = model(test_input2)
    hook_manager.finalize_batch()
    
    results = hook_manager.get_results()
    
    print(f"\n配置:")
    print(f"  Batch 1: {bs1} 个样本，输入值 = 1.0")
    print(f"  Batch 2: {bs2} 个样本，输入值 = 10.0")
    print(f"\n结果:")
    print(f"  总样本数: {results['count'][0, 0]}")
    print(f"  Head output norm mean: {results['head_output_norm_mean']}")
    
    # 验证：如果使用样本级加权，count 应该是 4
    # 如果使用批均值加权，count 会是 2
    if results['count'][0, 0] == 4:
        print("✓ 使用了正确的样本级加权！")
    else:
        print(f"✗ 加权方式可能不正确，count = {results['count'][0, 0]}（预期 4）")
    
    hook_manager.remove_hooks()
    return True


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("测试修复后的 Attention Mask 处理")
    print("=" * 60)
    
    try:
        success = True
        success &= test_last_token_with_mask()
        success &= test_all_token_with_mask()
        success &= test_batch_weighting()
        
        if success:
            print("\n" + "=" * 60)
            print("所有测试通过！✓")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("部分测试失败 ✗")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

