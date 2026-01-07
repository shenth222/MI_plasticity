# 代码简化说明

## 简化前的问题

原始实现 (`hooks.py` 旧版本) 存在以下问题：

1. **重复计算 attention**：在 hook 中重新计算了 Q、K、V 投影和 attention weights
2. **架构复杂**：使用 pre-hook 和 forward-hook 的组合，代码冗长
3. **内存开销大**：需要存储 hidden_states 等中间变量
4. **效率低**：每个 batch 都要重新计算一遍完整的 attention

```python
# 旧版本的核心逻辑（简化示意）
def _extract_head_outputs(module, hidden_states, ...):
    # 重新计算 Q、K、V
    q = module.q_proj(hidden_states)
    k = module.k_proj(hidden_states)
    v = module.v_proj(hidden_states)
    
    # 重新计算 attention weights
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)
    attn_weights = softmax(attn_weights, dim=-1)
    
    # 重新计算 head outputs
    head_outputs = torch.matmul(attn_weights, v)
    ...
```

## 简化后的方案

### 核心思路

根据 `transformers/models/llama/modeling_llama.py` 的实现：

```python
# LlamaAttention.forward 中的流程
def forward(hidden_states, ...):
    # 1. 计算 Q、K、V 并 reshape
    query_states = self.q_proj(hidden_states).view(...).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(...).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(...).transpose(1, 2)
    
    # 2. 调用 eager_attention_forward
    attn_output, attn_weights = eager_attention_forward(...)
    # attn_output shape: [bs, seq_len, num_heads, head_dim]
    
    # 3. Reshape 并通过 o_proj
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # 现在 shape: [bs, seq_len, hidden_size]
    
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
```

**关键观察**：
- `o_proj` 的输入就是 reshape 后的 per-head outputs：`[bs, seq_len, hidden_size]`
- 可以直接 view 为 `[bs, seq_len, num_heads, head_dim]`
- 不需要重新计算 attention！

### 新实现

```python
# 精简版本：在 o_proj 上添加 pre-hook
def _attach_hooks(self):
    for layer_idx, layer in enumerate(layers):
        o_proj = layer.self_attn.o_proj
        # 在 o_proj 的 pre-hook 中捕获输入
        handle = o_proj.register_forward_pre_hook(
            self._make_o_proj_hook(layer_idx, layer.self_attn)
        )

def _make_o_proj_hook(self, layer_idx, attn_module):
    def pre_hook(module, args):
        # args[0] 就是 o_proj 的输入: [bs, seq_len, hidden_size]
        attn_output_before_proj = args[0]
        
        # 直接 view 为 per-head
        head_outputs = attn_output_before_proj.view(
            bs, seq_len, num_heads, head_dim
        )
        
        # 计算指标
        self._compute_and_update_metrics(...)
    
    return pre_hook
```

## 简化效果

### 代码量减少
- **旧版本**：~535 行
- **新版本**：~310 行
- **减少**：~42%

### 性能提升
1. **无重复计算**：直接复用 LlamaAttention 的 forward 计算结果
2. **内存占用减少**：不需要存储 hidden_states、attention_mask 等中间变量
3. **逻辑更清晰**：只在 o_proj 处捕获，单一 hook 点

### 核心指标计算

#### 1. Head Output Norm
```python
# 输入: head_outputs [bs, seq_len, num_heads, head_dim]
# 输出: 每个 head 的 L2 范数

# "last" 模式：取最后一个 token
selected = head_outputs[:, -1, :, :]  # [bs, num_heads, head_dim]
norms = torch.norm(selected, p=2, dim=2)  # [bs, num_heads]

# "all" 模式：对所有 token 取平均
norms_per_token = torch.norm(head_outputs, p=2, dim=3)  # [bs, seq_len, num_heads]
norms = norms_per_token.mean(dim=1)  # [bs, num_heads]
```

#### 2. Head Residual Contribution Norm
```python
# 对每个 head，计算其经过 o_proj 后的贡献

for h in range(num_heads):
    head_out = selected[:, h, :]  # [bs, head_dim]
    
    # o_proj 权重的对应列切片
    o_proj_slice = o_proj_weight[:, h*head_dim:(h+1)*head_dim]  # [hidden_size, head_dim]
    
    # 计算贡献
    contrib = head_out @ o_proj_slice.T  # [bs, hidden_size]
    
    # L2 范数
    norm = torch.norm(contrib, p=2, dim=1)  # [bs]
```

## 理论依据

### 为什么可以在 o_proj 输入处捕获 per-head 输出？

根据 `LlamaAttention.forward` 的实现：

```python
# eager_attention_forward 返回 attn_output
attn_output = torch.matmul(attn_weights, value_states)  # [bs, num_heads, seq_len, head_dim]
attn_output = attn_output.transpose(1, 2)               # [bs, seq_len, num_heads, head_dim]

# 然后 reshape
attn_output = attn_output.reshape(*input_shape, -1)     # [bs, seq_len, hidden_size]
```

其中 `hidden_size = num_heads * head_dim`，所以：

```python
attn_output.view(bs, seq_len, hidden_size)
  ≡ attn_output.view(bs, seq_len, num_heads, head_dim).reshape(bs, seq_len, hidden_size)
```

反过来，从 `[bs, seq_len, hidden_size]` view 为 `[bs, seq_len, num_heads, head_dim]` 就恢复了原始的 per-head 输出！

### Head Residual Contribution 的正确性

o_proj 是一个线性层：`o_proj(x) = x @ W^T`

对于 per-head 的输入：
```
x = [head_0, head_1, ..., head_n]  # concatenated
```

每个 head 的贡献：
```
contrib_h = head_h @ W[:, h*d:(h+1)*d]^T
```

这正是我们的实现！

## 兼容性说明

### 与原有接口保持一致
- `HookManager` 的初始化参数不变
- `get_results()` 返回格式不变
- 与 `main.py` 的集成无需修改

### 支持的功能
- ✅ Token 聚合策略 ("last" / "all")
- ✅ 在线统计 (Welford 算法)
- ✅ 批处理
- ✅ 多层、多头

## 总结

这次简化遵循了**"利用已有计算结果，避免重复计算"**的原则，核心改进：

1. **找准 hook 点**：在 o_proj 的 pre-hook 中捕获
2. **直接 view**：利用 tensor 的内存布局，无需重新计算
3. **逻辑清晰**：单一 hook 点，代码更易维护

代码行数减少 42%，性能提升显著，同时保持了完整的功能和正确性。

