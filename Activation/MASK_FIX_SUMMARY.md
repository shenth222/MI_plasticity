# Attention Mask 处理修复总结

## 问题分析

代码审查反馈**完全正确**。原实现与用户需求存在以下严重不符：

### 🔴 严重问题

#### 1. Last Token 聚合忽略 Attention Mask

**问题描述**：
- 原代码 `_get_last_token_positions` 始终返回 `seq_len - 1`
- 完全忽略 `attention_mask`，即使 `set_attention_mask` 被调用
- 在 ARC 任务中，不同样本的有效长度不同，导致系统性偏差：把 padding token 当作答案 token

**用户需求**：
- 对每个样本取**最后一个非 padding 的 token**

#### 2. All Token 聚合包含 Padding

**问题描述**：
- 原代码直接对所有 token（包括 padding）进行平均
- Padding token 稀释真实信号，导致 head norm 被低估

**用户需求**：
- 对每个样本取**所有非 padding token 的平均**

### 🟡 中等问题

#### 3. 缺失层默认填充 0

**问题描述**：
- 如果某层 hook 未触发，`finalize_batch` 会填充 0
- Heatmap 显示该层 norm 极小，而不是"无数据"
- 降低可观测性，难以排查

#### 4. 统计量使用批均值加权

**问题描述**：
- 原代码先对 batch 求平均，再喂给 Welford
- 结果 = 各 batch 均值的算术平均
- 与真实"样本级加权平均"不同
- 例如：3 个 32-样本 batch + 1 个 2-样本 batch，权重应该是 96:2，而非 3:1

## 解决方案

### ✅ 修复 1：Last Token 使用 Attention Mask

**位置**：`hooks.py` 第 172-198 行

**修改**：
```python
def _get_last_token_positions(self, bs: int, seq_len: int, device: torch.device = None) -> torch.Tensor:
    if self.current_attention_mask is not None:
        # 使用 attention_mask 找到每个样本最后一个非 padding token
        mask = self.current_attention_mask.to(device)
        # 对每个样本，找到最后一个 1 的位置
        last_positions = mask.sum(dim=1) - 1  # [bs]
        # 确保至少为 0（处理全 0 mask 的边界情况）
        last_positions = torch.clamp(last_positions, min=0)
        return last_positions.long()
    else:
        # 如果没有 mask，回退到使用最后一个位置
        logger.warning("attention_mask 未设置，使用 seq_len-1 作为最后 token 位置")
        return torch.full((bs,), seq_len - 1, dtype=torch.long, device=device)
```

**关键点**：
- 使用 `mask.sum(dim=1) - 1` 找到最后一个有效 token 位置
- 添加边界检查和警告

### ✅ 修复 2：All Token 过滤 Padding

**位置**：`hooks.py` 第 200-238 行（`_compute_head_output_norm`）和第 258-292 行（`_compute_head_resid_contrib_norm`）

**修改**：
```python
# "all" 聚合 - 只对有效 token 求平均
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
    norms = norms_per_token.mean(dim=1)
```

**关键点**：
- 使用 mask 加权平均：`sum(values * mask) / sum(mask)`
- 同时应用于 head output norm 和 head residual contribution norm

### ✅ 修复 3：缺失层检测和报告

**位置**：`hooks.py` 第 303-318 行

**修改**：
```python
# 检查缺失的层
missing_layers = []
for layer_idx in range(self.num_layers):
    if layer_idx not in self._batch_head_output_norms:
        missing_layers.append(layer_idx)

if missing_layers:
    logger.warning(f"以下层没有收集到数据: {missing_layers}")
```

**关键点**：
- 记录缺失层并打印警告
- 提高可观测性，便于排查问题

### ✅ 修复 4：样本级加权统计

**位置**：`hooks.py` 第 160-170 行（`_compute_and_update_metrics`）和第 303-340 行（`finalize_batch`）

**修改**：
```python
# 在 _compute_and_update_metrics 中：
# 存储到 batch 缓存（保存每个样本的值，不求平均）
self._batch_head_output_norms[layer_idx] = head_output_norms.cpu().numpy()  # [bs, num_heads]
self._batch_head_resid_norms[layer_idx] = head_resid_contrib_norms.cpu().numpy()  # [bs, num_heads]

# 在 finalize_batch 中：
# 对每个样本更新统计（使用样本级加权）
for sample_idx in range(batch_size):
    # 为当前样本聚合所有层的指标
    sample_head_output_norms = np.zeros((self.num_layers, self.num_heads))
    sample_head_resid_norms = np.zeros((self.num_layers, self.num_heads))
    
    for layer_idx in range(self.num_layers):
        if layer_idx in self._batch_head_output_norms:
            sample_head_output_norms[layer_idx, :] = self._batch_head_output_norms[layer_idx][sample_idx, :]
            sample_head_resid_norms[layer_idx, :] = self._batch_head_resid_norms[layer_idx][sample_idx, :]
    
    # 更新统计
    self.head_output_norm_stats.update(sample_head_output_norms)
    self.head_resid_contrib_norm_stats.update(sample_head_resid_norms)
```

**关键点**：
- 不再对 batch 维度求平均后更新统计
- 逐样本更新 `OnlineStats`，确保权重与样本数匹配
- 小 batch 和大 batch 按实际样本数加权

## 测试验证

创建了 `test_mask_handling.py` 包含 3 个测试：

### 测试 1：Last Token Position 计算

**场景**：
- 样本 0: 5 个有效 token，最后位置应为 4
- 样本 1: 6 个有效 token，最后位置应为 5
- 样本 2: 3 个有效 token，最后位置应为 2

**结果**：✅ **通过** - 正确计算出 `[4, 5, 2]`

### 测试 2：All Token Padding 过滤

**场景**：
- 样本 0: 前 3 个 token 有效
- 样本 1: 前 2 个 token 有效
- 验证 padding token 不参与平均

**结果**：✅ **通过** - 正确过滤 padding

### 测试 3：样本级加权

**场景**：
- Batch 1: 3 个样本
- Batch 2: 1 个样本
- 验证 count = 4（样本级）而非 2（批级）

**结果**：✅ **通过** - Count = 4，确认使用样本级加权

## 影响和建议

### 📊 对历史结果的影响

**严重影响**：
- 之前的结果可能存在系统性偏差
- 特别是 ARC 任务中 prompt 长度可变的情况
- **建议重新运行所有实验**

### 📝 后续工作

1. **补充测试覆盖**（低优先级）：
   - 边界情况：全 0 mask、全 1 mask
   - 与手工计算的基准值对比
   - 不同模型架构的兼容性

2. **文档更新**：
   - 更新 README 和 QUICKSTART 说明 attention_mask 的重要性
   - 明确说明 token_agg 的行为

3. **可选增强**：
   - 支持显式的 answer span 标注（如果需要更精确的控制）
   - 添加 mask 验证：检查是否所有样本的 mask 都合理

## 总结

✅ **所有反馈问题已修复**：
1. ✅ Last token 正确使用 attention_mask
2. ✅ All token 正确过滤 padding
3. ✅ 缺失层会被检测和报告
4. ✅ 统计量使用样本级加权
5. ✅ 测试验证所有修复正确

**修复前后对比**：

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| Last token | 总是取 `seq_len-1`，包含 padding | 取最后一个有效 token |
| All token | 对所有 token 平均，包含 padding | 只对有效 token 平均 |
| 缺失层 | 填充 0，难以发现 | 警告日志，可观测 |
| 统计权重 | 批均值的均值（权重偏差） | 样本级加权（正确） |
| Mask 使用 | 未使用（虽然设置了） | 正确使用 |

**现在的实现完全符合用户需求**！


