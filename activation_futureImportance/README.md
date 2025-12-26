# 激活提取与分析工具

本工具用于对比微调前后模型各注意力头的激活情况，验证未微调模型中激活头强弱与未来重要性的关系。

## 功能特性

- **激活提取**: 从注意力层中间 hook 出每个头的输出张量
- **激活强度计算**: 对每个头 h 计算 R_h = E_batch[||y_h||_2]
- **对比分析**: 对比微调前后相同样本的激活情况
- **可视化**: 生成热力图展示激活强度和差异

## 核心公式

对每个头 h 计算激活强度：

```
R_h = E_batch[||y_h||_2]
```

其中：
- `y_h` 是该 head 的输出张量（在注意力中间 hook 出来）
- `||y_h||_2` 是 L2 范数
- `E_batch` 表示对批次求期望（平均）

## 使用方法

### 基本用法

```bash
python activation_extract.py \
    --pre_tuned_model /path/to/pre_tuned_model \
    --fine_tuned_model /path/to/fine_tuned_model \
    --data_path /path/to/data.json \
    --sample_num 100 \
    --batch_size 8 \
    --output_dir ./results
```

### 参数说明

- `--pre_tuned_model`: 微调前模型路径
- `--fine_tuned_model`: 微调后模型路径
- `--data_path`: 数据文件路径（JSON 格式）
- `--sample_num`: 使用的样本数量
- `--batch_size`: 批次大小
- `--output_dir`: 输出目录
- `--layers`: 要分析的层索引（可选，默认所有层）
- `--heads`: 要分析的头索引（可选，默认所有头）
- `--seed`: 随机种子（默认 42）

### 编程接口

#### 1. 创建激活提取器

```python
from activation_extract import ActivationExtractor
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("model_path")

# 创建提取器
extractor = ActivationExtractor(model, model_name="my_model")

# 如果自动检测失败，手动设置模型结构
extractor.set_model_structure(num_layers=12, num_heads=12)

# 注册 hooks（提取所有层和头）
extractor.register_hooks()

# 或者只提取特定层和头
extractor.register_hooks(layers=[0, 1, 2], heads=[0, 1, 2, 3])
```

#### 2. 提取激活值

```python
from torch.utils.data import DataLoader

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=8)

# 提取激活值
extractor.extract_activations(dataloader, tokenizer=tokenize_fn)

# 计算 R_h
R_h_dict = extractor.compute_R_h()
print(f"Layer 0, Head 0 的 R_h: {R_h_dict[(0, 0)]}")

# 获取所有统计信息
stats = extractor.get_all_stats()
for stat in stats:
    print(f"Layer {stat.layer_idx}, Head {stat.head_idx}: R_h = {stat.R_h:.4f}")

# 保存结果
extractor.save_results("results.json")

# 清理
extractor.remove_hooks()
```

#### 3. 对比分析

```python
from activation_extract import ActivationComparator

# 创建两个提取器
pre_extractor = ActivationExtractor(pre_model, "pre_tuned")
fine_extractor = ActivationExtractor(fine_model, "fine_tuned")

# 提取激活值
pre_extractor.extract_activations(dataloader)
fine_extractor.extract_activations(dataloader)

# 创建对比器
comparator = ActivationComparator(pre_extractor, fine_extractor)

# 获取对比结果
comparison = comparator.compare()
print("差异:", comparison['differences'])
print("比率:", comparison['ratios'])

# 可视化
comparator.visualize_comparison("comparison.png")

# 保存对比结果
comparator.save_comparison("comparison.json")
```

## 输出文件

运行后会生成以下文件：

1. **pre_tuned_activations.json**: 微调前模型的激活统计
2. **fine_tuned_activations.json**: 微调后模型的激活统计
3. **comparison.json**: 对比分析结果
4. **comparison_heatmap.png**: 可视化热力图

## 数据格式

输入数据应为 JSON 格式，每个样本包含以下字段之一：
- `input`: 输入文本
- `text`: 文本内容
- `question`: 问题文本

示例：
```json
[
    {"input": "What is the capital of France?"},
    {"text": "The sky is blue."},
    {"question": "How does photosynthesis work?"}
]
```

## 注意事项

1. **模型结构**: 工具会自动检测常见的模型结构（GPT、BERT 等），如果检测失败，需要手动调用 `set_model_structure()`

2. **Hook 位置**: 默认 hook 到注意力层的输出。如果模型结构特殊，可能需要修改 `_get_hook_point()` 方法

3. **内存使用**: 处理大量样本时注意内存使用，可以通过 `max_batches` 参数限制处理的批次数

4. **设备**: 默认使用 CUDA（如果可用），否则使用 CPU

## 自定义 Hook 位置

如果默认的 hook 位置不适用于你的模型，可以修改 `_get_hook_point()` 方法或直接指定 hook 点：

```python
# 自定义 hook 注册
hook = ActivationHook(layer_idx=0, head_idx=0)
hook.register(model, "transformer.h.0.attn.c_attn")  # 直接指定路径
```

## 扩展功能

### 提取特定层的激活

```python
# 只提取前 3 层
extractor.register_hooks(layers=[0, 1, 2])

# 只提取特定头
extractor.register_hooks(heads=[0, 1, 2, 3])
```

### 分析特定样本

```python
# 创建只包含特定样本的数据加载器
specific_samples = [samples[0], samples[5], samples[10]]
specific_dataloader = create_simple_dataloader(specific_samples, tokenizer)
extractor.extract_activations(specific_dataloader)
```

