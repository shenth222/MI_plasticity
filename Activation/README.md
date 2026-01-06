# LLaMA Attention Head Activation Collection

用于在 LLaMA 3.2-1B 模型上采集每层每个 attention head 的激活强度信号的 Python 项目。

## 项目简介

本项目实现了在 decoder-only 模型（LLaMA 3.2-1B）上采集两类 attention head 强度信号：

1. **Head Output Norm**: 每个 attention head 输出的 L2 范数（合并前）
2. **Head Residual Contribution Norm**: 每个 head 经过输出投影层后对残差流的贡献的 L2 范数

### 主要特性

- ✅ 支持 LLaMA 3.2-1B（transformers 4.5x）
- ✅ 支持 ARC-Challenge 数据集（4-5 个选项，A-E）
- ✅ 灵活的 Prompt 模板系统（适配 SFT/LoRA 微调）
- ✅ 在线统计（Welford 算法），避免内存溢出
- ✅ 支持两种 token 聚合策略：last / all
- ✅ 自动生成热力图可视化
- ✅ 可扩展到其他数据集

## 项目结构

```
project/
├── README.md                  # 本文档
├── requirements.txt           # Python 依赖
├── configs/
│   └── default.yaml          # 默认配置文件
├── src/
│   ├── __init__.py
│   ├── main.py               # 主程序入口
│   ├── config.py             # 配置管理
│   ├── data/                 # 数据层
│   │   ├── __init__.py
│   │   ├── dataset_base.py   # 数据集基类
│   │   ├── arc.py            # ARC-Challenge 数据集
│   │   └── prompt.py         # Prompt 模板构建器
│   ├── model/                # 模型层
│   │   ├── __init__.py
│   │   ├── loader.py         # 模型加载
│   │   ├── hooks.py          # Hook 管理器（核心）
│   │   └── metrics.py        # 在线统计
│   └── utils/                # 工具函数
│       ├── __init__.py
│       ├── seed.py           # 随机种子
│       ├── io.py             # 文件 I/O
│       └── logging.py        # 日志
├── scripts/
│   └── run_arc_collect.sh    # 运行脚本
└── outputs/                  # 输出目录（自动生成）
```

## 环境安装

### 系统要求

- Python >= 3.8
- CUDA >= 11.8（推荐）
- GPU 内存 >= 8GB（LLaMA 3.2-1B bf16）

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

```
torch>=2.0.0
transformers>=4.50.0,<4.60.0
datasets>=2.14.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

## 数据准备

### ARC-Challenge 数据集

本项目支持两种数据加载方式：

#### 方式 1: 本地 JSONL 文件（推荐）

将 ARC-Challenge 数据保存为 JSONL 格式，每行一个 JSON 对象：

```jsonl
{"id": "Mercury_7220990", "question": "Which property of a mineral...", "choices": {"text": ["color", "hardness", "luster", "streak"], "label": ["A", "B", "C", "D"]}, "answerKey": "D"}
```

**字段说明**：

- `id`: 问题 ID
- `question`: 问题文本
- `choices`: 选项字典
  - `text`: 选项文本列表
  - `label`: 选项标签列表（可以是字母或数字）
- `answerKey`: 正确答案（字母 A/B/C/D/E 或数字 1/2/3/4/5）

文件命名：`test.jsonl` 或 `ARC-Challenge-test.jsonl`

#### 方式 2: HuggingFace Datasets 缓存

```python
from datasets import load_dataset

# 下载并缓存数据集
dataset = load_dataset("ai2_arc", "ARC-Challenge", cache_dir="/data/datasets/arc_challenge/")
```

然后将 `data_dir` 设置为缓存目录。

### 选项数量支持

- ✅ **4 个选项**（A/B/C/D）
- ✅ **5 个选项**（A/B/C/D/E）
- ❌ 其他数量的选项会被跳过并记录日志

### Answer Key 映射规则

本项目自动处理不同格式的 `answerKey`：

- **字母格式**: "A", "B", "C", "D", "E" → 直接使用
- **数字格式**: "1", "2", "3", "4", "5" → 映射到 A/B/C/D/E（1-based）

## 使用方法

### 快速开始

1. **修改配置文件** `configs/default.yaml`：

```yaml
model_path: "/data/models/llama-3.2-1b/"    # 模型路径
data_dir: "/data/datasets/arc_challenge/"    # 数据路径
output_dir: "./outputs"                       # 输出路径
max_samples: 5000                             # 最大样本数（-1 表示全部）
batch_size: 4                                 # 批大小
max_length: 384                               # 最大序列长度
dtype: "bf16"                                 # bf16/fp16/fp32
device_map: "auto"                            # auto/cuda/cpu
token_agg: "last"                             # last/all
template_name: "arc_mcq_v1"                   # Prompt 模板
seed: 42                                      # 随机种子
```

2. **运行脚本**：

```bash
# 使用配置文件
python -m src.main --config configs/default.yaml

# 或使用 bash 脚本
bash scripts/run_arc_collect.sh
```

3. **命令行参数覆盖**（可选）：

```bash
python -m src.main \
    --config configs/default.yaml \
    --model_path /data/models/llama-3.2-1b/ \
    --data_dir /data/datasets/arc_challenge/ \
    --max_samples 1000 \
    --batch_size 8
```

### Token 聚合策略

- **`last`** (默认): 每个样本取最后一个有效 token 的激活
- **`all`**: 对所有有效 token 取平均（排除 padding）

### 输出文件

运行完成后，在 `outputs/<experiment_name>_<timestamp>/` 目录下生成：

```
outputs/arc_head_activation_20250106_123456/
├── config.json                              # 运行配置
├── meta.json                                # 元数据（模型信息、统计数据等）
├── head_output_norm_mean.npy                # Head Output Norm 均值 [num_layers, num_heads]
├── head_output_norm_std.npy                 # Head Output Norm 标准差
├── head_resid_contrib_norm_mean.npy         # Head Residual Contribution Norm 均值
├── head_resid_contrib_norm_std.npy          # 标准差
├── head_output_norm_heatmap.png             # 热力图（Head Output Norm）
└── head_resid_contrib_norm_heatmap.png      # 热力图（Head Residual Contribution Norm）
```

### 读取结果

```python
import numpy as np
import json

# 加载激活数据
head_output_norm = np.load("outputs/.../head_output_norm_mean.npy")
head_resid_norm = np.load("outputs/.../head_resid_contrib_norm_mean.npy")

# 加载元数据
with open("outputs/.../meta.json", "r") as f:
    meta = json.load(f)

print(f"Shape: {head_output_norm.shape}")  # (num_layers, num_heads)
print(f"Processed samples: {meta['num_processed']}")
```

## Prompt 模板

### arc_mcq_v1（默认）

设计原则：
- ✅ 清晰的任务指令
- ✅ 强制单字母输出（A/B/C/D/E）
- ✅ 隐藏推理过程（训练友好）
- ✅ 适配全量微调和 LoRA 微调

示例输出：

```
You are a careful reasoner. Read the question and choose the single best answer from the options.
Think step-by-step privately, but do not reveal your reasoning.
Return only the letter of the correct option (A, B, C, or D).

Question: Which property of a mineral can be determined just by looking at it?
Options:
A. color
B. hardness
C. luster
D. streak

Answer: 
```

### 扩展自定义模板

在 `src/data/prompt.py` 中添加新模板：

```python
@PromptBuilder.register_template("my_template")
def my_template(question, option_labels, option_texts, few_shot=0):
    # 自定义 prompt 构建逻辑
    return prompt_text
```

## 技术细节

### Hook 机制

本项目通过 forward hook 在每个 attention 层采集激活：

1. **Hook 位置**: `model.model.layers[i].self_attn`
2. **采集方式**: 重计算 attention 以获取 per-head 输出
3. **内存优化**: 使用在线统计（Welford 算法），不存储所有激活

### Head Output Norm

计算公式：

```
对每个 head h：
  attn_output_h = attention_weights_h @ V_h   # [bs, seq_len, head_dim]
  norm_h = ||attn_output_h[token_pos]||_2     # L2 范数
```

### Head Residual Contribution Norm

计算公式：

```
对每个 head h：
  o_proj_slice = W_o[:, h*head_dim:(h+1)*head_dim]   # 输出投影的对应切片
  contribution_h = attn_output_h @ o_proj_slice^T    # [bs, seq_len, hidden_size]
  norm_h = ||contribution_h[token_pos]||_2           # L2 范数
```

### 性能优化

- 仅在指定 token 位置计算（`token_agg="last"`）
- 批量矩阵运算
- 在线统计避免存储大张量
- `use_cache=False` 减少内存占用

## 扩展到其他数据集

### 步骤 1: 创建数据集类

在 `src/data/` 中创建新文件（如 `commonsenseqa.py`）：

```python
from .dataset_base import DatasetBase

class CommonsenseQADataset(DatasetBase):
    def __init__(self, data_dir, template_name, ...):
        # 实现数据加载逻辑
        pass
    
    def __getitem__(self, idx):
        # 返回格式化样本
        return {
            "prompt_text": ...,
            "answer_letter": ...,
            "option_labels": ...,
            "target_text": ...,
            "meta": {...}
        }
```

### 步骤 2: 注册 Prompt 模板

在 `src/data/prompt.py` 中添加：

```python
@PromptBuilder.register_template("commonsenseqa_v1")
def commonsenseqa_template(question, option_labels, option_texts, few_shot=0):
    # 构建 prompt
    return prompt_text
```

### 步骤 3: 修改配置

在 `configs/default.yaml` 中修改：

```yaml
template_name: "commonsenseqa_v1"
```

## 兼容性说明

### Transformers 版本

- ✅ 支持 transformers 4.50 - 4.59
- ⚠️ 不同版本的 `LlamaAttention` 内部实现可能略有差异
- 💡 如遇到问题，检查 hook 捕获的张量形状

### 验证形状

在 `hooks.py` 中添加调试代码：

```python
logger.debug(f"head_outputs shape: {head_outputs.shape}")  # 应为 [bs, seq_len, num_heads, head_dim]
```

### 已知问题

- Flash Attention 2 可能改变内部计算流程，导致 hook 失效
- 解决方法：设置 `attn_implementation: null` 使用标准实现

## 常见问题

### Q1: 内存不足

**解决方法**：
- 减小 `batch_size`
- 减小 `max_length`
- 使用 `dtype: "fp16"` 或 `"bf16"`
- 减小 `max_samples`

### Q2: 数据集加载失败

**检查**：
- 数据文件路径是否正确
- JSONL 格式是否正确（每行一个有效 JSON）
- HuggingFace datasets 是否已下载

### Q3: Hook 无法捕获激活

**检查**：
- 模型架构是否为 LLaMA 系列
- Transformers 版本是否在 4.50-4.59
- 尝试设置 `attn_implementation: null`

### Q4: 输出全零或异常值

**可能原因**：
- Attention mask 设置错误
- Token 聚合位置计算错误
- 模型权重未正确加载

**调试**：添加 logger.debug 打印中间结果

## 未来训练（SFT/LoRA）

虽然本项目仅做推理采集，但设计考虑了后续训练的便利性：

### 数据集输出

每个样本包含 `target_text` 字段（与 `answer_letter` 相同），可直接用于训练：

```python
# 训练时的 label 对齐
def build_sft_example(prompt_text, target_text, tokenizer):
    full_text = prompt_text + target_text
    input_ids = tokenizer(full_text)["input_ids"]
    
    # 只训练 target_text 部分
    prompt_ids = tokenizer(prompt_text)["input_ids"]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {"input_ids": input_ids, "labels": labels}
```

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{llama_head_activation_collection,
  title = {LLaMA Attention Head Activation Collection},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**: 2025-01-06

