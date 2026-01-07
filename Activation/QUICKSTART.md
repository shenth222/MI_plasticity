# 快速开始指南

> **💡 2025-01-06 更新**：代码已大幅简化（~42% 代码减少），性能提升。详见 [SIMPLIFICATION_NOTE.md](SIMPLIFICATION_NOTE.md)

## 5 分钟上手

### 1️⃣ 安装依赖

```bash
cd /data1/shenth/work/MI_plasticity/Activation
pip install -r requirements.txt
```

### 2️⃣ 准备数据

将 ARC-Challenge 数据放到指定目录（JSONL 格式）：

```bash
# 数据目录结构
/data/datasets/arc_challenge/
└── test.jsonl
```

**JSONL 格式示例**：
```jsonl
{"id": "Mercury_7220990", "question": "Which property...", "choices": {"text": ["color", "hardness", "luster", "streak"], "label": ["A", "B", "C", "D"]}, "answerKey": "D"}
```

### 3️⃣ 准备模型

下载 LLaMA 3.2-1B 模型到本地：

```bash
# 模型目录结构
/data/models/llama-3.2-1b/
├── config.json
├── pytorch_model.bin (或 model.safetensors)
├── tokenizer.json
└── tokenizer_config.json
```

### 4️⃣ 修改配置

编辑 `configs/default.yaml`：

```yaml
model_path: "/data/models/llama-3.2-1b/"      # 修改为你的模型路径
data_dir: "/data/datasets/arc_challenge/"      # 修改为你的数据路径
max_samples: 5000                              # 处理样本数（-1 表示全部）
batch_size: 4                                  # 根据 GPU 内存调整
dtype: "bf16"                                  # bf16/fp16/fp32
token_agg: "last"                              # last/all
```

### 5️⃣ 运行采集

```bash
# 方法 1: 使用 bash 脚本（推荐）
bash scripts/run_arc_collect.sh

# 方法 2: 直接运行
python -m src.main --config configs/default.yaml

# 方法 3: 命令行覆盖参数
python -m src.main \
    --config configs/default.yaml \
    --max_samples 1000 \
    --batch_size 2
```

### 6️⃣ 查看结果

```bash
# 输出目录
ls outputs/arc_head_activation_<timestamp>/

# 包含文件：
# - head_output_norm_mean.npy            # 激活数据
# - head_resid_contrib_norm_mean.npy     # 激活数据
# - head_output_norm_heatmap.png         # 可视化
# - head_resid_contrib_norm_heatmap.png  # 可视化
# - meta.json                            # 元数据
# - config.json                          # 运行配置
```

---

## 🔧 常用参数调整

### 内存不足？

```yaml
batch_size: 2          # 减小批大小
max_length: 256        # 减小序列长度
dtype: "fp16"          # 使用半精度
```

### 加速处理？

```yaml
batch_size: 8          # 增加批大小
max_samples: 1000      # 处理部分样本
device_map: "auto"     # 多卡并行
```

### 改变聚合策略？

```yaml
token_agg: "all"       # 对所有 token 平均（更全面）
token_agg: "last"      # 只用最后 token（更快）
```

---

## 📊 读取和分析结果

```python
import numpy as np
import json
import matplotlib.pyplot as plt

# 加载数据
head_output_norm = np.load("outputs/.../head_output_norm_mean.npy")
head_resid_norm = np.load("outputs/.../head_resid_contrib_norm_mean.npy")

# 加载元数据
with open("outputs/.../meta.json", "r") as f:
    meta = json.load(f)

# 数据形状
print(f"Shape: {head_output_norm.shape}")  # (num_layers, num_heads)

# 统计信息
print(f"Head Output Norm range: [{head_output_norm.min():.4f}, {head_output_norm.max():.4f}]")
print(f"Processed samples: {meta['num_processed']}")

# 分析特定层
layer_0_norms = head_output_norm[0, :]
print(f"Layer 0 head norms: {layer_0_norms}")

# 找出最强的 head
max_layer, max_head = np.unravel_index(head_output_norm.argmax(), head_output_norm.shape)
print(f"Strongest head: Layer {max_layer}, Head {max_head}")
```

---

## 🧪 运行示例验证

```bash
# 运行示例脚本验证安装
python example_usage.py
```

---

## ⚡ 故障排查

### 问题 1: 模块导入失败

```bash
# 确保在项目根目录
cd /data1/shenth/work/MI_plasticity/Activation

# 使用 -m 方式运行
python -m src.main --config configs/default.yaml
```

### 问题 2: 数据加载失败

```bash
# 检查数据文件
ls -lh /data/datasets/arc_challenge/
cat /data/datasets/arc_challenge/test.jsonl | head -1 | python -m json.tool
```

### 问题 3: GPU 内存不足

```yaml
# 修改配置
batch_size: 1          # 最小批大小
dtype: "fp16"          # 半精度
max_length: 256        # 短序列
```

### 问题 4: Hook 无法捕获

```yaml
# 尝试禁用 Flash Attention
attn_implementation: null
```

---

## 📖 进一步阅读

- **完整文档**: `README.md`
- **项目总结**: `PROJECT_SUMMARY.md`
- **使用示例**: `example_usage.py`
- **配置说明**: `configs/default.yaml`

---

## 💡 提示

1. **首次运行**: 建议先用少量样本测试（`max_samples: 100`）
2. **调试模式**: 在 `src/main.py` 中将 logger level 改为 DEBUG
3. **中间保存**: 设置 `save_every: 100` 每 100 步保存一次
4. **多次实验**: 修改 `experiment_name` 避免覆盖

---

**祝使用愉快！** 🚀

如有问题，请查看 README.md 的"常见问题"部分或提交 Issue。

