# 快速开始指南

## 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 2. 准备数据

### 方式 1：使用示例数据（测试）

项目已包含 `example_data.jsonl`，可直接用于测试。

### 方式 2：使用您的 CS170k 数据

确保数据格式为 JSONL 或 JSON，每条数据包含：

```json
{
  "question": "问题文本",
  "choices": {
    "text": ["选项A", "选项B", "选项C", "选项D"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "A"
}
```

如果字段不同，请修改 `configs/default.yaml` 中的 `field_mapping`。

## 3. 准备模型

确保您已下载 Llama-3.2-1B 模型到本地，并记下路径。

## 4. 修改配置

编辑 `configs/default.yaml`：

```yaml
model:
  path: "/your/path/to/Llama-3.2-1B"  # 修改这里

data:
  path: "/your/path/to/commonsense_170k.jsonl"  # 修改这里
  max_samples: 1024  # 可调整

inference:
  batch_size: 4  # 根据您的 GPU 显存调整
  device: "cuda:0"  # 或 "cpu"
```

## 5. 运行

### 方式 1：使用配置文件

```bash
python src/main.py --config configs/default.yaml
```

### 方式 2：使用命令行参数

```bash
python src/main.py \
  --model_path /path/to/Llama-3.2-1B \
  --data_path /path/to/commonsense_170k.jsonl \
  --output_dir outputs/run_001 \
  --max_samples 1024 \
  --batch_size 4 \
  --device cuda:0
```

### 方式 3：使用脚本

```bash
# 修改 run_example.sh 中的路径
chmod +x run_example.sh
./run_example.sh
```

## 6. 查看结果

运行完成后，在输出目录（如 `outputs/run_001/`）中查看：

- `run.log`: 运行日志
- `scores_raw.csv`: 原始分数
- `scores_norm.csv`: 归一化分数
- `scores_combined.csv`: 组合分数
- `topk_global.json`: 全局 Top-k heads
- `topk_per_layer.json`: 每层 Top-k heads

## 7. 常见问题

### GPU 显存不足

```bash
# 减小 batch size
python src/main.py --config configs/default.yaml --batch_size 1

# 或减小序列长度
python src/main.py --config configs/default.yaml --max_length 256

# 或使用 CPU（会很慢）
python src/main.py --config configs/default.yaml --device cpu
```

### 数据加载失败

检查数据文件格式，确保是有效的 JSON/JSONL。可以先用示例数据测试。

### 模型加载失败

确保模型路径正确，且包含必要的文件（config.json, tokenizer 等）。

## 8. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径 | 必需 |
| `--data_path` | 数据路径 | 必需 |
| `--output_dir` | 输出目录 | 必需 |
| `--max_samples` | 最大样本数 | 1024 |
| `--batch_size` | 批次大小 | 4 |
| `--max_length` | 最大序列长度 | 512 |
| `--device` | 设备 | cuda:0 |
| `--dtype` | 数据类型 | fp16 |
| `--seed` | 随机种子 | 42 |
| `--score_query_mode` | Query 模式 | last_token |
| `--norm_mode` | 归一化模式 | zscore |
| `--lambda_ent` | Entropy 权重 | 0.5 |
| `--lambda_task` | Task-align 权重 | 1.0 |

## 9. 输出文件说明

### scores_raw.csv

包含原始分数，列：

- `layer`: 层索引（0-based）
- `head`: head 索引（0-based）
- `out_raw`: Head output 强度原始分数
- `ent_raw`: Attention entropy 原始分数
- `task_raw`: Task alignment 原始分数

### scores_norm.csv

包含归一化分数（layer-wise），列同上但为 `_norm` 后缀。

### scores_combined.csv

包含组合分数，列：

- `layer`: 层索引
- `head`: head 索引
- `combined`: 组合分数

### topk_global.json

全局 Top-k heads（按组合分数排序），格式：

```json
[
  {"layer": 15, "head": 8, "score": 125.6},
  {"layer": 14, "head": 12, "score": 120.3},
  ...
]
```

### topk_per_layer.json

每层 Top-k heads，格式：

```json
{
  "0": [
    {"head": 5, "score": 12.3},
    ...
  ],
  "1": [...],
  ...
}
```

