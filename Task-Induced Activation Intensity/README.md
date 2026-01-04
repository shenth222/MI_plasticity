# Task-Induced Activation Intensity - Pre-Finetuning Head Scoring

本项目实现了对 Llama-3.2-1B 模型在 commonsense 任务上的任务诱导信号（task-induced signals）分析，计算每层每头的四种评分指标。

## 功能特性

实现并对比四种 pre-finetuning head scoring：

1. **Head Output / Activation 强度** (`S_out(h)`): 衡量每个 attention head 输出的激活强度
2. **Attention Entropy** (`S_ent(h)`): 衡量注意力分布的集中程度
3. **Attention to Task-Relevant Tokens** (`S_task(h)`): 衡量对任务相关 token（question span）的注意力
4. **组合分数**: 基于 rank 的融合分数，支持 layer-wise normalization

## 项目结构

```
project_root/
  README.md
  requirements.txt
  configs/
    default.yaml          # 配置文件
  src/
    main.py              # 主入口
    args.py              # 命令行参数解析
    utils/
      seed.py            # 随机种子设置
      io.py              # 文件 I/O 工具
      logging.py         # 日志工具
      span.py            # Prompt span 定位
      stats.py           # 统计归一化工具
    data/
      cs170k_dataset.py  # CS170k 数据集加载器
      prompt.py          # Prompt 模板
    model/
      load_model.py      # 模型加载
      hooks.py           # Forward hooks 捕获
      forward.py         # 推理逻辑
    scoring/
      out_norm.py        # Head output 强度计算
      entropy.py         # Attention entropy 计算
      task_align.py      # Task-relevant attention 计算
      combine.py         # 组合评分
    outputs/
      (运行时自动生成)
```

## 安装

### 环境要求

- Python >= 3.8
- CUDA 支持（可选，CPU 也可运行但较慢）
- PyTorch >= 2.0
- Transformers >= 4.40

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本运行

```bash
python src/main.py \
  --model_path /path/to/Llama-3.2-1B \
  --data_path /path/to/commonsense_170k.jsonl \
  --output_dir outputs/run_001 \
  --max_samples 1024 \
  --batch_size 4 \
  --device cuda:0
```

### 主要参数说明

- `--model_path`: 本地 Llama-3.2-1B 模型路径（必需）
- `--data_path`: CS170k 数据集路径，支持 jsonl/json 格式（必需）
- `--output_dir`: 输出目录（必需）
- `--max_samples`: 使用的最大样本数，默认 1024
- `--batch_size`: 批次大小，默认 4
- `--max_length`: 最大序列长度，默认 512
- `--device`: 设备，默认 `cuda:0`
- `--dtype`: 数据类型，可选 `fp16`/`bf16`/`fp32`，默认 `fp16`
- `--seed`: 随机种子，默认 42
- `--score_query_mode`: 评分时的 query token 模式，可选 `last_token`/`all_tokens`，默认 `last_token`
- `--norm_mode`: 归一化模式，可选 `zscore`/`percentile`，默认 `zscore`
- `--lambda_ent`: entropy 分数的融合权重，默认 0.5
- `--lambda_task`: task-align 分数的融合权重，默认 1.0

### 配置文件方式

也可以使用配置文件（推荐）：

```bash
# 编辑 configs/default.yaml
python src/main.py --config configs/default.yaml
```

### 数据集格式要求

CS170k 数据集应为 JSONL 或 JSON 格式，每条数据包含：

```json
{
  "question": "What happens when you drop a ball?",
  "choices": {
    "text": ["It falls down", "It flies up", "It disappears", "It stays still"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "A"
}
```

如果您的数据集字段不同，请修改 `src/data/cs170k_dataset.py` 中的字段映射。

## 输出文件

运行完成后，`outputs/run_xxx/` 目录将包含：

- `config.yaml`: 运行使用的完整配置
- `scores_raw.csv`: 原始分数（layer, head, out_raw, ent_raw, task_raw）
- `scores_norm.csv`: 归一化分数（layer, head, out_z, ent_z, task_z）
- `scores_combined.csv`: 组合分数（layer, head, combined）
- `topk_global.json`: 全局 Top-k heads
- `topk_per_layer.json`: 每层 Top-k heads
- `run.log`: 运行日志

## 常见问题

### 问题 1: 无法捕获 attention probabilities

**现象**: 报错 "Cannot capture attention probabilities" 或所有 attention 都是 None

**原因**: Llama 模型在使用 Flash Attention 或 SDPA 实现时不会返回 attention weights

**解决方案**:
1. 代码已自动处理：会强制使用 `attn_implementation="eager"` 来获取 attention probs
2. 如果仍有问题，请确保 transformers 版本 >= 4.40
3. 降级到 eager attention 会使推理变慢，这是正常现象

### 问题 2: CUDA out of memory

**解决方案**:
- 减小 `--batch_size`（如改为 1 或 2）
- 减小 `--max_length`（如改为 256）
- 使用 `--dtype fp16` 或 `bf16`
- 减少 `--max_samples`

### 问题 3: Span 提取失败

**现象**: 日志中显示大量 "Failed to extract question span"

**原因**: Prompt 模板与数据集格式不匹配

**解决方案**:
- 检查 `src/data/prompt.py` 中的 prompt 模板
- 确保模板中包含明确的 "Question:" 和 "Choices:" 标记
- 查看 `src/utils/span.py`，可以调整 span 提取逻辑

### 问题 4: 数据集加载失败

**解决方案**:
- 确保数据文件路径正确
- 确保数据文件是有效的 JSON/JSONL 格式
- 修改 `src/data/cs170k_dataset.py` 中的字段映射以适配您的数据格式

## 扩展与修改

### 修改 Prompt 模板

编辑 `src/data/prompt.py` 中的 `create_prompt` 函数。

### 添加新的评分方法

1. 在 `src/scoring/` 下创建新的评分模块
2. 实现评分函数，输入 attention probs/head outputs，输出分数
3. 在 `src/main.py` 中集成新的评分方法

### 适配其他数据集

修改 `src/data/cs170k_dataset.py` 中的 `CS170kDataset` 类，调整字段映射。

## 技术细节

### Attention Probabilities 捕获

- 对于 Llama 模型，强制使用 `attn_implementation="eager"` 以获取 attention weights
- 如果模型架构不支持，使用 forward hooks 捕获（见 `src/model/hooks.py`）

### Layer-wise Normalization

- 支持两种模式：z-score 和 percentile
- z-score: `(x - mean) / std`
- percentile: 将分数映射到 [0, 1] 的百分位数

### Rank-based Fusion

组合分数计算：
```
S_combined = rank(S_out) + λ1 * rank(S_ent) + λ2 * rank(S_task)
```

其中 rank 是在每层内部进行的。

## 引用

如果本项目对您的研究有帮助，请考虑引用相关工作。

## License

MIT License

