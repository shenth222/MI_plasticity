# 项目文件清单

## 目录结构

```
Task-Induced Activation Intensity/
├── README.md                    # 项目说明文档
├── QUICKSTART.md               # 快速开始指南
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略文件
├── example_data.jsonl          # 示例数据（用于测试）
├── run_example.sh              # 示例运行脚本
├── test_setup.py               # 项目结构测试脚本
│
├── configs/                    # 配置文件目录
│   └── default.yaml           # 默认配置
│
└── src/                        # 源代码目录
    ├── main.py                # 主程序入口
    ├── args.py                # 命令行参数解析
    │
    ├── utils/                 # 工具模块
    │   ├── seed.py           # 随机种子设置
    │   ├── io.py             # 文件 I/O 工具
    │   ├── logging.py        # 日志工具
    │   ├── span.py           # Prompt span 定位
    │   └── stats.py          # 统计归一化工具
    │
    ├── data/                  # 数据处理模块
    │   ├── cs170k_dataset.py # CS170k 数据集加载器
    │   └── prompt.py         # Prompt 模板
    │
    ├── model/                 # 模型相关模块
    │   ├── load_model.py     # 模型加载
    │   ├── hooks.py          # Forward hooks 捕获
    │   └── forward.py        # 推理逻辑
    │
    └── scoring/               # 评分模块
        ├── out_norm.py       # Head output 强度计算
        ├── entropy.py        # Attention entropy 计算
        ├── task_align.py     # Task-relevant attention 计算
        └── combine.py        # 组合评分
```

## 文件说明

### 核心文件

- **src/main.py**: 主程序，协调整个工作流程
- **src/args.py**: 处理配置文件和命令行参数
- **configs/default.yaml**: 默认配置文件，包含所有可配置参数

### 数据处理

- **src/data/cs170k_dataset.py**: 数据集加载器，支持自定义字段映射
- **src/data/prompt.py**: Prompt 模板生成，将问题和选项格式化

### 模型相关

- **src/model/load_model.py**: 加载 Llama 模型和 tokenizer
- **src/model/hooks.py**: 使用 PyTorch hooks 捕获中间层输出
- **src/model/forward.py**: 批量推理，返回 attention 和 hidden states

### 评分实现

- **src/scoring/out_norm.py**: 计算每个 head 的输出强度（L2 norm）
- **src/scoring/entropy.py**: 计算 attention 的熵（衡量分布集中程度）
- **src/scoring/task_align.py**: 计算对任务相关 tokens 的注意力
- **src/scoring/combine.py**: 基于 rank 的分数融合，生成 Top-k

### 工具函数

- **src/utils/seed.py**: 设置所有随机种子（确保可复现）
- **src/utils/io.py**: 文件读写（支持 JSON/JSONL/YAML/CSV）
- **src/utils/logging.py**: 日志配置
- **src/utils/span.py**: 从 prompt 中提取 question span
- **src/utils/stats.py**: 统计函数（z-score/percentile 归一化，rank 计算）

## 输出文件

运行后在 `outputs/run_xxx/` 目录生成：

- **config.yaml**: 本次运行的完整配置
- **scores_raw.csv**: 原始分数（未归一化）
- **scores_norm.csv**: 归一化分数（layer-wise）
- **scores_combined.csv**: 组合分数
- **topk_global.json**: 全局 Top-k heads
- **topk_per_layer.json**: 每层 Top-k heads
- **run.log**: 运行日志

## 运行流程

1. **配置加载** (args.py): 合并命令行参数和配置文件
2. **数据加载** (cs170k_dataset.py): 解析并适配数据格式
3. **模型加载** (load_model.py): 加载模型，强制 eager attention
4. **批量推理** (forward.py): 执行 forward pass，捕获 attention 和 hidden states
5. **评分计算**:
   - Head output 强度 (out_norm.py)
   - Attention entropy (entropy.py)
   - Task alignment (task_align.py)
6. **归一化** (stats.py): Layer-wise z-score 或 percentile
7. **组合分数** (combine.py): Rank-based fusion
8. **保存结果** (io.py): 输出 CSV 和 JSON 文件

## 测试与验证

运行以下命令验证项目设置：

```bash
python test_setup.py
```

这将检查：
- 目录结构完整性
- 依赖包安装情况
- 模块导入是否正常
- 示例数据加载

## 常用命令

```bash
# 验证项目设置
python test_setup.py

# 使用配置文件运行
python src/main.py --config configs/default.yaml

# 使用命令行参数运行
python src/main.py \
  --model_path /path/to/model \
  --data_path /path/to/data.jsonl \
  --output_dir outputs/test

# 使用脚本运行
./run_example.sh
```

## 依赖项

核心依赖（见 requirements.txt）：
- PyTorch >= 2.0
- Transformers >= 4.40
- NumPy, Pandas, SciPy
- tqdm, PyYAML

## 扩展性

代码设计考虑了扩展性：

1. **添加新的评分方法**: 在 `src/scoring/` 下创建新模块
2. **支持其他模型**: 修改 `src/model/load_model.py`
3. **适配其他数据集**: 修改 `src/data/cs170k_dataset.py` 的字段映射
4. **自定义 prompt**: 修改 `configs/default.yaml` 中的 template

