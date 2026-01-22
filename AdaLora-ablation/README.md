# AdaLoRA Signal-Replacement Ablation

这是一个基于 HuggingFace Transformers + PEFT 的研究项目，用于验证不同 scoring signal 对 AdaLoRA 动态 rank 分配的影响。

## 🎯 实验目标

在固定 AdaLoRA 训练与预算调度机制的情况下，仅替换用于 rank/budget 分配的 **scoring signal**，以验证：
- **Importance**（重要性）vs. **Plasticity**（可塑性）的区别
- 两者组合的效果

## 📊 实验设置

- **基座模型**: DeBERTa v3 base（本地加载）
- **任务**: GLUE MNLI（主任务）+ GLUE RTE（低资源任务）
- **训练框架**: Transformers Trainer（单机单卡）
- **PEFT 方法**: AdaLoRA (peft==0.18.1)
- **Rank 分配粒度**: module-level（Q/K/V/O + FFN dense）

## 🔬 Ablation 1: Signal Replacement

在完全相同的 AdaLoRA 配置下，仅替换 scoring signal：

| Signal Type | Description | Formula |
|------------|-------------|---------|
| `baseline_adalora` | AdaLoRA 原生机制 | PEFT 内置 |
| `importance_only` | 一阶 Taylor 重要性 | EMA(\|w·grad\|) |
| `plasticity_only` | 参数可塑性 | EMA(\|\|grad\|\|₂) |
| `combo` | 组合 signal | zscore(importance) + λ·zscore(plasticity) |

## 📦 安装

### 环境要求
- Python >= 3.8
- CUDA >= 11.8（推荐）
- peft == 0.18.1（严格要求）

### 依赖安装

```bash
cd /data1/shenth/work/MI_plasticity/AdaLora-ablation
pip install -r requirements.txt
```

### 模型准备

确保 DeBERTa v3 base 模型已下载到本地，修改 `src/config.py` 中的 `MODEL_PATH`：

```python
MODEL_PATH = "/path/to/deberta-v3-base"  # 修改为实际路径
```

## 🚀 快速开始

### 单任务运行示例

```bash
# MNLI 任务（baseline AdaLoRA）
bash scripts/run_mnli.sh baseline_adalora 42

# MNLI 任务（importance signal）
bash scripts/run_mnli.sh importance_only 42

# RTE 任务（plasticity signal）
bash scripts/run_rte.sh plasticity_only 42

# 组合 signal
bash scripts/run_mnli.sh combo 42
```

### Ablation 1 一键运行

运行所有 4 种 signal（MNLI 任务，seed=42）：

```bash
bash scripts/run_ablation_all.sh mnli 42
```

运行多个 seed：

```bash
# 在 MNLI 上运行 seed 42, 1, 2026
for seed in 42 1 2026; do
    bash scripts/run_ablation_all.sh mnli $seed
done
```

## 📂 输出目录结构

```
outputs/
└── <task>/              # mnli / rte
    └── <signal>/        # baseline_adalora / importance_only / plasticity_only / combo
        └── <seed>/      # 42 / 1 / 2026
            ├── metrics.jsonl           # 训练过程指标
            ├── rank_pattern.jsonl      # 每次 rank 更新的分配情况
            ├── signal_scores.jsonl     # 每个 module 的 score
            ├── final_summary.json      # 最终汇总
            └── checkpoint-*/           # 模型 checkpoint
```

## ✅ 验证 AdaLoRA 动态 Rank 分配

### 方法 1: 检查日志

训练过程中会输出类似信息：

```
[AdaLoRA Update] Step 200: Total budget=576, Active modules=144
  - layer.0.attention.self.query_proj: rank 8 → 6
  - layer.0.attention.self.key_proj: rank 8 → 4
  ...
```

### 方法 2: 分析输出文件

```python
import json

# 读取 rank 分配历史
with open("outputs/mnli/importance_only/42/rank_pattern.jsonl") as f:
    for line in f:
        record = json.loads(line)
        print(f"Step {record['step']}: {record['total_rank']}")
```

### 方法 3: 可视化

```bash
cd /data1/shenth/work/MI_plasticity/AdaLora-ablation
python src/plots.py --task mnli --signal importance_only --seed 42
```

将生成：
- `rank_evolution.png`: rank 随训练步数的变化
- `signal_heatmap.png`: module-level score 热力图

## 🔧 配置说明

### AdaLoRA 核心参数

```python
init_r = 12           # 初始 rank
target_r = 4          # 目标 rank（预算约束）
lora_alpha = 16       # scaling factor
tinit = 200           # 开始 rank 调整的步数
tfinal = 200          # 停止 rank 调整的步数（相对于 tinit）
deltaT = 10           # 每隔 deltaT 步调整一次
```

**注意**: 
- `tinit` 必须 < 总训练步数
- `tfinal` 是相对步数，实际停止步数 = `tinit + tfinal`
- `deltaT` 越小，调整越频繁

### Signal 参数

```python
signal_type = "importance_only"  # 选择 signal 类型
ema_decay = 0.9                  # EMA 衰减系数
combo_lambda = 1.0               # combo signal 的权重
normalize_method = "zscore"      # 归一化方法
```

### Target Modules

项目会自动探测 DeBERTa 的实际 Linear 模块名，默认覆盖：
- Attention: `query_proj`, `key_proj`, `value_proj`, `output.dense`
- FFN: `intermediate.dense`, `output.dense`

## 🐛 常见问题

### Q1: `update_and_allocate` 未调用？

**症状**: 训练结束后所有 module rank 仍为 `init_r`

**原因**: Trainer 默认不会调用 AdaLoRA 的 rank 更新逻辑

**解决**: 本项目通过 `AdaLoRACallback` 自动处理，无需手动干预

---

### Q2: Budget 不一致？

**症状**: 不同 signal 的总 rank 不同

**原因**: Scoring signal 影响了 mask_to_budget 逻辑

**检查**: 
```python
# 查看 final_summary.json
cat outputs/mnli/importance_only/42/final_summary.json | grep total_rank
```

本项目在每次更新后会打印 budget consistency check。

---

### Q3: PEFT 版本不匹配？

**症状**: `ImportError` 或 `AttributeError`

**解决**: 
```bash
pip show peft  # 确认版本
pip install peft==0.18.1 --force-reinstall
```

---

### Q4: tinit/tfinal 设置不当？

**症状**: rank 一直不变化

**检查**:
- 总训练步数 = `len(train_dataset) / batch_size / grad_accum * epochs`
- 确保 `tinit` < 总步数
- 确保 `tinit + tfinal` > `tinit`（否则立即停止）

**示例**（MNLI，batch=32，accum=1，epoch=3）:
- 总步数 ≈ 392702 / 32 * 3 ≈ 36878
- tinit=200, tfinal=200, deltaT=10 → 在 step 200~400 间每 10 步调整

---

### Q5: Target modules 不匹配？

**症状**: `ValueError: Target modules not found`

**调试**:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("path/to/deberta-v3-base")
for name, module in model.named_modules():
    print(name, type(module))
```

修改 `src/config.py` 中的 `TARGET_MODULES_MAP`。

## 📊 结果分析

### 对比不同 Signal

```bash
# 生成对比图表
python src/plots.py --compare \
    --task mnli \
    --signals baseline_adalora importance_only plasticity_only combo \
    --seed 42
```

### 导出统计表格

```python
import pandas as pd
import json

results = []
for signal in ["baseline_adalora", "importance_only", "plasticity_only", "combo"]:
    path = f"outputs/mnli/{signal}/42/final_summary.json"
    with open(path) as f:
        data = json.load(f)
        results.append({
            "signal": signal,
            "accuracy": data["eval_accuracy"],
            "final_rank": data["total_rank"]
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

## 📝 引用

本项目基于以下工作：
- [AdaLoRA](https://arxiv.org/abs/2303.10512): Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
- [PEFT](https://github.com/huggingface/peft): State-of-the-art Parameter-Efficient Fine-Tuning

## 📄 License

MIT License
