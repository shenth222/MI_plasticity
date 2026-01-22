# AdaLoRA Signal-Replacement Ablation - 项目总结

## 📋 项目概览

本项目实现了一个完整的、可直接运行的 AdaLoRA Signal-Replacement Ablation 实验框架。

**实验目标**: 在固定 AdaLoRA 训练与预算调度机制下，仅替换用于 rank/budget 分配的 scoring signal，验证 importance 与 plasticity 的区别及组合效果。

## ✅ 已实现功能

### 1. 核心模块 (src/)

| 模块 | 功能 | 状态 |
|-----|------|------|
| `config.py` | 配置管理，支持 CLI 和配置文件 | ✓ 完成 |
| `data.py` | GLUE MNLI/RTE 数据加载与预处理 | ✓ 完成 |
| `modeling.py` | 模型加载与 AdaLoRA 应用 | ✓ 完成 |
| `signal.py` | Scoring signals 计算（importance/plasticity/combo） | ✓ 完成 |
| `patch_adalora.py` | PEFT RankAllocator monkeypatch | ✓ 完成 |
| `callbacks.py` | TrainerCallback（调用 update_and_allocate） | ✓ 完成 |
| `logging_utils.py` | JSONL 日志记录与结果汇总 | ✓ 完成 |
| `main.py` | CLI 入口（train/eval/export） | ✓ 完成 |
| `plots.py` | 可视化（rank evolution, signal heatmap） | ✓ 完成 |

### 2. 运行脚本 (scripts/)

| 脚本 | 功能 | 状态 |
|-----|------|------|
| `run_mnli.sh` | 运行单个 MNLI 实验 | ✓ 完成 |
| `run_rte.sh` | 运行单个 RTE 实验 | ✓ 完成 |
| `run_ablation_all.sh` | 一键运行所有 4 种 signals | ✓ 完成 |
| `quick_test.sh` | 快速测试（小规模配置） | ✓ 完成 |
| `verify_setup.py` | 环境验证脚本 | ✓ 完成 |

### 3. 文档

| 文档 | 内容 | 状态 |
|-----|------|------|
| `README.md` | 完整文档（安装、使用、FAQ） | ✓ 完成 |
| `QUICKSTART.md` | 快速开始指南 | ✓ 完成 |
| `requirements.txt` | Python 依赖 | ✓ 完成 |
| `.gitignore` | Git 忽略规则 | ✓ 完成 |

## 🎯 实现的 4 种 Scoring Signals

| Signal Type | 计算公式 | 说明 |
|------------|---------|------|
| `baseline_adalora` | PEFT 内置 | 原生 AdaLoRA（不替换） |
| `importance_only` | EMA(\|w·grad\|) | 一阶 Taylor 重要性 |
| `plasticity_only` | EMA(\|\|grad\|\|₂) | 参数可塑性 |
| `combo` | zscore(importance) + λ·zscore(plasticity) | 组合 signal |

## 🔧 关键技术实现

### 1. Monkeypatch 机制

通过 `patch_adalora.py` 实现最小侵入式 patch：
- 保存原始 `RankAllocator.update_and_allocate` 方法
- 注入外部 scores（来自 `SignalTracker`）
- 不修改 AdaLoRA 的预算调度和 rank 裁剪逻辑

### 2. Signal Tracking

`SignalTracker` 在线计算：
- **Importance**: 基于参数范数与梯度范数的乘积
- **Plasticity**: 基于梯度范数
- **EMA 平滑**: 避免 signal 剧烈波动
- **Module-level 聚合**: 适配 AdaLoRA 的 rank 分配粒度

### 3. Callback 机制

`AdaLoRACallback` 确保：
- 仅在真正的 optimizer step 时调用 `update_and_allocate`
- 兼容 gradient accumulation
- 记录 rank 分配历史和 signal scores

### 4. Budget 一致性检查

`BudgetConsistencyCallback` 监控：
- 每次更新后的总 rank
- 与目标 budget 的偏差
- 生成一致性报告

## 📊 输出结构

```
outputs/
└── <task>/              # mnli / rte
    └── <signal>/        # baseline_adalora / importance_only / ...
        └── <seed>/      # 42 / 1 / 2026
            ├── metrics.jsonl           # 训练指标（每个 eval）
            ├── rank_pattern.jsonl      # Rank 分配历史
            ├── signal_scores.jsonl     # Signal scores 历史
            ├── final_summary.json      # 最终汇总
            ├── config.json             # 实验配置
            ├── training.log            # 完整日志
            ├── rank_evolution.png      # Rank 变化曲线
            ├── signal_heatmap.png      # Signal 热力图
            └── checkpoint-*/           # 模型 checkpoints
```

## 🚀 快速开始

### Step 1: 环境设置

```bash
cd /data1/shenth/work/MI_plasticity/AdaLora-ablation
pip install -r requirements.txt
python scripts/verify_setup.py
```

### Step 2: 修改模型路径

编辑 `src/config.py` 或使用 `--model_path` 参数。

### Step 3: 运行实验

```bash
# 单个实验
bash scripts/run_mnli.sh importance_only 42

# 完整 ablation（所有 4 种 signals）
bash scripts/run_ablation_all.sh mnli 42
```

### Step 4: 查看结果

```bash
# 查看汇总
cat outputs/mnli/importance_only/seed42/final_summary.json | jq .

# 生成对比图
python src/plots.py --compare \
    --task mnli \
    --signals baseline_adalora importance_only plasticity_only combo \
    --seed 42
```

## 📈 预期实验流程

### MNLI 任务（大规模）

```bash
# 训练集: ~393k 样本
# 推荐配置:
#   - epochs: 3
#   - batch_size: 32
#   - tinit: 200
#   - tfinal: 200
#   - total steps: ~36,750

bash scripts/run_ablation_all.sh mnli 42
```

### RTE 任务（低资源）

```bash
# 训练集: ~2.5k 样本
# 推荐配置:
#   - epochs: 5
#   - batch_size: 16
#   - tinit: 50
#   - tfinal: 100
#   - total steps: ~780

bash scripts/run_ablation_all.sh rte 42
```

### 多 seed 实验

```bash
for seed in 42 1 2026; do
    bash scripts/run_ablation_all.sh mnli $seed
    bash scripts/run_ablation_all.sh rte $seed
done
```

## 🔍 验证 AdaLoRA 动态性

### 方法 1: 检查训练日志

```bash
grep "AdaLoRA Update" outputs/mnli/importance_only/seed42/training.log
```

期望看到：
```
[AdaLoRA Update] Step 200: Total rank=1728, Active modules=144
[AdaLoRA Update] Step 210: Total rank=1680, Active modules=144
[AdaLoRA Update] Step 220: Total rank=1632, Active modules=144
...
```

### 方法 2: 分析 rank_pattern.jsonl

```python
import json

with open("outputs/mnli/importance_only/seed42/rank_pattern.jsonl") as f:
    for line in f:
        r = json.loads(line)
        print(f"Step {r['step']:4d}: Total rank = {r['total_rank']}")
```

### 方法 3: 可视化

```bash
python src/plots.py --task mnli --signal importance_only --seed 42
# 生成 rank_evolution.png 和 signal_heatmap.png
```

## ⚙️ 配置调优建议

### tinit / tfinal 设置

```python
# 计算总训练步数
total_steps = len(train_dataset) / batch_size / grad_accum * epochs

# 建议:
# - tinit < total_steps (确保开始调整)
# - tinit + tfinal < total_steps (确保有足够窗口)
# - deltaT: 10-20 (调整频率)

# 示例（MNLI）:
# total_steps ≈ 392702 / 32 * 3 ≈ 36750
# tinit=200, tfinal=200, deltaT=10
# → 在 step 200~400 之间每 10 步调整
```

### 内存优化

```bash
# 如果 OOM:
--batch_size 8 \
--gradient_accumulation_steps 4 \
--fp16  # 或 --bf16
```

### 加速训练

```bash
# 使用更大的 batch size
--batch_size 64 \
--gradient_accumulation_steps 2

# 减少 logging 频率
--logging_steps 100
```

## 📊 结果分析示例

### 导出对比表格

```python
import pandas as pd
import json

results = []
for signal in ["baseline_adalora", "importance_only", "plasticity_only", "combo"]:
    path = f"outputs/mnli/{signal}/seed42/final_summary.json"
    with open(path) as f:
        data = json.load(f)
        results.append({
            "Signal": signal,
            "Accuracy": data["metrics"]["final_eval_accuracy"],
            "Final Rank": data["rank_pattern"]["final_total_rank"],
            "Rank Reduction": data["rank_pattern"]["rank_reduction"],
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

预期输出：
```markdown
| Signal            | Accuracy | Final Rank | Rank Reduction |
|-------------------|----------|------------|----------------|
| baseline_adalora  | 0.8520   | 576        | 1152           |
| importance_only   | 0.8545   | 576        | 1152           |
| plasticity_only   | 0.8510   | 576        | 1152           |
| combo             | 0.8560   | 576        | 1152           |
```

## 🛠️ 故障排除

### 问题 1: "Target modules not found"

**原因**: DeBERTa 模块名与配置不匹配

**解决**:
```python
# 调试：打印所有 Linear 模块名
from transformers import AutoModel
model = AutoModel.from_pretrained("path/to/deberta-v3-base")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)

# 修改 src/config.py 中的 target_modules
```

### 问题 2: Rank 不变化

**原因**: tinit/tfinal 设置不当

**检查**:
```bash
# 查看总训练步数
grep "max_steps" outputs/.../training.log

# 确保 tinit < max_steps
```

### 问题 3: Budget 不一致

**原因**: Patch 未正确应用

**调试**:
```python
# 检查 patch 是否生效
grep "Patch" outputs/.../training.log
grep "AdaLoRA Patched" outputs/.../training.log
```

## 📝 代码质量

- ✓ 所有模块都有文档字符串
- ✓ 所有函数都有类型提示
- ✓ 详细的注释说明关键逻辑
- ✓ 异常处理和日志记录
- ✓ 配置与代码分离

## 🎓 扩展建议

### 1. 添加更多 signals

编辑 `src/signal.py`，在 `SignalTracker` 中添加新的 signal 类型。

### 2. 支持更多任务

编辑 `src/data.py`，添加新的 GLUE 任务或自定义数据集。

### 3. 超参数搜索

使用 Optuna 或 Ray Tune 进行超参数优化。

### 4. 分布式训练

修改 `TrainingArguments`，添加 DDP 支持。

## 📚 参考文献

- AdaLoRA: [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
- PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- DeBERTa: [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)

## 📧 联系

如有问题或建议，请检查：
1. `README.md` - 详细文档
2. `QUICKSTART.md` - 快速指南
3. 代码注释 - 内联说明

---

**项目状态**: ✅ 已完成，可直接运行

**最后更新**: 2026-01-22
