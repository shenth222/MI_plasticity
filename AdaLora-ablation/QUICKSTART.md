# 快速开始指南

## 1. 环境设置

### 安装依赖

```bash
cd /data1/shenth/work/MI_plasticity/AdaLora-ablation
pip install -r requirements.txt
```

### 验证环境

```bash
python scripts/verify_setup.py
```

如果所有检查都通过，即可开始实验。

## 2. 配置模型路径

编辑 `src/config.py`，修改 `MODEL_PATH`：

```python
model_name_or_path: str = "/data1/shenth/models/deberta-v3-base"  # 改为实际路径
```

或者在运行时通过命令行参数指定：

```bash
--model_path /path/to/deberta-v3-base
```

## 3. 运行单个实验

### MNLI 任务

```bash
# Baseline AdaLoRA
bash scripts/run_mnli.sh baseline_adalora 42

# Importance signal
bash scripts/run_mnli.sh importance_only 42

# Plasticity signal
bash scripts/run_mnli.sh plasticity_only 42

# Combo signal
bash scripts/run_mnli.sh combo 42
```

### RTE 任务

```bash
bash scripts/run_rte.sh importance_only 42
```

## 4. 运行完整 Ablation

一键运行所有 4 种 signal：

```bash
# MNLI
bash scripts/run_ablation_all.sh mnli 42

# RTE
bash scripts/run_ablation_all.sh rte 42
```

## 5. 查看结果

### 输出目录结构

```
outputs/
└── mnli/
    ├── baseline_adalora/
    │   └── seed42/
    │       ├── metrics.jsonl
    │       ├── rank_pattern.jsonl
    │       ├── signal_scores.jsonl
    │       ├── final_summary.json
    │       └── checkpoint-*/
    ├── importance_only/
    ├── plasticity_only/
    └── combo/
```

### 查看汇总

```bash
cat outputs/mnli/importance_only/seed42/final_summary.json | jq .
```

### 生成可视化

```bash
# 单个实验
python src/plots.py --task mnli --signal importance_only --seed 42

# 对比所有 signals
python src/plots.py --compare \
    --task mnli \
    --signals baseline_adalora importance_only plasticity_only combo \
    --seed 42
```

## 6. 快速测试

使用小规模配置快速测试（1 epoch）：

```bash
bash scripts/quick_test.sh
```

## 常见问题

### Q: 训练太慢？

A: 减小 batch size 或 epoch 数：

```bash
python src/main.py \
    --task mnli \
    --signal importance_only \
    --epochs 1 \
    --batch_size 16 \
    ...
```

### Q: OOM (Out of Memory)？

A: 
1. 减小 batch size: `--batch_size 8`
2. 增加 gradient accumulation: `--gradient_accumulation_steps 4`
3. 减小 max_seq_length (修改 `src/config.py`)

### Q: Rank 不变化？

A: 检查 `tinit` 和 `tfinal` 设置：

```bash
# 确保 tinit < 总训练步数
# 总步数 ≈ len(train_set) / batch_size * epochs

# 例如 MNLI (392k samples, batch=32, epoch=3):
# 总步数 ≈ 392000 / 32 * 3 ≈ 36750

# 设置 tinit=200, tfinal=200, deltaT=10
# 即在 step 200~400 间每 10 步调整
```

### Q: 如何确认 AdaLoRA 在工作？

A: 查看训练日志：

```bash
grep "AdaLoRA Update" outputs/mnli/importance_only/seed42/training.log
```

应该看到类似输出：

```
[AdaLoRA Update] Step 200: Total rank=576, Active modules=144
[AdaLoRA Update] Step 210: Total rank=560, Active modules=144
...
```

## 进阶使用

### 自定义配置

创建配置文件 `my_config.json`：

```json
{
  "model": {
    "model_name_or_path": "/path/to/model"
  },
  "adalora": {
    "init_r": 16,
    "target_r": 8,
    "tinit": 100,
    "tfinal": 300
  },
  "signal": {
    "signal_type": "combo",
    "combo_lambda": 0.5
  }
}
```

使用配置文件：

```bash
python src/main.py --config my_config.json
```

### 多 seed 运行

```bash
for seed in 42 1 2026; do
    bash scripts/run_ablation_all.sh mnli $seed
done
```

### 导出结果表格

```python
import pandas as pd
import json

results = []
for signal in ["baseline_adalora", "importance_only", "plasticity_only", "combo"]:
    with open(f"outputs/mnli/{signal}/seed42/final_summary.json") as f:
        data = json.load(f)
        results.append({
            "signal": signal,
            "accuracy": data["metrics"]["final_eval_accuracy"],
            "final_rank": data["rank_pattern"]["final_total_rank"],
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

## 联系与支持

如有问题，请检查：
1. `README.md` - 完整文档
2. 训练日志 - `outputs/.../training.log`
3. 代码注释 - 所有模块都有详细注释
