# LoRA微调实验指南

本文档介绍如何在原项目基础上使用**LoRA（Low-Rank Adaptation）**进行模型微调，并与全量微调（FFT）进行对比分析。

---

## 📚 目录

- [LoRA简介](#lora简介)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [实验流程](#实验流程)
- [对比分析](#对比分析)
- [输出说明](#输出说明)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

---

## LoRA简介

**LoRA（Low-Rank Adaptation）**是一种参数高效的微调方法：

- **原理**：在预训练权重旁边添加低秩矩阵 A 和 B，冻结原始权重，仅训练A和B
- **优势**：
  - 可训练参数量大幅减少（通常<1%）
  - 训练速度更快，显存占用更小
  - 便于多任务适配和模型分发

- **公式**：`h = W₀x + BAx`，其中 W₀ 是冻结的预训练权重，B 和 A 是可训练的低秩矩阵

---

## 环境配置

### 依赖安装

已在 `requirements.txt` 中添加 LoRA 支持：

```bash
pip install -r requirements.txt
```

新增的依赖：
- `peft>=0.7.0` - Hugging Face 的参数高效微调库
- `scipy` - 用于数值计算和统计分析

---

## 快速开始

### 一键运行LoRA实验

```bash
# 1. 训练LoRA模型
bash scripts/run_lora.sh 1 RTE 8 16

# 2. 测量指标（重要性、可塑性、更新量）
bash scripts/measure_lora.sh 1 RTE 8

# 3. 可视化分析
bash scripts/make_plots_lora.sh 1 RTE 8
```

### 对比FFT和LoRA

```bash
# 先确保已运行FFT实验
bash scripts/run_mnli.sh 1 RTE FFT
bash scripts/measure_mnli.sh 1 RTE FFT
bash scripts/make_plots.sh 1 RTE

# 然后运行对比脚本
bash scripts/compare_fft_lora.sh 1 RTE 8
```

---

## 实验流程

### 步骤1：LoRA训练

```bash
bash scripts/run_lora.sh [seed] [task] [lora_r] [lora_alpha]
```

**参数说明**：
- `seed`: 随机种子（默认：1）
- `task`: GLUE任务名称（默认：RTE，可选：MNLI, SST2等）
- `lora_r`: LoRA秩（默认：8，推荐范围：4-64）
- `lora_alpha`: LoRA缩放因子（默认：16，通常设为 2×r）

**示例**：

```bash
# 使用LoRA rank=8训练RTE任务
bash scripts/run_lora.sh 1 RTE 8 16

# 使用LoRA rank=16训练MNLI任务
bash scripts/run_lora.sh 1 MNLI 16 32
```

**输出**：
- `outputs/LoRA/RTE/seed1_r8/ckpt_init/` - 初始基础模型（θ0）
- `outputs/LoRA/RTE/seed1_r8/ckpt_final/` - LoRA微调后的模型（θ1）
- `outputs/LoRA/RTE/seed1_r8/run_config.json` - 训练配置

**训练时间**：约10-30分钟（取决于任务和数据集大小）

---

### 步骤2：测量指标

```bash
bash scripts/measure_lora.sh [seed] [task] [lora_r]
```

此步骤会依次测量：

1. **创建固定评估子集**（1024条，seed=999）
2. **重要性测量（微调前）** - 基于基础模型θ0的head ablation
3. **梯度与Fisher proxy（微调前）** - 测量每个head的梯度幅值
4. **更新量测量** - 计算LoRA权重合并后的参数变化
5. **重要性测量（微调后）** - 基于LoRA模型θ1的head ablation

**输出**：
- `eval_subset.json` - 固定的评估子集索引
- `importance_pre.jsonl` - 微调前重要性（144行，每个head一行）
- `gradfisher_pre.jsonl` - 梯度和Fisher proxy
- `update.jsonl` - 更新量（绝对值U和相对值Urel）
- `importance_post.jsonl` - 微调后重要性

**测量时间**：约1-2小时

---

### 步骤3：可视化分析

```bash
bash scripts/make_plots_lora.sh [seed] [task] [lora_r]
```

**示例**：

```bash
bash scripts/make_plots_lora.sh 1 RTE 8
```

**生成的图表**：
- `fig_I_vs_U.png` - 重要性 vs 更新量散点图
- `fig_I_vs_G.png` - 重要性 vs 梯度散点图
- `fig_stats.png` - 统计指标柱状图
- `fig_Ipre_vs_Ipost.png` - 微调前后重要性对比
- `fig_Ipost_corrs.png` - 微调后重要性与其他指标的相关性

**生成的数据**：
- `heads.csv` - 所有head的完整指标表
- `stats.json` - Spearman相关系数、top-K重叠度等统计量
- `cases.json` - 反例集合（important-but-static、plastic-but-unimportant）

---

## 对比分析

### FFT vs LoRA 对比

```bash
bash scripts/compare_fft_lora.sh [seed] [task] [lora_r]
```

**前置条件**：
1. 已完成FFT实验（训练+测量+可视化）
2. 已完成LoRA实验（训练+测量+可视化）

**输出目录**：`outputs/COMPARE/[task]/seed[seed]/`

**生成的对比图表**：

1. **compare_I_vs_U.png** - 并排对比FFT和LoRA的重要性vs更新量散点图
2. **compare_stats.png** - 4个子图对比：
   - Spearman相关系数
   - Top-K重叠度
   - 反例数量
   - 微调后重要性相关性

3. **compare_update_dist.png** - 更新量分布对比
4. **compare_headwise.png** - Head-wise对比（同一个head在FFT和LoRA中的指标）

**生成的对比数据**：

- **compare_metrics.json** - 两种方法的关键指标对比
- **compare_summary.txt** - 文字摘要报告

**摘要报告示例**：

```
============================================================
对比摘要: FFT vs LoRA-r8
============================================================

## 1. Spearman相关系数（重要性 vs 更新量）
  ρ(I_pre, U):
    FFT: 0.2341
    LoRA-r8: 0.1876
    差异: 0.0465

  ρ(I_pre, Urel):
    FFT: 0.2103
    LoRA-r8: 0.1654
    差异: 0.0449

## 2. Top-20重叠度
  Top-K overlap (I_pre, U):
    FFT: 0.1500
    LoRA-r8: 0.1250
    差异: 0.0250

## 3. 反例数量
  Important-but-static:
    FFT: 8
    LoRA-r8: 10

  Plastic-but-unimportant:
    FFT: 12
    LoRA-r8: 14

## 4. 主要发现
  - LoRA-r8显示更弱的相关性，更能说明'重要性≠可塑性'
  - LoRA-r8产生更多反例 (24 vs 20)

============================================================
```

---

## 输出说明

### LoRA实验目录结构

```
outputs/LoRA/RTE/seed1_r8/
├── ckpt_init/              # θ0: 初始基础模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── ckpt_final/             # θ1: LoRA适配器权重
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── run_config.json         # 训练配置
├── eval_subset.json        # 固定评估子集
├── importance_pre.jsonl    # 微调前重要性
├── gradfisher_pre.jsonl    # 梯度与Fisher
├── update.jsonl            # 更新量
├── importance_post.jsonl   # 微调后重要性
├── heads.csv               # 汇总表
├── stats.json              # 统计量
├── cases.json              # 反例集合
└── fig_*.png               # 可视化图表
```

### 对比实验目录结构

```
outputs/COMPARE/RTE/seed1/
├── compare_I_vs_U.png      # 散点图对比
├── compare_stats.png       # 统计指标对比
├── compare_update_dist.png # 更新量分布对比
├── compare_headwise.png    # Head-wise对比
├── compare_metrics.json    # 对比指标
└── compare_summary.txt     # 对比摘要
```

---

## 技术细节

### LoRA配置

在 `finetune_glue_lora.py` 中配置：

```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # 秩（rank）
    lora_alpha=16,          # 缩放因子
    lora_dropout=0.1,       # dropout率
    target_modules=[        # 目标模块
        "query_proj",       # Q投影
        "key_proj",         # K投影
        "value_proj",       # V投影
        "dense"             # 输出投影（O）
    ],
    bias="none",            # 不训练bias
    inference_mode=False,   # 训练模式
)
```

### LoRA更新量计算

LoRA的更新量测量需要特殊处理：

1. **加载LoRA模型**：基础模型 + LoRA适配器
2. **合并权重**：`W_new = W_base + B @ A * (alpha/r)`
3. **计算更新量**：与FFT相同的方式计算 `||W_new - W_base||`

**关键代码**（`update_magnitude_lora.py`）：

```python
# 加载LoRA模型并合并权重
base_model = AutoModelForSequenceClassification.from_pretrained(ckpt_init)
lora_model = PeftModel.from_pretrained(base_model, ckpt_final)
merged_model = lora_model.merge_and_unload()

# 计算head-level更新量
for layer, head in enumerate_heads:
    delta_q = merged_Wq[head_slice] - base_Wq[head_slice]
    delta_k = merged_Wk[head_slice] - base_Wk[head_slice]
    delta_v = merged_Wv[head_slice] - base_Wv[head_slice]
    delta_o = merged_Wo[:, head_slice] - base_Wo[:, head_slice]
    
    U = sqrt(||delta_q||² + ||delta_k||² + ||delta_v||² + ||delta_o||²)
```

### LoRA重要性测量

重要性测量使用**head gating**技术，与FFT相同：

1. **合并LoRA权重**以获得完整模型
2. **注入head gates**到attention层
3. **逐个ablation每个head**，测量loss变化
4. **重要性** = loss_ablate - loss_base

**注意**：由于head gating是在attention输出层面操作的，与权重是FFT还是LoRA无关，因此可以直接应用。

---

## 常见问题

### 1. LoRA和FFT的区别

**训练过程**：
- **FFT**：更新所有参数（~125M参数）
- **LoRA**：只更新低秩矩阵（~0.3M参数，r=8时）

**更新量**：
- **FFT**：直接修改原始权重
- **LoRA**：通过 `B @ A` 添加增量更新

**适用场景**：
- **FFT**：追求最佳性能，资源充足
- **LoRA**：参数高效，快速适配，多任务场景

### 2. 如何选择LoRA rank (r)?

**建议**：
- **r=4**: 极致参数效率，性能可能受限
- **r=8**: 推荐默认值，平衡性能和效率
- **r=16**: 更好的性能，接近全量微调
- **r=32-64**: 性能接近FFT，但参数量增加

**实验**：可以运行多个rank进行对比：

```bash
for r in 4 8 16 32; do
    bash scripts/run_lora.sh 1 RTE $r $((r*2))
    bash scripts/measure_lora.sh 1 RTE $r
    bash scripts/make_plots.sh 1 RTE LoRA_r${r}
done
```

### 3. LoRA模型加载失败

**症状**：`RuntimeError: Error(s) in loading state_dict`

**原因**：LoRA权重和基础模型不匹配

**解决**：
1. 确保 `ckpt_init` 和 `ckpt_final` 来自同一次训练
2. 检查 `adapter_config.json` 中的 `base_model_name_or_path`
3. 重新运行训练脚本

### 4. 内存不足 (OOM)

**LoRA优势**：显存占用比FFT少30-50%

**如果仍然OOM**：
- 减小 `batch_size`（默认128，可改为64或32）
- 减小 `max_len`（默认256，可改为128）
- 使用梯度累积：`--gradient_accumulation_steps 2`

### 5. 合并权重后精度下降

**症状**：`merge_and_unload()` 后精度与原LoRA模型不一致

**原因**：数值精度问题或归一化差异

**解决**：
1. 使用 `fp32` 进行合并：`model = model.float().merge_and_unload()`
2. 如果仅用于ablation，精度差异<0.1%是可接受的

### 6. 如何在推理中使用LoRA模型？

**不合并权重**（推荐用于生产）：

```python
from peft import PeftModel
base_model = AutoModelForSequenceClassification.from_pretrained("path/to/ckpt_init")
lora_model = PeftModel.from_pretrained(base_model, "path/to/ckpt_final")
lora_model.eval()
# 直接推理
outputs = lora_model(**inputs)
```

**合并权重**（单一模型部署）：

```python
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("path/to/merged_model")
# 后续可直接加载merged_model
```

---

## 进阶实验

### 多种子对比

```bash
#!/bin/bash
# 运行多个种子的FFT和LoRA实验

for seed in 1 2 3 4 5; do
    echo "=== Seed ${seed} ==="
    
    # FFT
    bash scripts/run_mnli.sh ${seed} RTE FFT
    bash scripts/measure_mnli.sh ${seed} RTE FFT
    bash scripts/make_plots.sh ${seed} RTE
    
    # LoRA
    bash scripts/run_lora.sh ${seed} RTE 8 16
    bash scripts/measure_lora.sh ${seed} RTE 8
    bash scripts/make_plots_lora.sh ${seed} RTE 8
    
    # 对比
    bash scripts/compare_fft_lora.sh ${seed} RTE 8
done

# 汇总多种子结果
python -m src.analysis.aggregate_seeds \
    --method FFT LoRA_r8 \
    --task RTE \
    --seeds 1 2 3 4 5 \
    --out_dir outputs/MULTI_SEED/
```

### 不同rank对比

```bash
#!/bin/bash
# 对比不同LoRA rank的效果

TASK="RTE"
SEED=1

for r in 4 8 16 32; do
    alpha=$((r * 2))
    
    echo "=== LoRA rank=${r} ==="
    bash scripts/run_lora.sh ${SEED} ${TASK} ${r} ${alpha}
    bash scripts/measure_lora.sh ${SEED} ${TASK} ${r}
    bash scripts/make_plots_lora.sh ${SEED} ${TASK} ${r}
done

# 对比不同rank
python -m src.analysis.compare_lora_ranks \
    --task ${TASK} \
    --seed ${SEED} \
    --ranks 4 8 16 32 \
    --out_dir outputs/RANK_COMPARISON/
```

---

## 相关论文

1. **LoRA: Low-Rank Adaptation of Large Language Models**
   - Hu et al., ICLR 2022
   - https://arxiv.org/abs/2106.09685

2. **Parameter-Efficient Transfer Learning for NLP**
   - Houlsby et al., ICML 2019
   - Adapter方法的原始论文

3. **The Power of Scale for Parameter-Efficient Prompt Tuning**
   - Lester et al., EMNLP 2021
   - Prompt tuning方法

---

## 总结

本LoRA扩展提供了：

✅ **完整的LoRA训练pipeline**  
✅ **与FFT等价的测量方法**（重要性、可塑性、更新量）  
✅ **详细的对比分析工具**（可视化+统计）  
✅ **灵活的配置选项**（rank、alpha、target_modules）  
✅ **清晰的文档和示例**  

通过对比FFT和LoRA，您可以：
- 验证"重要性≠可塑性"这一发现在不同微调方法下的普遍性
- 研究参数高效微调方法对模型可塑性的影响
- 探索不同LoRA配置对head更新模式的影响

**祝实验顺利！** 🎉

---

**生成时间**：2026-01-27  
**作者**：AI Assistant (Claude Sonnet 4.5)  
**项目路径**：`/data1/shenth/work/MI_plasticity/minimal-exp`
