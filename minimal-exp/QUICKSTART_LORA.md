# LoRA实验快速入门 🚀

本文档提供最快速的LoRA实验运行指南。

---

## 一键运行

### 完整实验（训练→测量→可视化）

```bash
# 进入项目目录
cd /data1/shenth/work/MI_plasticity/minimal-exp

# 1. LoRA训练（约10-30分钟）
bash scripts/run_lora.sh 1 RTE 8 16

# 2. 测量指标（约1-2小时）
bash scripts/measure_lora.sh 1 RTE 8

# 3. 可视化（约1分钟）
bash scripts/make_plots_lora.sh 1 RTE 8
```

### FFT vs LoRA 对比

```bash
# 运行FFT实验（如果还没有）
bash scripts/run_mnli.sh 1 RTE FFT
bash scripts/measure_mnli.sh 1 RTE FFT  
bash scripts/make_plots.sh 1 RTE

# 对比分析
bash scripts/compare_fft_lora.sh 1 RTE 8
```

---

## 查看结果

### LoRA实验结果

```bash
# 查看统计指标
cat outputs/LoRA/RTE/seed1_r8/stats.json

# 查看反例集合
cat outputs/LoRA/RTE/seed1_r8/cases.json

# 查看所有head的指标
head -20 outputs/LoRA/RTE/seed1_r8/heads.csv

# 查看图表
ls outputs/LoRA/RTE/seed1_r8/fig_*.png
```

### 对比结果

```bash
# 查看对比摘要
cat outputs/COMPARE/RTE/seed1/compare_summary.txt

# 查看对比指标
cat outputs/COMPARE/RTE/seed1/compare_metrics.json

# 查看对比图表
ls outputs/COMPARE/RTE/seed1/compare_*.png
```

---

## 参数说明

### run_lora.sh

```bash
bash scripts/run_lora.sh [seed] [task] [lora_r] [lora_alpha]
```

- `seed`: 随机种子（默认：1）
- `task`: 任务名称（默认：RTE，可选：MNLI, SST2, QNLI等）
- `lora_r`: LoRA秩（默认：8，建议范围：4-64）
- `lora_alpha`: LoRA缩放（默认：16，通常为2×r）

**示例**：

```bash
# 使用rank=16训练MNLI
bash scripts/run_lora.sh 1 MNLI 16 32

# 使用rank=4训练RTE（更快，参数更少）
bash scripts/run_lora.sh 2 RTE 4 8
```

### measure_lora.sh

```bash
bash scripts/measure_lora.sh [seed] [task] [lora_r]
```

**注意**：必须先运行 `run_lora.sh` 完成训练。

### compare_fft_lora.sh

```bash
bash scripts/compare_fft_lora.sh [seed] [task] [lora_r]
```

**前置条件**：
1. FFT实验已完成（训练+测量+可视化）
2. LoRA实验已完成（训练+测量+可视化）

---

## 常用命令

### 批量运行多个种子

```bash
# 运行3个种子的LoRA实验
for seed in 1 2 3; do
    bash scripts/run_lora.sh ${seed} RTE 8 16
    bash scripts/measure_lora.sh ${seed} RTE 8
    bash scripts/make_plots_lora.sh ${seed} RTE 8
done
```

### 对比不同rank

```bash
# 对比rank=4、8、16的效果
for r in 4 8 16; do
    alpha=$((r * 2))
    bash scripts/run_lora.sh 1 RTE ${r} ${alpha}
    bash scripts/measure_lora.sh 1 RTE ${r}
    bash scripts/make_plots_lora.sh 1 RTE ${r}
done
```

### 检查实验状态

```bash
# 查看所有实验输出目录
ls -lh outputs/

# 检查LoRA训练是否完成
ls outputs/LoRA/RTE/seed1_r8/ckpt_final/

# 检查测量是否完成
ls outputs/LoRA/RTE/seed1_r8/*.jsonl

# 检查可视化是否完成
ls outputs/LoRA/RTE/seed1_r8/fig_*.png
```

---

## 预期输出

### 训练完成后

```
outputs/LoRA/RTE/seed1_r8/
├── ckpt_init/              # ✓ 基础模型
├── ckpt_final/             # ✓ LoRA权重
└── run_config.json         # ✓ 配置文件
```

### 测量完成后

```
outputs/LoRA/RTE/seed1_r8/
├── eval_subset.json        # ✓ 评估子集
├── importance_pre.jsonl    # ✓ 144行
├── gradfisher_pre.jsonl    # ✓ 144行
├── update.jsonl            # ✓ 144行
└── importance_post.jsonl   # ✓ 144行
```

### 可视化完成后

```
outputs/LoRA/RTE/seed1_r8/
├── heads.csv               # ✓ 完整指标表
├── stats.json              # ✓ 统计量
├── cases.json              # ✓ 反例集合
├── fig_I_vs_U.png          # ✓
├── fig_I_vs_G.png          # ✓
├── fig_stats.png           # ✓
├── fig_Ipre_vs_Ipost.png   # ✓
└── fig_Ipost_corrs.png     # ✓
```

### 对比完成后

```
outputs/COMPARE/RTE/seed1/
├── compare_I_vs_U.png      # ✓ 散点图对比
├── compare_stats.png       # ✓ 统计对比
├── compare_update_dist.png # ✓ 更新量分布
├── compare_headwise.png    # ✓ Head-wise对比
├── compare_metrics.json    # ✓ 对比指标
└── compare_summary.txt     # ✓ 对比摘要
```

---

## 故障排除

### 问题1：训练时CUDA OOM

**解决**：减小batch size

```bash
# 修改 scripts/run_lora.sh 中的 --bsz 参数
--bsz 64  # 原来是128
```

### 问题2：测量时间过长

**优化**：
- 减小评估子集大小（修改 `measure_lora.sh` 中的 `--n` 参数）
- 使用更大的batch size（如果显存允许）

### 问题3：对比脚本报错

**检查**：
1. FFT和LoRA实验都已完成
2. 目录结构正确
3. 所有必需文件都存在

```bash
# 检查FFT实验
ls outputs/FFT/RTE/seed1/heads.csv
ls outputs/FFT/RTE/seed1/stats.json

# 检查LoRA实验
ls outputs/LoRA/RTE/seed1_r8/heads.csv
ls outputs/LoRA/RTE/seed1_r8/stats.json
```

### 问题4：Python模块导入错误

**解决**：确保使用 `python -m` 运行

```bash
# 正确
python -m src.train.finetune_glue_lora ...

# 错误
python src/train/finetune_glue_lora.py ...
```

---

## 下一步

- 📖 阅读 [README_LORA.md](README_LORA.md) 了解详细技术细节
- 🔬 运行多种子实验以验证结果稳定性
- 📊 尝试不同的LoRA rank进行对比
- 🚀 在其他GLUE任务上测试（MNLI、SST2、QNLI等）

---

**祝实验顺利！** 🎉
