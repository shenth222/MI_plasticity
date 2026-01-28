# make_plots脚本说明

## 回答：能否直接复用原项目的make_plots.sh？

**不能直接复用。** 原因是目录结构不同：

- **FFT输出目录**：`outputs/FFT/${TASK}/seed${SEED}`
- **LoRA输出目录**：`outputs/LoRA/${TASK}/seed${SEED}_r${RANK}`

---

## 解决方案

已创建专门的 **`make_plots_lora.sh`** 脚本，完全仿照FFT的实现风格。

### 实现对比

#### FFT版本 (make_plots.sh)

```bash
SEED=${1:-1}
TASK=${2:-"MNLI"}
OUT_DIR="outputs/FFT/${TASK}/seed${SEED}"

# 调用相同的Python脚本
python -m src.analysis.aggregate --exp_dir ${OUT_DIR} --topk 20
python -m src.analysis.plots --exp_dir ${OUT_DIR}
```

#### LoRA版本 (make_plots_lora.sh)

```bash
SEED=${1:-1}
TASK=${2:-"RTE"}
LORA_R=${3:-8}
OUT_DIR="outputs/LoRA/${TASK}/seed${SEED}_r${LORA_R}"

# 调用相同的Python脚本
python -m src.analysis.aggregate --exp_dir ${OUT_DIR} --topk 20
python -m src.analysis.plots --exp_dir ${OUT_DIR}
```

### 关键相似点

✅ **完全相同的实现逻辑**
- 调用相同的Python模块（`aggregate.py` 和 `plots.py`）
- 使用相同的参数（`--exp_dir`, `--topk`）
- 输出相同的文件（5张图表 + 3个数据文件）

✅ **相同的输入要求**
- 需要4个JSONL文件：
  - `importance_pre.jsonl`
  - `gradfisher_pre.jsonl`
  - `update.jsonl`
  - `importance_post.jsonl`

✅ **相同的输出结果**
- `heads.csv` - 完整指标表
- `stats.json` - 统计量
- `cases.json` - 反例集合
- 5张PNG图表

---

## 使用方法

### FFT可视化

```bash
bash scripts/make_plots.sh [seed] [task]

# 示例
bash scripts/make_plots.sh 1 RTE
```

### LoRA可视化

```bash
bash scripts/make_plots_lora.sh [seed] [task] [lora_r]

# 示例
bash scripts/make_plots_lora.sh 1 RTE 8
```

---

## 为什么不能合并成一个脚本？

虽然逻辑相同，但有以下原因建议保持分离：

### 1. 目录结构差异
- FFT：`seed${SEED}`
- LoRA：`seed${SEED}_r${LORA_R}`（需要额外的rank参数）

### 2. 默认参数差异
- FFT默认任务：MNLI
- LoRA默认任务：RTE
- LoRA需要额外的rank参数（默认8）

### 3. 清晰性和易用性
- 两个独立脚本更清晰，避免参数混淆
- 用户明确知道使用哪个脚本
- 错误提示更准确

---

## 完整实验流程

### FFT完整流程

```bash
# 1. 训练
bash scripts/run_mnli.sh 1 RTE FFT

# 2. 测量
bash scripts/measure_mnli.sh 1 RTE FFT

# 3. 可视化
bash scripts/make_plots.sh 1 RTE
```

### LoRA完整流程

```bash
# 1. 训练
bash scripts/run_lora.sh 1 RTE 8 16

# 2. 测量
bash scripts/measure_lora.sh 1 RTE 8

# 3. 可视化
bash scripts/make_plots_lora.sh 1 RTE 8
```

### 对比分析

```bash
# 确保两个实验都已完成上述3步
bash scripts/compare_fft_lora.sh 1 RTE 8
```

---

## 底层Python脚本的复用

虽然Shell脚本是分开的，但**底层Python脚本是完全复用的**：

### aggregate.py
- 用于FFT和LoRA
- 读取JSONL文件，生成CSV和统计量
- 检测反例集合

### plots.py
- 用于FFT和LoRA
- 生成5张可视化图表
- 无需知道数据来自FFT还是LoRA

### 这体现了良好的代码设计：
- **Shell脚本**：处理目录结构和参数差异
- **Python脚本**：处理通用的数据分析逻辑
- **分层清晰**：上层适配，下层复用

---

## 总结

| 方面 | FFT | LoRA | 是否共享 |
|------|-----|------|----------|
| Shell脚本 | `make_plots.sh` | `make_plots_lora.sh` | ❌ 分开 |
| Python脚本 | `aggregate.py`, `plots.py` | `aggregate.py`, `plots.py` | ✅ 共享 |
| 输入格式 | JSONL | JSONL | ✅ 相同 |
| 输出文件 | 5图+3数据 | 5图+3数据 | ✅ 相同 |
| 目录结构 | `outputs/FFT/...` | `outputs/LoRA/...` | ❌ 不同 |

**结论**：通过创建 `make_plots_lora.sh`，既保持了与FFT实现风格的一致性，又适配了LoRA的目录结构，同时最大化复用了底层Python代码。✅

---

**创建时间**：2026-01-27
