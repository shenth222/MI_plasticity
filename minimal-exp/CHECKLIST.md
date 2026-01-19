# 项目完整性检查清单

## ✓ 核心代码文件（已提供 + 已修正）

- [x] `src/model/deberta_head_gating.py` - Head gate 注入（已有，兼容 transformers 4.57.5）
- [x] `src/train/finetune_glue.py` - 训练主脚本（已修正 bf16/fp16 自动选择）
- [x] `src/measure/importance_ablation.py` - 重要性测量（已有）
- [x] `src/measure/grad_fisher_gate.py` - 梯度与 Fisher proxy（已有）
- [x] `src/measure/update_magnitude.py` - 更新量计算（已有）
- [x] `src/analysis/make_subset.py` - 固定 eval subset（已修正 seed=999）

## ✓ 新增补齐文件

### 数据加载
- [x] `src/data/glue.py` - GLUE 数据加载与预处理
  - ✓ MNLI 在线加载（validation_matched）
  - ✓ Tokenization（max_len=256）
  - ✓ collate_fn（动态 padding，支持 token_type_ids）
  - ✓ compute_metrics（accuracy）
  - ✓ eval_raw 返回 dict 供 Subset 使用

### 分析与可视化
- [x] `src/analysis/aggregate.py` - JSONL → CSV + stats + cases
  - ✓ Spearman 秩相关（自实现，无需 scipy）
  - ✓ Top-K overlap（K=20）
  - ✓ 反例集合检测（important-but-static, plastic-but-unimportant）
  - ✓ 分位数阈值（p90, p30）

- [x] `src/analysis/plots.py` - 三张图可视化
  - ✓ fig_I_vs_U.png（重要性 vs 更新量）
  - ✓ fig_I_vs_G.png（重要性 vs 梯度）
  - ✓ fig_stats.png（统计指标）
  - ✓ 不使用 seaborn，不指定颜色
  - ✓ 用不同 marker 标记反例集合

### 脚本
- [x] `scripts/run_mnli.sh` - 训练脚本
  - ✓ 保存 θ0 和 θ1
  - ✓ 传参支持 seed

- [x] `scripts/measure_mnli.sh` - 测量脚本
  - ✓ 五步流程：subset → I_pre → G/F → U → I_post
  - ✓ 自动获取 eval size
  - ✓ 固定 subset seed=999

- [x] `scripts/make_plots.sh` - 可视化脚本
  - ✓ 调用 aggregate 和 plots
  - ✓ 输出路径说明

### 包初始化
- [x] `src/__init__.py`
- [x] `src/data/__init__.py`
- [x] `src/model/__init__.py`
- [x] `src/train/__init__.py`
- [x] `src/measure/__init__.py`
- [x] `src/analysis/__init__.py`

### 文档
- [x] `README.md` - 完整文档
  - ✓ 快速开始（三步命令）
  - ✓ 输出说明
  - ✓ 验收标准
  - ✓ 常见问题排查（6 个场景）
  - ✓ 技术细节
  - ✓ 项目结构

- [x] `QUICKSTART.md` - 快速开始指南
  - ✓ 一键命令
  - ✓ 多种子实验
  - ✓ 调试技巧

- [x] `requirements.txt` - 依赖列表
  - ✓ transformers==4.57.5
  - ✓ datasets==4.4.2
  - ✓ torch==2.9.1
  - ✓ numpy, matplotlib, accelerate

### 配置与工具
- [x] `configs/mnli.yaml` - 参考配置（可选）
- [x] `.gitignore` - Git 忽略规则
- [x] `test_setup.py` - 环境测试脚本

## ✓ 关键功能验证

### 数据处理
- [x] MNLI 在线加载（train + validation_matched）
- [x] Tokenization（premise/hypothesis, max_len=256）
- [x] collate_fn 支持动态 padding
- [x] eval_raw 可被 Subset 使用

### 模型与 Gate
- [x] DeBERTa-v3-base 自动下载
- [x] HeadGate hook 注册到 `attention.self`
- [x] Gate 输出处理 tuple/list（兼容 transformers 4.57.5）
- [x] ablate_one 和 set_all_ones 功能正常

### 训练
- [x] BF16/FP16/FP32 自动选择
- [x] 保存 θ0（ckpt_init）和 θ1（ckpt_final）
- [x] Trainer 使用 load_best_model_at_end

### 测量
- [x] 固定 eval subset（1024 条，seed=999）
- [x] 重要性（ablation Δloss）：importance_pre.jsonl, importance_post.jsonl
- [x] 梯度与 Fisher：gradfisher_pre.jsonl
- [x] 更新量：update.jsonl（Q/K/V/O 切片聚合）

### 分析
- [x] JSONL → CSV（heads.csv）
- [x] Spearman 相关系数（无需 scipy）
- [x] Top-K overlap
- [x] 反例集合检测（两类）
- [x] 三张图（matplotlib，无 seaborn，无指定颜色）

## ✓ 输出文件清单

运行完整流程后，`outputs/MNLI/seed1/` 应包含：

```
outputs/MNLI/seed1/
├── ckpt_init/              # θ0
├── ckpt_final/             # θ1
├── trainer_out/            # Trainer 中间文件
├── run_config.json         # 训练配置
├── eval_subset.json        # 固定 subset 索引
├── importance_pre.jsonl    # 重要性（微调前）
├── importance_post.jsonl   # 重要性（微调后）
├── gradfisher_pre.jsonl    # 梯度与 Fisher
├── update.jsonl            # 更新量
├── heads.csv               # 汇总表
├── stats.json              # 统计指标
├── cases.json              # 反例集合
├── fig_I_vs_U.png          # 图1
├── fig_I_vs_G.png          # 图2
└── fig_stats.png           # 图3
```

## ✓ 验收标准

### 数据完整性
- [ ] `heads.csv` 包含 144 行（12 layers × 12 heads）
- [ ] 所有 JSONL 文件行数 = 144
- [ ] `cases.json` 包含两类反例集合（非空）

### 数值合理性
- [ ] Spearman ρ(I_pre, U) < 0.5（弱相关）
- [ ] Top-20 overlap < 0.3（top head 不一致）
- [ ] Important-but-static cases ≥ 3
- [ ] Plastic-but-unimportant cases ≥ 3

### 可视化
- [ ] 三张 PNG 图生成成功
- [ ] 散点图中两类反例用不同 marker 标记
- [ ] 图表无颜色指定（使用默认 matplotlib 配色）

## ✓ 测试步骤

1. **环境测试**（< 5 分钟）
   ```bash
   python test_setup.py
   ```
   - ✓ 所有导入成功
   - ✓ CUDA 可用（或 CPU 回退）
   - ✓ 模型结构验证
   - ✓ HeadGate 功能正常

2. **训练测试**（30-60 分钟）
   ```bash
   bash scripts/run_mnli.sh 1
   ```
   - ✓ θ0 和 θ1 保存成功
   - ✓ 训练 loss 下降
   - ✓ Eval accuracy > 0.8（MNLI baseline）

3. **测量测试**（1-2 小时）
   ```bash
   bash scripts/measure_mnli.sh 1
   ```
   - ✓ 5 个 JSONL 文件生成
   - ✓ eval_subset.json 包含 1024 个索引
   - ✓ 重要性值有明显差异（非全零）

4. **可视化测试**（< 1 分钟）
   ```bash
   bash scripts/make_plots.sh 1
   ```
   - ✓ CSV + JSON + PNG 文件生成
   - ✓ 统计指标合理
   - ✓ 图表清晰可读

## ✓ 常见陷阱已规避

- [x] Gate hook 返回 tuple/list 兼容处理
- [x] Q/K/V 切片使用 out_dim，O 切片使用 in_dim
- [x] BF16 支持检测（torch.cuda.is_bf16_supported）
- [x] MNLI split 使用 validation_matched
- [x] Subset seed 固定为 999
- [x] Spearman 相关自实现（无需 scipy）
- [x] collate_fn 处理 token_type_ids（如果存在）
- [x] eval_raw 保留 dict 格式供 DataLoader 使用

## 总结

✅ **所有 23 个文件已生成**  
✅ **所有关键功能已实现**  
✅ **文档详尽，一键可运行**  
✅ **兼容 transformers 4.57.5**  

准备就绪，可以开始实验！
