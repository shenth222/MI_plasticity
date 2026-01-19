# 项目文件清单

本文档列出了完整项目中的所有文件及其用途。

## 📁 项目结构概览

```
minimal-exp/
├── configs/              # 配置文件
├── scripts/              # 运行脚本
├── src/                  # 源代码
│   ├── data/            # 数据加载
│   ├── model/           # 模型相关
│   ├── train/           # 训练
│   ├── measure/         # 测量
│   └── analysis/        # 分析
├── outputs/             # 输出（运行时自动生成）
└── [文档和配置]
```

## 📄 文件列表（共 25 个）

### 📘 文档文件（4 个）

1. **README.md** - 主文档
   - 完整的项目说明
   - 快速开始指南
   - 输出说明、验收标准
   - 常见问题排查（6 个场景）
   - 技术细节

2. **QUICKSTART.md** - 快速开始
   - 一键命令
   - 预期输出示例
   - 调试技巧
   - 多种子实验

3. **CHECKLIST.md** - 完整性检查清单
   - 文件清单（✓）
   - 功能验证（✓）
   - 测试步骤
   - 验收标准

4. **PROJECT_FILES.md** - 本文件
   - 完整文件清单
   - 每个文件的用途说明

### ⚙️ 配置文件（3 个）

5. **requirements.txt** - Python 依赖
   - transformers==4.57.5
   - datasets==4.4.2
   - torch==2.9.1
   - numpy, matplotlib, accelerate

6. **configs/mnli.yaml** - MNLI 实验配置（可选）
   - 超参数参考
   - 当前未被脚本使用，仅供参考

7. **.gitignore** - Git 忽略规则
   - Python 缓存、虚拟环境
   - IDE 配置、输出文件

### 🔧 工具脚本（4 个）

8. **test_setup.py** - 环境测试脚本
   - 验证所有导入
   - 检查 CUDA/BF16/FP16
   - 测试模型加载
   - 验证 HeadGate
   - 测试工具函数

9. **scripts/run_mnli.sh** - 训练脚本
   - 训练 DeBERTa-v3-base on MNLI
   - 保存 θ0（ckpt_init）和 θ1（ckpt_final）
   - 支持传参 seed

10. **scripts/measure_mnli.sh** - 测量脚本
    - 步骤 1：固定 eval subset
    - 步骤 2：重要性（微调前）
    - 步骤 3：梯度与 Fisher
    - 步骤 4：更新量
    - 步骤 5：重要性（微调后）

11. **scripts/make_plots.sh** - 可视化脚本
    - 汇总 JSONL → CSV
    - 生成统计指标
    - 生成三张图

### 🐍 Python 源代码（16 个）

#### 包初始化（6 个）

12. **src/__init__.py**
13. **src/data/__init__.py**
14. **src/model/__init__.py**
15. **src/train/__init__.py**
16. **src/measure/__init__.py**
17. **src/analysis/__init__.py**

#### 数据加载（1 个）

18. **src/data/glue.py** ✨ 新增
    - `load_glue_dataset()` - 加载并预处理 GLUE 任务
    - 支持 MNLI 和 RTE
    - 在线加载 datasets（validation_matched for MNLI）
    - Tokenization（max_len=256）
    - collate_fn（动态 padding，支持 token_type_ids）
    - compute_metrics（accuracy）
    - 返回 train/eval/eval_raw/collate_fn/num_labels

#### 模型相关（1 个）

19. **src/model/deberta_head_gating.py** ✅ 已有（已验证兼容）
    - `HeadGatingConfig` - Gate 配置
    - `DebertaV2HeadGate` - Head gate 注入
    - Hook 注册到 `attention.self`
    - 兼容 transformers 4.57.5（处理 tuple/list 输出）
    - `set_all_ones()` / `ablate_one()` / `remove()`

#### 训练（1 个）

20. **src/train/finetune_glue.py** ✅ 已有（已修正）
    - 主训练脚本
    - 自动选择精度：BF16 > FP16 > FP32
    - 保存 θ0（ckpt_init）和 θ1（ckpt_final）
    - 使用 HuggingFace Trainer
    - load_best_model_at_end=True

#### 测量（3 个）

21. **src/measure/importance_ablation.py** ✅ 已有
    - 重要性测量（ablation Δloss）
    - Head-level ablation
    - 固定 eval subset
    - 输出 JSONL（layer, head, I, loss_base, loss_ablate）

22. **src/measure/grad_fisher_gate.py** ✅ 已有
    - 梯度幅值（G = mean |∂L/∂gate|）
    - Fisher 近似（F = mean (∂L/∂gate)^2）
    - 预测可塑性（Ppred = G^2 / (F + ε)）
    - 输出 JSONL（layer, head, G, F, Ppred）

23. **src/measure/update_magnitude.py** ✅ 已有
    - 更新量计算（θ1 - θ0）
    - Q/K/V 切片（out_dim）+ O 切片（in_dim）
    - U = sqrt(Uq^2 + Uk^2 + Uv^2 + Uo^2)
    - Urel = U / (初始范数之和)
    - 输出 JSONL（layer, head, U, Urel, Uq, Uk, Uv, Uo）

#### 分析（3 个）

24. **src/analysis/make_subset.py** ✅ 已有（已修正）
    - 固定 eval subset（默认 1024 条）
    - 固定随机种子 999
    - 输出 JSON 索引列表

25. **src/analysis/aggregate.py** ✨ 新增
    - 加载所有 JSONL 文件
    - Join on (layer, head)
    - 生成 heads.csv（完整指标表）
    - 计算 Spearman 相关（自实现）
    - 计算 Top-K overlap（K=20）
    - 检测反例集合：
      - important-but-static: I_pre top10% & Urel bottom30%
      - plastic-but-unimportant: Urel top10% & I_pre bottom30%
    - 输出 stats.json + cases.json

26. **src/analysis/plots.py** ✨ 新增
    - 生成三张图（matplotlib，无 seaborn，无颜色指定）
    - fig_I_vs_U.png：I_pre vs Urel 散点图
    - fig_I_vs_G.png：I_pre vs G 散点图
    - fig_stats.png：统计指标柱状图
    - 用不同 marker 标记反例集合

## 📊 输出文件（运行后自动生成）

运行完整流程后，`outputs/MNLI/seed{seed}/` 包含：

| 文件名 | 生成步骤 | 说明 |
|--------|---------|------|
| `ckpt_init/` | run_mnli.sh | θ0（初始模型） |
| `ckpt_final/` | run_mnli.sh | θ1（微调后最佳模型） |
| `trainer_out/` | run_mnli.sh | Trainer 中间文件 |
| `run_config.json` | run_mnli.sh | 训练配置 |
| `eval_subset.json` | measure_mnli.sh | 固定 subset 索引（1024 条） |
| `importance_pre.jsonl` | measure_mnli.sh | 重要性（微调前） |
| `gradfisher_pre.jsonl` | measure_mnli.sh | 梯度与 Fisher |
| `update.jsonl` | measure_mnli.sh | 更新量 |
| `importance_post.jsonl` | measure_mnli.sh | 重要性（微调后） |
| `heads.csv` | make_plots.sh | 汇总表（144 行） |
| `stats.json` | make_plots.sh | 统计指标 |
| `cases.json` | make_plots.sh | 反例集合 |
| `fig_I_vs_U.png` | make_plots.sh | 图1：重要性 vs 更新量 |
| `fig_I_vs_G.png` | make_plots.sh | 图2：重要性 vs 梯度 |
| `fig_stats.png` | make_plots.sh | 图3：统计指标 |

## 🎯 文件状态说明

- ✅ **已有（已验证）**：用户提供的代码，已验证兼容性
- ✅ **已有（已修正）**：用户提供的代码，进行了最小修改
- ✨ **新增**：本次生成的补齐文件

### 修正说明

1. **finetune_glue.py**
   - 修改前：`fp16=torch.cuda.is_available()`
   - 修改后：自动选择 BF16 > FP16 > FP32

2. **make_subset.py**
   - 修改前：`--seed` 默认 123
   - 修改后：`--seed` 默认 999（用户要求）

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试环境（可选）
python test_setup.py

# 3. 运行完整实验
bash scripts/run_mnli.sh 1
bash scripts/measure_mnli.sh 1
bash scripts/make_plots.sh 1

# 4. 查看结果
cat outputs/MNLI/seed1/stats.json
cat outputs/MNLI/seed1/cases.json
ls outputs/MNLI/seed1/*.png
```

## 📝 注意事项

1. **运行时间**
   - 训练：30-60 分钟（单 V100/A100）
   - 测量：1-2 小时（144 heads × 1024 samples）
   - 可视化：< 1 分钟

2. **硬件需求**
   - 推荐：单卡 V100/A100（16GB+）
   - 最低：单卡 GPU（调小 batch size）
   - CPU：可运行但极慢（不推荐）

3. **存储需求**
   - 模型检查点：约 1.5GB × 2 = 3GB
   - 中间文件：约 100MB
   - 总计：约 3-4GB / seed

4. **网络需求**
   - 首次运行需下载：
     - DeBERTa-v3-base 模型（约 700MB）
     - MNLI 数据集（约 300MB）
   - 后续运行使用缓存

## 📧 支持

遇到问题？请查阅：
1. **README.md** - 完整文档 + 常见问题排查
2. **QUICKSTART.md** - 快速开始 + 调试技巧
3. **CHECKLIST.md** - 验收标准 + 测试步骤

---

**项目生成时间**：2026-01-16  
**环境要求**：transformers 4.57.5, datasets 4.4.2, torch 2.9.1  
**适用模型**：DeBERTa-v3-base  
**适用任务**：MNLI（可扩展到其他 GLUE 任务）
