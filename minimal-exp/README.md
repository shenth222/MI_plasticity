# 最小可复现实验：Head 粒度下"重要性 ≠ 可塑性/更新量"

本项目用于在 **DeBERTa-v3-base** 上进行全量微调（MNLI 任务），证明在 head 粒度下 **重要性 (Importance) ≠ 可塑性 (Plasticity) / 更新量 (Update)**。

---

## 环境要求

- Python 3.8+
- CUDA（推荐，自动选择 bf16/fp16/fp32）
- 单机单卡

### 依赖安装

```bash
pip install -r requirements.txt
```

**requirements.txt** 包含：
- `transformers==4.57.5`
- `datasets==4.4.2`
- `torch==2.9.1`
- `numpy`
- `matplotlib`
- `accelerate`

---

## 快速开始

运行完整实验只需三步命令：

### 1. 训练模型（保存 θ0 和 θ1）

```bash
bash scripts/run_mnli.sh 1
```

- 参数 `1` 是随机种子（seed）
- 训练会保存：
  - `outputs/MNLI/seed1/ckpt_init` (θ0: 初始模型)
  - `outputs/MNLI/seed1/ckpt_final` (θ1: 微调后最佳模型)
- 默认配置：lr=2e-5, epochs=3, batch_size=16
- 自动选择精度：优先 bf16，其次 fp16，否则 fp32

### 2. 测量重要性、可塑性、更新量

```bash
bash scripts/measure_mnli.sh 1
```

此步骤依次执行：
1. 固定 eval subset（1024 条，seed=999）→ `eval_subset.json`
2. 重要性（微调前）→ `importance_pre.jsonl`
3. 梯度与 Fisher proxy（微调前）→ `gradfisher_pre.jsonl`
4. 更新量（θ1 - θ0）→ `update.jsonl`
5. 重要性（微调后）→ `importance_post.jsonl`

### 3. 汇总与可视化

```bash
bash scripts/make_plots.sh 1
```

此步骤生成：
- `heads.csv`：所有 head 的指标汇总表
- `stats.json`：Spearman 相关系数、top-K overlap 等统计量
- `cases.json`：反例集合（important-but-static、plastic-but-unimportant）
- `fig_I_vs_U.png`：重要性 vs 更新量散点图
- `fig_I_vs_G.png`：重要性 vs 梯度散点图
- `fig_stats.png`：统计指标柱状图

---

## 输出说明

所有输出保存在 `outputs/MNLI/seed{seed}/` 目录下：

| 文件名 | 说明 |
|--------|------|
| `ckpt_init/` | θ0（初始模型检查点） |
| `ckpt_final/` | θ1（微调后最佳模型检查点） |
| `eval_subset.json` | 固定 eval subset 索引（1024 条，seed=999） |
| `importance_pre.jsonl` | 微调前重要性（ablation Δloss） |
| `gradfisher_pre.jsonl` | 微调前梯度与 Fisher proxy |
| `update.jsonl` | 更新量（θ1-θ0，head-level） |
| `importance_post.jsonl` | 微调后重要性 |
| `heads.csv` | 所有 head 指标汇总表 |
| `stats.json` | Spearman ρ、top-K overlap 等统计量 |
| `cases.json` | 反例集合 |
| `fig_I_vs_U.png` | 散点图：重要性 vs 更新量 |
| `fig_I_vs_G.png` | 散点图：重要性 vs 梯度 |
| `fig_stats.png` | 统计指标可视化 |

---

## 验收标准

### 1. 反例集合（cases.json）

`cases.json` 包含两类反例：

- **important-but-static**：`I_pre` top 10% 且 `Urel` bottom 30%
  - 说明：重要但不动的 head
- **plastic-but-unimportant**：`Urel` top 10% 且 `I_pre` bottom 30%
  - 说明：更新大但不重要的 head

示例：
```json
{
  "important_but_static": [
    {"layer": 5, "head": 3, "I_pre": 0.15, "Urel": 0.002, ...},
    ...
  ],
  "plastic_but_unimportant": [
    {"layer": 2, "head": 7, "I_pre": 0.001, "Urel": 0.08, ...},
    ...
  ],
  "thresholds": {
    "I_pre_p90": 0.12,
    "I_pre_p30": 0.01,
    "Urel_p90": 0.05,
    "Urel_p30": 0.002
  }
}
```

### 2. 统计指标（stats.json）

- `spearman_rho_Ipre_U`：Spearman 秩相关系数（I_pre vs U）
- `topk_overlap_Ipre_U`：top-K 重叠度（K=20）
- `n_important_but_static`：反例 A 数量
- `n_plastic_but_unimportant`：反例 B 数量

示例：
```json
{
  "spearman_rho_Ipre_U": 0.23,
  "spearman_rho_Ipre_Urel": 0.19,
  "topk_overlap_Ipre_U": 0.15,
  "topk_overlap_Ipre_Urel": 0.12,
  "topk": 20,
  "n_heads": 144,
  "n_important_but_static": 8,
  "n_plastic_but_unimportant": 12
}
```

### 3. 可视化图表

- **fig_I_vs_U.png**：散点图，X轴=I_pre，Y轴=Urel，用不同形状标记两类反例
- **fig_I_vs_G.png**：散点图，X轴=I_pre，Y轴=G（梯度），同样标记反例
- **fig_stats.png**：柱状图，展示 Spearman ρ 与 top-K overlap

---

## 常见问题排查

### 1. Gate hook 未生效

**症状**：`importance_pre.jsonl` 中所有 `I` 值接近 0，或 ablation 后 loss 没有明显变化。

**排查**：
- 检查 gate 是否正确注入：在 `deberta_head_gating.py` 中，hook 应该在 `model.deberta.encoder.layer[i].attention.self` 上注册。
- Sanity check：设置所有 gate=0（`gatewrap.gates.fill_(0.0)`），loss 应该大幅变坏。

### 2. View/Reshape 报错

**症状**：`RuntimeError: shape ... is invalid for input of size ...`

**原因**：`hidden_size` 不能被 `num_heads` 整除，或 hook 输出 shape 不匹配。

**排查**：
- 打印模型配置：`print(model.config.hidden_size, model.config.num_attention_heads)`
- DeBERTa-v3-base 默认：hidden_size=768, num_heads=12, head_dim=64

### 3. MNLI split 名称错误

**症状**：加载 eval 数据时报错 `Split validation not found`。

**解决**：MNLI 的 eval split 是 `validation_matched`（代码中已正确设置）。

### 4. BF16/FP16 切换失败

**症状**：训练时报错 `BFloat16 not supported`。

**排查**：
- 检查 GPU 是否支持 bf16：`torch.cuda.is_bf16_supported()`
- 如果不支持，代码会自动回退到 fp16 或 fp32。

### 5. 运行时间过长

**影响因素**：
- Eval subset 大小（默认 1024 条）
- Ablation 次数：L × H（DeBERTa-v3-base: 12 layers × 12 heads = 144 次）
- 每次 ablation 需要过一遍 eval subset

**优化建议**：
- 减少 eval subset 大小：`--n 512`（在 `make_subset.py` 中修改）
- 使用更小的 batch size：`--bsz 8`（加速单次前向传播）

### 6. 模型结构不匹配

**症状**：`AttributeError: 'DebertaV2Model' object has no attribute 'deberta'`

**排查**：
- 确认模型是 `AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")`
- 打印模型结构：`print(model)`，确认路径：
  ```
  model.deberta.encoder.layer[i].attention.self.query_proj
  model.deberta.encoder.layer[i].attention.self.key_proj
  model.deberta.encoder.layer[i].attention.self.value_proj
  model.deberta.encoder.layer[i].attention.output.dense
  ```

---

## 项目结构

```
minimal-exp/
├── requirements.txt
├── README.md
├── scripts/
│   ├── run_mnli.sh          # 训练
│   ├── measure_mnli.sh      # 测量
│   └── make_plots.sh        # 汇总与可视化
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── glue.py          # GLUE 数据加载
│   ├── model/
│   │   ├── __init__.py
│   │   └── deberta_head_gating.py  # Head gate 注入
│   ├── train/
│   │   ├── __init__.py
│   │   └── finetune_glue.py        # 训练主脚本
│   ├── measure/
│   │   ├── __init__.py
│   │   ├── importance_ablation.py  # 重要性测量
│   │   ├── grad_fisher_gate.py     # 梯度与 Fisher
│   │   └── update_magnitude.py     # 更新量
│   └── analysis/
│       ├── __init__.py
│       ├── make_subset.py          # 固定 eval subset
│       ├── aggregate.py            # 汇总 JSONL → CSV
│       └── plots.py                # 可视化
└── outputs/                        # 自动生成
    └── MNLI/
        └── seed1/
            ├── ckpt_init/
            ├── ckpt_final/
            ├── eval_subset.json
            ├── importance_pre.jsonl
            ├── gradfisher_pre.jsonl
            ├── update.jsonl
            ├── importance_post.jsonl
            ├── heads.csv
            ├── stats.json
            ├── cases.json
            ├── fig_I_vs_U.png
            ├── fig_I_vs_G.png
            └── fig_stats.png
```

---

## 技术细节

### 1. Head 粒度的定义

在 DeBERTa-v3-base 中：
- **Q/K/V 投影**：`query_proj`, `key_proj`, `value_proj` 的 weight shape 为 `[hidden_size, input_dim]`
  - Head 粒度：切分 **out_features 维度**（每个 head 产生 `head_dim` 个输出）
- **O 投影**（output.dense）：weight shape 为 `[hidden_size, hidden_size]`
  - Head 粒度：切分 **in_features 维度**（混合拼接后的 head 输出）

### 2. 重要性测量（Ablation）

- 通过 gate hook 动态置零某个 head 的输出
- 计算 ablation 后的 loss 变化：`I = loss_ablate - loss_base`
- 固定 eval subset（1024 条，seed=999）确保可重复性

### 3. 可塑性 Proxy

- **梯度幅值（G）**：`mean(|∂L/∂gate|)` over eval subset
- **Fisher 近似（F）**：`mean((∂L/∂gate)^2)`
- **预测可塑性（Ppred）**：`G^2 / (F + ε)`

### 4. 更新量（Update）

- 对每个 head，计算 Q/K/V/O 切片的 Frobenius 范数差：
  - `U = sqrt(Uq^2 + Uk^2 + Uv^2 + Uo^2)`
- 相对更新量（Urel）：归一化到初始切片范数之和
  - `Urel = U / (||Wq0|| + ||Wk0|| + ||Wv0|| + ||Wo0||)`

