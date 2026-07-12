# Goodput-LoRA：统一 LoRA 预算分配实验框架

本目录是论文工作的**规范代码库**（canonical harness）。研究定位与实验设计见
[`docs/2026-06-28-research-proposal.md`](docs/2026-06-28-research-proposal.md)。

核心论点：把 LoRA 的 rank 预算分配统一在「单位预算学习进展（Goodput）」视角下，
沿**分配时机**坐标轴比较不同策略，并以极简在线信号在固定预算下达到 baseline 级精度、
更优的 Goodput（rank-step / wall-clock / FLOP）。

## 方法（`--method`，分配时机轴）

| method | 分配时机 | 说明 |
| --- | --- | --- |
| `lora` | 无（静态） | 所有模块固定均匀 rank（vanilla LoRA，预算对齐参照） |
| `gora` | 训练前一次性 | 用预训练梯度重要性 `|W·∇W|` 一次性分配 rank 后冻结（GoRA 式） |
| `goodput` | 训练中在线 | warmup 后周期性估计 Module Goodput 并在固定预算下动态重分配 |

三者共用同一套数据加载 / 评估 / 日志 / 预算记账，保证可比性。
（`adalora`=训练中复杂派、`para`=训练后剪枝，作为外部 baseline 后续接入。）

## 目录结构

```
.
├── train_ipd_lora.py        # 统一训练入口（--method 分发）
├── ipd_lora.py              # 核心库：动态 rank LoRA、Goodput、各分配策略
├── plot_gp_only_results.py  # 结果可视化
├── docs/                    # proposal、实现说明、历史 README
├── scripts/
│   ├── run_glue.sh          # 单次运行：bash scripts/run_glue.sh <method> <task> [seed]
│   ├── run_multiseed.sh     # 方法×seed 扫描，多 GPU 并行
│   └── legacy/              # 旧 GP-only 单任务脚本（仅存档参考，路径已失效）
└── outputs/                 # 实验输出（gitignore）；旧 GP-only run 仍保留其中
```

> 同级 `../archive/` 存放已被取代的历史版本：`IPD-LoRA`、`IPD-LoRA-v3`、`IPD-LoRA-GP`。

## 快速开始

```bash
# 单次运行（GPU 0，RTE，在线 goodput）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_glue.sh goodput rte 42

# 同预算 baseline 对照
CUDA_VISIBLE_DEVICES=1 bash scripts/run_glue.sh lora rte 42
CUDA_VISIBLE_DEVICES=2 bash scripts/run_glue.sh gora rte 42

# 方法×seed 扫描（4 卡并行）
bash scripts/run_multiseed.sh rte "lora gora goodput" "42 1 2" "0 1 2 3"
```

公共预算：`--target_rank 6 --max_lora_rank 12`（三方法总 rank 预算一致）。
默认 `WANDB_MODE=disabled`，需要上报时设 `WANDB_MODE=online`。

## 评测产物

每个 run 的 `output_dir` 下：`eval_results.json`（best/final 指标、rank 预算、参数量、
`method`/`rank_allocation`）、`training_log.jsonl`（含 `global_goodput`）、
`module_scores.jsonl`、`rank_history.csv`、`best_model/`、`final_model/`。
