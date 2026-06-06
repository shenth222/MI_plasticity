# IPD-GP-LoRA：Goodput-aware IPD-LoRA

> 在固定训练预算下，最大化“单位预算带来的有效学习进展”（Learning Progress per Adaptation Budget），
> 而不是单纯追求最终 Accuracy。本项目在原 IPD-LoRA（Importance–Plasticity Dynamics LoRA）基础上，
> 引入受 **SLAQ** 与 **Pollux** 的 *Learning Goodput* 思想启发的新指标 **Module Learning Goodput**，
> 并据此实现 Goodput 感知的动态 Rank 分配策略。

---

## 1. 项目概览

本目录在 DeBERTa-v3-base + GLUE 上做参数高效微调（PEFT），核心是把每个 LoRA 模块视为可独立调度的“学习单元”，
在训练过程中动态测量每个模块的：

- **I（Importance / 重要性）**：前向消融损失增量，`I_m = L_calib(module off) − L_calib(normal)`。
- **P（Plasticity / 可塑性）**：把训练资源转化为有效适应的能力（多信号代理比值）。
- **G（Goodput / 学习吞吐，本次新增）**：单位 rank-step 预算带来的验证损失下降。

并在固定总 Rank 预算下，按这些信号动态地给每个模块分配 rank、设置更新频率、必要时冻结/解冻。

### 文件结构

| 文件 | 作用 |
| --- | --- |
| `ipd_lora.py` | 核心库：动态 rank LoRA 模块、I/P/G 度量、象限 & Goodput 两套 rank 分配、early-stop。 |
| `train_ipd_lora.py` | 训练编排：注入 LoRA、训练循环、评分事件、Goodput 估计、评估与日志。 |
| `plot_ipd_results.py` | 可视化：训练曲线、Goodput 曲线、I/P/G 相关性、I-P-G 象限图、效率汇总指标。 |
| `run_gp_examples.sh` | IPD-GP-LoRA 示例脚本（proxy / probing / baseline 三种模式）。 |
| `run_train_examples.sh` / `rte.sh` | 原始 IPD-LoRA 示例脚本。 |
| `PARAMETERS.md` | 全部命令行参数说明（含本次新增的 Goodput 参数）。 |

---

## 2. Module Learning Goodput 定义

对每个 LoRA 模块 m：

```
G_m(t) = ΔQ_m(t) / ΔC_m(t)
```

- **质量提升 ΔQ_m**：验证集 Loss Reduction，`ΔQ_m = L_val_before − L_val_after`。
- **预算成本 ΔC_m**：`ΔC_m = r_m × steps_m`（当前 rank × 当前阶段训练步数）。

最终：

```
G_m(t) = (L_val_before − L_val_after) / (r_m · steps_m)
```

> 含义：模块 m 每消耗一个 rank-step 能带来多少验证集损失下降。

由于真实 Goodput 难以直接精确计算，项目实现了两种近似估计方法（可通过 `--goodput_method` 切换）。

### 方案 A：在线代理 Goodput（低成本，`proxy`）

把窗口内的**全局**验证损失下降按一个 I/P 派生的 share 分配到各模块，再除以 rank-step 成本：

```
s_m(t)     = norm_I_m^α · norm_P_m^β / Σ_j (norm_I_j^α · norm_P_j^β)
Ĝ_m(t)     = ΔL_val(t−K, t) · s_m(t) / (r_m · K)
```

- 优点：几乎不增加计算开销（复用已维护的 I/P 信号）。
- 缺点：不是严格的因果收益（是按权重分摊的全局改进）。
- 实现：`ipd_lora.compute_proxy_goodput(...)`。`norm_I/norm_P` 采用 min-max 归一化到 `[0,1]` 并加 `1e-3` 下限避免完全置零；`ΔL_val` 用 calibration 集损失在相邻 Goodput 事件间的差分近似。

### 方案 B：One-Step Probing Goodput（主实验推荐，`probing`）

每个评估窗口，对每个模块 m：

1. 记录 baseline 验证损失 `L_val(θ_t)`；
2. 冻结其它模块，仅让模块 m 走一步；
3. 测验证损失 `L_val(θ_t − η·g_m)`；
4. 回滚参数。

```
Ĝ_m(t) = (L_val(θ_t) − L_val(θ_t − η·g_m)) / r_m
```

- 实现：`ipd_lora.compute_probing_goodput(...)`。
- **关键工程取舍**：探针**复用主训练反向传播已经产生的梯度**（不额外做 backward），且只临时改写 LoRA 参数并立即回滚、**不触碰梯度**，因此既不破坏待执行的 `optimizer.step()`，也避免重复 backward。损失在 calibration loader 上测量；可选 `--goodput_probe_use_adam` 用 Adam 预条件方向代替原始梯度。
- 成本：每个 Goodput 事件 = 1 次 baseline 评估 + 每个被探针模块 1 次评估。可用 `--goodput_every_n_scoring` 与 `--goodput_probe_max_batches` 控制开销。

---

## 3. IPD-GP-LoRA 训练流程

```
Step 1  初始化所有模块，rank = 初始 active rank
Step 2  每 K steps（score_interval）：计算 I 与 P
Step 3  每 E 个评分事件（goodput_every_n_scoring）：估计 Goodput
Step 4  根据 Goodput 重新分配 Rank
Step 5  固定总预算，继续训练
```

在 `train_ipd_lora.py` 的评分事件内，顺序为：

1. `compute_plasticity_scores`（读取主反向的梯度，计算 P）；
2. `compute_importance_scores`（前向消融，计算 I）；
3. **Goodput 估计**（proxy 或 probing，更新 `ema_G`、`G_z`、`G_rank`）——放在 rank 重分配**之前**，让分配用上最新 G，并让 probing 观察到产生当前梯度的状态；
4. `update_quadrants_and_budget`（始终执行，保留象限标签 / 更新周期，供分析与 early-stop）；
5. 若 `--rank_alloc_mode goodput`：`update_goodput_rank_allocation` 覆盖 rank；
6. early-stop、写日志。

---

## 4. Rank Allocation（Goodput 感知）

综合评分（每项先在活跃模块间 min-max 归一化到 `[0,1]`，记为 `~`）：

```
S_m = λ_G · G̃_m + λ_I · Ĩ_m + λ_P · P̃_m
```

便捷模式（`--goodput_score_mode`）：

- `G`：`S_m = G̃_m`（Goodput 已直接刻画收益，避免与 I/P 过度重叠）。
- `GxI`：`S_m = G̃_m · Ĩ_m`（收益按重要性加权）。
- `combo`：上面的 λ 加权和。

分配公式（保底 `r_min` + 按 `S_m` 比例瓜分剩余预算）：

```
r_m = r_min + floor( B · S_m / Σ_j S_j )
B   = 总 Rank 预算 − 冻结模块占用 − r_min · n_active
```

- 实现：`ipd_lora.update_goodput_rank_allocation(...)`。
- 预算严格守恒：先地板取整，再把余量按小数部分从大到小补给；冻结模块保持当前 rank 且不参与瓜分；rank 经 `active_rank_choices` 与 `max_rank` 约束。
- 该函数只改写 `target_rank/active_rank`，不动象限标签与更新周期，因此与原 IPD 调度兼容。

---

## 5. 评估指标

训练中（`train_ipd_lora.py`）每个评估点记录到 `training_log.jsonl` 的 `global_goodput`：

| 指标 | 含义 |
| --- | --- |
| `goodput/rank_step` | `ΔValLoss / Rank-Step` |
| `goodput/per_second` | `ΔValLoss / Second`（Time Goodput） |
| `goodput/per_flop` | `ΔValLoss / FLOPs`（FLOP Goodput，按活跃 LoRA cost 近似） |

其中 `ΔValLoss = L_val(prev_eval) − L_val(cur_eval)`；rank-step / FLOP 为相邻评估点之间的累计消耗，wall-clock 为评估点间真实耗时。

绘图脚本（`plot_ipd_results.py`）额外汇总到 `efficiency_metrics.json`：

- **Final / Best Accuracy**
- **AULC**（Area Under Learning Curve，按 step 归一化的曲线下面积）
- **Steps-to-Threshold**（首次达到 `90% × best_acc` 的 step）
- **Accuracy @ Fixed Steps**（25% / 50% / 75% / 100% 训练进度处）
- **全局 Goodput 均值**（rank-step / second / flop）

---

## 6. 关键可视化

| 文件 | 对应 | 内容 / 预期 |
| --- | --- | --- |
| `eval_curve.png` | 图1 | 训练曲线（Accuracy / Loss）。验证 IPD-GP-LoRA 收敛更快。 |
| `goodput_curve.png` | 图2 | 平均 Module Learning Goodput 随 step。验证单位预算收益更高。 |
| `global_goodput_curve.png` | 补充 | 全局 Goodput / Time / FLOP 三联图。 |
| `ipg_correlation.png` | 图3 | `Spearman(I,G)` / `Spearman(P,G)` / `Spearman(I×P,G)`。预期 `corr(I,G)` 低、`corr(P,G)` 中、`corr(I×P,G)` 最高。 |
| `ipg_quadrant.png` | 图4 | X=Importance，Y=Plasticity，颜色=Goodput。预期高 I 高 P → Goodput 最高；低 I 低 P → 最低。 |

---

## 7. 快速开始

环境：使用 conda 中的 `MI` 环境。

```bash
# 方案 A（低成本在线代理 goodput）+ goodput 分配
bash run_gp_examples.sh proxy

# 方案 B（One-Step Probing，主实验推荐）+ goodput 分配
bash run_gp_examples.sh probing

# 对照组：仍记录 goodput，但保持原 quadrant 分配
bash run_gp_examples.sh baseline

# 生成全部图表与效率汇总指标
python plot_ipd_results.py --output_dir outputs/<your_run_dir>
```

最小可用命令示例：

```bash
python train_ipd_lora.py \
  --task_name rte \
  --dataset_path /data/shenth/datasets/glue \
  --model_name_or_path /data/shenth/models/deberta/v3-base \
  --output_dir outputs/ipd_gp_lora_rte \
  --target_rank 6 --max_lora_rank 12 \
  --score_interval 100 --warmup_steps_for_ipd 100 \
  --enable_goodput --goodput_method probing \
  --rank_alloc_mode goodput --goodput_score_mode G \
  --evaluation_strategy epoch
```

> 关闭 `--enable_goodput` 或设 `--rank_alloc_mode quadrant` 即退化为原始 IPD-LoRA，便于做消融对照。

---

## 8. 与原 IPD-LoRA 的兼容性

- 所有 Goodput 功能默认关闭，不影响原有训练行为与产物。
- `IPDLoRALinear` 新增 `current_G / ema_G / G_z / G_rank / current_share / cumulative_update_steps` 状态，
  并已纳入 `collect_module_rows`、运行时 `snapshot/restore`，因此 best/final 模型恢复时 Goodput 状态一致。
- `module_scores.jsonl` 新增 `current_G / ema_G / G_z / G_rank / current_share` 列；旧的可视化函数对缺列做了空值保护。

---

## 9. 实现要点速查（代码索引）

| 功能 | 函数 / 位置 |
| --- | --- |
| Goodput 运行时状态 | `IPDLoRALinear.__init__`（`ipd_lora.py`） |
| 方案 A 在线代理 | `compute_proxy_goodput`（`ipd_lora.py`） |
| 方案 B One-Step Probing | `compute_probing_goodput`（`ipd_lora.py`） |
| G 的 z-score / 排名 | `finalize_goodput_stats`（`ipd_lora.py`） |
| Goodput 感知分配 | `update_goodput_rank_allocation`（`ipd_lora.py`） |
| 训练集成与模式切换 | `main()` 评分事件块（`train_ipd_lora.py`） |
| 全局 Goodput/Time/FLOP | `compute_global_goodput` + 评估分支（`train_ipd_lora.py`） |
| 图2/3/4 与效率指标 | `plot_goodput_curve` / `plot_ipg_correlation` / `plot_ipg_quadrant` / `compute_efficiency_metrics`（`plot_ipd_results.py`） |
