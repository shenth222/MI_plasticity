# IPD-LoRA 参数说明

本文档覆盖 `train_ipd_lora.py` 的全部参数，并补充 `plot_ipd_results.py` 与 `run_train_examples.sh` 的参数入口说明。

## 1. 数据与任务参数

- `--task_name`：任务名（必填），如 `mnli`/`rte`/`stsb`。
- `--dataset_name`：数据集名称，默认 `glue`。
- `--dataset_config_name`：数据集配置名（可选）。
- `--dataset_path`：本地 `load_from_disk` 数据路径（优先于线上数据集）。
- `--train_file`：本地训练文件路径（csv/json/jsonl）。
- `--validation_file`：本地验证文件路径（csv/json/jsonl）。
- `--local_train_split`：本地数据训练 split 名，默认 `train`。
- `--local_eval_split`：本地数据验证 split 名，默认 `validation`。
- `--text_column1`：文本字段1（可手动指定）。
- `--text_column2`：文本字段2（可手动指定）。
- `--label_column`：标签列名，默认 `label`。

## 2. 模型与输出参数

- `--model_name_or_path`：预训练模型路径或名称。
- `--output_dir`：输出目录（必填）。
- `--max_length`：最大序列长度。

## 3. 基础训练参数

- `--per_device_train_batch_size`：训练 batch size（每设备）。
- `--per_device_eval_batch_size`：评估 batch size（每设备）。
- `--learning_rate`：学习率。
- `--weight_decay`：权重衰减。
- `--num_train_epochs`：训练 epoch 数。
- `--warmup_ratio`：学习率 warmup 比例。
- `--seed`：随机种子。
- `--logging_steps`：训练日志步长。
- `--evaluation_strategy`：评估策略，`no|steps|epoch`（默认 `epoch`，与 AdaLoRA 对齐）。
- `--eval_steps`：仅在 `evaluation_strategy=steps` 时生效的评估步长。
- `--save_steps`：checkpoint 保存步长（`<=0` 禁用）。

## 4. LoRA 结构参数

- `--max_lora_rank`：LoRA 最大 rank。
- `--initial_active_rank`：初始化 active rank（实际会保证每个模块至少为 1）。
- `--lora_alpha`：LoRA alpha。
- `--lora_dropout`：LoRA dropout。

## 5. IPD 评分参数（含近似/稀疏）

- `--score_interval`：IPD 评分/重分配触发周期（step）。
- `--importance_update_interval`：I 分数更新稀疏周期（按评分事件计）。
- `--importance_exact_interval`：I 分数精确评估周期（其余周期用分组近似）。
- `--importance_group_size`：I 分组近似时每组模块数量。
- `--score_module_batch_size`：每次评分事件抽取参与 I 评估的模块数量。
- `--min_importance_scores_per_module`：每个模块在本次训练中至少应经历的 I 打分次数（默认 `2`）。  
  - 训练开始前会根据“可用 I 更新次数”自动计算最小分组规模；
  - 若 `importance_group_size` 已满足该覆盖要求则保持不变，否则自动增大到计算值。
- `--warmup_steps_for_ipd`：IPD 策略开始前 warmup 步数。
- `--calibration_size`：每次评分时随机抽样 calibration 样本数。
- `--calibration_resample_stride`：calibration 重采样随机步长（与 step 共同决定随机种子偏移）。
- `--calibration_max_batches`：单次 I 评估使用的 calibration 最大 batch 数。
- `--beta_I`：I 的 EMA 平滑系数。
- `--beta_P`：P 的 EMA 平滑系数。
- `--plasticity_task_weight`：P 计算中 `max(ema_I,0)` 任务改进代理项的权重（默认 `0.1`，用于降低 I->P 信息泄漏）。
- `--high_i_quantile`：定义 high-I 的分位数阈值（默认 `0.5`）。
- `--high_p_quantile`：定义 high-P 的分位数阈值（默认 `0.5`）。
- `--low_i_low_p_update_interval`：low-I/low-P 象限的更新周期（默认 `32`，不再默认“几乎永久不更新”）。

### I / P 实现定义（当前代码）

- `I`（Importance）定义：在 calibration 数据上的前向消融损失增量  
  `I_m = L_calib(module off) - L_calib(normal)`  
  - exact 模式：逐模块消融精确计算。  
  - grouped 模式：按组消融后按模块 proxy cost 分摊近似。  
  - 运行时维护 `current_I` 与 `ema_I`（由 `beta_I` 控制平滑）。

- `P`（Plasticity）定义：模块把训练资源转化为有效任务适应的能力  
  参考形式：`P_i = |Delta L_i| / (||Delta W_i|| + eps)`。  
  实际实现采用多信号代理比值：
  - 分子（有效适应）：  
    - 一阶预测 loss decrease（`-<g, Delta W>`）  
    - logit change 代理（`||g|| * ||Delta W||`）  
    - task improvement 代理（`max(ema_I, 0)`）  
    - representation adaptation 代理（`||Delta W||`）  
  - 分母（资源投入）：  
    - optimizer effort（`||adam_direction||`）  
    - parameter update（`||Delta W||`）  
    - training budget 代理（active LoRA cost）  
  - 运行时维护 `current_P` 与 `ema_P`（由 `beta_P` 控制平滑）。

## 6. Rank 预算与目标参数（AdaLoRA 风格）

- `--target_rank`：目标平均 rank（每模块）。
- `--total_rank_budget`：总 rank 预算上限。
  - 若 `>0`：实际预算 = `min(total_rank_budget, target_rank * 模块数)`。
  - 若 `=0`：实际预算 = `target_rank * 模块数`。
- `--high_i_min_rank`：高 I 模块保底 rank；`-1` 表示跟随 `target_rank`。
- `--avoid_zero_rank`：尽量避免模块 rank 为 0（默认开启）。

## 6.5 Module Learning Goodput 参数（IPD-GP-LoRA）

> 受 SLAQ / Pollux 的 Learning Goodput 思想启发：目标不是单纯提高最终 Accuracy，
> 而是在固定训练预算下最大化“单位预算带来的有效学习进展”。

- `--enable_goodput`：启用 Module Learning Goodput 估计（默认关闭，关闭时与原 IPD-LoRA 行为一致）。
- `--goodput_method`：Goodput 估计方法，`proxy|probing`。
  - `proxy`（方案 A，低成本在线代理）：把窗口内的全局验证损失下降按 I/P 派生的 share 分配到模块，再除以 rank-step 成本。
  - `probing`（方案 B，主实验推荐）：对每个模块做 One-Step Probing，单独让该模块走一步并测验证损失变化后回滚。
- `--rank_alloc_mode`：Rank 分配模式，`quadrant|goodput`。
  - `quadrant`：沿用原 IPD 象限分配。
  - `goodput`：在象限分配之后用综合评分 `S_m` 覆盖 rank（保留象限标签/更新周期用于分析与 early-stop）。
- `--goodput_every_n_scoring`：每 N 个评分事件估计一次 Goodput（即以“评分事件”为单位的 E 窗口，默认 `1`）。
- `--beta_G`：Goodput 的 EMA 平滑系数（默认 `0.9`）。
- `--goodput_alpha`：方案 A 中 `norm_I` 的指数 α（默认 `1.0`）。
- `--goodput_beta`：方案 A 中 `norm_P` 的指数 β（默认 `1.0`）。
- `--goodput_probe_max_batches`：方案 B 每次探针损失评估使用的最大 calibration batch 数（默认 `8`）。
- `--goodput_probe_use_adam`：方案 B 探针使用 Adam 预条件方向而非原始梯度（默认关闭）。
- `--goodput_score_mode`：分配评分 `S_m` 形式，`G|GxI|combo`。
  - `G`：`S_m = G̃_m`（Goodput 已直接刻画收益）。
  - `GxI`：`S_m = G̃_m · Ĩ_m`（收益按重要性加权）。
  - `combo`：`S_m = λ_G·G̃_m + λ_I·Ĩ_m + λ_P·P̃_m`。
- `--lambda_G` / `--lambda_I` / `--lambda_P`：`combo` 模式下各归一化项权重。
- `--goodput_min_rank`：`r_min`，goodput 分配时每个活跃模块的保底 rank（默认 `1`）。

### Goodput 定义（当前代码）

- 模块级 Goodput：`G_m = ΔQ_m / ΔC_m = (L_val_before - L_val_after) / (r_m · steps_m)`，含义为“模块 m 每消耗一个 rank-step 能带来多少验证集损失下降”。
- 方案 A：`Ĝ_m = ΔL_val(t-K,t) · s_m / (r_m · K)`，其中 `s_m = norm_I^α · norm_P^β / Σ_j(...)`。
- 方案 B：`Ĝ_m = (L_val(θ_t) - L_val(θ_t - η·g_m)) / r_m`（复用主反向传播的梯度，不额外做 backward，也不破坏当前 optimizer step）。
- 运行时维护 `current_G`、`ema_G`（由 `beta_G` 平滑），并在每个评分事件计算 `G_z`/`G_rank` 供可视化使用。

### 全局 Goodput 指标（每个评估点记录到 `training_log.jsonl` 的 `global_goodput`）

- `goodput/rank_step`：`ΔValLoss / Rank-Step`
- `goodput/per_second`：`ΔValLoss / Second`（Time Goodput）
- `goodput/per_flop`：`ΔValLoss / FLOPs`（FLOP Goodput，按活跃 LoRA cost 近似）

## 7. 收敛稳定参数（tfinal）

- `--tfinal_steps`：末尾固定 rank 微调步数（优先级高）。
- `--tfinal_ratio`：当 `tfinal_steps=0` 时，按总步数比例计算末尾固定 rank 步数。

## 8. Early Stop 与解冻参数

- `--early_stop_patience`：早停耐心值。
- `--early_stop_i_tolerance`：I 增长容忍阈值。
- `--early_stop_unfreeze_interval`：冻结后解冻周期（按评分事件周期换算）。
- `--early_stop_max_freeze_cycles`：模块最多冻结-解冻循环次数。
- `--early_stop_unfreeze_rank`：解冻后恢复的最小 rank。
- `--disable_module_early_stop`：关闭模块级 early-stop 冻结策略。

## 9. W&B 参数

- `--report_to_wandb`：是否启用 W&B 上报。
- `--wandb_project`：W&B 项目名。
- `--wandb_entity`：W&B 实体（可选）。
- `--wandb_run_name`：W&B 运行名（可选）。
- `--wandb_mode`：`online`/`offline`/`disabled`。

## 10. 评估与日志行为

- GLUE 任务：会上报评估器返回的全部指标，不只 accuracy。
- MNLI：只要数据中存在 `validation_matched`/`validation_mismatched`，就会同时上报两者所有指标到 W&B。
- best model 选择：
  - MNLI：以 `matched` split 的主指标为准。
  - 其他任务：以默认验证 split 主指标为准。

## 11. 绘图脚本参数（`plot_ipd_results.py`）

- `--output_dir`：读取训练输出并保存图像的目录。
- `--future_eval_points`：用于计算未来验证损失下降的评估点偏移。
- `--future_score_points`：用于计算未来 `ema_I` 增量的评分点偏移。

新增的 Goodput 相关产物：

- `goodput_curve.png`（图2）：平均 Module Learning Goodput 随 step 变化。
- `global_goodput_curve.png`：全局 Goodput / Time-Goodput / FLOP-Goodput 三联图。
- `ipg_correlation.png`（图3）：`Spearman(I,G)`、`Spearman(P,G)`、`Spearman(I×P,G)` 随 step 变化；`ipg_correlation_summary.csv` 给出整体相关性。
- `ipg_quadrant.png`（图4）：X=Importance、Y=Plasticity、颜色=Goodput 的四象限散点。
- `efficiency_metrics.json`：Final/Best Accuracy、AULC、Steps-to-Threshold(90%)、Accuracy@Fixed-Steps、全局 Goodput 均值。

## 12. 示例脚本入口

### `run_train_examples.sh`（原始 IPD-LoRA）

- `MODE=rte|mnli|local`：示例训练模式。
- `rte/mnli`：预置了对应任务参数模板。
- `local`：给出本地数据训练示例命令模板。

### `run_gp_examples.sh`（IPD-GP-LoRA）

- `bash run_gp_examples.sh proxy`：方案 A 在线代理 Goodput + goodput 分配（`GxI`）。
- `bash run_gp_examples.sh probing`：方案 B One-Step Probing Goodput + goodput 分配（`G`，主实验推荐）。
- `bash run_gp_examples.sh baseline`：仍记录 Goodput 但保持 quadrant 分配，作为对照组。
- 支持 `CUDA_VISIBLE_DEVICES` / `MODEL_PATH` / `DATA_ROOT` 环境变量覆盖。