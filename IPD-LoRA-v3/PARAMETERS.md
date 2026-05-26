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

## 12. 示例脚本入口（`run_train_examples.sh`）

- `MODE=rte|mnli|local`：示例训练模式。
- `rte/mnli`：预置了对应任务参数模板。
- `local`：给出本地数据训练示例命令模板。