# GP-only 关键实现说明

## 1. 训练循环不再计算 I/P

`train_ipd_lora.py` 的评分事件只做三件事：

1. 构造 calibration loader。
2. 调用 `compute_probing_goodput` 或 `compute_proxy_goodput` 更新 `current_G / ema_G`。
3. 调用 `update_goodput_rank_allocation` 仅按 `ema_G` 分配 rank。

因此原 `IPD-LoRA-GP` 中的这些路径已移除：

- `compute_importance_scores`
- `compute_plasticity_scores`
- `update_quadrants_and_budget`
- `apply_module_early_stopping`
- `rank_alloc_mode`
- `goodput_score_mode`
- `lambda_I / lambda_P`

## 2. Goodput 状态最小化

`IPDLoRALinear` 只保留 GP-only 需要的运行时状态：

```text
active_rank
target_rank
update_interval
current_G
ema_G
G_z
G_rank
current_share
```

`snapshot_ipd_runtime_state`、`restore_ipd_runtime_state`、`collect_module_rows` 也只保存/记录这些字段，避免 I/P 字段继续出现在日志和 checkpoint runtime state 中。

## 3. Probing Goodput

`compute_probing_goodput` 直接参考原 `IPD-LoRA-GP` 的 one-step probing 思路：

```text
baseline = L_calib(theta)
after_m  = L_calib(theta - lr * grad_m)
G_m      = (baseline - after_m) / active_rank_m
```

实现要点：

- 复用主训练 batch 的 backward 结果。
- 对每个模块只临时更新 LoRA 参数。
- 每个模块测完后立即回滚参数。
- 不调用 `optimizer.step()`，不清空梯度，不破坏随后主训练的 optimizer step。

## 4. Proxy Goodput 不再依赖 I/P

原 proxy goodput 使用 `norm_I^alpha * norm_P^beta` 分摊全局窗口 loss reduction。GP-only 版本改为梯度一阶进展代理：

```text
proxy_m = max(sum_p -<grad_p, -lr * direction_p>, 0)
share_m = proxy_m / sum_j proxy_j
G_m     = delta_val_loss * share_m / (active_rank_m * interval_steps)
```

这样 proxy 仍然是低成本在线估计，但不再需要 I/P。`direction_p` 默认是原始梯度；启用 `--goodput_probe_use_adam` 后可使用 Adam 预条件方向。

## 5. Goodput-only Rank 分配

`update_goodput_rank_allocation` 的输入分数只有 `ema_G`：

```text
S_m = minmax(ema_G_m)
```

随后在固定预算下分配：

```text
distributable = total_rank_budget - goodput_min_rank * num_modules
rank_m = goodput_min_rank + floor(distributable * S_m / sum(S))
```

预算余数按小数部分补齐，保证总 rank 尽量贴近预算；最终 rank 受 `active_rank_choices` 和 `max_lora_rank` 约束。

## 6. 全局 Goodput 保留

训练评估分支仍保留全局 goodput 统计：

```text
goodput/rank_step = delta_eval_loss / delta_rank_steps
goodput/per_second = delta_eval_loss / wall_time
goodput/per_flop = delta_eval_loss / active_lora_cost_proxy
```

这些指标写入 `training_log.jsonl`，用于比较不同 rank 分配策略的整体效率。

## 7. 显存优化实现

本版本增加了几处针对 GPU 峰值显存的优化：

- `_avg_loss_over_loader` 使用 `torch.no_grad()` 包裹 calibration / probing loss forward，避免评估时构建计算图。
- `compute_probing_goodput` 将每个模块的临时 LoRA 参数备份保存到 CPU，回滚时再拷回 GPU。
- `train_ipd_lora.py` 新增 `--gradient_checkpointing`，启用后调用 HuggingFace 模型的 `gradient_checkpointing_enable()`，并配合 `enable_input_require_grads()` 支持 LoRA-only 训练。
- best checkpoint 的内存副本统一使用 `cpu_state_dict(model)`，避免 `deepcopy(model.state_dict())` 在 GPU 上复制整模型。
- `run_goodput_only.sh` 默认启用 `--bf16`、`--gradient_checkpointing`，并把默认训练/评估 batch 和 calibration size 调低；可用 `TRAIN_BATCH_SIZE`、`EVAL_BATCH_SIZE`、`CALIBRATION_SIZE` 覆盖。

## 8. 画图设计

GP-only 的实验目的不是验证 I/P 象限，而是验证“只用 Goodput 能否把 rank 分给单位预算收益更高的模块”。因此 `plot_gp_only_results.py` 保留并扩展了原 `IPD-LoRA-GP` 中与效率相关的图，同时移除 I/P scatter、I/P correlation、quadrant distribution。

核心图表对应的实验问题：

- `goodput_curve.png`：模块平均 Goodput 是否随训练稳定，proxy/probing 是否有可比较的趋势。
- `global_goodput_curve.png`：整体训练在 rank-step、时间和 FLOP proxy 上的收益。
- `module_goodput_heatmap.png`：哪些层/投影长期贡献较高 Goodput。
- `rank_heatmap.png`：rank 是否随训练被动态迁移，而不是静态平均分配。
- `goodput_rank_alignment.png`：`ema_G` 与 `active_rank` 的 Spearman 相关性，直接检验 rank 分配是否跟随 Goodput。
- `final_goodput_rank_scatter.png`：最终时刻高 Goodput 模块是否获得更高 rank。
- `rank_budget_efficiency.png`：评估性能提升是否发生在可控 rank 预算下。
- `efficiency_metrics.json`：保存 AULC、best/final accuracy、mean global goodput，便于跨 run 比较。
